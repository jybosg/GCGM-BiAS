import torch.cuda
import torch.optim as optim
import time
import xlwt
from datetime import datetime, timedelta
from pathlib import Path
from tensorboardX import SummaryWriter

from src.dataset.data_loader import GMDataset, get_dataloader
from src.displacement_layer import Displacement
from src.loss_func import *
from src.evaluation_metric import matching_accuracy, matching_f1
from src.parallel import DataParallel
from src.utils.model_sl import load_model, save_model
from eval_baseline import eval_model
from src.utils.data_to_cuda import data_to_cuda

from src.utils.config import cfg

# * wandb
import wandb
from wandb import AlertLevel
import math

# * reproductivity
import random
import numpy as np

def train_eval_model(cfg,
                     model,
                     criterion,
                     optimizer,
                     dataloader,
                     tfboard_writer,
                     run,
                     dataset_len,
                     data_splits,
                     num_epochs=25,
                     start_epoch=0,
                     xls_wb=None):
    print('Start training...')

    since = time.time()
    displacement = Displacement()

    try: 
        device = next(model.parameters()).device
        print('model on device: {}'.format(device))
    except:
        # * LF methods and some other methods with no parameters
        device = torch.device("cuda:{}".format(cfg.GPUS[0]) if torch.cuda.is_available else "cpu")

    checkpoint_path = Path(cfg.OUTPUT_PATH) / 'params'
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True)

    model_path, optim_path = '',''
    if start_epoch > 0:
        model_path = str(checkpoint_path / 'params_{:04}.pt'.format(start_epoch))
        optim_path = str(checkpoint_path / 'optim_{:04}.pt'.format(start_epoch))
    if len(cfg.PRETRAINED_PATH) > 0:
        model_path = cfg.PRETRAINED_PATH
    if len(model_path) > 0:
        print('Loading model parameters from {}'.format(model_path))
        load_model(model, model_path, strict=False)
    if optimizer is not None:
        if len(optim_path) > 0:
            print('Loading optimizer state from {}'.format(optim_path))
            optimizer.load_state_dict(torch.load(optim_path))
    
    if optimizer is not None:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                milestones=cfg.TRAIN.LR_STEP,
                                                gamma=cfg.TRAIN.LR_DECAY,
                                                last_epoch=-1)
    else:
        scheduler = None
        
    # * early stopping
    val_acc_history = []

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        if cfg.TRAIN.LR != 0:
            model.train()  # Set model to training mode
        else:
            model.eval() # Evaluate

        if optimizer is not None:
            print('lr = ' + ', '.join(['{:.2e}'.format(x['lr']) for x in optimizer.param_groups]))

        epoch_loss = 0.0
        epoch_cl_loss = 0.0
        running_loss = 0.0
        running_since = time.time()
        training_start = time.time()
        iter_num = 0

        # Iterate over data.
        batch_size = cfg.BATCH_SIZE
        EPOCH_ITERS = dataset_len.get('train', 0) // batch_size
        if dataloader.get('train') != None:
            for inputs in dataloader['train']:
                if iter_num >= EPOCH_ITERS:
                    break
                    
                if device != torch.device('cpu'):
                    inputs = data_to_cuda(inputs)

                iter_num = iter_num + 1

                # zero the parameter gradients
                if optimizer is not None:
                    optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    # forward
                    outputs = model(inputs)

                    if cfg.PROBLEM.TYPE in ['2GM', 'GCL']:
                        assert 'ds_mat' in outputs
                        assert 'perm_mat' in outputs
                        assert 'gt_perm_mat' in outputs

                        # compute loss
                        if cfg.TRAIN.LOSS_FUNC[-1] == 'offset':
                            d_gt, grad_mask = displacement(outputs['gt_perm_mat'], *outputs['Ps'], outputs['ns'][0])
                            d_pred, _ = displacement(outputs['ds_mat'], *outputs['Ps'], outputs['ns'][0])
                            loss = criterion(d_pred, d_gt, grad_mask)
                        elif cfg.TRAIN.LOSS_FUNC[-1] in ['perm', 'ce', 'hung']:
                            if cfg.PROBLEM.SSL and not cfg.SSL.MIX_DETACH:
                                loss_old = criterion(outputs['ds_mat'], outputs['gt_perm_mat_old'], *outputs['ns'])
                                loss_new = criterion(outputs['ds_mat'], outputs['gt_perm_mat_new'], *outputs['ns'])
                                loss = loss_old * (1 - cfg.SSL.MIX_RATE) + loss_new * cfg.SSL.MIX_RATE
                            else:
                                loss = criterion(outputs['ds_mat'], outputs['gt_perm_mat'], *outputs['ns'])
                        elif cfg.TRAIN.LOSS_FUNC[-1] == 'hamming':
                            if cfg.PROBLEM.SSL and not cfg.SSL.MIX_DETACH:
                                loss_old = criterion(outputs['perm_mat'], outputs['gt_perm_mat_old'])
                                loss_new = criterion(outputs['perm_mat'], outputs['gt_perm_mat_new'])
                                loss = loss_old * (1 - cfg.SSL.MIX_RATE) + loss_new * cfg.SSL.MIX_RATE
                            else:
                                loss = criterion(outputs['perm_mat'], outputs['gt_perm_mat'])
                        elif cfg.TRAIN.LOSS_FUNC[-1] == 'plain':
                            loss = torch.sum(outputs['loss'])
                        else:
                            raise ValueError('Unsupported loss function {} for problem type {}'.format(cfg.TRAIN.LOSS_FUNC,
                                                                                                    cfg.PROBLEM.TYPE))

                        if cfg.PROBLEM.SSL and cfg.SSL.C_LOSS:
                            loss += outputs['c_loss'] * cfg.SSL.C_LOSS_RATE
                            cl_loss = outputs['c_loss'] * cfg.SSL.C_LOSS_RATE

                        # compute accuracy
                        if cfg.PROBLEM.SSL and not cfg.SSL.MIX_DETACH:
                            gt_mat = outputs['gt_perm_mat_old']
                        else:
                            gt_mat = outputs['gt_perm_mat']
                        acc = matching_accuracy(outputs['perm_mat'], gt_mat, outputs['ns'][0])

                    elif cfg.PROBLEM.TYPE in ['MGM', 'MGMC']:
                        assert 'ds_mat_list' in outputs
                        assert 'graph_indices' in outputs
                        assert 'perm_mat_list' in outputs
                        assert 'gt_perm_mat_list' in outputs

                        # compute loss & accuracy
                        if cfg.TRAIN.LOSS_FUNC[-1] in ['perm', 'ce' 'hung']:
                            # loss = torch.zeros(1, device=model.module.device)
                            loss = torch.zeros(1, device=device)
                            ns = outputs['ns']
                            for s_pred, x_gt, (idx_src, idx_tgt) in \
                                    zip(outputs['ds_mat_list'], outputs['gt_perm_mat_list'], outputs['graph_indices']):
                                l = criterion(s_pred, x_gt, ns[idx_src], ns[idx_tgt])
                                loss += l
                            loss /= len(outputs['ds_mat_list'])
                        elif cfg.TRAIN.LOSS_FUNC[-1] == 'plain':
                            loss = torch.sum(outputs['loss'])
                        else:
                            raise ValueError('Unsupported loss function {} for problem type {}'.format(cfg.TRAIN.LOSS_FUNC, cfg.PROBLEM.TYPE))

                        # compute accuracy
                        acc = torch.zeros(1, device=model.module.device)
                        for x_pred, x_gt, (idx_src, idx_tgt) in \
                                zip(outputs['perm_mat_list'], outputs['gt_perm_mat_list'], outputs['graph_indices']):
                            a = matching_accuracy(x_pred, x_gt, ns[idx_src])
                            acc += torch.sum(a)
                        acc /= len(outputs['perm_mat_list'])
                    else:
                        raise ValueError('Unknown problem type {}'.format(cfg.PROBLEM.TYPE))

                    # backward + optimize
                    if optimizer is not None:
                        if cfg.FP16:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()
                        optimizer.step()

                    batch_num = inputs['batch_size']

                    # tfboard writer
                    loss_dict = dict()
                    loss_dict['loss'] = loss.item()
                    # tfboard_writer.add_scalars('loss', loss_dict, epoch * cfg.TRAIN.EPOCH_ITERS + iter_num)
                    tfboard_writer.add_scalars('loss', loss_dict, EPOCH_ITERS + iter_num)

                    accdict = dict()
                    accdict['matching accuracy'] = torch.mean(acc)
                    tfboard_writer.add_scalars(
                        'training accuracy',
                        accdict,
                        # epoch * cfg.TRAIN.EPOCH_ITERS + iter_num
                        epoch * EPOCH_ITERS + iter_num
                    )

                    # statistics
                    running_loss += loss.item() * batch_num
                    epoch_loss += loss.item() * batch_num
                    epoch_cl_loss += cl_loss.item() * batch_num if cfg.PROBLEM.SSL and cfg.SSL.C_LOSS else 0
                    
                    # * wandb
                    run.log({"running_loss": running_loss})
                    # * log acc
                    run.log({"matching_accuracy": accdict['matching accuracy']})

                    if iter_num % cfg.STATISTIC_STEP == 0:
                        running_speed = cfg.STATISTIC_STEP * batch_num / (time.time() - running_since)
                        print('Epoch {:<4} Iteration {:<4} {:>4.2f}sample/s Loss={:<8.4f}'
                            .format(epoch, iter_num, running_speed, running_loss / cfg.STATISTIC_STEP / batch_num))
                        tfboard_writer.add_scalars(
                            'speed',
                            {'speed': running_speed},
                            # epoch * cfg.TRAIN.EPOCH_ITERS + iter_num
                            epoch * EPOCH_ITERS + iter_num
                        )
                        
                        if optimizer is not None:
                            tfboard_writer.add_scalars(
                                'learning rate',
                                {'lr_{}'.format(i): x['lr'] for i, x in enumerate(optimizer.param_groups)},
                                # epoch * cfg.TRAIN.EPOCH_ITERS + iter_num
                                epoch * EPOCH_ITERS + iter_num
                            )

                        running_loss = 0.0
                        running_since = time.time()

            # epoch_loss = epoch_loss / (cfg.TRAIN.SPLIT * cfg.TRAIN.EPOCH_ITERS) / batch_num
            # epoch_cl_loss = epoch_cl_loss / (cfg.TRAIN.SPLIT * cfg.TRAIN.EPOCH_ITERS) / batch_num
            # * wandb
            if dataset_len['train'] > 0:
                epoch_loss = epoch_loss / dataset_len['train'] 
                epoch_cl_loss = epoch_cl_loss / dataset_len['train'] 
            run.log({"loss": epoch_loss})
            if cfg.PROBLEM.TYPE == 'GCL':
                run.log({"cl_loss": epoch_cl_loss})
                
            print('Epoch {:<4} Loss: {:.4f}'.format(epoch, epoch_loss))
            # * runtime for each training epoch
            train_elapsed = time.time() - training_start
            print('Epoch {:<4} Finished in {:.0f}m {:.0f}s'.format(epoch, *divmod(train_elapsed, 60)))
            print('Now: {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M")))
            if cfg.PROBLEM.TYPE == 'GCL':
                print('Estimated complete time: {}'.format((datetime.now() + timedelta(seconds=train_elapsed * (num_epochs - epoch - 1))).strftime("%Y-%m-%d %H:%M")))
            print()
        else:
            print('No training set provided')
            print()
        
        if (epoch + 1) % cfg.TRAIN.PATIENCE == 0:
            # * validate every cfg.TRAIN.PATIENCE epochs
            if 'validation' in data_splits:
                print('Start validation...')
                # * validation set
                val_elapsed = 0
                val_acc = 0.0
                val_iter_num = 0
                val_start = time.time()
                was_training = model.training
                model.eval()
                
                for inputs in dataloader['validation']:
                    # if val_iter_num >= (1 - cfg.TRAIN.SPLIT) * cfg.TRAIN.EPOCH_ITERS:
                    if val_iter_num >= dataset_len['validation'] // cfg.BATCH_SIZE:
                        break
                    
                    if device != torch.device('cpu'):
                        inputs = data_to_cuda(inputs)
                    
                    val_iter_num += 1
                    
                    # * evaluation using matching accuracy, precision, and recall (the latter two are used when there are outliers)
                    with torch.set_grad_enabled(False):
                        outputs = model(inputs)
                        gt_mat = outputs['gt_perm_mat']
                        f1_score = matching_f1(outputs['perm_mat'], gt_mat, outputs['ns'][0])
                        val_acc += torch.sum(f1_score).item()
                        
                val_elapsed = time.time() - val_start
                model.train(mode=was_training)
                # * validation acc
                # val_acc = val_acc / ((1 - cfg.TRAIN.SPLIT) * cfg.TRAIN.EPOCH_ITERS) / batch_num
                val_acc = val_acc / dataset_len['validation']
                run.log({"val_acc": val_acc})
                print('Epoch {:<4} Validation F1 score: {:.4f}'.format(epoch, val_acc))
                print('Epoch {:<4} Validation finished in {:.0f}m {:.0f}s'.format(epoch, *divmod(val_elapsed, 60)))
                print('Now: {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M")))
                print()
                
                # * save the weights of the model
                if len(model_path) == 0:
                    save_model(model, str(checkpoint_path / 'params_{:04}.pt'.format(epoch + 1)))
                # * early-stopping based on validation accuracy
                if len(val_acc_history) >= 1 and val_acc - val_acc_history[-1] < cfg.TRAIN.DELTA:
                    print('Early stopping at epoch {}'.format(epoch))
                    val_acc_history.append(val_acc)
                    break
                val_acc_history.append(val_acc)
                
                # * remove the weights of the model if the validation accuracy is improved
                saved_epoch = epoch
                if epoch > 0:
                    try:
                        os.remove(str(checkpoint_path / 'params_{:04}.pt'.format(epoch + 1 - cfg.TRAIN.PATIENCE)))
                    except:
                        pass
            else: # if no validation set is provided
                # * save the weights of the model
                save_model(model, str(checkpoint_path / 'params_{:04}.pt'.format(epoch + 1)))
                saved_epoch = epoch
                if epoch > 0:
                    try:
                        os.remove(str(checkpoint_path / 'params_{:04}.pt'.format(epoch + 1 - cfg.TRAIN.PATIENCE)))
                    except:
                        pass
        
        if scheduler is not None:
            scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'
          .format(time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60))

    # * Eval
    if cfg.TRAIN.CLASS in ['all', 'none']:
        clss = dataloader['test'].dataset.classes
    else:
        clss = cfg.TRAIN.CLASS.split(', ')
    if cfg.PROBLEM.FILTER == 'difference':
        if cfg.DATASET_NAME == 'voc':
            if not isinstance(clss, list):
                clss = [clss]
            clss = list(filter(lambda x: x not in ['bicycle', 'bottle', 'chair', 'pottedplant', 'sofa', 'diningtable'], clss))
    
    if len(model_path) == 0:
        # * load the final model saved
        print('Loading the final model saved at epoch {}'.format(saved_epoch))
        saved_path = str(checkpoint_path / 'params_{:04}.pt'.format(saved_epoch + 1))
        load_model(model, saved_path, strict=False)
                
    # * return normalized objective scores
    precisions, recalls, f1s, objs = eval_model(model, dataloader['test'], clss, xls_sheet=xls_wb.add_sheet('epoch{}'.format(epoch + 1)))
    # * esitmated finish time
    print('Completion: {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M")))
    print()
    
    metrics = dict(zip(['precision', 'recall', 'f1'], [precisions, recalls, f1s]))
    for metric_name, metric in metrics.items():
        metric_dict = {"{}_{}".format(cls, metric_name): single_metric for cls, single_metric in zip(clss, metric)}
        metric_dict['average_{}'.format(metric_name)] = torch.mean(metric)
        tfboard_writer.add_scalars(
            'Eval {}'.format(metric_name),
            metric_dict,
            # (epoch + 1) * cfg.TRAIN.EPOCH_ITERS
            (epoch + 1) * EPOCH_ITERS
        )
        # * wandb
        run.log(metric_dict)
    
    # * record normalized objective scores
    obj_dict = {"{}_obj".format(cls): single_obj for cls, single_obj in zip(clss, objs)}
    obj_dict['average_obj'] = torch.mean(objs)
    tfboard_writer.add_scalars(
        'Eval obj',
        obj_dict,
        # (epoch + 1) * cfg.TRAIN.EPOCH_ITERS
        (epoch + 1) * EPOCH_ITERS
    )
    # * wandb
    run.log(obj_dict)
    wb.save(wb.__save_path)
    return model


if __name__ == '__main__':
    from src.utils.dup_stdout_manager import DupStdoutFileManager
    from src.utils.parse_args import parse_args
    from src.utils.print_easydict import print_easydict
    from src.utils.count_model_params import count_parameters

    args = parse_args('Deep learning of graph matching training & evaluation code.')
    
    # * wandb
    run = wandb.init(project='2GM', config=cfg, name=cfg.CONFIG_NAME)
    
    import os
    # * set visible GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(lambda x: str(x), cfg.GPUS)) if len(cfg.GPUS) > 1 else str(cfg.GPUS[0])
    # * specify cuda device
    device = torch.device("cuda:{}".format(cfg.GPUS[0]) if torch.cuda.is_available else "cpu")
    
    # * reproducibility
    random.seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)
    torch.manual_seed(cfg.RANDOM_SEED)
    os.environ["PYTHONHASHSEED"] = str(cfg.RANDOM_SEED) # latest
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.RANDOM_SEED)
        torch.cuda.manual_seed_all(cfg.RANDOM_SEED) # latest
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    import importlib
    mod = importlib.import_module(cfg.MODULE)
    Net = mod.Net

    model = Net()
    model = model.to(device)

    # * supervised training loss
    if cfg.TRAIN.LOSS_FUNC[-1].lower() == 'offset':
        criterion = OffsetLoss(norm=cfg.TRAIN.RLOSS_NORM)
    elif cfg.TRAIN.LOSS_FUNC[-1].lower() == 'perm':
        criterion = PermutationLoss()
    elif cfg.TRAIN.LOSS_FUNC[-1].lower() == 'ce':
        criterion = CrossEntropyLoss()
    elif cfg.TRAIN.LOSS_FUNC[-1].lower() == 'focal':
        criterion = FocalLoss(alpha=.5, gamma=0.)
    elif cfg.TRAIN.LOSS_FUNC[-1].lower() == 'hung':
        criterion = PermutationLossHung()
    elif cfg.TRAIN.LOSS_FUNC[-1].lower() == 'hamming':
        criterion = HammingLoss()
    elif cfg.TRAIN.LOSS_FUNC[-1].lower() == 'custom':
        criterion = None
        print('NOTE: You are setting the loss function as \'custom\', please ensure that there is a tensor with key '
              '\'loss\' in your model\'s returned dictionary.')
    else:
        criterion = None
        # raise ValueError('Unknown loss function {}'.format(cfg.TRAIN.LOSS_FUNC[-1]))

    # * update the projection head during training
    model_params = []
    param_keys = ['backbone_params', 'sconv_params']
    param_ids = {}
    for key in param_keys:
        param_ids[key] = [id(item) for item in getattr(model, key, [])]

    backbone_params = getattr(model, 'backbone_params', [])
    sconv_params = getattr(model, 'sconv_params', [])
    other_params = [param for param in model.parameters() if id(param) not in param_ids['backbone_params'] and id(param) not in param_ids['sconv_params']]
    # * backbone
    if cfg.TRAIN.FINETUNE_BACKBONE and backbone_params:
        model_params.append({'params': backbone_params, 'lr': cfg.TRAIN.BACKBONE_LR if cfg.TRAIN.SEPARATE_BACKBONE_LR else cfg.TRAIN.LR})
    else:
        for param in backbone_params:
            param.requires_grad = False
    # * encoder
    if cfg.TRAIN.FINETUNE_ENCODER and sconv_params:
        model_params.append({'params': sconv_params, 'lr': cfg.TRAIN.ENCODER_LR if cfg.TRAIN.SEPARATE_ENCODER_LR else cfg.TRAIN.LR})
    else:
        for param in sconv_params:
            param.requires_grad = False
    # * other params
    if cfg.TRAIN.LR == 0:
        for param in other_params:
            param.requires_grad = False
    else:
        model_params.append({'params': other_params})
    
    print('{} sets of params will be tuned'.format(len(model_params)))
        
    optimizer = None
    if len(model_params) > 0:
        if cfg.TRAIN.OPTIMIZER.lower() == 'sgd':
            optimizer = optim.SGD(model_params, lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, nesterov=True)
        elif cfg.TRAIN.OPTIMIZER.lower() == 'adam':
            optimizer = optim.Adam(model_params, lr=cfg.TRAIN.LR)
        else:
            raise ValueError('Unknown optimizer {}'.format(cfg.TRAIN.OPTIMIZER))

        if cfg.FP16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to enable FP16.")
            model, optimizer = amp.initialize(model, optimizer)

    model = DataParallel(model, device_ids=cfg.GPUS)

    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)
        
    # * overide dataset length i.e., the number of pairs
    if cfg.TRAIN.SAMPLES is not None and isinstance(cfg.TRAIN.SAMPLES, list):
        dataset_len = dict(zip(['train', 'validation', 'test'], cfg.TRAIN.SAMPLES + [cfg.EVAL.SAMPLES]))
    else:
        dataset_len = {'train': int(cfg.TRAIN.SPLIT * cfg.TRAIN.EPOCH_ITERS * cfg.BATCH_SIZE), 
                       'validation': math.ceil((1 - cfg.TRAIN.SPLIT) * cfg.TRAIN.EPOCH_ITERS * cfg.BATCH_SIZE),
                       'test': cfg.EVAL.SAMPLES}
    print(dataset_len)
    data_splits = list({k: v for k, v in dataset_len.items() if v != 0}.keys())

     # * modified based on the new train val split implementation
    rate_1 = 1.0 # args.rate
    rate_2 = 1.0
    image_dataset = {
        x: GMDataset(cfg.DATASET_FULL_NAME,
                    dataset_len[x],
                    rate_1,
                    rate_2,
                    cfg.TRAIN.CLASS if x in ['train','validation'] else cfg.EVAL.CLASS,
                    cfg.PROBLEM.TYPE if x == 'train' else 'GCLTS',
                    sets=x,
                    obj_resize=cfg.PROBLEM.RESCALE,
                    )
        for x in data_splits}
        
    dataloader = {x: get_dataloader(image_dataset[x], shuffle=(x == 'train'), fix_seed=(x in ['test', 'validation']))
                for x in data_splits} # * only shuffle the training set

    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')[:-3]
    tfboardwriter = SummaryWriter(logdir=str(Path(cfg.OUTPUT_PATH) / 'tensorboard' / 'training_{}'.format(now_time)))
    wb = xlwt.Workbook()
    wb.__save_path = str(Path(cfg.OUTPUT_PATH) / ('train_eval_result_' + now_time + '.xls'))

    with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH) / ('train_log_' + now_time + '.log'))) as _:
        print('rate : ', rate_1, rate_2)
        print_easydict(cfg)
        print('Number of parameters: {:.2f}M'.format(count_parameters(model) / 1e6))
        model = train_eval_model(cfg,
                                 model, criterion, optimizer, dataloader, tfboardwriter, run,
                                 dataset_len, data_splits,
                                 num_epochs=cfg.TRAIN.NUM_EPOCHS,
                                 start_epoch=cfg.TRAIN.START_EPOCH,
                                 xls_wb=wb)

    wb.save(wb.__save_path)
