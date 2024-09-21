import os
import torch.cuda
import torch.optim as optim
import time
import xlwt
from datetime import datetime, timedelta
from pathlib import Path
from tensorboardX import SummaryWriter

from src.dataset.data_loader import GMDataset, get_dataloader
from src.displacement_layer import Displacement
# * modified loss function
from src.loss_func_gcgm import *
from src.evaluation_metric import matching_accuracy, matching_f1
from src.parallel import DataParallel
from src.utils.model_sl import load_model, save_model
from eval_gcgm import eval_model
from src.utils.data_to_cuda import data_to_cuda

from src.utils.config import cfg

# * wandb
import wandb
from wandb import AlertLevel
import math

# * reproductivity
import random
import numpy as np

# * gradient accumulation
ACCUMULATION_STEPS = 1

# * add support for wandb
def train_eval_model(cfg, 
                     model,
                     hidden_criterion, global_criterion,
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

    device = next(model.parameters()).device
    print('model on device: {}'.format(device))

    checkpoint_path = Path(cfg.OUTPUT_PATH) / 'params'
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True)

    model_path, optim_path = '', ''
    if start_epoch != 0:
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

    scheduler = None
    if optimizer is not None:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=cfg.TRAIN.LR_STEP,
                                                   gamma=cfg.TRAIN.LR_DECAY,
                                                   last_epoch=cfg.TRAIN.START_EPOCH - 1)
        
    epoch_loss_history = []
    val_acc_history = []
    saved_epoch = 0 # record epoch number of the best model saved
    # * BiAS updating every PATIENCE epochs and PATIENCE decreasing over time 
    total_batches = 0
    decrease_factor = cfg.BiAS.DECREASE_FACTOR
    update_every_n_batches = cfg.BiAS.PATIENCE
    next_update_batch = update_every_n_batches
    
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

        batch_size = cfg.BATCH_SIZE
        EPOCH_ITERS = dataset_len.get('train', 0) // batch_size
        if dataloader.get('train') != None:
            # Iterate over data.
            for inputs in dataloader['train']:
                total_batches += 1
                # if iter_num >= cfg.TRAIN.SPLIT * cfg.TRAIN.EPOCH_ITERS:
                if iter_num >= EPOCH_ITERS:
                    break
                
                if model.module.device != torch.device('cpu'):
                    inputs = data_to_cuda(inputs)

                iter_num = iter_num + 1

                # zero the parameter gradients
                if optimizer is not None:
                    optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    # forward
                    outputs = model(inputs)
                    
                    if cfg.PROBLEM.TYPE == 'GCL':
                        cl_loss = 0.0
                        if hidden_criterion is not None:
                            cl_loss = cfg.GCGM.C_LOSS_RATE * hidden_criterion(outputs['views_1'], outputs['views_2'], outputs['hidden_1'], outputs['hidden_2'])  
                        
                        if global_criterion is not None:
                            cl_loss += cfg.GCGM.C_LOSS_RATE * global_criterion(outputs['Glob_1'], outputs['Glob_2'])
                        
                        # * mixup rates for the augmentations in sampled in the batch
                        cur_mixup = torch.tensor([model.module.augmentation.aug_pairs[i][0][-1][0] if model.module.augmentation.aug_pairs[i][0][0].__name__.upper() == 'MIXUP' else 0 for i in outputs['cur_indices']], dtype=torch.float32).to("cuda")
                        if cfg.TRAIN.LOSS_FUNC[-1].lower() == 'perm':
                            loss = torch.mean((1 - cur_mixup) * criterion(outputs['ds_mat'], outputs['new_perm_mat'], outputs['ns_1'], outputs['ns_2']))
                            loss += torch.mean(cur_mixup * criterion(outputs['ds_mat'], outputs['mix_perm_mat'], outputs['ns_1'], outputs['ns_2']))
                        elif cfg.TRAIN.LOSS_FUNC[-1].lower() == 'hamming':
                            loss = torch.mean((1 - cur_mixup) * criterion(outputs['perm_mat'], outputs['new_perm_mat']))
                            loss += torch.mean(cur_mixup * criterion(outputs['perm_mat'], outputs['mix_perm_mat']))
                        
                        loss += cl_loss
                        loss = loss / ACCUMULATION_STEPS # normalize the loss because gradients are accumulated
                        # compute accuracy
                        acc = matching_accuracy(outputs['perm_mat'], outputs['new_perm_mat'], outputs['ns_1'])
                        prev_counts, cur_indices, avg_f1s, weights = outputs['prev_counts'], outputs['cur_indices'].cpu().numpy(), outputs['avg_f1s'], outputs['weights']
                        prev_counts = prev_counts[: len(model.module.augmentation.aug_pairs)].cpu().numpy()
                        cur_counts = model.module.augmentation.update_counts(prev_counts, cur_indices)
                        avg_f1s = avg_f1s[: len(model.module.augmentation.aug_pairs)].cpu().numpy()
                        weights = weights[: len(model.module.augmentation.aug_pairs)].cpu().numpy()
                        # * BiAS updating
                        if total_batches == next_update_batch:
                            weights, avg_f1s, cur_counts = model.module.augmentation.update(prev_counts, cur_counts, cur_indices, avg_f1s, weights, acc, cfg.BiAS.LAMBDA, cfg.BiAS.ALPHA)
                            model.module.augmentation.weights = weights
                            model.module.augmentation.avg_f1s = avg_f1s
                            model.module.augmentation.prev_counts = cur_counts
                            update_every_n_batches = int(decrease_factor * update_every_n_batches)
                            next_update_batch = total_batches + max(1, update_every_n_batches)
                        else:
                            # * update average f1 scores only
                            weights, avg_f1s, cur_counts = model.module.augmentation.update(prev_counts, cur_counts, cur_indices, avg_f1s, weights, acc, cfg.BiAS.LAMBDA, cfg.BiAS.ALPHA, update_weights=False)
                            model.module.augmentation.weights = weights
                            model.module.augmentation.avg_f1s = avg_f1s
                            model.module.augmentation.prev_counts = cur_counts
                        weights = dict(zip(model.module.augmentation.aug_pair_names, model.module.augmentation.weights))
                        run.log(weights)
                                    
                    elif cfg.PROBLEM.TYPE in ['2GM', 'GCLTS', 'GCLE2E']:
                        assert 'ds_mat' in outputs
                        assert 'perm_mat' in outputs
                        assert 'gt_perm_mat' in outputs

                        # compute loss
                        if cfg.TRAIN.LOSS_FUNC[-1].lower() == 'offset':
                            d_gt, grad_mask = displacement(outputs['gt_perm_mat'], *outputs['Ps'], outputs['ns'][0])
                            d_pred, _ = displacement(outputs['ds_mat'], *outputs['Ps'], outputs['ns'][0])
                            loss = criterion(d_pred, d_gt, grad_mask)
                        # elif cfg.TRAIN.LOSS_FUNC[-1].lower() in ['perm', 'ce', 'hung']:
                        #     loss = criterion(outputs['ds_mat'], outputs['gt_perm_mat'], *outputs['ns'])
                        # elif cfg.TRAIN.LOSS_FUNC[-1].lower() == 'hamming':
                        #     loss = criterion(outputs['perm_mat'], outputs['gt_perm_mat'])
                        elif cfg.TRAIN.LOSS_FUNC[-1].lower() in ['ce', 'hung']:
                            loss = criterion(outputs['ds_mat'], outputs['gt_perm_mat'], *outputs['ns'])
                        elif cfg.TRAIN.LOSS_FUNC[-1].lower() == 'perm':
                            loss = criterion(outputs['ds_mat'], outputs['gt_perm_mat'], *outputs['ns'])
                            loss = torch.sum(loss) / torch.sum(outputs['ns'][0])
                        elif cfg.TRAIN.LOSS_FUNC[-1].lower() == 'hamming':
                            loss = criterion(outputs['perm_mat'], outputs['gt_perm_mat']).mean()
                        elif cfg.TRAIN.LOSS_FUNC[-1].lower() == 'custom':
                            loss = torch.sum(outputs['loss'])
                        else:
                            raise ValueError(
                                'Unsupported loss function {} for problem type {}'.format(cfg.TRAIN.LOSS_FUNC[-1].lower(),
                                                                                        cfg.PROBLEM.TYPE))

                        # compute accuracy
                        acc = matching_accuracy(outputs['perm_mat'], outputs['gt_perm_mat'], outputs['ns'][0])
                        
                        if cfg.PROBLEM.TYPE == 'GCLE2E':
                            if hidden_criterion is not None:
                                loss += hidden_criterion(outputs['views_1'], outputs['views_2'], outputs['hidden_1'], outputs['hidden_2'])
                            
                            if global_criterion is not None:
                                loss += global_criterion(outputs['Glob_1'], outputs['Glob_2'])

                    elif cfg.PROBLEM.TYPE in ['MGM', 'MGM3']:
                        assert 'ds_mat_list' in outputs
                        assert 'graph_indices' in outputs
                        assert 'perm_mat_list' in outputs
                        if not 'gt_perm_mat_list' in outputs:
                            assert 'gt_perm_mat' in outputs
                            gt_perm_mat_list = [outputs['gt_perm_mat'][idx] for idx in outputs['graph_indices']]
                        else:
                            gt_perm_mat_list = outputs['gt_perm_mat_list']

                        # compute loss & accuracy
                        if cfg.TRAIN.LOSS_FUNC[-1].lower() in ['perm', 'ce' 'hung']:
                            loss = torch.zeros(1, device=model.module.device)
                            ns = outputs['ns']
                            for s_pred, x_gt, (idx_src, idx_tgt) in \
                                    zip(outputs['ds_mat_list'], gt_perm_mat_list, outputs['graph_indices']):
                                l = criterion(s_pred, x_gt, ns[idx_src], ns[idx_tgt])
                                loss += l
                            loss /= len(outputs['ds_mat_list'])
                        elif cfg.TRAIN.LOSS_FUNC[-1].lower() == 'hamming':
                            loss = torch.zeros(1, device=model.module.device)
                            ns = outputs['ns']
                            for s_pred, x_gt in zip(outputs['ds_mat_list'], gt_perm_mat_list):
                                l = criterion(s_pred, x_gt)
                                loss += l
                            loss /= len(outputs['ds_mat_list'])
                        elif cfg.TRAIN.LOSS_FUNC[-1].lower() == 'plain':
                            loss = torch.sum(outputs['loss'])
                        else:
                            raise ValueError(
                                'Unsupported loss function {} for problem type {}'.format(cfg.TRAIN.LOSS_FUNC[-1].lower(),
                                                                                        cfg.PROBLEM.TYPE))
                        # compute accuracy
                        acc = torch.zeros(1, device=model.module.device)
                        for x_pred, x_gt, (idx_src, idx_tgt) in \
                                zip(outputs['perm_mat_list'], gt_perm_mat_list, outputs['graph_indices']):
                            a = matching_accuracy(x_pred, x_gt, ns[idx_src])
                            acc += torch.sum(a)
                        acc /= len(outputs['perm_mat_list'])
                    else:
                        raise ValueError('Unknown problem type {}'.format(cfg.PROBLEM.TYPE))
                    
                    if torch.isnan(loss).any():
                        wandb.alert(
                        title='Loss is NaN',
                        text='At Epoch {}, loss is NaN'.format(epoch),
                        # text=f'Loss {loss} is below the acceptable threshold {threshold}',
                        level=AlertLevel.WARN,
                        wait_duration=timedelta(minutes=5)
                        )

                    # backward + optimize
                    if optimizer is not None:
                        if cfg.FP16:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()
                            
                        if (iter_num + 1) % ACCUMULATION_STEPS == 0:
                            optimizer.step()
                            optimizer.zero_grad() # reset gradients

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
                    running_loss += loss.item() * batch_size
                    epoch_loss += loss.item() * batch_size
                    if cfg.PROBLEM.TYPE == 'GCL':
                        epoch_cl_loss += cl_loss * batch_size
                    
                    # * wandb
                    run.log({"running_loss": running_loss})
                    # * log acc
                    run.log({"matching_accuracy": accdict['matching accuracy']})

                    if iter_num % cfg.STATISTIC_STEP == 0:
                        running_speed = cfg.STATISTIC_STEP * batch_size / (time.time() - running_since)
                        print('Epoch {:<4} Iteration {:<4} {:>4.2f}sample/s Loss={:<8.4f}'.format(epoch, iter_num, running_speed, running_loss / cfg.STATISTIC_STEP / batch_size))
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

            # epoch_loss = epoch_loss / (cfg.TRAIN.SPLIT * cfg.TRAIN.EPOCH_ITERS) / batch_size
            # epoch_cl_loss = epoch_cl_loss / (cfg.TRAIN.SPLIT * cfg.TRAIN.EPOCH_ITERS) / batch_size
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
        
        # * early stopping the pretraining based on the loss
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
                    
                    if model.module.device != torch.device('cpu'):
                        inputs = data_to_cuda(inputs)
                    
                    val_iter_num += 1
                    
                    # * evaluation using matching accuracy, precision, and recall (the latter two are used when there are outliers)
                    with torch.set_grad_enabled(False):
                        outputs = model(inputs)
                        f1_score = matching_f1(outputs['perm_mat'], outputs['gt_perm_mat'], outputs['ns'][0])
                        val_acc += torch.sum(f1_score).item()
                        
                val_elapsed = time.time() - val_start
                model.train(mode=was_training)
                # * validation acc
                # val_acc = val_acc / ((1 - cfg.TRAIN.SPLIT) * cfg.TRAIN.EPOCH_ITERS) / batch_size
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
        
        torch.cuda.empty_cache() # * empty the cache to train the model with large batch size

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

    args = parse_args('Deep learning of graph matching training & evaluation code.')
    
    # * wandb
    run = wandb.init(project='GCGM', config=cfg, name=cfg.CONFIG_NAME)
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
    
    # * initialize the model
    import importlib
    mod = importlib.import_module(cfg.MODULE)
    Net = mod.Net
    
    model = Net()
    # nn.init.trunc_normal_(model.weight.data)
    model = model.to(device)
    
    hidden_criterion = None
    global_criterion = None
    
    # * contrastive loss 
    if 'nodeagreement' in cfg.TRAIN.LOSS_FUNC:
        hidden_criterion = NodeAgreementLoss(cfg.GCGM.TAU, cfg.GCGM.OUT_CHANNELS, cfg.GCGM.PROJECTION_CHANNELS, cfg.GCGM.PROJECTION_HEAD, cfg.GCGM.BIAS, cfg.GCGM.PROJECTION_STRUCTURE)
        if isinstance(hidden_criterion, tuple):
            hidden_criterion = hidden_criterion[0]
    
    if 'mutualinfo' in cfg.TRAIN.LOSS_FUNC:
        global_criterion = MutualInfoLoss(cfg.GCGM.TAU, cfg.GCGM.OUT_CHANNELS, cfg.GCGM.PROJECTION_CHANNELS, cfg.GCGM.PROJECTION_HEAD, cfg.GCGM.BIAS, cfg.GCGM.PROJECTION_STRUCTURE)
        if isinstance(global_criterion, tuple):
            global_criterion = global_criterion[0]
    
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
    if 'nodeagreement' in cfg.TRAIN.LOSS_FUNC and cfg.GCGM.PROJECTION_HEAD:
        model_params.append({'params': hidden_criterion.parameters()})  
            
    if 'mutualinfo' in cfg.TRAIN.LOSS_FUNC and cfg.GCGM.PROJECTION_HEAD:
        model_params.append({'params': global_criterion.parameters()})
    
    param_keys = ['backbone_params', 'encoder_params', 'sconv_params']
    param_ids = {}
    for key in param_keys:
        param_ids[key] = [id(item) for item in getattr(model, key, [])]

    backbone_params = getattr(model, 'backbone_params', [])
    encoder_params = getattr(model, 'encoder_params', [])
    sconv_params = getattr(model, 'sconv_params', [])
    other_params = [param for param in model.parameters() if id(param) not in param_ids['backbone_params'] and id(param) not in param_ids['encoder_params'] and id(param) not in param_ids['sconv_params']]
    # * backbone
    if cfg.TRAIN.FINETUNE_BACKBONE and backbone_params:
        model_params.append({'params': backbone_params, 'lr': cfg.TRAIN.BACKBONE_LR / ACCUMULATION_STEPS if cfg.TRAIN.SEPARATE_BACKBONE_LR else cfg.TRAIN.LR / ACCUMULATION_STEPS})
    else:
        for param in backbone_params:
            param.requires_grad = False
    # * encoder
    if cfg.TRAIN.FINETUNE_ENCODER and (encoder_params or sconv_params):
        combined_params = encoder_params + sconv_params
        model_params.append({'params': combined_params, 'lr': cfg.TRAIN.ENCODER_LR / ACCUMULATION_STEPS if cfg.TRAIN.SEPARATE_ENCODER_LR else cfg.TRAIN.LR / ACCUMULATION_STEPS})
    else:
        for param in encoder_params + sconv_params:
            param.requires_grad = False
    # * other params
    if cfg.TRAIN.LR == 0:
        for param in other_params:
            param.requires_grad = False
    else:
        model_params.append({'params': other_params})

    optimizer = None
    if len(model_params) > 0:
        if cfg.TRAIN.OPTIMIZER.lower() == 'sgd':
            optimizer = optim.SGD(model_params, lr=cfg.TRAIN.LR / ACCUMULATION_STEPS, momentum=cfg.TRAIN.MOMENTUM, nesterov=True)
        elif cfg.TRAIN.OPTIMIZER.lower() == 'adam':
            optimizer = optim.Adam(model_params, lr=cfg.TRAIN.LR / ACCUMULATION_STEPS, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        else:
            raise ValueError('Unknown optimizer {}'.format(cfg.TRAIN.OPTIMIZER))

        if cfg.FP16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to enable FP16.")
            model, optimizer = amp.initialize(model, optimizer)

    model = DataParallel(model, device_ids=cfg.GPUS)
    criterion = criterion.to(device)
    
    # * if node agreement loss is used, we need to also put the criterion onto cuda device
    if hidden_criterion is not None:
        hidden_criterion = hidden_criterion.to(device)
        
    if global_criterion is not None:
        global_criterion = global_criterion.to(device)
        
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
    print(data_splits)
    
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
        
    # dataloader = {x: get_dataloader(image_dataset[x], shuffle=True, fix_seed=(x in ['test', 'validation']))
    #             for x in data_splits}
    dataloader = {x: get_dataloader(image_dataset[x], shuffle=(x == 'train'), fix_seed=(x in ['test', 'validation']))
                for x in data_splits} # * only shuffle the training set

    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    tfboardwriter = SummaryWriter(logdir=str(Path(cfg.OUTPUT_PATH) / 'tensorboard' / 'training_{}'.format(now_time)))
    wb = xlwt.Workbook()
    wb.__save_path = str(Path(cfg.OUTPUT_PATH) / ('train_eval_result_' + now_time + '.xls'))

    with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH) / ('train_log_' + now_time + '.log'))) as _:
        print_easydict(cfg)
        model = train_eval_model(cfg, 
                                 model, 
                                 hidden_criterion, global_criterion,
                                 criterion, optimizer, dataloader, tfboardwriter, run,
                                 dataset_len, data_splits,
                                 num_epochs=cfg.TRAIN.NUM_EPOCHS,
                                 start_epoch=cfg.TRAIN.START_EPOCH,
                                 xls_wb=wb)
    if cfg.PROBLEM.TYPE != 'GCL':
        wb.save(wb.__save_path)
    
    run.finish()