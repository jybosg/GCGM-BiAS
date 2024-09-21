import torch
import torch.nn as nn
import torch.nn.functional as F
from src.lap_solvers.hungarian import hungarian
from torch import Tensor
import torch_geometric as pyg
# * for node agreement loss
from models.GCGM.encoder import ProjectionHead

# maximize the node agreement btw the same nodes from two views
# the negative samples are drawn from both inter- and intra-view

class MutualInfoLoss(nn.Module):
    r"""
    maximize the mutual information btw views pair within the minibatch
    For batched input, this loss function computes the average loss among all instances in the batch.
    """
    def __init__(self, 
                 tau,
                 out_channels, 
                 projection_channels,
                 projection_head: bool = False,
                 bias: bool = True,
                 structure: str = 'GRACE'
                 ):
        
        super(MutualInfoLoss, self).__init__()
        self.tau = tau # cfg.CLGM.TAU
        if projection_head: # cfg.CLGM.PROJECTION_HEAD
            self.projection = ProjectionHead(out_channels, projection_channels, bias, structure)
        else:
            self.projection = None
        
    # * pass the variables from two views separately
    def forward(self, y_1: Tensor, y_2: Tensor) -> Tensor:
        r"""
        :param y_1: :math:`(2b \times d)` the global embedding of Views_1, b is the batch size, d is the embedding dimension
        :param y_2: :math:`(2b \times d)` the global embedding of Views_2, b is the batch size, d is the embedding dimension
        :return: :math:`(1)` averaged MI
        """
        
        batch_num = y_1.size(0)

        y_1 = y_1.to(dtype=torch.float32, device=y_1.device)
        y_2 = y_2.to(dtype=torch.float32, device=y_2.device)
        
        loss = torch.tensor(0.).to(y_1.device)
        loss = self.obj(y_1, y_2)
        # return loss / batch_num
        return loss
    
    @staticmethod
    def cosine_sim(z1: torch.Tensor, z2: torch.Tensor):
        """cosine similarity"""
        return F.cosine_similarity(z1, z2, dim=1)
    
    @staticmethod
    def weighted_exponential(z1: torch.Tensor, z2: torch.Tensor, tau: float):
        """weighted exponential similarity"""
        return torch.exp(NodeAgreementLoss.cosine_sim(z1, z2) / tau)
    
    @staticmethod
    def dot_product(z1: torch.Tensor, z2: torch.Tensor):
        # l2 normalization
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())
    
    # * maximize the MI between the two corrupted views
    def pairwise_obj(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        pos_sim = f(MutualInfoLoss.dot_product(z1, z2))
        neg_sim = f(MutualInfoLoss.dot_product(z1, z1))
        return -torch.log(pos_sim.diag() / (pos_sim.sum(1) + neg_sim.sum(1) - neg_sim.diag()))

    def obj(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True):
        if self.projection is not None:
            z1, z2  = self.projection(z1), self.projection(z2)
        l1 = self.pairwise_obj(z1, z2)
        l2 = self.pairwise_obj(z2, z1)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
        return ret

class NodeAgreementLoss(nn.Module):
    r"""
    softmax cross-entropy also known as InfoNCE loss
    for maximizing the node agreement, the formulation could be found in the paper ``Deep Graph Contrastive Representation Learning``
    https://arxiv.org/pdf/2006.04131.pdf

    .. math::
        the pairwise objective
        https://cdn.mathpix.com/snip/images/VWQSDSyHt8ls43vfYoQdvlbqTe7fa7vp3GY32eyxBR8.original.fullsize.png
        
        and the overall objective
        \mathcal{J}=\frac{1}{2 N} \sum_{i=1}^N\left[\ell\left(\boldsymbol{u}_i, \boldsymbol{v}_i\right)+\ell\left(\boldsymbol{v}_i, \boldsymbol{u}_i\right)\right]
        
    .. note::
        For batched input, this loss function computes the average loss among all instances in the batch.
    """
    def __init__(self, 
                 tau,
                 out_channels, 
                 projection_channels,
                 projection_head: bool = False,
                 bias: bool = True,
                 structure: str = 'GRACE'
                 ):
        
        super(NodeAgreementLoss, self).__init__()
        self.tau = tau # cfg.CLGM.TAU
        if projection_head: # cfg.CLGM.PROJECTION_HEAD
            self.projection = ProjectionHead(out_channels, projection_channels, bias, structure)
        else:
            self.projection = None
        
    # * pass the variables from two views separately
    def forward(self, views_1, views_2, x_1: Tensor, x_2: Tensor) -> Tensor:
        r"""
        :param views_1: :type: `torch_geometric.data.batch.DataBatch` :math:`(2b)` the first augmented views of the batch
        :param views_2: :type: `torch_geometric.data.batch.DataBatch` :math:`(2b)` the second augmented views of the batch
        :param x_1: :math:`(n \times d)` the embedding of Views_1, n is the number of nodes of the DataBatch, d is the embedding dimension
        :param x_2: :math:`(n \times d)` the embedding of Views_2, n is the number of nodes of the DataBatch, d is the embedding dimension
        :return: :math:`(1)` averaged node agreement loss
        
        .. note::
            support batched instances with different number of nodes, therefore ``n_1`` and ``n_2`` are
            required to specify the exact number of nodes of two views of each instance in the batch.
        """
        # number of graph pairs = 2 * BATCH_SIZE
        if isinstance(views_1, pyg.data.Batch):
            batch_num = views_1.num_graphs
        else:
            batch_num = len(views_1)

        x_1 = x_1.to(dtype=torch.float32, device=x_1.device)
        x_2 = x_2.to(dtype=torch.float32, device=x_2.device)
        
        loss = torch.tensor(0.).to(x_1.device)
        # src_count, tgt_count = 0, 0
        for b in range(batch_num):
            src_graph, tgt_graph = views_1[b], views_2[b]
            src_dummy, tgt_dummy = src_graph.dummy_label, tgt_graph.dummy_label
            src_x, tgt_x = x_1[:src_graph.num_nodes, :], x_2[:tgt_graph.num_nodes, :]
            x_1, x_2 = x_1[src_graph.num_nodes:, :], x_2[tgt_graph.num_nodes:, :]
            # * correspondence_dummy = set(src_dummy.detach().cpu().tolist()).intersection(set(tgt_dummy.detach().cpu().tolist()))
            correspondence_dummy = set(src_dummy[:, 0].detach().cpu().tolist()).intersection(set(tgt_dummy[:, 0].detach().cpu().tolist()))
            if -1 in correspondence_dummy:
                correspondence_dummy.remove(-1)
            
            # no correspondence nodes in the two views
            if len(correspondence_dummy) == 0:
                continue
            
            # * inter-view positive and intra-view negative samples
            # * create positive and negative samples from two views only
            # * form z1 and z2 where z2 is the feature embedding of the correspondence points from the other view
            z1, z2, z3 = [], [], []
            # src_index = torch.tensor([(src_dummy==label).nonzero().item() for label in correspondence_dummy]).to(x_1.device)
            # tgt_index = torch.tensor([(tgt_dummy==label).nonzero().item() for label in correspondence_dummy]).to(x_2.device)
            src_index = torch.tensor([(src_dummy[:, 0]==label).nonzero().item() for label in correspondence_dummy]).to(x_1.device)
            tgt_index = torch.tensor([(tgt_dummy[:, 0]==label).nonzero().item() for label in correspondence_dummy]).to(x_2.device)
            # * inter-view positive sample 
            z1 = src_x[src_index]
            z2 = tgt_x[tgt_index]
            # * inter-view negative samples
            # * and intra-view negative samples
            for label in correspondence_dummy:
                # neg = torch.cat((src_graph.x[src_dummy != label], tgt_graph.x[tgt_dummy != label]), dim=0)
                # z3.append(neg)
                # z3.extend([src_x[src_dummy != label], tgt_x[tgt_dummy != label]])
                z3.extend([src_x[src_dummy[:, 0] != label], tgt_x[tgt_dummy[:, 0] != label]])
            
            loss += self.obj(z1, z2, torch.cat(z3, dim=0))
        return loss / batch_num
    
    ### discriminators
    @staticmethod
    def cosine_sim(z1: torch.Tensor, z2: torch.Tensor):
        """cosine similarity"""
        return F.cosine_similarity(z1, z2, dim=1)
    
    @staticmethod
    def weighted_exponential(z1: torch.Tensor, z2: torch.Tensor, tau: float):
        """weighted exponential similarity"""
        return torch.exp(NodeAgreementLoss.cosine_sim(z1, z2) / tau)
    
    @staticmethod
    def dot_product(z1: torch.Tensor, z2: torch.Tensor):
        # l2 normalization
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())
    
    ### maximize the node agreement between the same nodes from two views
    # consider both intra- and inter-view negative samples
    def pairwise_obj(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        pos_sim = f(NodeAgreementLoss.dot_product(z1, z2))
        z4 = z1.repeat(1, z3.size(0)//z1.size(0)).reshape(z3.shape)
        neg_sim = f(NodeAgreementLoss.dot_product(z3, z4))
        return -torch.log(pos_sim.diag() / (pos_sim.diag() + neg_sim.diag().reshape(z1.size(0), z3.size(0)//z1.size(0)).sum(1)))
    
    def obj(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor, mean: bool = True):
        if self.projection is not None:
            z1, z2, z3 = self.projection(z1), self.projection(z2), self.projection(z3)
        l1 = self.pairwise_obj(z1, z2, z3)
        l2 = self.pairwise_obj(z2, z1, z3)
    
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
        return ret

    
class PermutationLoss(nn.Module):
    r"""
    Binary cross entropy loss between two permutations, also known as "permutation loss".
    Proposed by `"Wang et al. Learning Combinatorial Embedding Networks for Deep Graph Matching. ICCV 2019."
    <http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Learning_Combinatorial_Embedding_Networks_for_Deep_Graph_Matching_ICCV_2019_paper.pdf>`_

    .. math::
        L_{perm} =- \sum_{i \in \mathcal{V}_1, j \in \mathcal{V}_2}
        \left(\mathbf{X}^{gt}_{i,j} \log \mathbf{S}_{i,j} + (1-\mathbf{X}^{gt}_{i,j}) \log (1-\mathbf{S}_{i,j}) \right)

    where :math:`\mathcal{V}_1, \mathcal{V}_2` are vertex sets for two graphs.

    .. note::
        For batched input, this loss function computes the averaged loss among all instances in the batch.
    """
    def __init__(self):
        super(PermutationLoss, self).__init__()

    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor, src_ns: Tensor, tgt_ns: Tensor) -> Tensor:
        r"""
        :param pred_dsmat: :math:`(b\times n_1 \times n_2)` predicted doubly-stochastic matrix :math:`(\mathbf{S})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :param src_ns: :math:`(b)` number of exact pairs in the first graph (also known as source graph).
        :param tgt_ns: :math:`(b)` number of exact pairs in the second graph (also known as target graph).
        # ! :return: :math:`(b)` averaged permutation loss

        .. note::
            We support batched instances with different number of nodes, therefore ``src_ns`` and ``tgt_ns`` are
            required to specify the exact number of nodes of each instance in the batch.
        """
        batch_num = pred_dsmat.shape[0]

        pred_dsmat = pred_dsmat.to(dtype=torch.float32)

        try:
            assert torch.all((pred_dsmat >= 0) * (pred_dsmat <= 1))
            assert torch.all((gt_perm >= 0) * (gt_perm <= 1))
        except AssertionError as err:
            print(pred_dsmat)
            raise err

        loss = torch.zeros(batch_num).to(pred_dsmat.device)
        for b in range(batch_num):
            batch_slice = [b, slice(src_ns[b]), slice(tgt_ns[b])]
            loss[b] = F.binary_cross_entropy(
                pred_dsmat[batch_slice],
                gt_perm[batch_slice],
                reduction='sum') / src_ns[b].to(pred_dsmat.device)

        return loss

class CrossEntropyLoss(nn.Module):
    r"""
    Multi-class cross entropy loss between two permutations.

    .. math::
        L_{ce} =- \sum_{i \in \mathcal{V}_1, j \in \mathcal{V}_2} \left(\mathbf{X}^{gt}_{i,j} \log \mathbf{S}_{i,j}\right)

    where :math:`\mathcal{V}_1, \mathcal{V}_2` are vertex sets for two graphs.

    .. note::
        For batched input, this loss function computes the averaged loss among all instances in the batch.
    """
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor, src_ns: Tensor, tgt_ns: Tensor) -> Tensor:
        r"""
        :param pred_dsmat: :math:`(b\times n_1 \times n_2)` predicted doubly-stochastic matrix :math:`(\mathbf{S})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :param src_ns: :math:`(b)` number of exact pairs in the first graph (also known as source graph).
        :param tgt_ns: :math:`(b)` number of exact pairs in the second graph (also known as target graph).
        :return: :math:`(1)` averaged cross-entropy loss

        .. note::
            We support batched instances with different number of nodes, therefore ``src_ns`` and ``tgt_ns`` are
            required to specify the exact number of nodes of each instance in the batch.
        """
        batch_num = pred_dsmat.shape[0]

        pred_dsmat = pred_dsmat.to(dtype=torch.float32)

        try:
            assert torch.all((pred_dsmat >= 0) * (pred_dsmat <= 1))
            assert torch.all((gt_perm >= 0) * (gt_perm <= 1))
        except AssertionError as err:
            print(pred_dsmat)
            raise err

        loss = torch.tensor(0.).to(pred_dsmat.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            batch_slice = [b, slice(src_ns[b]), slice(tgt_ns[b])]
            gt_index = torch.max(gt_perm[batch_slice], dim=-1).indices
            loss += F.nll_loss(
                torch.log(pred_dsmat[batch_slice]),
                gt_index,
                reduction='sum')
            n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)

        return loss / n_sum

# * BCE loss btw probability score matrix P and Hungarian matrix premutation matrix Z
# * refer to IA-SSGM
class MatchingLossHung(nn.Module):
    r"""
    Binary cross entropy loss between probability score matrix P and Hungarian matrix premutation matrix Z.
    .. math::
        \mathcal{L}_{\text {mat}}=-\sum_{i \in \mathcal{V}_s, j \in \mathcal{V}_t}\left(z_{i, j} \log p_{i, j}+\left(1-z_{i, j}\right) \log \left(1-p_{i, j}\right)\right)

    where :math:`\mathcal{V}_s, \mathcal{V}_t` are vertex sets for source graph and target graph respectively.
    .. note::
        For batched input, this loss function computes the averaged loss among all instances in the batch.

    """
    def __init__(self):
        super(MatchingLossHung, self).__init__()

    def forward(self, pred_prob: Tensor, dis_pred: Tensor, src_ns: Tensor, tgt_ns: Tensor) -> Tensor:
        r"""
        :param pred_prob: :math:`(b\times n_1 \times n_2)`  probability score matrix :math:`(\mathbf{P})`
        :param dis_pred: :math:`(b\times n_1 \times n_2)`  Hungarian matching matrix :math:`(\mathbf{Z})`
        :param src_ns: :math:`(b)` number of exact pairs in the first graph (also known as source graph).
        :param tgt_ns: :math:`(b)` number of exact pairs in the second graph (also known as target graph).
        :does not return: :math:`(1)` averaged permutation loss

        .. note::
            We support batched instances with different number of nodes, therefore ``src_ns`` and ``tgt_ns`` are
            required to specify the exact number of nodes of each instance in the batch.
        """
        batch_num = pred_prob.shape[0]
        assert torch.all((pred_prob >= 0) * (pred_prob <= 1))

        # dis_pred = hungarian(pred_prob, src_ns, tgt_ns)
        loss = torch.zeros(batch_num).to(pred_prob.device)
        for b in range(batch_num):
            batch_slice = [b, slice(src_ns[b]), slice(tgt_ns[b])]
            loss[b] = F.binary_cross_entropy(
                pred_prob[batch_slice],
                dis_pred[batch_slice],
                reduction='sum') / src_ns[b].to(pred_prob.device)
        return loss

class PermutationLossHung(nn.Module):
    r"""
    Binary cross entropy loss between two permutations with Hungarian attention. The vanilla version without Hungarian
    attention is :class:`~src.loss_func.PermutationLoss`.

    .. math::
        L_{hung} &=-\sum_{i\in\mathcal{V}_1,j\in\mathcal{V}_2}\mathbf{Z}_{ij}\left(\mathbf{X}^\text{gt}_{ij}\log \mathbf{S}_{ij}+\left(1-\mathbf{X}^{\text{gt}}_{ij}\right)\log\left(1-\mathbf{S}_{ij}\right)\right) \\
        \mathbf{Z}&=\mathrm{OR}\left(\mathrm{Hungarian}(\mathbf{S}),\mathbf{X}^\text{gt}\right)

    where :math:`\mathcal{V}_1, \mathcal{V}_2` are vertex sets for two graphs.

    Hungarian attention highlights the entries where the model makes wrong decisions after the Hungarian step (which is
    the default discretization step during inference).

    Proposed by `"Yu et al. Learning deep graph matching with channel-independent embedding and Hungarian attention.
    ICLR 2020." <https://openreview.net/forum?id=rJgBd2NYPH>`_

    .. note::
        For batched input, this loss function computes the averaged loss among all instances in the batch.

    A working example for Hungarian attention:

    .. image:: ../../images/hungarian_attention.png
    """
    def __init__(self):
        super(PermutationLossHung, self).__init__()

    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor, src_ns: Tensor, tgt_ns: Tensor) -> Tensor:
        r"""
        :param pred_dsmat: :math:`(b\times n_1 \times n_2)` predicted doubly-stochastic matrix :math:`(\mathbf{S})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :param src_ns: :math:`(b)` number of exact pairs in the first graph (also known as source graph).
        :param tgt_ns: :math:`(b)` number of exact pairs in the second graph (also known as target graph).
        :return: :math:`(1)` averaged permutation loss

        .. note::
            We support batched instances with different number of nodes, therefore ``src_ns`` and ``tgt_ns`` are
            required to specify the exact number of nodes of each instance in the batch.
        """
        batch_num = pred_dsmat.shape[0]

        assert torch.all((pred_dsmat >= 0) * (pred_dsmat <= 1))
        assert torch.all((gt_perm >= 0) * (gt_perm <= 1))

        dis_pred = hungarian(pred_dsmat, src_ns, tgt_ns)
        ali_perm = dis_pred + gt_perm
        ali_perm[ali_perm > 1.0] = 1.0 # Hung
        pred_dsmat = torch.mul(ali_perm, pred_dsmat)
        gt_perm = torch.mul(ali_perm, gt_perm)
        loss = torch.tensor(0.).to(pred_dsmat.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            loss += F.binary_cross_entropy(
                pred_dsmat[b, :src_ns[b], :tgt_ns[b]],
                gt_perm[b, :src_ns[b], :tgt_ns[b]],
                reduction='sum')
            n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)
        return loss / n_sum


class OffsetLoss(nn.Module):
    r"""
    OffsetLoss Criterion computes a robust loss function based on image pixel offset.
    Proposed by `"Zanfir et al. Deep Learning of Graph Matching. CVPR 2018."
    <http://openaccess.thecvf.com/content_cvpr_2018/html/Zanfir_Deep_Learning_of_CVPR_2018_paper.html>`_

    .. math::
        \mathbf{d}_i =& \sum_{j \in V_2} \left( \mathbf{S}_{i, j} P_{2j} \right)- P_{1i} \\
        L_{off} =& \sum_{i \in V_1} \sqrt{||\mathbf{d}_i - \mathbf{d}^{gt}_i||^2 + \epsilon}

    :math:`\mathbf{d}_i` is the displacement vector. See :class:`src.displacement_layer.Displacement` or more details

    :param epsilon: a small number for numerical stability
    :param norm: (optional) division taken to normalize the loss
    """
    def __init__(self, epsilon: float=1e-5, norm=None):
        super(OffsetLoss, self).__init__()
        self.epsilon = epsilon
        self.norm = norm

    def forward(self, d1: Tensor, d2: Tensor, mask: float=None) -> Tensor:
        """
        :param d1: predicted displacement matrix
        :param d2: ground truth displacement matrix
        :param mask: (optional) dummy node mask
        :return: computed offset loss
        """
        # Loss = Sum(Phi(d_i - d_i^gt))
        # Phi(x) = sqrt(x^T * x + epsilon)
        if mask is None:
            mask = torch.ones_like(mask)
        x = d1 - d2
        if self.norm is not None:
            x = x / self.norm

        xtx = torch.sum(x * x * mask, dim=-1)
        phi = torch.sqrt(xtx + self.epsilon)
        loss = torch.sum(phi) / d1.shape[0]

        return loss


class FocalLoss(nn.Module):
    r"""
    Focal loss between two permutations.

    .. math::
        L_{focal} =- \sum_{i \in \mathcal{V}_1, j \in \mathcal{V}_2}
        \left((1-\mathbf{S}_{i,j})^{\gamma} \mathbf{X}^{gt}_{i,j} \log \mathbf{S}_{i,j} +
        \mathbf{S}_{i,j}^{\gamma} (1-\mathbf{X}^{gt}_{i,j}) \log (1-\mathbf{S}_{i,j}) \right)

    where :math:`\mathcal{V}_1, \mathcal{V}_2` are vertex sets for two graphs, :math:`\gamma` is the focal loss
    hyper parameter.

    :param gamma: :math:`\gamma` parameter for focal loss
    :param eps: a small parameter for numerical stability

    .. note::
        For batched input, this loss function computes the averaged loss among all instances in the batch.
    """
    def __init__(self, gamma=0., eps=1e-15):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor, src_ns: Tensor, tgt_ns: Tensor) -> Tensor:
        r"""
        :param pred_dsmat: :math:`(b\times n_1 \times n_2)` predicted doubly-stochastic matrix :math:`(\mathbf{S})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :param src_ns: :math:`(b)` number of exact pairs in the first graph (also known as source graph).
        :param tgt_ns: :math:`(b)` number of exact pairs in the second graph (also known as target graph).
        :return: :math:`(1)` averaged focal loss

        .. note::
            We support batched instances with different number of nodes, therefore ``src_ns`` and ``tgt_ns`` are
            required to specify the exact number of nodes of each instance in the batch.
        """
        batch_num = pred_dsmat.shape[0]

        pred_dsmat = pred_dsmat.to(dtype=torch.float32)

        assert torch.all((pred_dsmat >= 0) * (pred_dsmat <= 1))
        assert torch.all((gt_perm >= 0) * (gt_perm <= 1))

        loss = torch.tensor(0.).to(pred_dsmat.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            x = pred_dsmat[b, :src_ns[b], :tgt_ns[b]]
            y = gt_perm[b, :src_ns[b], :tgt_ns[b]]
            loss += torch.sum(
                - (1 - x) ** self.gamma * y * torch.log(x + self.eps)
                - x ** self.gamma * (1 - y) * torch.log(1 - x + self.eps)
            )
            n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)

        return loss / n_sum


class InnerProductLoss(nn.Module):
    r"""
    Inner product loss for self-supervised problems.

    .. math::
        L_{ce} =- \sum_{i \in \mathcal{V}_1, j \in \mathcal{V}_2} \left(\mathbf{X}^{gt}_{i,j} \mathbf{S}_{i,j}\right)

    where :math:`\mathcal{V}_1, \mathcal{V}_2` are vertex sets for two graphs.

    .. note::
        For batched input, this loss function computes the averaged loss among all instances in the batch.
    """
    def __init__(self):
        super(InnerProductLoss, self).__init__()

    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor, src_ns: Tensor, tgt_ns: Tensor) -> Tensor:
        r"""
        :param pred_dsmat: :math:`(b\times n_1 \times n_2)` predicted doubly-stochastic matrix :math:`(\mathbf{S})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :param src_ns: :math:`(b)` number of exact pairs in the first graph (also known as source graph).
        :param tgt_ns: :math:`(b)` number of exact pairs in the second graph (also known as target graph).
        :return: :math:`(1)` averaged inner product loss

        .. note::
            We support batched instances with different number of nodes, therefore ``src_ns`` and ``tgt_ns`` are
            required to specify the exact number of nodes of each instance in the batch.
        """
        batch_num = pred_dsmat.shape[0]

        pred_dsmat = pred_dsmat.to(dtype=torch.float32)

        try:
            assert torch.all((gt_perm >= 0) * (gt_perm <= 1))
        except AssertionError as err:
            raise err

        loss = torch.tensor(0.).to(pred_dsmat.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            batch_slice = [b, slice(src_ns[b]), slice(tgt_ns[b])]
            loss -= torch.sum(pred_dsmat[batch_slice] * gt_perm[batch_slice])
            n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)

        return loss / n_sum
    
class HammingLoss(torch.nn.Module):
    r"""
    Hamming loss between two permutations.

    .. math::
        L_{hamm} = \sum_{i \in \mathcal{V}_1, j \in \mathcal{V}_2}
        \left(\mathbf{X}_{i,j} (1-\mathbf{X}^{gt}_{i,j}) +  (1-\mathbf{X}_{i,j}) \mathbf{X}^{gt}_{i,j}\right)

    where :math:`\mathcal{V}_1, \mathcal{V}_2` are vertex sets for two graphs.

    Firstly adopted by `"Rolinek et al. Deep Graph Matching via Blackbox Differentiation of Combinatorial Solvers.
    ECCV 2020." <https://arxiv.org/abs/2003.11657>`_

    .. note::
        Hamming loss is defined between two discrete matrices, and discretization will in general truncate gradient. A
        workaround may be using the `blackbox differentiation technique <https://arxiv.org/abs/1912.02175>`_.
    """
    def __init__(self):
        super(HammingLoss, self).__init__()

    def forward(self, pred_perm: Tensor, gt_perm: Tensor) -> Tensor:
        r"""
        :param pred_perm: :math:`(b\times n_1 \times n_2)` predicted permutation matrix :math:`(\mathbf{X})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :return:
        """
        errors = pred_perm * (1.0 - gt_perm) + (1.0 - pred_perm) * gt_perm
        return errors.sum(dim=[1,2]) # .mean(dim=0).sum()
    
# * matching acc for the validation set
class MatchingAccuracy(torch.nn.Module):
    r"""
    matching acc between two permutations.
    https://pygmtools.readthedocs.io/en/latest/_modules/pygmtools/benchmark.html#Benchmark.eval
    """
    def __init__(self):
        super(MatchingAccuracy, self).__init__()

    def forward(self, pred_perm: Tensor, gt_perm: Tensor) -> Tensor:
        r"""
        :param pred_perm: :math:`(b\times n_1 \times n_2)` predicted permutation matrix :math:`(\mathbf{X})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :return:
        """
        precision = (pred_perm * gt_perm).sum() / pred_perm.sum()
        recall = (pred_perm * gt_perm).sum() / gt_perm.sum()
        f1_score = (2 * precision * recall) / (precision + recall)
        return precision, recall, f1_score