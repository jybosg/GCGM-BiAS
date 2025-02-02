import torch
import torch.nn as nn
from torch import Tensor


class Sinkhorn(nn.Module):
    r"""
    Sinkhorn algorithm turns the input matrix into a bi-stochastic matrix.

    Sinkhorn algorithm firstly applies an ``exp`` function with temperature :math:`\tau`:

    .. math::
        \mathbf{S}_{i,j} = \exp \left(\frac{\mathbf{s}_{i,j}}{\tau}\right)

    And then turns the matrix into doubly-stochastic matrix by iterative row- and column-wise normalization:

    .. math::
        \mathbf{S} &= \mathbf{S} \oslash (\mathbf{1}_{n_2} \mathbf{1}_{n_2}^\top \cdot \mathbf{S}) \\
        \mathbf{S} &= \mathbf{S} \oslash (\mathbf{S} \cdot \mathbf{1}_{n_2} \mathbf{1}_{n_2}^\top)

    where :math:`\oslash` means element-wise division, :math:`\mathbf{1}_n` means a column-vector with length :math:`n`
    whose elements are all :math:`1`\ s.

    :param max_iter: maximum iterations (default: ``10``)
    :param tau: the hyper parameter :math:`\tau` controlling the temperature (default: ``1``)
    :param epsilon: a small number for numerical stability (default: ``1e-4``)
    :param log_forward: apply log-scale computation for better numerical stability (default: ``True``)
    :param batched_operation: apply batched_operation for better efficiency (but may cause issues for back-propagation,
     default: ``False``)

    .. note::
        ``tau`` is an important hyper parameter to be set for Sinkhorn algorithm. ``tau`` controls the distance between
        the predicted doubly-stochastic matrix, and the discrete permutation matrix computed by Hungarian algorithm (see
        :func:`~src.lap_solvers.hungarian.hungarian`). Given a small ``tau``, Sinkhorn performs more closely to
        Hungarian, at the cost of slower convergence speed and reduced numerical stability.

    .. note::
        We recommend setting ``log_forward=True`` because it is more numerically stable. It provides more precise
        gradient in back propagation and helps the model to converge better and faster.

    .. warning::
        If you set ``log_forward=False``, this function behaves a little bit differently: it does not include the
        ``exp`` part.

    .. note::
        Setting ``batched_operation=True`` may be preferred when you are doing inference with this module and do not
        need the gradient.
    """
    def __init__(self, max_iter: int=10, tau: float=1., epsilon: float=1e-4,
                 log_forward: bool=True, batched_operation: bool=False):
        super(Sinkhorn, self).__init__()
        self.max_iter = max_iter
        self.tau = tau
        self.epsilon = epsilon
        self.log_forward = log_forward
        if not log_forward:
            print('Warning: Sinkhorn algorithm without log forward is deprecated because log_forward is more stable.')
        self.batched_operation = batched_operation # batched operation may cause instability in backward computation,
                                                   # but will boost computation.

    def forward(self, s: Tensor, nrows: Tensor=None, ncols: Tensor=None, dummy_row: bool=False) -> Tensor:
        r"""
        :param s: :math:`(b\times n_1 \times n_2)` input 3d tensor. :math:`b`: batch size
        :param nrows: :math:`(b)` number of objects in dim1
        :param ncols: :math:`(b)` number of objects in dim2
        :param dummy_row: whether to add dummy rows (rows whose elements are all 0) to pad the matrix to square matrix.
         default: ``False``
        :return: :math:`(b\times n_1 \times n_2)` the computed doubly-stochastic matrix

        .. note::
            We support batched instances with different number of nodes, therefore ``nrows`` and ``ncols`` are
            required to specify the exact number of objects of each dimension in the batch. If not specified, we assume
            the batched matrices are not padded.

        .. note::
            The original Sinkhorn algorithm only works for square matrices. To handle cases where the graphs to be
            matched have different number of nodes, it is a common practice to add dummy rows to construct a square
            matrix. After the row and column normalizations, the padded rows are discarded.

        .. note::
            We assume row number <= column number. If not, the input matrix will be transposed.
        """
        if self.log_forward:
            return self.forward_log(s, nrows, ncols, dummy_row)
        else:
            return self.forward_ori(s, nrows, ncols, dummy_row) # deprecated

    def forward_log(self, s, nrows=None, ncols=None, dummy_row=False):
        """Compute sinkhorn with row/column normalization in the log space."""
        if len(s.shape) == 2:
            s = s.unsqueeze(0)
            matrix_input = True
        elif len(s.shape) == 3:
            matrix_input = False
        else:
            raise ValueError('input data shape not understood.')

        batch_size = s.shape[0]

        if s.shape[2] >= s.shape[1]:
            transposed = False
        else:
            s = s.transpose(1, 2)
            transposed = True

        if nrows is None:
            nrows = [s.shape[1] for _ in range(batch_size)]
        if ncols is None:
            ncols = [s.shape[2] for _ in range(batch_size)]
        if transposed:
            n = nrows
            nrows = ncols
            ncols = n
        # operations are performed on log_s
        s = s / self.tau

        if dummy_row:
            assert s.shape[2] >= s.shape[1]
            dummy_shape = list(s.shape)
            dummy_shape[1] = s.shape[2] - s.shape[1]
            ori_nrows = nrows
            nrows = ncols
            s = torch.cat((s, torch.full(dummy_shape, -float('inf')).to(s.device)), dim=1)
            for b in range(batch_size):
                s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -100
                s[b, nrows[b]:, :] = -float('inf')
                s[b, :, ncols[b]:] = -float('inf')

        if self.batched_operation:
            log_s = s

            for i in range(self.max_iter):
                if i % 2 == 0:
                    log_sum = torch.logsumexp(log_s, 2, keepdim=True)
                    log_s = log_s - log_sum
                    log_s[torch.isnan(log_s)] = -float('inf')
                else:
                    log_sum = torch.logsumexp(log_s, 1, keepdim=True)
                    log_s = log_s - log_sum
                    log_s[torch.isnan(log_s)] = -float('inf')

                # ret_log_s[b, row_slice, col_slice] = log_s

            if dummy_row and dummy_shape[1] > 0:
                log_s = log_s[:, :-dummy_shape[1]]
                for b in range(batch_size):
                    log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -float('inf')

            if matrix_input:
                log_s.squeeze_(0)

            return torch.exp(log_s)
        else:
            ret_log_s = torch.full((batch_size, s.shape[1], s.shape[2]), -float('inf'), device=s.device, dtype=s.dtype)

            for b in range(batch_size):
                row_slice = slice(0, nrows[b])
                col_slice = slice(0, ncols[b])
                log_s = s[b, row_slice, col_slice]

                for i in range(self.max_iter):
                    if i % 2 == 0:
                        log_sum = torch.logsumexp(log_s, 1, keepdim=True)
                        log_s = log_s - log_sum
                    else:
                        log_sum = torch.logsumexp(log_s, 0, keepdim=True)
                        log_s = log_s - log_sum

                ret_log_s[b, row_slice, col_slice] = log_s

            if dummy_row:
                if dummy_shape[1] > 0:
                    ret_log_s = ret_log_s[:, :-dummy_shape[1]]
                for b in range(batch_size):
                    ret_log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -float('inf')

            if transposed:
                ret_log_s = ret_log_s.transpose(1, 2)
            if matrix_input:
                ret_log_s.squeeze_(0)

            return torch.exp(ret_log_s)

        # ret_log_s = torch.full((batch_size, s.shape[1], s.shape[2]), -float('inf'), device=s.device, dtype=s.dtype)

        # for b in range(batch_size):
        #    row_slice = slice(0, nrows[b])
        #    col_slice = slice(0, ncols[b])
        #    log_s = s[b, row_slice, col_slice]

    ### pygm.sinkhorn
    def forward_log_ori(self, s, nrows=None, ncols=None, dummy_row=False):
        """
        Pytorch implementation of Sinkhorn algorithm
        """
        batch_size = s.shape[0]

        if s.shape[2] >= s.shape[1]:
            transposed = False
        else:
            s = s.transpose(1, 2)
            nrows, ncols = ncols, nrows
            unmatchrows, unmatchcols = unmatchcols, unmatchrows
            transposed = True

        if nrows is None:
            nrows = torch.tensor([s.shape[1] for _ in range(batch_size)], device=s.device)
        if ncols is None:
            ncols = torch.tensor([s.shape[2] for _ in range(batch_size)], device=s.device)

        # ensure that in each dimension we have nrow < ncol
        transposed_batch = nrows > ncols
        if torch.any(transposed_batch):
            s_t = s.transpose(1, 2)
            s_t = torch.cat((
                s_t[:, :s.shape[1], :],
                torch.full((batch_size, s.shape[1], s.shape[2]-s.shape[1]), -float('inf'), device=s.device)), dim=2)
            s = torch.where(transposed_batch.view(batch_size, 1, 1), s_t, s)

            new_nrows = torch.where(transposed_batch, ncols, nrows)
            new_ncols = torch.where(transposed_batch, nrows, ncols)
            nrows = new_nrows
            ncols = new_ncols

            if unmatchrows is not None and unmatchcols is not None:
                unmatchrows_pad = torch.cat((
                    unmatchrows,
                    torch.full((batch_size, unmatchcols.shape[1] - unmatchrows.shape[1]), -float('inf'), device=s.device)),
                dim=1)
                new_unmatchrows = torch.where(transposed_batch.view(batch_size, 1), unmatchcols, unmatchrows_pad)[:, :unmatchrows.shape[1]]
                new_unmatchcols = torch.where(transposed_batch.view(batch_size, 1), unmatchrows_pad, unmatchcols)
                unmatchrows = new_unmatchrows
                unmatchcols = new_unmatchcols

        # operations are performed on log_s
        log_s = s / self.tau
        if unmatchrows is not None and unmatchcols is not None:
            unmatchrows = unmatchrows / self.tau
            unmatchcols = unmatchcols / self.tau

        if dummy_row:
            assert log_s.shape[2] >= log_s.shape[1]
            dummy_shape = list(log_s.shape)
            dummy_shape[1] = log_s.shape[2] - log_s.shape[1]
            ori_nrows = nrows
            nrows = ncols.clone()
            log_s = torch.cat((log_s, torch.full(dummy_shape, -float('inf'), device=log_s.device, dtype=log_s.dtype)), dim=1)
            if unmatchrows is not None:
                unmatchrows = torch.cat((unmatchrows, torch.full((dummy_shape[0], dummy_shape[1]), -float('inf'), device=log_s.device, dtype=log_s.dtype)), dim=1)
            for b in range(batch_size):
                log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -100

        # assign the unmatch weights
        if unmatchrows is not None and unmatchcols is not None:
            new_log_s = torch.full((log_s.shape[0], log_s.shape[1]+1, log_s.shape[2]+1), -float('inf'), device=log_s.device, dtype=log_s.dtype)
            new_log_s[:, :-1, :-1] = log_s
            log_s = new_log_s
            for b in range(batch_size):
                log_s[b, :nrows[b], ncols[b]] = unmatchrows[b, :nrows[b]]
                log_s[b, nrows[b], :ncols[b]] = unmatchcols[b, :ncols[b]]
        row_mask = torch.zeros(batch_size, log_s.shape[1], 1, dtype=torch.bool, device=log_s.device)
        col_mask = torch.zeros(batch_size, 1, log_s.shape[2], dtype=torch.bool, device=log_s.device)
        for b in range(batch_size):
            row_mask[b, :nrows[b], 0] = 1
            col_mask[b, 0, :ncols[b]] = 1
        if unmatchrows is not None and unmatchcols is not None:
            ncols += 1
            nrows += 1

        if self.batched_operation:
            for b in range(batch_size):
                log_s[b, nrows[b]:, :] = -float('inf')
                log_s[b, :, ncols[b]:] = -float('inf')

            for i in range(self.max_iter):
                if i % 2 == 0:
                    log_sum = torch.logsumexp(log_s, 2, keepdim=True)
                    log_s = log_s - torch.where(row_mask, log_sum, torch.zeros_like(log_sum))
                    assert not torch.any(torch.isnan(log_s))
                else:
                    log_sum = torch.logsumexp(log_s, 1, keepdim=True)
                    log_s = log_s - torch.where(col_mask, log_sum, torch.zeros_like(log_sum))
                    assert not torch.any(torch.isnan(log_s))

            ret_log_s = log_s
        else:
            ret_log_s = torch.full((batch_size, log_s.shape[1], log_s.shape[2]), -float('inf'), device=log_s.device, dtype=log_s.dtype)

            for b in range(batch_size):
                row_slice = slice(0, nrows[b])
                col_slice = slice(0, ncols[b])
                log_s_b = log_s[b, row_slice, col_slice]
                row_mask_b = row_mask[b, row_slice, :]
                col_mask_b = col_mask[b, :, col_slice]

                for i in range(self.max_iter):
                    if i % 2 == 0:
                        log_sum = torch.logsumexp(log_s_b, 1, keepdim=True)
                        log_s_b = log_s_b - torch.where(row_mask_b, log_sum, torch.zeros_like(log_sum))
                    else:
                        log_sum = torch.logsumexp(log_s_b, 0, keepdim=True)
                        log_s_b = log_s_b - torch.where(col_mask_b, log_sum, torch.zeros_like(log_sum))

                ret_log_s[b, row_slice, col_slice] = log_s_b

        if unmatchrows is not None and unmatchcols is not None:
            ncols -= 1
            nrows -= 1
            for b in range(batch_size):
                ret_log_s[b, :nrows[b] + 1, ncols[b]] = -float('inf')
                ret_log_s[b, nrows[b], :ncols[b]] = -float('inf')
            ret_log_s = ret_log_s[:, :-1, :-1]

        if dummy_row:
            if dummy_shape[1] > 0:
                ret_log_s = ret_log_s[:, :-dummy_shape[1]]
            for b in range(batch_size):
                ret_log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -float('inf')

        if torch.any(transposed_batch):
            s_t = ret_log_s.transpose(1, 2)
            s_t = torch.cat((
                s_t[:, :ret_log_s.shape[1], :],
                torch.full((batch_size, ret_log_s.shape[1], ret_log_s.shape[2]-ret_log_s.shape[1]), -float('inf'), device=log_s.device)), dim=2)
            ret_log_s = torch.where(transposed_batch.view(batch_size, 1, 1), s_t, ret_log_s)

        if transposed:
            ret_log_s = ret_log_s.transpose(1, 2)

        return torch.exp(ret_log_s)
    
    def forward_ori(self, s, nrows=None, ncols=None, dummy_row=False):
        r"""
        Computing sinkhorn with row/column normalization.

        .. warning::
            This function is deprecated because :meth:`~src.lap_solvers.sinkhorn.Sinkhorn.forward_log` is more
            numerically stable.
        """
        if len(s.shape) == 2:
            s = s.unsqueeze(0)
            matrix_input = True
        elif len(s.shape) == 3:
            matrix_input = False
        else:
            raise ValueError('input data shape not understood.')

        batch_size = s.shape[0]

        #s = s.to(dtype=dtype)

        if nrows is None:
            nrows = [s.shape[1] for _ in range(batch_size)]
        if ncols is None:
            ncols = [s.shape[2] for _ in range(batch_size)]

        # tau scaling
        ret_s = torch.zeros_like(s)
        for b, n in enumerate(nrows):
            ret_s[b, 0:n, 0:ncols[b]] = \
                nn.functional.softmax(s[b, 0:n, 0:ncols[b]] / self.tau, dim=-1)
        s = ret_s

        # add dummy elements
        if dummy_row:
            dummy_shape = list(s.shape)
            dummy_shape[1] = s.shape[2] - s.shape[1]
            #s = torch.cat((s, torch.full(dummy_shape, self.epsilon * 10).to(s.device)), dim=1)
            #nrows = nrows + dummy_shape[1] # non in-place
            s = torch.cat((s, torch.full(dummy_shape, 0.).to(s.device)), dim=1)
            ori_nrows = nrows
            nrows = ncols
            for b in range(batch_size):
                s[b, ori_nrows[b]:nrows[b], :ncols[b]] = self.epsilon

        row_norm_ones = torch.zeros(batch_size, s.shape[1], s.shape[1], device=s.device, dtype=s.dtype)  # size: row x row
        col_norm_ones = torch.zeros(batch_size, s.shape[2], s.shape[2], device=s.device, dtype=s.dtype)  # size: col x col
        for b in range(batch_size):
            row_slice = slice(0, nrows[b])
            col_slice = slice(0, ncols[b])
            row_norm_ones[b, row_slice, row_slice] = 1
            col_norm_ones[b, col_slice, col_slice] = 1

        s += self.epsilon

        for i in range(self.max_iter):
            if i % 2 == 0:
                # column norm
                #ones = torch.ones(batch_size, s.shape[1], s.shape[1], device=s.device)
                sum = torch.sum(torch.mul(s.unsqueeze(3), col_norm_ones.unsqueeze(1)), dim=2)
            else:
                # row norm
                # ones = torch.ones(batch_size, s.shape[2], s.shape[2], device=s.device)
                sum = torch.sum(torch.mul(row_norm_ones.unsqueeze(3), s.unsqueeze(1)), dim=2)

            tmp = torch.zeros_like(s)
            for b in range(batch_size):
                row_slice = slice(0, nrows[b] if nrows is not None else s.shape[2])
                col_slice = slice(0, ncols[b] if ncols is not None else s.shape[1])
                tmp[b, row_slice, col_slice] = 1 / sum[b, row_slice, col_slice]
            s = s * tmp

        if dummy_row:
            if dummy_shape[1] > 0:
                s = s[:, :-dummy_shape[1]]
            for b in range(batch_size):
                s[b, ori_nrows[b]:nrows[b], :ncols[b]] = 0

        if matrix_input:
            s.squeeze_(0)

        return s


class GumbelSinkhorn(nn.Module):
    """
    Gumbel Sinkhorn Layer turns the input matrix into a bi-stochastic matrix.
    See details in `"Mena et al. Learning Latent Permutations with Gumbel-Sinkhorn Networks. ICLR 2018"
    <https://arxiv.org/abs/1802.08665>`_

    :param max_iter: maximum iterations (default: ``10``)
    :param tau: the hyper parameter :math:`\tau` controlling the temperature (default: ``1``)
    :param epsilon: a small number for numerical stability (default: ``1e-4``)
    :param batched_operation: apply batched_operation for better efficiency (but may cause issues for back-propagation,
     default: ``False``)

    .. note::
        This module only supports log-scale Sinkhorn operation.
    """
    def __init__(self, max_iter=10, tau=1., epsilon=1e-4, batched_operation=False):
        super(GumbelSinkhorn, self).__init__()
        self.sinkhorn = Sinkhorn(max_iter, tau, epsilon, batched_operation=batched_operation)

    def forward(self, s: Tensor, nrows: Tensor=None, ncols: Tensor=None,
                sample_num=5, dummy_row=False) -> Tensor:
        r"""
        :param s: :math:`(b\times n_1 \times n_2)` input 3d tensor. :math:`b`: batch size
        :param nrows: :math:`(b)` number of objects in dim1
        :param ncols: :math:`(b)` number of objects in dim2
        :param sample_num: number of samples
        :param dummy_row: whether to add dummy rows (rows whose elements are all 0) to pad the matrix to square matrix.
         default: ``False``
        :return: :math:`(b m\times n_1 \times n_2)` the computed doubly-stochastic matrix. :math:`m`: number of samples
         (``sample_num``)

        The samples are stacked at the fist dimension of the output tensor. You may reshape the output tensor ``s`` as:

        ::

            s = torch.reshape(s, (-1, sample_num, s.shape[1], s.shape[2]))

        .. note::
            We support batched instances with different number of nodes, therefore ``nrows`` and ``ncols`` are
            required to specify the exact number of objects of each dimension in the batch. If not specified, we assume
            the batched matrices are not padded.

        .. note::
            The original Sinkhorn algorithm only works for square matrices. To handle cases where the graphs to be
            matched have different number of nodes, it is a common practice to add dummy rows to construct a square
            matrix. After the row and column normalizations, the padded rows are discarded.

        .. note::
            We assume row number <= column number. If not, the input matrix will be transposed.
        """
        def sample_gumbel(t_like, eps=1e-20):
            """
            randomly sample standard gumbel variables
            """
            u = torch.empty_like(t_like).uniform_()
            return -torch.log(-torch.log(u + eps) + eps)

        s_rep = torch.repeat_interleave(s, sample_num, dim=0)
        s_rep = s_rep + sample_gumbel(s_rep)
        nrows_rep = torch.repeat_interleave(nrows, sample_num, dim=0)
        ncols_rep = torch.repeat_interleave(ncols, sample_num, dim=0)
        s_rep = self.sinkhorn(s_rep, nrows_rep, ncols_rep, dummy_row)
        #s_rep = torch.reshape(s_rep, (-1, sample_num, s_rep.shape[1], s_rep.shape[2]))
        return s_rep


if __name__ == '__main__':
    bs = Sinkhorn(max_iter=8, epsilon=1e-4)
    inp = torch.tensor([[[1., 0, 1.],
                         [1., 0, 3.],
                         [2., 0, 1.],
                         [4., 0, 2.]]], requires_grad=True)
    outp = bs(inp, (3, 4))

    print(outp)
    l = torch.sum(outp)
    l.backward()
    print(inp.grad * 1e10)

    outp2 = torch.tensor([[0.1, 0.1, 1],
                          [2, 3, 4.]], requires_grad=True)

    l = torch.sum(outp2)
    l.backward()
    print(outp2.grad)
