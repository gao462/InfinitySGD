import torch
import numpy as np
import autograd.numpy as dnp
from autograd import jacobian
import scipy.stats as stats
import time


# time cost monitor
TIME_COST = {
    'Forward::Shape_Check' : [],
    'Forward::Build_P'     : [],
    'Forward::Solve_PI'    : [],
    'Backward::Malloc'     : [],
    'Backward::Jacob_Q'    : [],
    'Backward::Jacob_Gamma': [],
    'Backward::Jacob_P'    : [],
    'Backward::Jacob_PI'   : [],
    'Backward::Inf_Sum'    : [],
    'Backward::Grad_P'     : [],
    'Backward::Grad_PI'    : [],
}


r"""
PyTorch and PyTorch Autograd Version
====================================
This part provides differentiable discrete markov process implemented for PyTorch autograd.

It supports batching, but do not support CUDA acceleration.

The gap of CUDA supporting is CUDA linear solver. Indeed, there are libraries such as pycuda and scikit-cuda that
provides a Python wrapper for cuSOLVER. However, I failed to install either of them by conda on all my available
environments.

Easy plug-in functions to linear solver and infinite power summation is supported, so that I can easily update it
for CUDA supporting and other potential infinite power summation tricks.

- numpy_solver
  Linear system solver by Numpy for PyTorch batch input.

- _russian_roulette
  Compute expectation of infinite power summation by only Russian Roulette trick.

- _inf_split
  Compute expectation of infinite power summation by only Infinite Split trick.

- _russian_roulette_inf_split
  Compute expectation of infinite power summation by Russian Roulette and Infinite Split tricks.

- DDMP
  Forwarding and backwarding functions of differentiable discrete markov process.
  `DDMP.solver` is used to control the linear solver plug-in function. It can be thoroughly modified during running
  by calling `DDMP.set_solver(...)`.
  `DDMP.infusm` is used to control the infinite power summation plug-in function. It can be thoroughly modified
  during running by calling `DDMP.set_infsum(...)`.
  It is possible that the tricks need hyperparameters, so `DDMP.dkargs` is used to control those hyperparameters as
  keyword arguments. It can be thoroughly modified during running by calling `DDMP.set_dkargs(...)`.

- stdy_dist
  Common function call interface for DDMP autograd function class.
------------------------------------------------------------------
"""


def _numpy_solver(A, b):
    r"""Numpy Solver Function

    Args
    ----
    A : torch.Tensor
        A of linear system $Ax=b$.
    b : torch.Tensor
        b of linear system $Ax=b$.

    Returns
    -------
    x : torch.Tensor
        x of linear system $Ax=b$.

    """
    # go to numpy environment
    bat_A = A.data.cpu().numpy()
    bat_b = b.data.cpu().numpy()

    # solve for the whole batch
    bat_x = []
    for A_np, b_np in zip(bat_A, bat_b):
        bat_x.append(np.linalg.lstsq(A_np, b_np, rcond=None)[0])
    bat_x = np.stack(bat_x)

    # go to torch environment
    x = torch.from_numpy(bat_x).to(A.device)
    return x


def _russian_roulette(pi, coeff, P, rvdist):
    r"""Only Russian Roulette

    Args
    ----
    pi : torch.Tensor
        Steady state distrbution as a row of infinite power.
    coeff : torch.Tensor
        Dimension swapped (input first, then probability matrix) Jacobian tensors.
    P : torch.Tensor
        Probability matrix.
    rvdist : tuple
        Argument tuple for random variable sampling.

    Returns
    -------
    S : torch.Tensor
        Expectation of infinite power summation.

    """
    # P^{\infty} is a repetition of \pi
    inf = pi.repeat((1, pi.size(-1), 1))

    # get random variable
    rv = rvdist[0].rvs(*rvdist[1:])
    rv = 2

    # resize for batch and broadcasting
    bsz, k = pi.size(0), pi.size(-1)
    coeff = coeff.view(bsz, k * k, k, k)

    # get infinite summation by Russian Roulette
    S = torch.zeros(pi.size(-1), pi.size(-1), dtype=P.dtype, device=P.device)
    for i in range(1, rv + 1):
        cdf_above = rvdist[0].sf(i, *rvdist[1:])
        part1 = torch.matrix_power(P, rv - i).view(bsz, 1, k, k)
        part2 = torch.matrix_power(P, i - 1).view(bsz, 1, k, k)
        S = S + torch.matmul(torch.matmul(part1, coeff), part2) / cdf_above
    return S.view(bsz, k, k, k, k)


def _inf_split(pi, coeff, P, rv):
    r"""Only Infinite Split

    Args
    ----
    pi : torch.Tensor
        Steady state distrbution as a row of infinite power.
    coeff : torch.Tensor
        Dimension swapped (input first, then probability matrix) Jacobian tensors.
    P : torch.Tensor
        Probability matrix.
    rv : int
        Constant exponent.

    Returns
    -------
    S : torch.Tensor
        Expectation of infinite power summation.

    """
    # P^{\infty} is a repetition of \pi
    inf = pi.repeat((1, pi.size(-1), 1))

    # get infinite summation by approximation
    M = torch.eye(pi.size(-1), dtype=P.dtype, device=P.device)
    S = torch.zeros(pi.size(-1), pi.size(-1), dtype=P.dtype, device=P.device)
    for _ in range(rv):
        S = S + M
        M = torch.matmul(P, M)

    # resize for batch and broadcasting
    bsz, k = pi.size(0), pi.size(-1)
    inf = inf.view(bsz, 1, k, k)
    coeff = coeff.view(bsz, k * k, k, k)
    S = S.view(bsz, 1, k, k)
    return torch.matmul(torch.matmul(inf, coeff), S).view(bsz, k, k, k, k)


def _russian_roulette_inf_split(pi, coeff, P, rvdist):
    r"""Russian Roulette and Infinite Split

    Args
    ----
    pi : torch.Tensor
        Steady state distrbution as a row of infinite power.
    coeff : torch.Tensor
        Dimension swapped (input first, then probability matrix) Jacobian tensors.
    P : torch.Tensor
        Probability matrix.
    rvdist : tuple
        Argument tuple for random variable sampling.

    Returns
    -------
    S : torch.Tensor
        Expectation of infinite power summation.

    """
    # P^{\infty} is a repetition of \pi
    inf = pi.repeat((1, pi.size(-1), 1))

    # get random variable
    rv = rvdist[0].rvs(*rvdist[1:])

    # get infinite summation by Russian Roulette
    M = torch.eye(pi.size(-1), dtype=P.dtype, device=P.device)
    S = torch.zeros(pi.size(-1), pi.size(-1), dtype=P.dtype, device=P.device)
    for i in range(1, rv + 1):
        cdf_above = rvdist[0].sf(i, *rvdist[1:])
        S = S + M / cdf_above
        M = torch.matmul(P, M)

    # resize for batch and broadcasting
    bsz, k = pi.size(0), pi.size(-1)
    inf = inf.view(bsz, 1, k, k)
    coeff = coeff.view(bsz, k * k, k, k)
    S = S.view(bsz, 1, k, k)
    return torch.matmul(torch.matmul(inf, coeff), S).view(bsz, k, k, k, k)


class DDMP(torch.autograd.Function):
    """Differentiable Discrete Markov Process"""
    # Default Constants
    C = 0.001
    solver = _numpy_solver
    infsum = _russian_roulette_inf_split
    dkargs = dict(rvdist=(stats.geom, 0.1))

    @staticmethod
    def set_c(c):
        r"""Set uniform normalization bias

        Args
        ----
        c : float
            Uniform normalization bias.

        """
        # change the class
        DDMP.C = c

    @staticmethod
    def set_solver(solver):
        r"""Set linear system solver

        Args
        ----
        solver : func
            Solver function.

        """
        # change the class
        DDMP.solver = solver

    @staticmethod
    def set_infsum(infsum):
        r"""Set infinite power summation

        Args
        ----
        infsum : func
            Summation function.

        """
        # change the class
        DDMP.infsum = infsum

    @staticmethod
    def set_dkargs(dkargs):
        r"""Set random variable distrbution keyword arguments

        Args
        ----
        dkargs : dict
            Distrbution keyword arguments.

        """
        # change the class
        DDMP.dkargs = dkargs

    @staticmethod
    def forward(ctx, trans_rate_mx):
        r"""Forwarding

        Args
        ----
        ctx : object
            Temporary buffer to from forwarding to backwarding.
        trans_rate_mx : torch.Tensor
            Transition rate matrix.

        Returns
        -------
        P : torch.Tensor
            Transition probability matrix.
        pi : torch.Tensor
            Steady state distribution vector.
        gamma : torch.Tensor
            Uniformization denominator vector.

        """
        # claim class
        cls = DDMP

        # enforce matrix shape
        chkpt = time.time()
        if len(trans_rate_mx.size()) == 3:
            ctx.bsz, ctx.k, k2 = trans_rate_mx.size()
            if ctx.k != k2:
                raise RuntimeError('transition rate matrix must be square.')
            else:
                pass
        else:
            raise RuntimeError('transition rate matrix must have (bsz, k, k) shape.')
        TIME_COST['Forward::Shape_Check'].append(time.time() - chkpt)

        # diagonal lines must be empty and be auto-filled
        chkpt = time.time()
        if len(torch.nonzero(torch.diagonal(trans_rate_mx, dim1=-2, dim2=-1))) > 0:
            raise RuntimeError('transition rate matrix must only have zero diagonal elements.')
        else:
            ctx.X = trans_rate_mx
            diag = torch.sum(ctx.X, dim=-1)
            ctx.did = torch.argmax(diag, dim=-1)
            D = torch.diag_embed(diag, dim1=-2, dim2=-1)
            ctx.Q = ctx.X - D
            ctx.gamma = diag[torch.arange(ctx.bsz, dtype=ctx.did.dtype, device=ctx.did.device), ctx.did] + cls.C
            I = torch.eye(ctx.k, dtype=ctx.Q.dtype, device=ctx.Q.device).view(1, ctx.k, ctx.k)
            ctx.P = ctx.Q / ctx.gamma.view(-1, 1, 1) + I
        TIME_COST['Forward::Build_P'].append(time.time() - chkpt)

        # nan exclusion
        assert len(torch.nonzero(ctx.P != ctx.P)) == 0, \
               'Transition probability matrix has nan.'

        # get steady state distribution by linear system
        chkpt = time.time()
        v = torch.ones(ctx.bsz, ctx.k, 1, dtype=ctx.P.dtype, device=ctx.P.device)
        A = torch.cat([ctx.P - I, v], dim=-1)
        b = torch.zeros(ctx.bsz, 1, ctx.k + 1, dtype=A.dtype, device=A.device)
        b[:, :, -1] = 1
        A, b = torch.transpose(A, -2, -1), torch.transpose(b, -2, -1)
        x = cls.solver(A, b)
        TIME_COST['Forward::Solve_PI'].append(time.time() - chkpt)

        # nan exclusion
        assert len(torch.nonzero(x != x)) == 0, \
               'linear system solution has nan.'
        ctx.pi = torch.transpose(x, -2, -1)
        return ctx.P, ctx.pi, ctx.gamma

    @staticmethod
    def backward(ctx, grad_from_P=None, grad_from_pi=None, grad_from_gamma=None):
        r"""Backwarding

        Args
        ----
        grad_from_P : torch.Tensor or None
            Previous gradients from transactions.
        grad_from_pi : torch.Tensor or None
            Previous gradients from states.
        grad_from_pi : None
            Previous gradients from uniformization demoniator.

        Returns
        -------
        grad_to_mx : torch.Tensor or None
            Gradients to transition rate matrix.

        """
        # claim class
        cls = DDMP

        # enforce at least one source gradient
        assert not (grad_from_P is None and grad_from_pi is None)

        # gamma never have gradient
        assert grad_from_gamma is None or len(torch.nonzero(grad_from_gamma)) == 0

        # allocate Q to X Jacobian gradients
        chkpt = time.time()
        jacob_size = (ctx.bsz, ctx.k, ctx.k, ctx.k, ctx.k)
        jacob_Q = torch.zeros(*jacob_size, dtype=ctx.P.dtype, device=ctx.P.device)

        # allocata gamma to X Jacobian graidents
        jacob_size = (ctx.bsz, ctx.k, ctx.k)
        jacob_gamma = torch.zeros(*jacob_size, dtype=ctx.P.dtype, device=ctx.P.device)

        # allocata P to X Jacobian graidents
        jacob_size = (ctx.bsz, ctx.k, ctx.k, ctx.k, ctx.k)
        jacob_P = torch.zeros(*jacob_size, dtype=ctx.P.dtype, device=ctx.P.device)

        # allocata P^{\infty} to \pi Jacobian gradients (can share)
        jacob_size = (1, ctx.k, ctx.k, ctx.k)
        jacob_pi = torch.zeros(*jacob_size, dtype=ctx.P.dtype, device=ctx.P.device)
        TIME_COST['Backward::Malloc'].append(time.time() - chkpt)

        # fill Q to X Jacobian gradients
        chkpt = time.time()
        for i in range(ctx.k):
            for j in range(ctx.k):
                if i == j:
                    jacob_Q[:, i, i, i, :] = -1
                    jacob_Q[:, i, i, i, i] = 0
                else:
                    jacob_Q[:, i, j, i, j] = 1
        TIME_COST['Backward::Jacob_Q'].append(time.time() - chkpt)

        # fill gamma to X Jacobian gradients
        chkpt = time.time()
        jacob_gamma[torch.arange(ctx.bsz, dtype=ctx.did.dtype, device=ctx.did.device), ctx.did, :] = 1
        for i in range(ctx.k):
            jacob_gamma[:, i, i] = 0
        TIME_COST['Backward::Jacob_Gamma'].append(time.time() - chkpt)

        # fill P to X Jacobian gradients
        chkpt = time.time()
        gamma = ctx.gamma.view(-1, 1, 1)
        for i in range(ctx.k):
            for j in range(ctx.k):
                q = ctx.Q[:, i, j].view(-1, 1, 1)
                dQ = jacob_Q[:, i, j]
                dgamma = jacob_gamma
                jacob_P[:, i, j] = (dQ * gamma - q * dgamma) / (gamma ** 2)
        TIME_COST['Backward::Jacob_P'].append(time.time() - chkpt)

        # nan exclusion
        assert len(torch.nonzero(jacob_P != jacob_P)) == 0, \
               'Jacobian matrix of transition matrix has nan.'

        # fill P^{\infty} to \pi Jacobian gradients
        chkpt = time.time()
        for i in range(ctx.k):
            jacob_pi[:, i, :, i] = 1 / ctx.k
        TIME_COST['Backward::Jacob_PI'].append(time.time() - chkpt)

        # nan exclusion
        assert len(torch.nonzero(jacob_pi != jacob_pi)) == 0, \
               'Jacobian matrix of selection matrix has nan.'

        # get infinite Jacobian gradients
        chkpt = time.time()
        coeff_P = jacob_P.permute(0, 3, 4, 1, 2)
        coeff_inf = cls.infsum(ctx.pi, coeff_P, ctx.P, **cls.dkargs)
        TIME_COST['Backward::Inf_Sum'].append(time.time() - chkpt)

        # nan exclusion
        assert len(torch.nonzero(coeff_inf != coeff_inf)) == 0, \
               'Coefficient matrix of infinite power has nan.'

        # backward from each transaction
        chkpt = time.time()
        if grad_from_P is None:
            grad_of_trans = torch.zeros(ctx.bsz, ctx.k, ctx.k, dtype=ctx.P.dtype, device=ctx.P.device)
        else:
            grad_of_trans = grad_from_P.view(ctx.bsz, ctx.k, ctx.k, 1, 1) * jacob_P
            grad_of_trans = torch.sum(grad_of_trans, dim=(1, 2))
        TIME_COST['Backward::Grad_P'].append(time.time() - chkpt)

        # backward from each state
        chkpt = time.time()
        if grad_from_pi is None:
            grad_of_states = torch.zeros(ctx.bsz, ctx.k, ctx.k, dtype=ctx.P.dtype, device=ctx.P.device)
        else:
            coeff_inf = coeff_inf.view(ctx.bsz, ctx.k * ctx.k, ctx.k, ctx.k)
            grad_of_states = []
            for i in range(ctx.k):
                coeff = jacob_pi[:, i] * coeff_inf
                coeff = torch.sum(coeff, dim=(-2, -1))
                grad = (grad_from_pi[:, :, i] * coeff).view(-1, 1, ctx.k, ctx.k)
                grad_of_states.append(grad)
            grad_of_states = torch.cat(grad_of_states, dim=1)
            grad_of_states = torch.sum(grad_of_states, dim=1)
        TIME_COST['Backward::Grad_PI'].append(time.time() - chkpt)

        # aggregate two backwarding
        return grad_of_trans + grad_of_states


# Support for DC-BPTT
def approx_stdy_dist(trans_rate_mx, rv):
    r"""Forwarding

    Args
    ----
    trans_rate_mx : torch.Tensor
        Transition rate matrix.
    rv : int
        Constant exponent used to approximate infinite power.

    Returns
    -------
    P : torch.Tensor
        Transition probability matrix.
    pi : torch.Tensor
        Steady state distribution vector.
    gamma : torch.Tensor
        Uniformization denominator vector.

    """
    # claim class
    _const = DDMP

    # enforce matrix shape
    if len(trans_rate_mx.size()) == 3:
        bsz, k, k2 = trans_rate_mx.size()
        if k != k2:
            raise RuntimeError('transition rate matrix must be square.')
        else:
            pass
    else:
        raise RuntimeError('transition rate matrix must have (bsz, k, k) shape.')

    # diagonal lines must be empty and be auto-filled
    if len(torch.nonzero(torch.diagonal(trans_rate_mx, dim1=-2, dim2=-1))) > 0:
        raise RuntimeError('transition rate matrix must only have zero diagonal elements.')
    else:
        X = trans_rate_mx
        diag = torch.sum(X, dim=-1)
        did = torch.argmax(diag, dim=-1)
        D = torch.diag_embed(diag, dim1=-2, dim2=-1)
        Q = X - D
        gamma = diag[torch.arange(bsz, dtype=did.dtype, device=did.device), did] + _const.C
        I = torch.eye(k, dtype=Q.dtype, device=Q.device).view(1, k, k)
        P = Q / gamma.view(-1, 1, 1) + I

    # get approximation of infinite power
    EXPO = P
    for i in range(rv):
        EXPO = torch.matmul(EXPO, EXPO)

    # average all rows
    pi = torch.mean(EXPO, dim=1, keepdim=True)
    return P, pi, gamma


# Common functon call interface for DDMP
def stdy_dist(trans_rate_mx, c=0.001, solver=_numpy_solver, config='rrinf', infsum=None,
              dkargs=None):
    r"""Get steady state distribution vector for given transition rate matrix

    Args
    ----
    trans_rate_mx : torch.Tensor
        Transition rate matrix.
    c : float
        Uniformization offset.
    solver : func
        Linear solver.
    config : str
        Specify an already defined configuration for infsum and dkargs.
    infsum : func
        Infinite power summation trick.
    dkargs : dict
        Keyword arguments for infinite power summation trick.

    Returns
    -------
    P : torch.Tensor
        Transition probability matrix.
    pi : torch.Tensor
        Steady state distribution vector.
    gamma : torch.Tensor
        Uniformization denominator vector.

    """
    # set DDMP parameters
    DDMP.set_c(c)
    DDMP.set_solver(solver)
    if config is None:
        DDMP.set_infsum(infsum)
        DDMP.set_dkargs(dkargs)
    elif config == 'dc':
        return approx_stdy_dist(trans_rate_mx, dkargs['rv'])
    elif config == 'rrinf':
        DDMP.set_infsum(infsum or _russian_roulette_inf_split)
        DDMP.set_dkargs(dkargs or dict(rvdist=(stats.geom, 0.1)))
    elif config == 'rr':
        DDMP.set_infsum(infsum or _russian_roulette)
        DDMP.set_dkargs(dkargs or dict(rvdist=(stats.geom, 0.1)))
    elif config == 'inf':
        DDMP.set_infsum(infsum or _inf_split)
        DDMP.set_dkargs(dkargs or dict(rv=10))
    else:
        raise NotImplementedError

    # compute steady state distribution
    return DDMP.apply(trans_rate_mx)


r"""
Numpy and Numpy Autograd Version
================================
This part provides differentiable discrete markov process implemented for Numpy. Numpy autograd is applied
somewhere to get ground truth partial derivatives. It is used as a ground truth result to verify PyTorch
implementation. Two things are not verified, and their math and code are assumed to be true under all cases.
1. Steady state distribution from transition probability matrix.
   To verify this, we need compute infinite power of transition probability matrix which is impossible to
   implement.
2. Infinite power summation and related tricks for gradients.
   To verify this, we need either backprop through a infinite matrix power, or compute infinite power summation
   directly.
   Neither of them is feasible.

It is used as ground truth to verify PyTorch implementation. It only support single sample on CPU to avoid
encounter any risk of broadcasting matrices for batching.

Easy plug-in functions are supported as PyTorch implementation.

- numpy_solver_numpy
  Linear system solver by Numpy for Numpy single-sample input.

- _russian_roulette_inf_split_numpy
  Compute expectation of infinite power summation by Russian Roulette and Infinite Split tricks for Numpy
  single-sample input.

- _func_X2Q_dnp
  Numpy autograd function to add negative diagonal line based on transition rate matrix.

- _func_X2P_dnp
  Numpy autograd function to get transition probability matrix based on transition rate matrix.

- _grad_X2Q_dnp
  Autograd of _func_X2Q_dnp. Pay attention that forwarding is matrix-to-matrix, so it is indeed computing Jacobian.

- _grad_X2P_dnp
  Autograd of _func_X2P_dnp. Pay attention that forwarding is matrix-to-matrix, so it is indeed computing Jacobian.

- _DDMP_numpy
  Forwarding and backwarding functions of differentiable discrete markov process for Numpy.
  To keep consistency with PyTorch implementation, hyperparameters for this class is directly fetched from members
  of the same variable names in `DDMP`.
  Unlike Pytorch autograd function classes, an instance must be created to use it for forwarding and backwarding.
  Since it can not trace forwarding and backwarding graph, an instance is bounded with only one input. If you do
  forwarding with an instance, you can operate the instance unless you do backwarding.
--------------------------------------------------------------------------------------
"""


def _numpy_solver_numpy(A, b):
    r"""Numpy Solver Function

    Args
    ----
    A : numpy.ndarray
        A of linear system $Ax=b$.
    b : numpy.ndarray
        b of linear system $Ax=b$.

    Returns
    -------
    x : numpy.ndarray
        x of linear system $Ax=b$.

    """
    # solve for the whole batch
    return np.linalg.lstsq(A, b, rcond=None)[0]


def _russian_roulette_inf_split_numpy(pi, coeff, P, rvdist):
    r"""Russian Roulette and Infinite Split

    Args
    ----
    pi : numpy.ndarray
        Steady state distrbution as a row of infinite power.
    coeff : numpy.ndarray
        Dimension swapped (input first, then probability matrix) Jacobian tensors.
    P : numpy.ndarray
        Probability matrix.
    rvdist : tuple
        Argument tuple for random variable sampling.

    Returns
    -------
    S : numpy.ndarray
        Expectation of infinite power summation.

    """
    # P^{\infty} is a repetition of \pi
    inf = np.tile(pi, (pi.shape[-1], 1))

    # get random variable
    rv = rvdist[0].rvs(*rvdist[1:])

    # get infinite summation by Russian Roulette
    M = np.eye(pi.shape[-1], dtype=P.dtype)
    S = np.zeros(shape=(pi.shape[-1], pi.shape[-1]), dtype=P.dtype)
    for i in range(1, rv + 1):
        cdf_above = rvdist[0].sf(i, *rvdist[1:])
        S = S + M / cdf_above
        M = P @ M
    return (inf @ coeff) @ S


def _func_X2Q_dnp(X):
    r"""A clone of function from rate to rate by autograd numpy

    Args
    ----
    X : numpy.ndarray
        Transition rate matrix.

    Returns
    -------
    Q : numpy.ndarray
        Diagonally filled transition rate matrix.

    """
    # autograd numpy
    diag = dnp.sum(X, axis=1)
    diagsum = dnp.zeros(shape=X.shape, dtype=X.dtype)
    for i in range(diag.shape[0]):
        diagbase = dnp.zeros(shape=X.shape, dtype=X.dtype)
        diagbase[i, i] = 1
        diagsum = diagsum + diagbase * diag[i]
    D = diagsum
    Q = X - D
    return Q


def _func_X2P_dnp(X, did):
    r"""A clone of function from rate to probability by autograd numpy

    Args
    ----
    X : numpy.ndarray
        Transition rate matrix.
    did : int
        Index for gamma.

    Returns
    -------
    P : numpy.ndarray
        Transition probability matrix.

    Appendix
    --------
    One wierd thing is that replacing Q, gamma by
    ```python
    diag = dnp.sum(X, axis=1)
    diagsum = dnp.zeros(shape=X.shape, dtype=X.dtype)
    for i in range(diag.shape[0]):
        diagbase = dnp.zeros(shape=X.shape, dtype=X.dtype)
        diagbase[i, i] = 1
        diagsum = diagsum + diagbase * diag[i]
    D = diagsum
    Q = X - D
    gamma = diag[did] + _const.C
    ```
    can not pass the test, but mathematically, they are equivalent.
    So, autograd numpy may have problem on computing that.

    """
    # claim class
    _const = DDMP

    # autograd numpy
    Q = _func_X2Q_dnp(X)
    gamma = -Q[did, did] + _const.C
    I = dnp.eye(X.shape[0])
    P = Q / gamma + I
    return P


# Gradient functions by autograd numpy
_grad_X2Q_dnp = jacobian(_func_X2Q_dnp, 0)
_grad_X2P_dnp = jacobian(_func_X2P_dnp, 0)


class _DDMP_numpy(object):
    """Differentiable Discrete Markov Process"""
    # Private Buffer Class
    class _Buffer(object):
        bsz   = None
        k     = None
        X     = None
        did   = None
        Q     = None
        gamma = None
        P     = None
        pi    = None

    # Define Constants
    NULL      = 0
    FORWARDED = 1

    def __init__(self):
        r"""Initialize the class"""
        # create buffer
        self._buffer = self._Buffer()

        # create forwarding/backwarding signal
        self._signal = self.NULL

    def forward(self, trans_rate_mx):
        r"""Forwarding

        Args
        ----
        trans_rate_mx : numpy.ndarray
            Transition rate matrix.

        Returns
        -------
        P : numpy.ndarray
            Transition probability matrix.
        pi : numpy.ndarray
            Steady state distribution vector.
        gamma : numpy.ndarray
            Uniformization denominator vector.

        """
        # security check
        if self._signal != self.NULL:
            raise RuntimeError('there is uncleared forwarding.')
        else:
            self._signal = self.FORWARDED

        # claim class
        _const = DDMP

        # claim temporary buffer
        _buf = self._buffer

        # enforce matrix shape
        if len(trans_rate_mx.shape) == 2:
            _buf.k, k2 = trans_rate_mx.shape
            if _buf.k != k2:
                raise RuntimeError('transition rate matrix must be square.')
            else:
                pass
        else:
            raise RuntimeError('transition rate matrix must have (k, k) shape.')

        # diagonal lines must be empty and be auto-filled
        if len(np.nonzero(np.diagonal(trans_rate_mx))[0]) > 0:
            raise RuntimeError('transition rate matrix must only have zero diagonal elements.')
        else:
            _buf.X = trans_rate_mx
            diag = np.sum(_buf.X, axis=1)
            _buf.did = torch.argmax(torch.from_numpy(diag).to(device)).data.item()
            D = np.diagflat(diag)
            _buf.Q = _buf.X - D
            _buf.gamma = diag[_buf.did] + _const.C
            I = np.eye(_buf.k)
            _buf.P = _buf.Q / _buf.gamma + I

        # check autograd numpy by numpy
        assert np.all(_func_X2P_dnp(_buf.X.copy(), _buf.did) - _buf.P == 0)

        # get steady state distribution by linear system
        v = np.ones(shape=(_buf.k, 1))
        A = np.concatenate([_buf.P - I, v], axis=1)
        b = np.zeros(shape=(1, _buf.k + 1))
        b[:, -1] = 1
        A, b = A.T, b.T
        x = _const.solver(A, b)
        _buf.pi = x.T
        return _buf.P, _buf.pi, _buf.gamma

    def backward(self, grad_from_P=None, grad_from_pi=None, grad_from_gamma=None):
        r"""Backwarding

        Args
        ----
        grad_from_P : numpy.ndarray or None
            Previous gradients from transactions.
        grad_from_pi : numpy.ndarray or None
            Previous gradients from states.
        grad_from_pi : numpy.ndarray or None
            Previous gradients from uniformalization denominator.

        Returns
        -------
        grad_to_mx : numpy.ndarray or None
            Gradients to transition rate matrix.

        """
        # security check
        if self._signal != self.FORWARDED:
            raise RuntimeError('there is no forwarding to backprop.')
        else:
            self._signal = self.NULL

        # claim class
        _const = DDMP

        # claim temporary buffer
        _buf = self._buffer

        # enforce at least one source gradient
        assert not (grad_from_P is None and grad_from_pi is None)

        # gamma never have gradient
        assert grad_from_gamma is None

        # fill P to X Jacobian gradients
        jacob_P = _grad_X2P_dnp(_buf.X, _buf.did)

        # allocate and fill P^{\infty} to \pi Jacobian gradients
        jacob_pi = np.zeros(shape=(_buf.k, _buf.k, _buf.k), dtype=_buf.P.dtype)
        for i in range(_buf.k):
            jacob_pi[i, :, i] = 1 / _buf.k

        # get infinite Jacobian gradients
        coeff_P = np.transpose(jacob_P, (2, 3, 0, 1))
        coeff_inf = _const.infsum(_buf.pi, coeff_P, _buf.P, **_const.dkargs)

        # backward from each transaction
        if grad_from_P is None:
            grad_of_trans = np.zeros(shape=(_buf.k, _buf.k), dtype=_buf.P.dtype)
        else:
            grad_of_trans = grad_from_P.reshape(_buf.k, _buf.k, 1, 1) * jacob_P
            grad_of_trans = np.sum(grad_of_trans, axis=(0, 1))

        # backward from each state
        if grad_from_pi is None:
            grad_of_states = np.zeros(shape=(_buf.k, _buf.k), dtype=_buf.P.dtype)
        else:
            jacob_pi = jacob_pi.reshape(_buf.k, 1, 1, _buf.k, _buf.k)
            grad_of_states = []
            for i in range(_buf.k):
                coeff = jacob_pi[i] * coeff_inf
                coeff = np.sum(coeff, axis=(-2, -1))
                grad = (grad_from_pi[0, i] * coeff).reshape(1, _buf.k, _buf.k)
                grad_of_states.append(grad)
            grad_of_states = np.concatenate(grad_of_states, axis=0)
            grad_of_states = np.sum(grad_of_states, axis=0)

        # aggregate two backwarding
        return grad_of_trans + grad_of_states


r"""
Benchmark
=========
Verify the PyTorch implementation on random input generated from fixed random seed. Since the bottom implementation
may be different for PyTorch and Numpy autograd, noise is allowed between PyTorch result and ground truth numpy
result.

In some infinite power summation tricks, we have random variable sampling. To avoid introducing noise by this, a
constant variable generator is defined.

We use absolute tolerance to control the noise to ensure result availability. Tolerance is examined over following
metrics.
1. Transition probability matrix (Batch).
2. Steady state distrbution vector (Batch).
3. Gradients from both transition probability matrix and steady state distrbution vector (Scalar:Batch-Summation).
------------------------------------------------------------------------------------------------------------------
"""


if __name__ == '__main__':
    # set tolerance
    atot = 1e-10

    # test environment
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    # set random settings
    seed = 43
    np.set_printoptions(precision=8, suppress=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # define constant geometryvariable for testing
    class ConstGeo(object):
        @staticmethod
        def rvs():
            return 10
        @staticmethod
        def sf(x):
            return stats.geom.sf(x, 0.1)

    # define birth-death basis
    bsz = 16
    k = 16
    birth_base = np.zeros(shape=(1, k, k)).astype(int)
    death_base = np.zeros(shape=(1, k, k)).astype(int)
    for i in range(k - 1):
        birth_base[:, i, i + 1] = 1
        death_base[:, i + 1, i] = 1

    # generate a batch to test
    vmin, vmax = 64, 128
    birth_rate = np.random.randint(vmin, vmax + 1)
    death_rate = np.random.randint(birth_rate - vmin // 2, birth_rate + vmin // 2, size=(bsz, 1, 1))
    true_X = (birth_rate * birth_base + death_rate * death_base).astype(float)

    # set ddmp numpy autograd hyperparameters
    DDMP.set_c(128)
    DDMP.set_solver(_numpy_solver_numpy)
    DDMP.set_infsum(_russian_roulette_inf_split_numpy)
    DDMP.set_dkargs(dict(rvdist=(ConstGeo(),)))

    # compute ground truth result by numpy
    df_ddmp_numpy, true_P, true_pi = [], [], []
    for i in range(bsz):
        df_ddmp_numpy.append(_DDMP_numpy())
        _P, _pi, _ = df_ddmp_numpy[i].forward(true_X[i])
        true_P.append(_P)
        true_pi.append(_pi)
    true_P, true_pi = np.stack(true_P), np.stack(true_pi)

    # set ddmp pytorch autograd hyperparameters
    DDMP.set_c(128)
    DDMP.set_solver(_numpy_solver)
    DDMP.set_infsum(_russian_roulette_inf_split)
    DDMP.set_dkargs(dict(rvdist=(ConstGeo(),)))

    # compute ground truth result by pytorch
    batch_th = torch.from_numpy(true_X)
    batch_th = batch_th.to(device)
    P, pi, _ = DDMP.apply(batch_th)

    # verify pytorch by numpy as ground truth
    assert np.all(np.fabs(P.data.cpu().numpy() - true_P) < atot), \
           'Transition probability matrix is intolerable.'
    assert np.all(np.fabs(pi.data.cpu().numpy() - true_pi) < atot), \
           'Steady state distribution vector is intolerable.'

    # generate an initial batch to test autograd
    init_X = (np.ones(shape=(bsz, 1, 1)) * birth_base + death_rate * death_base).astype(float)

    # set ddmp numpy autograd hyperparameters
    DDMP.set_c(128)
    DDMP.set_solver(_numpy_solver_numpy)
    DDMP.set_infsum(_russian_roulette_inf_split_numpy)
    DDMP.set_dkargs(dict(rvdist=(ConstGeo(),)))

    # define loss function
    def mse_loss_numpy(output, target):
        return np.sum(np.square(output - target))

    # compute gradients by numpy
    df_ddmp_numpy, init_P, init_pi = [], [], []
    init_loss_P, init_loss_pi = [], []
    init_grad = []
    for i in range(bsz):
        df_ddmp_numpy.append(_DDMP_numpy())
        _P, _pi, _ = df_ddmp_numpy[i].forward(init_X[i])
        init_P.append(_P)
        init_pi.append(_pi)
        init_loss_P.append(mse_loss_numpy(_P, true_P[i]))
        init_loss_pi.append(mse_loss_numpy(_pi, true_pi[i]))
        _grad = df_ddmp_numpy[i].backward(2 * (_P - true_P[i]), 2 * (_pi - true_pi[i]))
        init_grad.append(_grad)
    init_P, init_pi = np.stack(init_P), np.stack(init_pi)
    init_loss_P, init_loss_pi = np.sum(init_loss_P), np.sum(init_loss_pi)
    init_loss = init_loss_P + init_loss_pi
    init_grad = np.sum(np.stack(init_grad), axis=0)

    # set ddmp pytorch autograd hyperparameters
    DDMP.set_c(128)
    DDMP.set_solver(_numpy_solver)
    DDMP.set_infsum(_russian_roulette_inf_split)
    DDMP.set_dkargs(dict(rvdist=(ConstGeo(),)))

    # compute gradients by pytorch
    birth_rate_th = torch.nn.Parameter(torch.DoubleTensor([1]).to(device))
    birth_base_th = torch.from_numpy(birth_base).double().to(device)
    death_rate_th = torch.from_numpy(death_rate).double().to(device)
    death_base_th = torch.from_numpy(death_base).double().to(device)
    batch_th = birth_rate_th * birth_base_th + death_rate_th * death_base_th
    P, pi, _ = DDMP.apply(batch_th)
    loss_P = torch.sum((P - torch.from_numpy(true_P).to(device)) ** 2)
    loss_pi = torch.sum((pi - torch.from_numpy(true_pi).to(device)) ** 2)
    loss = loss_P + loss_pi
    loss.backward()

    # loss should be the same
    assert np.fabs(loss.data.item() - init_loss) < atot, \
           'Loss is intolerable.'

    # gradient should be the same
    assert birth_rate_th.grad.data.item() - (init_grad * birth_base).sum() < atot, \
           'Gradient is intolerable.'

    # pass the verification
    print('PyTorch Differentiable Discrete Markov Process is Verified!')
    print('Please Continue to Verify Convergence On real tasks!')