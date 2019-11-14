import numpy as np
import torch


r"""
Prior
=====
This part contains prior structure used to construct transition rate matrices for multiple Markov processes. It is
specified written for PyTorch, so all arguments and returns in transition rate matrix construction (except
initialization parameters) must be torch.Tensor.

- MMmKPrior
  Prior information class for M/M/$m$/$K$ Markov process.

- WebBrowsePrior
  Prior information class for Web Browsing Markov process.

- UpTriPrior
  Prior information class for Upper Triangular Markov process.
---------------------------------------------------------
"""


class MMmKPrior(object):
    r"""M/M/$m$/$K$ Structure Prior"""
    def __init__(self, k, m, c, tensor=torch.Tensor, device='cpu'):
        r"""Initialize the class

        Args
        ----
        k : int
            $k$ for M/M/$m$/$K$.
        m : int
            $m$ for M/M/$m$/$K$.
        c : float
            Uniformalization offset.
        tensor : type
            Tensor type used to create basis.
        device : str
            Device to work on.

        """
        # save necessary attributes
        self.k = k
        self.m = m
        self.c = c
        self.dtype = tensor.dtype
        self.device = device

        # allocate structure matrices
        self.birth_basis = torch.zeros(self.k, self.k, dtype=self.dtype, device=self.device)
        self.death_basis = torch.zeros(self.k, self.k, dtype=self.dtype, device=self.device)

        # fill structure matrices
        for i in range(self.k - 1):
            self.birth_basis.data[i, i + 1] = 1
            self.death_basis.data[i + 1, i] = min(i + 1, self.m)

        # get useful transitions
        self.useful = self.get_useful_trans()

    def trans_rate_mx(self, birth_rate, death_rate, noise=None):
        r"""Get transition rate matrix

        Args
        ----
        birth_rate : torch.Tensor
            Birth rate.
        death_rate : torch.Tensor
            Death rate.
        noise : torch.Tensor
            Noise to transition rate matrix.

        Returns
        -------
        mx : torch.Tensor
            Transition rate matrix.

        """
        # safety check
        assert len(torch.nonzero(birth_rate < 0)) == 0, \
               "negative birth rate is invalid."
        assert len(torch.nonzero(death_rate < 0)) == 0, \
               "negative death rate is invalid."
        assert noise is None or len(torch.nonzero(noise < 0)) == 0, \
               "negative noise in invalid."
        assert len(torch.nonzero(birth_rate != birth_rate)) == 0, \
               "nan birth rate is invalid."
        assert len(torch.nonzero(death_rate != death_rate)) == 0, \
               "nan death rate is invalid."
        assert noise is None or len(torch.nonzero(noise != noise)) == 0, \
               "nan noise in invalid."

        # broadcast
        birth_rate = birth_rate.view(-1, 1, 1)
        death_rate = death_rate.view(-1, 1, 1)

        # get raw transition rate matrix
        mx = birth_rate * self.birth_basis + death_rate * self.death_basis

        # add valid noise
        if noise is not None:
            # broadcast
            noise = noise.view(-1, self.k, self.k)

            # clean noise on diagonal and basis overlapping positions
            noise = noise * (1 - self.useful)

            # add clean noise
            mx = mx + noise
        else:
            pass
        return mx

    def get_useful_trans(self):
        """Get useful transactions
        
        Returns
        -------
        mx : torch.Tensor
            Useful transition as 01 matrix.

        """
        # generate diagonal line first
        mx = torch.eye(self.k, dtype=self.dtype, device=self.device)

        # add other basis
        mx = mx + self.birth_basis
        mx = mx + self.death_basis
        return (mx > 0).to(self.dtype).to(self.device)


class WebBrowsePrior(object):
    r"""Web Browsing Structure Prior"""
    def __init__(self, u, r, b, c, tensor=torch.Tensor, device='cpu'):
        r"""Initialize the class

        Args
        ----
        u : int
            Number of users.
        r : int
            Size of request buffer.
        b : int
            Size of bucket buffer.
        c : float
            Uniformalization offset.
        tensor : type
            Tensor type used to create basis.
        device : str
            Device to work on.

        """
        # save necessary attributes
        self.u = u
        self.r = r
        self.b = b
        self.c = c
        self.dtype = tensor.dtype
        self.device = device

        # allocate all states
        self.k = (self.r + 1) * (self.b + 1)

        # allocate structure matrices
        self.birth_basis = torch.zeros(self.k, self.k, dtype=self.dtype, device=self.device)
        self.buckt_basis = torch.zeros(self.k, self.k, dtype=self.dtype, device=self.device)
        self.death_basis = torch.zeros(self.k, self.k, dtype=self.dtype, device=self.device)

        # fill structure matrices
        for i in range(self.r + 1):
            for j in range(self.b + 1):
                src = i * (self.b + 1) + j

                # birth
                if i < self.r:
                    dst = (i + 1) * (self.b + 1) + j
                    self.birth_basis.data[src, dst] = max(0, self.u - i)
                else:
                    pass

                # bucket
                if j < self.b:
                    dst = i * (self.b + 1) + (j + 1)
                    self.buckt_basis.data[src, dst] = 1
                else:
                    pass

                # death
                if i > 0 and j > 0:
                    dst = (i - 1) * (self.b + 1) + (j - 1)
                    self.death_basis.data[src, dst] = 1
                else:
                    pass

        # get useful transitions
        self.useful = self.get_useful_trans()

    def trans_rate_mx(self, birth_rate, buckt_rate, death_rate, noise=None):
        r"""Get transition rate matrix

        Args
        ----
        birth_rate : torch.Tensor
            Birth rate.
        buckt_rate : torch.Tensor
            Bucket rate.
        death_rate : torch.Tensor
            Death rate.
        noise : torch.Tensor
            Noise to transition rate matrix.

        Returns
        -------
        mx : torch.Tensor
            Transition rate matrix.

        """
        # safety check
        assert len(torch.nonzero(birth_rate < 0)) == 0, \
               "negative birth rate is invalid."
        assert len(torch.nonzero(buckt_rate < 0)) == 0, \
               "negative bucket rate is invalid."
        assert len(torch.nonzero(death_rate < 0)) == 0, \
               "negative death rate is invalid."
        assert noise is None or len(torch.nonzero(noise < 0)) == 0, \
               "negative noise in invalid."
        assert len(torch.nonzero(birth_rate != birth_rate)) == 0, \
               "nan birth rate is invalid."
        assert len(torch.nonzero(buckt_rate != buckt_rate)) == 0, \
               "nan bucket rate is invalid."
        assert len(torch.nonzero(death_rate != death_rate)) == 0, \
               "nan death rate is invalid."
        assert noise is None or len(torch.nonzero(noise != noise)) == 0, \
               "nan noise in invalid."

        # broadcast
        birth_rate = birth_rate.view(-1, 1, 1)
        buckt_rate = buckt_rate.view(-1, 1, 1)
        death_rate = death_rate.view(-1, 1, 1)

        # get raw transition rate matrix
        mx = birth_rate * self.birth_basis + buckt_rate * self.buckt_basis + death_rate * self.death_basis

        # add valid noise
        if noise is not None:
            # broadcast
            noise = noise.view(-1, self.k, self.k)

            # clean noise on diagonal and basis overlapping positions
            noise = noise * (1 - self.useful)

            # add clean noise
            mx = mx + noise
        else:
            pass
        return mx

    def get_useful_trans(self):
        """Get useful transactions
        
        Returns
        -------
        mx : torch.Tensor
            Useful transition as 01 matrix.

        """
        # generate diagonal line first
        mx = torch.eye(self.k, dtype=self.dtype, device=self.device)

        # add other basis
        mx = mx + self.birth_basis
        mx = mx + self.buckt_basis
        mx = mx + self.death_basis
        return (mx > 0).to(self.dtype).to(self.device)


class UpTriPrior(object):
    r"""Upper Triangular Structure Prior"""
    def __init__(self, k, c, tensor=torch.Tensor, device='cpu'):
        r"""Initialize the class

        Args
        ----
        k : int
            $k$ for upper triangular.
        c : float
            Uniformalization offset.
        tensor : type
            Tensor type used to create basis.
        device : str
            Device to work on.

        """
        # save necessary attributes
        self.k = k
        self.c = c
        self.dtype = tensor.dtype
        self.device = device

        # allocate structure matrices
        self.birth_basis = torch.zeros(self.k, self.k, dtype=self.dtype, device=self.device)
        self.lowtr_basis = torch.ones(self.k, self.k, dtype=self.dtype, device=self.device)

        # fill structure matrices
        for i in range(self.k - 1):
            self.birth_basis.data[i, i + 1] = 1
        self.lowtr_basis = torch.tril(self.lowtr_basis, diagonal=-1)

        # get useful transitions
        self.useful = self.get_useful_trans()

    def trans_rate_mx(self, birth_rate, lowtr_rate, noise):
        r"""Get transition rate matrix

        Args
        ----
        birth_rate : torch.Tensor
            Birth rate.
        lowtr_rate : torch.Tensor
            Lower triangular tensor.
        noise : torch.Tensor
            Noise to transition rate matrix.

        Returns
        -------
        mx : torch.Tensor
            Transition rate matrix.

        """
        # safety check
        assert len(torch.nonzero(birth_rate < 0)) == 0, \
               "negative birth rate is invalid."
        assert len(torch.nonzero(lowtr_rate[torch.tril(torch.ones(self.k, self.k), diagonal=-1) == 1] < 0)) == 0, \
               "negative lower triangular rate is invalid."
        assert noise is None or len(torch.nonzero(noise < 0)) == 0, \
               "negative noise in invalid."
        assert len(torch.nonzero(birth_rate != birth_rate)) == 0, \
               "nan birth rate is invalid."
        assert len(torch.nonzero(lowtr_rate != lowtr_rate)) == 0, \
               "nan lower triangular rate is invalid."
        assert noise is None or len(torch.nonzero(noise != noise)) == 0, \
               "nan noise in invalid."

        # broadcast
        birth_rate = birth_rate.view(-1, 1, 1)
        lowtr_rate = lowtr_rate.view(-1, self.k, self.k)

        # get raw transition rate matrix
        mx = birth_rate * self.birth_basis + lowtr_rate * self.lowtr_basis

        # add valid noise
        if noise is not None:
            # broadcast
            noise = noise.view(-1, self.k, self.k)

            # clean noise on diagonal and basis overlapping positions
            noise = noise * (1 - self.useful)

            # add clean noise
            mx = mx + noise
        else:
            pass
        return mx

    def get_useful_trans(self):
        """Get useful transactions
        
        Returns
        -------
        mx : torch.Tensor
            Useful transition as 01 matrix.

        """
        # generate diagonal line first
        mx = torch.eye(self.k, dtype=self.dtype, device=self.device)

        # add other basis
        mx = mx + self.birth_basis
        mx = mx + self.lowtr_basis
        return (mx > 0).to(self.dtype).to(self.device)


class MMMulKPrior(object):
    r"""M/M/Multiple/$K$ Structure Prior"""
    def __init__(self, k, d, c, tensor=torch.Tensor, device='cpu'):
        r"""Initialize the class

        Args
        ----
        k : int
            $k$ for M/M/1/$K$.
        d : int
            Number of multiple diagonal lines.
        c : float
            Uniformalization offset.
        tensor : type
            Tensor type used to create basis.
        device : str
            Device to work on.

        """
        # save necessary attributes
        self.k = k
        self.d = d
        self.c = c
        self.dtype = tensor.dtype
        self.device = device

        # allocate structure matrices
        self.birth_basis = torch.zeros(self.k, self.k, dtype=self.dtype, device=self.device)
        self.death_basis = torch.zeros(self.d, self.k, self.k, dtype=self.dtype, device=self.device)

        # fill structure matrices
        for i in range(self.k - 1):
            self.birth_basis.data[i, i + 1] = 1
        for i in range(self.d):
            for j in range(self.k - i - 1):
                self.death_basis.data[i, j + i + 1, j] = 1

        # get useful transitions
        self.useful = self.get_useful_trans()

    def trans_rate_mx(self, birth_rate, death_rate, noise=None):
        r"""Get transition rate matrix

        Args
        ----
        birth_rate : torch.Tensor
            Birth rate.
        death_rate : torch.Tensor
            Death rate.
        noise : torch.Tensor
            Noise to transition rate matrix.

        Returns
        -------
        mx : torch.Tensor
            Transition rate matrix.

        """
        # safety check
        assert len(torch.nonzero(birth_rate < 0)) == 0, \
               "negative birth rate is invalid."
        assert len(torch.nonzero(death_rate < 0)) == 0, \
               "negative death rate is invalid."
        assert noise is None or len(torch.nonzero(noise < 0)) == 0, \
               "negative noise in invalid."
        assert len(torch.nonzero(birth_rate != birth_rate)) == 0, \
               "nan birth rate is invalid."
        assert len(torch.nonzero(death_rate != death_rate)) == 0, \
               "nan death rate is invalid."
        assert noise is None or len(torch.nonzero(noise != noise)) == 0, \
               "nan noise in invalid."

        # broadcast
        birth_rate = birth_rate.view(-1, 1, 1)
        death_rate = death_rate.view(-1, self.d)

        # get raw transition rate matrix
        death_mx = death_rate[:, 0] * self.death_basis[[0]]
        for i in range(1, self.d):
            death_mx = death_mx + death_rate[:, i] * self.death_basis[[i]]
        mx = birth_rate * self.birth_basis + death_mx.view(-1, self.k, self.k)

        # add valid noise
        if noise is not None:
            # broadcast
            noise = noise.view(-1, self.k, self.k)

            # clean noise on diagonal and basis overlapping positions
            noise = noise * (1 - self.useful)

            # add clean noise
            mx = mx + noise
        else:
            pass
        return mx

    def get_useful_trans(self):
        """Get useful transactions
        
        Returns
        -------
        mx : torch.Tensor
            Useful transition as 01 matrix.

        """
        # generate diagonal line first
        mx = torch.eye(self.k, dtype=self.dtype, device=self.device)

        # add other basis
        mx = mx + self.birth_basis
        for i in range(self.d):
            mx = mx + self.death_basis[i]
        return (mx > 0).to(self.dtype).to(self.device)



r"""
Benchmark
=========
Verify the PyTorch implementation of structure priors used to generate transition rate matrices for specified
Markov processes. The construction given by the prior should exactly match the manual constructed transition rate
matrices.

For each prior, three possible usages are examined:
1. Single sample generation.
2. Batch generation without broadcasting.
2. Batch generation with broadcasting.

Following priors are examined:
1. M/M/$m$/$K$
--------------
"""


if __name__ == '__main__':
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

    # set prior configurations
    k = 5
    m = 2
    tensor = torch.DoubleTensor

    # create ground truth
    mx1 = torch.zeros(k, k).type(tensor).to(device)
    mx2 = torch.zeros(k, k).type(tensor).to(device)
    noise1 = torch.from_numpy(np.random.randint(0, 3, size=(k, k))).type(tensor).to(device)
    noise2 = torch.from_numpy(np.random.randint(0, 3, size=(k, k))).type(tensor).to(device)
    clean1 = noise1.clone()
    clean2 = noise2.clone()
    for i in range(k):
        clean1.data[i, i] = 0
        clean2.data[i, i] = 0
    for i in range(k - 1):
        mx1.data[i, i + 1] = 7
        mx2.data[i, i + 1] = 9
        clean1.data[i, i + 1] = 0
        clean2.data[i, i + 1] = 0
        mx1.data[i + 1, i] = 3 * min(i + 1, m)
        mx2.data[i + 1, i] = 3 * min(i + 1, m)
        clean1.data[i + 1, i] = 0
        clean2.data[i + 1, i] = 0

    # create prior
    prior = MMmKPrior(k=k, m=m, c=10, tensor=tensor, device=device)

    # test single matrix generation
    b = tensor([7]).to(device)
    d = tensor([3]).to(device)
    noise = noise1
    res = prior.trans_rate_mx(b, d, noise)
    diff = res - mx1 - clean1
    assert np.all(diff.data.cpu().numpy() == 0), \
           'Single prior transition rate matrix construction is wrong.'

    # test batch matrices generation
    b = tensor([7, 9]).to(device)
    d = tensor([3, 3]).to(device)
    noise = torch.stack([noise1, noise2])
    res = prior.trans_rate_mx(b, d, noise)
    diff = res - torch.stack([mx1, mx2]) - torch.stack([clean1, clean2])
    assert np.all(diff.data.cpu().numpy() == 0), \
           'Batch prior transition rate matrix construction is wrong.'

    # test batch matrices generation with broadcasting
    b = tensor([7, 9]).to(device)
    d = tensor([3]).to(device)
    noise = noise1
    res = prior.trans_rate_mx(b, d, noise)
    diff = res - torch.stack([mx1, mx2]) - torch.stack([clean1, clean1])
    assert np.all(diff.data.cpu().numpy() == 0), \
           'Batch prior transition rate matrix construction with broadcasting is wrong.'

    # create prior
    prior = WebBrowsePrior(u=5, r=3, b=2, c=43, tensor=tensor, device=device)

    # test single matrix generation
    br = tensor([7]).to(device)
    bu = tensor([5]).to(device)
    de = tensor([3]).to(device)
    res = prior.trans_rate_mx(br, bu, de)
    st2id = {}
    for i in range(prior.r + 1):
        for j in range(prior.b + 1):
            st2id[(i, j)] = len(st2id)
    for i in range(prior.r + 1):
        for j in range(prior.b + 1):
            src = st2id[(i, j)]
            if i + 1 < prior.r:
                dst = st2id[(i + 1, j)]
                assert prior.birth_basis[src, dst] == max(0, prior.c - i), \
                       'Prior (birth) transition rate matrix construction is wrong.'
            else:
                pass
            if j + 1 < prior.b:
                dst = st2id[(i, j + 1)]
                assert prior.buckt_basis[src, dst] == 1, \
                       'Prior (bucket) transition rate matrix construction is wrong.'
            else:
                pass
            if i > 0 and j > 0:
                dst = st2id[(i - 1, j - 1)]
                assert prior.death_basis[src, dst] == 1, \
                       'Prior (death) transition rate matrix construction is wrong.'
            else:
                pass