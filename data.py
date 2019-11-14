import os
import numpy as np
import torch
import pandas as pd
import ddmp
import mcpro
import mcsim


r"""
Data Prototype
==============
This part provides deep learning dataset prototype used for PyTorch implemented models.

The prototype support data generation and loading for simulation data, and just loading for published datasets.
It only supports iterable batch generation with shuffling. Batch size can be either a constant given by
iterator initialization or a variable given on every batch requirement iteration.

It should also, but not necessarily, provide a print function for generated batch.

- _Data
  Data prototype.
-----------------
"""


class _Data(object):
    r"""Class Prototype for Data"""
    def __init__(self, *args, **kargs):
        r"""Initialize the class"""
        raise NotImplementedError

    def __len__(self):
        r"""Get length"""
        # length is number of samples
        return len(self.samples)

    class Iterator(object):
        r"""Iterator for Data"""
        def __init__(self, data, bsz=None, order=None):
            r"""Initialize the class

            Args
            ----
            data : data._Data
                Data source.
            bsz : int or None
                Batch size.
                If it is None, then batch size should be given on requirement.
            order : numpy.ndarray or None
                Specify the iteration order over data samples.
                If it is None, order is a random permutation.

            """
            # save necessary attributes
            self.data = data
            self.bsz = bsz

            # initialize the iteration order
            self.order = np.random.permutation(len(self.data)) if order is None else order
            self.ptr = 0

        def __next__(self, bsz=None):
            r"""Get next iteration

            Args
            ----
            bsz : int
                Batch size on requirement.

            Returns
            -------
            batch : object
                All data in batch of next iteration.

            """
            # batch size should exist and not conflict
            assert not (self.bsz is None and bsz is None)
            assert self.bsz is None or bsz is None

            # get safe batch size
            bsz = min(bsz or self.bsz, len(self.data) - self.ptr)

            # empty batch is a signal to stop
            if bsz == 0:
                raise StopIteration
            else:
                pass

            # get indices used in next iteration
            indices = self.order[self.ptr:self.ptr + bsz]

            # move pointer
            self.ptr += bsz

            # construct batch
            return self.data.construct_batch(indices)

        # alternative call
        next = __next__

    def __iter__(self, bsz=None, order=None):
        r"""Get an iterator

        Args
        ----
        data : data._Data
            Data source.
        bsz : int or None
            Batch size.
            If it is None, then batch size should be given on requirement.
        order : numpy.ndarray or None
            Specify the iteration order over data samples.
            If it is None, order is a random permutation.

        Returns
        -------
        iter : data._Data.Iterator
            Iterator.

        """
        # create an iterator
        return self.Iterator(self, bsz, order)

    # alternative call
    iter = __iter__

    def construct_batch(self, indices):
        r"""Construct a batch for given indices

        Args
        ----
        indices : numpy.ndarray
            Indices used to construct the batch.

        """
        # leave to exact case
        raise NotImplementedError

    def print_batch(self, batch):
        r"""Print batch for the class

        Args
        ----
        batch : object
            Batch data.

        """
        # leave to exact case
        raise NotImplementedError


r"""
Markov Process Data
===================
This part provides all Markov process related data generation.

- spectral_gap
  Compute the spectral gap for a given matrix.

- PriorData
  Data prototype with prior.

- MMmKData
  M/M/$m$/$k$ Data.

- WebBrowseData
  Web Browsing Data.
--------------------
"""


def spectral_gap(mx):
    r"""Compute the spectral gap for the given matrix

    Args
    ----
    mx : numpy.ndarray
        Matrix.

    Returns
    -------
    gap : float
        Spectral gap.

    """
    # eigendecompose
    vals = np.linalg.eigvals(mx)

    # get sepctral gap
    vals = np.sort(np.fabs(np.real(vals)))
    gap = vals[-1] - vals[-2]
    return gap


class PriorData(_Data):
    r"""Data Prototype with Prior"""
    def __init__(self, *args, **kargs):
        r"""Initialize the class"""
        raise NotImplementedError

    def rand_trans(self, num, useful=True):
        r"""Get random transitions to sample

        Args
        ----
        num : int or None
            Number of transitions to sample.
        useful : bool
            Sample only useful transitions.

        Returns
        -------
        ind : torch.Tensor or None
            Indices tensor of transitions to sample.

        """
        # null case
        if num is None or num == 0:
            return None
        else:
            pass

        # get sample binary
        src = self.prior.useful if useful else torch.ones(self.k, self.k)
        for i in range(self.k):
            src.data[i, i] = 0
        src = torch.nonzero(src)

        # randomly select given number of transitions
        ind = list(range(len(src)))
        np.random.shuffle(ind)
        assert 0 < num and num <= len(ind), \
               "too many transitions to sample."
        ind = np.sort(ind[0:num])
        ind = src[ind]
        return ind

    def construct_batch(self, indices):
        r"""Construct a batch for given indices

        Args
        ----
        indices : numpy.ndarray
            Indices used to construct the batch.

        Returns
        -------
        input : object
            Input part of the batch.
        target : object
            Target part of the batch.

        """
        # allocate batch memory
        input_birth_rate  = []
        target_cnt_trans  = []
        target_cnt_states = []
        target_P          = []
        target_pi         = []

        # traverse all indices
        for idx in indices:
            birth_rate, cnt_trans, cnt_states, P, pi = self.samples[idx]
            input_birth_rate.append(birth_rate)
            target_cnt_trans.append(cnt_trans)
            target_cnt_states.append(cnt_states)
            target_P.append(P)
            target_pi.append(pi)

        # transform data into valid tensors
        input_birth_rate = torch.LongTensor(input_birth_rate).type(self.tensor).to(self.device)
        target_cnt_trans = torch.stack(target_cnt_trans).type(self.tensor).to(self.device)
        target_cnt_states = torch.stack(target_cnt_states).type(self.tensor).to(self.device)
        target_P = torch.stack(target_P).type(self.tensor).to(self.device)
        target_pi = torch.stack(target_pi).type(self.tensor).to(self.device)
        return input_birth_rate, ((target_cnt_trans, target_cnt_states), (target_P, target_pi))

    def print_batch(self, batch):
        r"""Print batch for the class

        Args
        ----
        batch : object
            Batch data.

        """
        # parse batch input and target
        input_birth_rate, ((target_cnt_trans, target_cnt_states), (target_P, target_pi)) = batch

        # print in designed convention
        print('Birth Rate')
        print('----------')
        print(input_birth_rate.data.cpu().numpy())
        print(":: \'{}\' :: \'{}\'".format(input_birth_rate.dtype, input_birth_rate.device))
        print()
        print('Count of Transitions')
        print('--------------------')
        print(target_cnt_trans.data.cpu().numpy())
        print(":: \'{}\' :: \'{}\'".format(target_cnt_trans.dtype, target_cnt_trans.device))
        print()
        print('Transition Probability Matrix')
        print('-----------------------------')
        print(target_P.data.cpu().numpy())
        print()
        print('Count of States')
        print('---------------')
        print(target_cnt_states.data.cpu().numpy())
        print(":: \'{}\' :: \'{}\'".format(target_cnt_states.dtype, target_cnt_states.device))
        print()
        print(":: \'{}\' :: \'{}\'".format(target_P.dtype, target_P.device))
        print('Steady State Distribution Vector')
        print('--------------------------------')
        print(target_pi.data.cpu().numpy())
        print(":: \'{}\' :: \'{}\'".format(target_pi.dtype, target_pi.device))


class MMmKData(PriorData):
    r"""M/M/$m$/$k$ Data"""
    def __init__(self, k=None, m=None, death_rate=None, birth_vmin=None, birth_vmax=None, sample_times=None,
                 save=None, load=None, device='cpu', t=None, num_trans=None, useful_trans=None):
        r"""Initialize the class

        Args
        ----
        k : int
            Number of states.
        m : int
            Number of servers.
        death_rate : int
            Constant death rate of the dataset.
        birth_vmin : int
            Minimum value of variahle birth rate of the dataset.
        birth_vmax : int
            Maximum value of variahle birth rate of the dataset.
        sample_times : int
            Number of sample times from the same process configuration.
        save : str
            Path to save generated data.
        load : str or None
            Path to load generated data.
            If it is not None, real generation process will be ignored.
        device : str
            Device to work on.
        t : None or int
            Simulation duration for each sample.
        num_trans : int
            Number of transitions to sample.
        useful_trans : bool
            Sample only useful transitions.

        """
        # set generation settings
        self.tensor = torch.DoubleTensor

        # load directly if necessary
        if load is not None:
            buffer = torch.load(load)
            self.k = buffer['k']
            self.m = buffer['m']
            self.cmax = buffer['cmax']
            self.prior = mcpro.MMmKPrior(k=self.k, m=self.m, c=self.cmax, tensor=self.tensor, device=device)
            self.train_trans  = buffer['train_trans']
            self.test_trans   = None
            self.train_states = [0, 1]
            self.test_states  = [self.k - 1]
            self.samples = buffer['samples']
            self.device = device
            self.min_gap = buffer['min_gap']
            self.max_gap = buffer['max_gap']
            return
        else:
            pass

        # save necessary attributes
        self.k = k
        self.m = m
        self.cmax = (birth_vmax + death_rate) * self.m
        self.prior = mcpro.MMmKPrior(k=self.k, m=self.m, c=self.cmax, tensor=self.tensor, device=device)
        self.train_trans  = self.rand_trans(num=num_trans, useful=useful_trans)
        self.test_trans   = None
        self.train_states = [0, 1]
        self.test_states  = [self.k - 1]

        # allocate sample buffer
        self.samples = []
        self.min_gap = None
        self.max_gap = None

        # generate samples ara saved only on CPU
        sim = mcsim.CMPLogSim(prior=self.prior)
        for birth_rate in range(birth_vmin, birth_vmax + 1):
            # // # scale birth rate with respect to m
            # // birth_rate *= self.m

            for i in range(sample_times):
                # simulate once
                b = self.tensor([birth_rate]).to(device)
                d = self.tensor([death_rate]).to(device)
                _trans, _states, P, pi = sim.simulate(t=t, birth_rate=b, death_rate=d)
                # // if len(torch.nonzero(pi[self.test_states] < 1e-8)) > 0:
                # //     print("distribution is less than precision 1e-8 ({}, {})".format(birth_rate, death_rate))
                # //     exit()
                # // else:
                # //     pass
                _trans = _trans.data.cpu()
                _states = _states.data.cpu()
                P = P.data.cpu()
                pi = pi.data.cpu()
                self.samples.append((birth_rate, _trans, _states, P, pi))

                # update spectral gap
                gap = spectral_gap(P.numpy())
                self.min_gap = gap if self.min_gap is None else min(self.min_gap, gap)
                self.max_gap = gap if self.max_gap is None else max(self.max_gap, gap)

        # save generation
        state_buffer = dict(
            k=self.k, m=self.m, cmax=self.cmax, train_trans=self.train_trans, samples=self.samples,
            min_gap=self.min_gap, max_gap=self.max_gap)
        torch.save(state_buffer, save)

        # save device settings
        self.device = device


class WebBrowseData(PriorData):
    r"""Web Browse Data"""
    def __init__(self, u=None, r=None, b=None, bucket_rate=None, death_rate=None, birth_vmin=None, birth_vmax=None,
                 sample_times=None, save=None, load=None, device='cpu', t=None, num_trans=None, useful_trans=None):
        r"""Initialize the class

        Args
        ----
        u : int
            Number of users.
        r : int
            Size of request buffer.
        b : int
            Size of bucket buffer.
        bucket_rate : int
            Constant bucket rate of the dataset.
        death_rate : int
            Constant death rate of the dataset.
        birth_vmin : int
            Minimum value of variahle birth rate of the dataset.
        birth_vmax : int
            Maximum value of variahle birth rate of the dataset.
        sample_times : int
            Number of sample times from the same process configuration.
        save : str
            Path to save generated data.
        load : str or None
            Path to load generated data.
            If it is not None, real generation process will be ignored.
        device : str
            Device to work on.
        t : None or int
            Simulation duration for each sample.
        num_trans : int
            Number of transitions to sample.
        useful_trans : bool
            Sample only useful transitions.

        """
        # set generation settings
        self.tensor = torch.DoubleTensor

        # load directly if necessary
        if load is not None:
            buffer = torch.load(load)
            self.u = buffer['u']
            self.r = buffer['r']
            self.b = buffer['b']
            self.cmax = buffer['cmax']
            self.prior = mcpro.WebBrowsePrior(
                u=self.u, r=self.r, b=self.b, c=self.cmax, tensor=self.tensor, device=device)
            self.k = self.prior.k
            self.train_trans  = buffer['train_trans']
            self.test_trans   = None
            self.train_states = [0, 1, self.b + 1, self.b + 2]
            self.test_states  = [self.r * (self.b + 1)]
            self.samples = buffer['samples']
            self.device = device
            self.min_gap = buffer['min_gap']
            self.max_gap = buffer['max_gap']
            return
        else:
            pass

        # save necessary attributes
        self.u = u
        self.r = r
        self.b = b
        self.cmax = birth_vmax * self.u + bucket_rate
        self.prior = mcpro.WebBrowsePrior(
            u=self.u, r=self.r, b=self.b, c=self.cmax, tensor=self.tensor, device=device)
        self.k = self.prior.k
        self.train_trans  = self.rand_trans(num=num_trans, useful=useful_trans)
        self.test_trans   = None
        self.train_states = [0, 1, self.b + 1, self.b + 2]
        self.test_states  = [self.r * (self.b + 1)]

        # allocate sample buffer
        self.samples = []
        self.min_gap = None
        self.max_gap = None

        # generate samples ara saved only on CPU
        sim = mcsim.CMPLogSim(prior=self.prior)
        for birth_rate in range(birth_vmin, birth_vmax + 1):
            for i in range(sample_times):
                # simulate once
                br = self.tensor([birth_rate]).to(device)
                bu = self.tensor([bucket_rate]).to(device)
                de = self.tensor([death_rate]).to(device)
                _trans, _states, P, pi = sim.simulate(t=t, birth_rate=br, buckt_rate=bu, death_rate=de)
                # // if len(torch.nonzero(pi[self.test_states] < 1e-8)) > 0:
                # //     print("distribution is less than precision 1e-8 ({}, {})".format(birth_rate, death_rate))
                # //     exit()
                # // else:
                # //     pass
                _trans = _trans.data.cpu()
                _states = _states.data.cpu()
                P = P.data.cpu()
                pi = pi.data.cpu()
                self.samples.append((birth_rate, _trans, _states, P, pi))

                # update spectral gap
                gap = spectral_gap(P.numpy())
                self.min_gap = gap if self.min_gap is None else min(self.min_gap, gap)
                self.max_gap = gap if self.max_gap is None else max(self.max_gap, gap)

        # save generation
        state_buffer = dict(
            u=self.u, r=self.r, b=self.b, cmax=self.cmax, train_trans=self.train_trans, samples=self.samples,
            min_gap=self.min_gap, max_gap=self.max_gap)
        torch.save(state_buffer, save)

        # save device settings
        self.device = device


class EmuData(PriorData):
    r"""Emulation Data"""
    def __init__(self, qsz=None, lamin=None, lamax=None, save=None, load=None, device='cpu'):
        r"""Initialize the class

        Args
        ----
        qsz : int
            
        lamin : int
            Minimum (inclusive) of lambda to load.
        lamax : int
            Maximum (inclusive) of lambda to load.
        save : str
            Path to save generated data.
        load : str or None
            Path to load generated data.
            If it is not None, real generation process will be ignored.
        device : str
            Device to work on.

        """
        # set generation settings
        self.tensor = torch.DoubleTensor

        # load directly if necessary
        if load is not None:
            buffer = torch.load(load)
            self.k = buffer['k']
            self.m = buffer['m']
            self.cmax = buffer['cmax']
            self.prior = mcpro.MMmKPrior(k=self.k, m=self.m, c=self.cmax, tensor=self.tensor, device=device)
            self.k = self.prior.k
            self.train_trans  = None
            self.test_trans   = None
            self.train_states = [1, 2]
            self.test_states  = [self.k - 1]
            self.samples = buffer['samples']
            self.device = device
            self.min_gap = buffer['min_gap']
            self.max_gap = buffer['max_gap']
            return
        else:
            pass

        # save necessary attributes
        self.k = qsz + 2
        self.m = 1
        self.cmax = 0.001
        self.prior = mcpro.MMmKPrior(k=self.k, m=self.m, c=self.cmax, tensor=self.tensor, device=device)
        self.k = self.prior.k
        self.train_trans  = None
        self.test_trans   = None
        self.train_states = [1, 2]
        self.test_states  = [self.k - 1]

        # allocate sample buffer
        self.samples = []
        self.min_gap = None
        self.max_gap = None

        # load raw data
        raw_data = pd.read_csv('RTT_ESIMATE_Q_LEN_test_sample_110.csv')
        raw_data = raw_data[(lamin <= raw_data['callrate']) & (raw_data['callrate'] <= lamax)]
        for itr in raw_data.keys():
            if '=' in itr:
                ch = itr[0]
                break
            else:
                pass

        # sample observations for all lambda candidates
        for _, row in raw_data.iterrows():
            # parse data row
            lambd = row['callrate']

            # load observations
            obvs = [float('nan') for i in range(self.k)]
            pi   = [float('nan') for i in range(self.k)]
            for i in self.train_states:
                obvs[i] = row["{}={}".format(ch, i)]
                pi[i]   = float(obvs[i]) / lambd

            # parse dropping states
            drop_lst = []
            cnt = qsz + 1
            while True:
                key = "{}={}".format(ch, cnt)
                if key in row:
                    drop_lst.append(row[key])
                else:
                    break
                cnt += 1
            obvs[self.k - 1] = sum(drop_lst)
            pi[self.k - 1]   = float(obvs[self.k - 1]) / lambd

            # get tensors
            obvs = self.tensor(obvs)
            pi   = self.tensor(pi)
            mx   = self.tensor(self.k, self.k)
            mx.fill_(float('nan'))

            # append parsed data
            self.samples.append((lambd, mx, obvs, mx, pi))

        # set spectral gap
        self.min_gap = float('nan')
        self.max_gap = float('nan')

        # save generation
        state_buffer = dict(
            k=self.k, m=self.m, cmax=self.cmax, samples=self.samples,
            min_gap=self.min_gap, max_gap=self.max_gap)
        torch.save(state_buffer, save)

        # save device settings
        self.device = device


class MMMulKData(PriorData):
    r"""M/M/Multiple/$k$ Data"""
    def __init__(self, k=None, d=None, death_rate=None, birth_vmin=None, birth_vmax=None, sample_times=None,
                 save=None, load=None, device='cpu', t=None, num_trans=None, useful_trans=None):
        r"""Initialize the class

        Args
        ----
        k : int
            Number of states.
        d : int
            Number of multiple diagonal lines.
        death_rate : int
            Constant death rate of the dataset.
        birth_vmin : int
            Minimum value of variahle birth rate of the dataset.
        birth_vmax : int
            Maximum value of variahle birth rate of the dataset.
        sample_times : int
            Number of sample times from the same process configuration.
        save : str
            Path to save generated data.
        load : str or None
            Path to load generated data.
            If it is not None, real generation process will be ignored.
        device : str
            Device to work on.
        t : None or int
            Simulation duration for each sample.
        num_trans : int
            Number of transitions to sample.
        useful_trans : bool
            Sample only useful transitions.

        """
        # set generation settings
        self.tensor = torch.DoubleTensor

        # load directly if necessary
        if load is not None:
            buffer = torch.load(load)
            self.k = buffer['k']
            self.d = buffer['d']
            self.cmax = buffer['cmax']
            self.prior = mcpro.MMMulKPrior(k=self.k, d=self.d, c=self.cmax, tensor=self.tensor, device=device)
            self.train_trans  = buffer['train_trans']
            self.test_trans   = None
            self.train_states = list(range(self.d + 1))
            self.test_states  = [self.k - 1]
            self.samples = buffer['samples']
            self.device = device
            self.min_gap = buffer['min_gap']
            self.max_gap = buffer['max_gap']
            return
        else:
            pass

        # save necessary attributes
        self.k = k
        self.d = d
        self.cmax = birth_vmax + sum(death_rate)
        self.prior = mcpro.MMMulKPrior(k=self.k, d=self.d, c=self.cmax, tensor=self.tensor, device=device)
        self.train_trans  = self.rand_trans(num=num_trans, useful=useful_trans)
        self.test_trans   = None
        self.train_states = list(range(self.d + 1))
        self.test_states  = [self.k - 1]

        # allocate sample buffer
        self.samples = []
        self.min_gap = None
        self.max_gap = None

        # generate samples ara saved only on CPU
        sim = mcsim.CMPLogSim(prior=self.prior)
        for birth_rate in range(birth_vmin, birth_vmax + 1):
            for i in range(sample_times):
                # simulate once
                b = self.tensor([birth_rate]).to(device)
                d = self.tensor(death_rate).to(device)
                _trans, _states, P, pi = sim.simulate(t=t, birth_rate=b, death_rate=d)
                # // if len(torch.nonzero(pi[self.test_states] < 1e-8)) > 0:
                # //     print("distribution is less than precision 1e-8 ({}, {})".format(birth_rate, death_rate))
                # //     exit()
                # // else:
                # //     pass
                _trans = _trans.data.cpu()
                _states = _states.data.cpu()
                P = P.data.cpu()
                pi = pi.data.cpu()
                self.samples.append((birth_rate, _trans, _states, P, pi))

                # update spectral gap
                gap = spectral_gap(P.numpy())
                self.min_gap = gap if self.min_gap is None else min(self.min_gap, gap)
                self.max_gap = gap if self.max_gap is None else max(self.max_gap, gap)

        # save generation
        state_buffer = dict(
            k=self.k, d=self.d, cmax=self.cmax, train_trans=self.train_trans, samples=self.samples,
            min_gap=self.min_gap, max_gap=self.max_gap)
        torch.save(state_buffer, save)

        # save device settings
        self.device = device


r"""
Benchmark
=========
Verify data generation and loading.
-----------------------------------
"""


if __name__ == '__main__':
    # set tolerance
    atot = 5e-2

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

    # test long run
    print('[Check] Long Run')
    print('----------------')
    path = 'mm1k.pt'
    data = MMmKData(k=5, m=1, death_rate=25, birth_vmin=27, birth_vmax=27, sample_times=1, save=path, t=200)
    data = MMmKData(load=path, device=device)
    assert len(data.samples) == 1
    target_pi = data.samples[0][4]
    output_pi = data.samples[0][2].to(target_pi.dtype)
    output_pi = output_pi / output_pi.sum()
    assert len(torch.nonzero(torch.abs(target_pi - output_pi) > atot)) == 0
    os.remove(path)
    print()
    
    # test multiple run
    print('[Check] Mutiple Run')
    print('-------------------')
    path = 'mm1k.pt'
    data = MMmKData(k=20, m=1, death_rate=25, birth_vmin=27, birth_vmax=27, sample_times=200, save=path)
    data = MMmKData(load=path, device=device)
    target_pi = data.samples[0][4]
    output_pi = torch.zeros(target_pi.size(0), dtype=target_pi.dtype, device=target_pi.device)
    for itr in data.samples:
        assert len(torch.nonzero(target_pi - itr[4] != 0)) == 0
        sample = itr[2].to(target_pi.dtype)
        sample = sample / sample.sum()
        output_pi = output_pi + sample
    output_pi = output_pi / len(data.samples)
    assert len(torch.nonzero(torch.abs(target_pi - output_pi) > atot)) == 0
    os.remove(path)
    print()
    
    # generate, load and clean data
    path = 'mm1k.pt'
    data = MMmKData(k=20, m=1, death_rate=25, birth_vmin=21, birth_vmax=30, sample_times=5, save=path)
    data = MMmKData(load=path, device=device)
    
    # check sepctral gap
    print('Spectral Gap')
    print('------------')
    print("[{:.8f}, {:.8f}]".format(data.min_gap, data.max_gap))
    print()
    
    # check batch operations
    batch = data.iter(bsz=4)
    data.print_batch(batch.next())
    print()
    os.remove(path)
    
    # generate, load and clean data
    path = 'lbwb.pt'
    data = WebBrowseData(
        u=5, r=3, b=2, death_rate=25, bucket_rate=15, birth_vmin=1, birth_vmax=50, sample_times=4, save=path)
    data = WebBrowseData(load=path, device=device)
    
    # check sepctral gap
    print('Spectral Gap')
    print('------------')
    print("[{:.8f}, {:.8f}]".format(data.min_gap, data.max_gap))
    print()
    
    # check batch operations
    batch = data.iter(bsz=4)
    data.print_batch(batch.next())
    print()
    os.remove(path)

    # generate, load and clean data
    path = 'emu.pt'
    data = EmuData(qsz=20, lamin=1001, lamax=9999, save=path, device=device)
    data = EmuData(load=path, device=device)

    # check sepctral gap
    print('Spectral Gap')
    print('------------')
    print("[{:.8f}, {:.8f}]".format(data.min_gap, data.max_gap))
    print()

    # check batch operations
    batch = data.iter(bsz=4)
    data.print_batch(batch.next())
    print()
    os.remove(path)