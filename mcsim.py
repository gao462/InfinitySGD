import numpy as np
import torch
import ddmp
import mcpro


"""
Markov Process Simulation
=========================
This part provides simulator of Markov processes.

- CMPLogSim
  Count the states and transactions in a logging interval after reaching steady state for a given continuous
  Markov Process. It only counts the state and transition only on logging, not those of every arrival of the
  real process.
  Pay attention that logging rate should be large enough to avoid missing anything. 
-----------------------------------------------------------------------------------
"""


# flag to control data sampling
PROCESS = 'simulate'


class CMPLogSim(object):
    r"""Continuous Markov Process Logging Simulation"""
    def __init__(self, prior):
        r"""Initialize the class

        Args
        ----
        prior : object
            Structure prior information holder for construction.

        """
        # save necessary attributes
        self.prior = prior
        self.k = self.prior.k
        self.c = self.prior.c

    def simulate(self, t=None, *args, **kargs):
        r"""Simulate Markov process inside an interval

        Args
        ----
        t : None or int
            Simulation duration.

        Returns
        -------
        cnt_trans : torch.LongTensor
            Count of transactions in the interval.
        cnt_states : torch.LongTensor
            Count of states in the interval.
        P : torch.Tensor
            Transition probability matrix used in simulation.
        pi : torch.Tensor
            Steady state distribution vector used in simulation.

        """
        # get transition rate matrix
        X = self.prior.trans_rate_mx(*args, **kargs)

        # get steady state distribution
        P, pi, gamma = ddmp.stdy_dist(X, c=self.c)

        # simulation can only work on single sample
        X, P, pi = X.squeeze(), P.squeeze(), pi.squeeze()
        assert len(pi.size()) == 1, \
               "Simulation can only work on single sample."

        # set time interval as one unit time
        T = t or 1

        # logging rate is equivalent to gamma
        log_rate = gamma.item()

        # logging rate should be large enough
        assert len(torch.nonzero(log_rate < X.sum(dim=-1) * 2)) == 0, \
               "Logging rate should be large enough."

        # initialize count by zero
        cnt_states = torch.zeros(self.k, dtype=torch.int64, device=pi.device)
        cnt_trans  = torch.zeros(self.k, self.k, dtype=torch.int64, device=P.device)

        # provide different sampling process
        if PROCESS == 'simulate':
            # get random steady state to start
            cumsum = torch.cumsum(pi, dim=0)
            rv = torch.rand(1).to(pi.dtype).to(pi.device)
            s = torch.nonzero(cumsum > rv).min().item()
            flag = True
    
            # get next arrival time after starting
            t = np.random.exponential(1 / log_rate)
    
            # loop until jump out of the interval
            trace, stamp = [], []
            while t < T:
                # update trace including self-loop
                trace.append(s)
                stamp.append(t)
    
                # get next arrival time
                x = np.random.exponential(1 / log_rate)
                t = t + x
    
                # sample next state based on current state transition probabilities
                states = torch.nonzero(P[s]).squeeze()
                probas = P[s, states]
                cumsum = torch.cumsum(probas, dim=0)
                rv = torch.rand(1).to(probas.dtype).to(probas.device)
                next_s = states[torch.nonzero(cumsum > rv).min().item()].item()
    
                # update next state as future
                flag = (next_s != s)
                s = next_s
    
            # count state from trace
            for itr in trace:
                cnt_states.data[itr] += 1
    
            # count transaction from trace
            for src, dst in zip(trace[:-1], trace[1:]):
                cnt_trans.data[src, dst] += 1
        elif PROCESS == 'poisson':
            # disable Poisson process
            raise RuntimeError('Poisson sampling is disabled.')

            # directly sample from Poisson
            for i in range(self.k):
                cnt_states.data[i] = np.random.poisson(pi[i].item() * T * log_rate)
        else:
            raise NotImplementedError
        return cnt_trans, cnt_states, P, pi


r"""
Benchmark
=========
Simulation is hard to verify. If number of test case is too small, we will have randomness; if too high, we will
suffer time cost. So, we only verify the coding and suppose the algorithm is always correct.
--------------------------------------------------------------------------------------------
"""


if __name__ == '__main__':
    # set tolerance
    num = None
    atot = None

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

    # create prior
    k = 5
    m = 2
    tensor = torch.DoubleTensor
    prior = mcpro.MMmKPrior(k=k, m=m, c=13, tensor=tensor, device=device)

    # create simulation
    sim = CMPLogSim(prior=prior)

    # simulate on a configuration
    birth_rate = tensor([7]).to(device)
    death_rate = tensor([3]).to(device)
    for i in range(10):
        buffer = sim.simulate(birth_rate=birth_rate, death_rate=death_rate)
    print('Simulation is hard to check by considering both randomness and time cost.')