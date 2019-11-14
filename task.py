import os
import time
import numpy as np
import torch
import ddmp
import data


r"""
Learning Task
=============
This part provide a general learning task controller for all experiments.

It supports simulation data generation and parameter tuning on generated data.

- def_train_loss_func
  Define a training loss function (NLL) on observable transitions and states.

- def_eval_loss_func
  Define a tuple of evaluation loss functions (MSE, MAPE) on observable transitions and states.

- Task
  Learning task controller.
  It provides both data generation and parameter fitting.
---------------------------------------------------------
"""


# define training loss function
def def_train_loss_func(trans=None, states=None):
    r"""Define training loss function

    Args
    ----
    trans : [int, ...] or int
        Observable transitions in training loss.
    states : [int, ...] or int
        Observable states in training loss.

    Returns
    -------
    train_loss_func : func
        Training loss function.

    """
    # define loss function
    def train_loss_func(output, target):
        # parse output and target
        output_P, output_pi = output
        target_P, target_pi = target[1]
        target_cnt_P, target_cnt_pi = target[0]

        # get batch size
        bsz = output_P.size(0)
        assert output_P.size(0) == bsz and output_pi.size(0) == bsz
        assert target_P.size(0) == bsz and target_pi.size(0) == bsz
        assert target_cnt_P.size(0) == bsz and target_cnt_pi.size(0) == bsz

        # clamp results
        output_pi = torch.clamp(output_pi, 1e-20, 1)
        output_P  = torch.clamp(output_P, 1e-20, 1)

        # filter, condition and flatten states for loss function
        condit_pi = output_pi[:, :, states].sum(dim=-1, keepdim=True)
        condit_pi = (output_pi[:, :, states] / condit_pi).view(-1)
        target_pi = target_pi[:, states].view(-1)
        target_cnt_pi = target_cnt_pi[:, states].view(-1)

        # nll loss function
        nll_states = torch.sum(-target_cnt_pi * torch.log(condit_pi))
        nll_states = nll_states / bsz

        # filter, condition and flatten transitions for nll loss function
        if trans is not None:
            edge_P       = output_P[:, trans[:, 0], trans[:, 1]].view(bsz, len(trans))
            node_P       = output_pi[:, :, trans[:, 0]].view(bsz, len(trans))
            product_P    = edge_P * node_P
            condit_P     = product_P.sum(dim=-1, keepdim=True)
            condit_P     = (product_P / condit_P).view(-1)
            target_P     = target_P[:, trans[:, 0], trans[:, 1]].view(-1)
            target_cnt_P = target_cnt_P[:, trans[:, 0], trans[:, 1]].view(-1)
            nll_trans = torch.sum(-target_cnt_P * torch.log(condit_P))
            nll_trans = nll_trans / bsz
            return nll_states + nll_trans
        else:
            return nll_states
    return train_loss_func


# define a tuple of evaluation loss functions
def def_eval_loss_funcs(trans=None, states=None):
    r"""Define evaluation loss functions

    Args
    ----
    trans : [int, ...] or int
        Observable transitions in evaluation loss.
    states : [int, ...] or int
        Observable states in evaluation loss.

    Returns
    -------
    eval_loss_funcs : (func, ...)
        A tuple of all evaluation loss functions.

    """
    # define absolute loss function
    def eval_aloss_func(output, target):
        # parse output and target
        output_P, output_pi = output
        target_P, target_pi = target[1]
        target_cnt_P, target_cnt_pi = target[0]

        # get batch size
        bsz = output_P.size(0)
        assert output_P.size(0) == bsz and output_pi.size(0) == bsz
        assert target_P.size(0) == bsz and target_pi.size(0) == bsz
        assert target_cnt_P.size(0) == bsz and target_cnt_pi.size(0) == bsz

        # clamp results
        output_pi = torch.clamp(output_pi, 1e-20, 1)
        output_P  = torch.clamp(output_P, 1e-20, 1)

        # filter and flatten states for loss function
        output_pi = output_pi[:, :, states].view(-1)
        target_pi = target_pi[:, states].view(-1)
        target_cnt_pi = target_cnt_pi[:, states].view(-1)

        # mse loss function
        mse_states = torch.nn.functional.mse_loss(output_pi, target_pi, reduction='sum')
        mse_states = mse_states / bsz

        # // # filter and flatten transitions for loss function
        # // output_P = output_P.view(-1) 
        # // target_P = target_P.view(-1)
        # // target_cnt_P = target_cnt_P.view(-1)
        return mse_states

    # define relative loss function
    def eval_rloss_func(output, target):
        # parse output and target
        output_P, output_pi = output
        target_P, target_pi = target[1]
        target_cnt_P, target_cnt_pi = target[0]

        # get batch size
        bsz = output_P.size(0)
        assert output_P.size(0) == bsz and output_pi.size(0) == bsz
        assert target_P.size(0) == bsz and target_pi.size(0) == bsz
        assert target_cnt_P.size(0) == bsz and target_cnt_pi.size(0) == bsz

        # filter and flatten transitions for loss function
        output_P = output_P.view(-1) 
        target_P = target_P.view(-1)
        target_cnt_P = target_cnt_P.view(-1)

        # filter and flatten states for loss function
        output_pi = torch.clamp(output_pi, 1e-20, 1)
        output_pi = output_pi[:, :, states].view(-1)
        target_pi = target_pi[:, states].view(-1)
        target_cnt_pi = target_cnt_pi[:, states].view(-1)

        # mape loss function
        mape_states = 100 * torch.abs((target_pi - output_pi) / (target_pi + 1e-20)).sum()
        mape_states = mape_states / bsz
        return mape_states
    return (eval_aloss_func, eval_rloss_func)


class Task(object):
    r"""Learning Task"""
    def __init__(self, data_cls, root):
        r"""Initialize the class

        Args
        ----
        data_cls : type
            Data class.
        root : str
            Root directory.

        """
        # save necessary attributes
        self.data_cls = data_cls
        self.root = root

        # validate folder
        if os.path.isdir(self.root):
            pass
        else:
            os.makedirs(self.root)

    def set_seed(self, seed):
        r"""Set random seed

        Args
        ----
        seed : int
            Random seed.

        """
        # set random seed for all libraries
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def generate(self, seed, train_kargs=None, valid_kargs=None, test_kargs=None):
        r"""Generate simulation data

        Args
        ----
        seed : int
            Random seed.
        train_kargs : dict
            Training generation keyword arguments.
        valid_kargs : dict
            Validation generation keyword arguments.
        test_kargs : dict
            Test generation keyword arguments.

        """
        # set random seed for generation
        self.set_seed(seed)

        # generate for each given arguments
        for kargs, name in zip((train_kargs, valid_kargs, test_kargs), ('Train', 'Valid', 'Test')):
            if kargs is not None:
                kargs['save'] = os.path.join(self.root, os.path.basename(kargs['save']))
                data = self.data_cls(**kargs)
                print("({})\t{}\t(Spec Gap)\t[{:.8f}, {:.8f}]".format(name, len(data.samples), data.min_gap, data.max_gap))
            else:
                pass

    def fit(self, seed, device, train_path, valid_path, model_cls, optim_cls, lr, num_epochs, alpha,
            save, model_kargs={}, optim_kargs={}):
        r"""Fit model parameters
        
        Args
        ----
        seed : int
            Random seed.
        device : str
            Device to work on.
        train_path : str
            Path to training data.
        valid_path : str
            Path to validation data.
        model_cls : type
            Model class.
        optim_cls : type
            Optimizer class.
        lr : float
            Leaning rate
        num_epochs : int
            Number of traning epochs.
        alpha : float
            Prior regularization strength.
        save : path
            Path to save.
        model_kargs : dict
            Model keyword arguments.
        optim_kargs : dict
            Optimizer keyword arguments.

        """
        # set random seed for fitting
        self.set_seed(seed)

        # load data first
        train_path = os.path.join(self.root, os.path.basename(train_path))
        train_data = self.data_cls(load=train_path, device=device)
        valid_path = os.path.join(self.root, os.path.basename(valid_path))
        valid_data = self.data_cls(load=valid_path, device=device)

        # create model
        model = model_cls(data_prior=train_data.prior, tensor=train_data.tensor, **model_kargs).to(device)
        optim = optim_cls(model.parameters(), lr=lr, **optim_kargs)

        # create loss functions
        train_loss_func = def_train_loss_func(trans=train_data.train_trans, states=train_data.train_states)
        eval_loss_funcs = def_eval_loss_funcs(trans=train_data.test_trans , states=train_data.test_states)
        eval_id         = 0

        # update saving path
        log_save = os.path.join(self.root, 'log_' + os.path.basename(save))
        best1_save = os.path.join(self.root, 'best1_' + os.path.basename(save))
        best2_save = os.path.join(self.root, 'best2_' + os.path.basename(save))

        # evaluate initialization performance
        train_loss = self.evaluate(train_data, model, [train_loss_func])[0]
        train_eval = self.evaluate(train_data, model, eval_loss_funcs)
        valid_eval = self.evaluate(valid_data, model, eval_loss_funcs)
        track_list = [(train_loss, train_eval, valid_eval, model.track_params, 0)]
        torch.save(track_list, log_save)

        # update best loss
        best_loss1 = train_loss
        torch.save(model.state_dict(), best1_save)
        best_loss2 = train_eval[eval_id]
        torch.save(model.state_dict(), best2_save)

        # print performance
        fmt = "[{:d}]\tBest Perf: {:.6f}\tTrain Perf: {:.6f}\tValid Perf: {:.6f}\t(Train Loss: {:.6f})" \
              "\t(Death Rate: {:.6f})\tTime: {}"
        print(fmt.format(
            0, best_loss2, train_eval[eval_id], valid_eval[eval_id], train_loss,
            model.track_params[0], '(null)'))

        # loop training epochs
        all_time = time.time()
        for epc in range(1, num_epochs + 1):
            # tune parameters
            epoch_time = time.time()
            self.train(train_data, model, alpha, train_loss_func, optim)

            # evaluate performance
            train_loss = self.evaluate(train_data, model, [train_loss_func])[0]
            train_eval = self.evaluate(train_data, model, eval_loss_funcs)
            valid_eval = self.evaluate(valid_data, model, eval_loss_funcs)
            epoch_time = int(np.ceil(time.time() - epoch_time))
            track_list.append((train_loss, train_eval, valid_eval, model.track_params, epoch_time))
            torch.save(track_list, log_save)

            # update best loss 1 and best model 1
            if train_loss < best_loss1:
                best_loss1 = train_loss
                torch.save(model.state_dict(), best1_save)
            else:
                pass

            # update best loss 2 and best model 2
            if train_eval[eval_id] < best_loss2:
                best_loss2 = train_eval[eval_id]
                torch.save(model.state_dict(), best2_save)
            else:
                pass

            # print performance
            print(fmt.format(
                epc, best_loss2, train_eval[eval_id], valid_eval[eval_id], train_loss,
                model.track_params[0], self.timerepr(epoch_time)))
        all_time = int(np.ceil(time.time() - all_time))
        print("(Time)\t{}".format(self.timerepr(all_time)))

    @staticmethod
    def timerepr(sec):
        r"""Translate seconds into time string

        Args
        ----
        sec : int
            Seconds.

        Returns
        -------
        msg : str
            Time string.

        """
        # split time units
        remain = sec
        s, remain = remain % 60, remain // 60
        m, remain = remain % 60, remain // 60
        h, remain = remain % 60, remain // 60
        time_vals = [h, m, s]
        time_units = ['hr', 'min', 'sec']

        # merge time units
        time_msgs = []
        for val, unit in zip(time_vals, time_units):
            if val > 0:
                time_msgs.append("{} {}".format(val, unit))
            else:
                pass
        return ', '.join(time_msgs)

    def evaluate(self, data, model, loss_funcs):
        r"""Evaluate model

        Args
        ----
        data : data._Data
            Data.
        model : torch.nn.Module
            Model.
        loss_funcs : (func, ...)
            A tuple of Loss functions.

        Returns
        -------
        loss_avg : (float, ...)
            A tuple of loss averages.

        """
        # create batch
        batch_iter = data.iter(bsz=1, order=np.arange(len(data)))

        # traverse all batch
        loss_sums = [0 for _ in range(len(loss_funcs))]
        loss_cnts = [0 for _ in range(len(loss_funcs))]
        while True:
            try:
                batch = batch_iter.next()
            except StopIteration:
                break
            input, target = batch
            output = model.forward(input)
            for i, loss_func in enumerate(loss_funcs):
                loss = loss_func(output, target)
                loss_sums[i] += loss.data.item()
                loss_cnts[i] += 1
        loss_avgs = tuple([s / c for s, c in zip(loss_sums, loss_cnts)])
        return loss_avgs

    def train(self, data, model, alpha, loss_func, optim):
        r"""Evaluate model

        Args
        ----
        data : data._Data
            Data.
        model : torch.nn.Module
            Model.
        alpha : float
            Prior regularization strength.
        loss_func : func
            Loss function.
        optim : torch.optim.Optimizer
            Optimizer.

        """
        # create batch
        batch_iter = data.iter(bsz=1)

        # traverse all batch
        while True:
            try:
                batch = batch_iter.next()
            except StopIteration:
                break
            input, target = batch
            optim.zero_grad()
            output = model.forward(input)
            reg = 0 if model.noise is None else alpha * torch.norm(model.noise)
            loss = reg + loss_func(output, target)
            loss.backward()

            # ensure gradient safety
            for param in model.parameters():
                param.grad[param.grad != param.grad] = 0

            # noise update should be controlled
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            # optimization should not give negative parameters
            optim.step()
            for name, param in model.named_parameters():
                if name == 'noise':
                    param.data[param.data < 0] = 0
                else:
                    param.data[param.data <= 0] = 1e-20