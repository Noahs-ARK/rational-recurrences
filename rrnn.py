import sys
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from termcolor import colored



def RRNN_Ngram_Compute_CPU(d, k, semiring, bidirectional=False):
    class TraceElement():
        def __init__(self, f, u, prev_traces, i, t, pattern_index, sample_index):
            self.u_indices = np.zeros((int(k / 2)), dtype=int)

            if t < i:
                self.score = float('-inf')
                return

            # Previous trace values
            prev_u = prev_traces[i-1][pattern_index][sample_index] if i > 0 else None

            # print("in te, t={}, i={}. all prevs is : {}".format(t, i, [x is None for x in prev_traces]))

            # if u[y][x].data.numpy() > 0 or f[y][x].data.numpy() > 0:
            #     print("before: ", x, y, u[y][x].data.numpy(), f[y][x].data.numpy(), is_u[y][x].data.numpy(), i, t,
            #           prev_traces[i][y][x].u_indices if is_u[y][x].data.numpy() and prev_traces[i] is not None else None,
            #           prev_traces[i+1][y][x].u_indices if not is_u[y][x].data.numpy() and prev_traces[i+1] is not None else None,
            #           self.u_indices)

            # Two candidates: u (read token) and f (forget token)
            u_score = u[sample_index][pattern_index].data.numpy()

            if prev_u is not None:
                u_score *= prev_u.score

            prev_f = prev_traces[i][pattern_index][sample_index] if t > i else None
            f_score = f[sample_index][pattern_index].data.numpy()

            if prev_f is not None:
                f_score *= prev_f.score
            else:
                f_score = float('-inf')


            # print("in te, doc_ind={}, patt_ind={}, t={}, i={}. u_score={}, f_score={} (u>v={}), all prevs is : {}".format(sample_index, pattern_index, t, i,
            #         u_score, f_score, u_score >= f_score, [x is None for x in prev_traces]))

            if u_score >= f_score:
                self.score = u_score
                if prev_u is not None:
                    # self.score *= prev1.score
                    self.u_indices = prev_u.u_indices
                # else:
                #     assert i == 0
                self.u_indices[i] = t
            else:
                self.score = f_score
                if prev_f is not None:
                    self.u_indices = prev_f.u_indices
                    # self.score *= prev2.score
                    # else:
                    #     assert i == 0

            # if u[y][x].data.numpy() > 0 or f[y][x].data.numpy() > 0:
            #     print("after", self.u_indices)

        def print(self, index, doc):
            print("{}. {}.".format(index, self.u_indices), end=' ')
            self.print_rec(doc, 0)
            print(float(self.score))

        def print_rec(self, doc, u_index):
            doc_index = self.u_indices[u_index]
            print(colored(doc[doc_index], 'red'), end='_MP ')

            u_index += 1

            if u_index == len(self.u_indices):
                return

            doc_index += 1

            while (doc_index < self.u_indices[u_index]):
                print(doc[doc_index], end='_SL ')
                doc_index += 1

            self.print_rec(doc, u_index)

    def get_trace(f, u, prev_traces, i, t):
        traces = [
            [
                TraceElement(f, u, prev_traces,
                             i, t, pattern_index, sample_index)
                for sample_index in range(u.size()[0])
            ]
            for pattern_index in range(u.size()[1])
        ]

        return traces



    def rrnn_compute_cpu(u, cs_init=None, eps=None, keep_trace=False):
        assert eps is None, "haven't implemented epsilon steps with arbitrary n-grams. Please set command line param to False."
        bidir = 2 if bidirectional else 1
        assert u.size(-1) == k
        length, batch = u.size(0), u.size(1)

        for i in range(len(cs_init)):
            cs_init[i] = cs_init[i].contiguous().view(batch, bidir, d)

        us = []
        for i in range(0,int(k/2)):
            us.append(u[..., i])
        forgets = []
        for i in range(int(k/2), k):
            forgets.append(u[..., i])

        cs_final = [[] for i in range(int(k/2))]

        css = [Variable(u.data.new(length, batch, bidir, d)) for i in range(int(k/2))]

        traces = None
        prev_traces = None

        for di in range(bidir):
            if di == 0:
                time_seq = range(length)
            else:
                time_seq = range(length - 1, -1, -1)

            cs_prev = [cs_init[i][:, di, :] for i in range(len(cs_init))]

            if keep_trace:
                prev_traces = [None for i in range(len(cs_init))]

            for t in time_seq:
                cs_t = []
                # ind = 0
                if keep_trace:
                    all_traces = []

                for i in range(len(cs_prev)):
                    first_term = cs_prev[i] * forgets[i][t, :, di, :]
                    second_term = us[i][t, :, di, :]

                    if i > 0:
                        second_term = second_term * cs_prev[i-1]

                    cs_t.append(first_term + second_term)

                    # print(second_term.size(),forgets[i][t, :, di, :].size(),traces[ind+1].size())
                    if keep_trace:
                        traces = get_trace(forgets[i][t, :, di, :], us[i][t, :, di, :], prev_traces, i, t)
                        all_traces.append(traces)

                if keep_trace:
                    prev_traces = all_traces

                cs_prev = cs_t
                
                for i in range(len(cs_prev)):
                    css[i][t,:,di,:] = cs_t[i]

            for i in range(len(cs_prev)):
                cs_final[i].append(cs_t[i])

        for i in range(len(cs_final)):
            cs_final[i] = torch.stack(cs_final[i], dim=1).view(batch, -1)
        
        return css, cs_final, traces
    if semiring.type == 0:
        # plus times
        return rrnn_compute_cpu
    else:
        assert False, "OTHER SEMIRINGS NOT IMPLEMENTED!"

        

def RRNN_Bigram_Compute_CPU(d, k, semiring, bidirectional=False):
    """CPU version of the core RRNN computation.

    Has the same interface as RRNN_Compute_GPU() but is a regular Python function
    instead of a torch.autograd.Function because we don't implement backward()
    explicitly.
    """

    def rrnn_semiring_compute_cpu(u, c1_init=None, c2_init=None, eps=None):
        bidir = 2 if bidirectional else 1
        assert u.size(-1) == k
        length, batch = u.size(0), u.size(1)
        if c1_init is None:
            assert False
        else:
            c1_init = c1_init.contiguous().view(batch, bidir, d)
            c2_init = c2_init.contiguous().view(batch, bidir, d)

        # this is not a typo. inputn is the gate for x_tilden
        u1, u2, forget1, forget2 = u[..., 0], u[..., 1],u[..., 2], u[..., 3]
        c1_final, c2_final = [], []
        c1s = Variable(u.data.new(length, batch, bidir, d))
        c2s = Variable(u.data.new(length, batch, bidir, d))
        for di in range(bidir):
            if di == 0:
                time_seq = range(length)
            else:
                time_seq = range(length - 1, -1, -1)

            c1_prev = c1_init[:, di, :]
            c2_prev = c2_init[:, di, :]

            for t in time_seq:
                c1_t = semiring.plus(
                        semiring.times(c1_prev, forget1[t, :, di, :]),
                        u1[t, :, di, :]
                )

                tmp = semiring.plus(eps[di, :], c1_prev)
                c2_t = semiring.plus(
                    semiring.times(c2_prev, forget2[t, :, di, :]),
                    semiring.times(tmp, u2[t, :, di, :])
                )
                c1_prev, c2_prev = c1_t, c2_t
                c1s[t,:,di,:], c2s[t,:,di,:] = c1_t, c2_t

            c1_final.append(c1_t)
            c2_final.append(c2_t)

        return c1s, c2s, \
               torch.stack(c1_final, dim=1).view(batch, -1), \
               torch.stack(c2_final, dim=1).view(batch, -1)
    
    def rrnn_compute_cpu(u, c1_init=None, c2_init=None, eps=None):
        bidir = 2 if bidirectional else 1
        assert u.size(-1) == k
        length, batch = u.size(0), u.size(1)
        if c1_init is None:
            assert False
        else:
            c1_init = c1_init.contiguous().view(batch, bidir, d)
            c2_init = c2_init.contiguous().view(batch, bidir, d)

        u1, u2, forget1, forget2 = u[..., 0], u[..., 1],u[..., 2],u[..., 3]

        c1_final, c2_final = [], []
        c1s = Variable(u.data.new(length, batch, bidir, d))
        c2s = Variable(u.data.new(length, batch, bidir, d))

        for di in range(bidir):
            if di == 0:
                time_seq = range(length)
            else:
                time_seq = range(length - 1, -1, -1)

            c1_prev = c1_init[:, di, :]
            c2_prev = c2_init[:, di, :]

            for t in time_seq:
                c1_t = c1_prev* forget1[t, :, di, :] + u1[t, :, di, :]
                if eps is not None:
                    tmp = eps[di, :] + c1_prev
                else:
                    tmp = c1_prev
                c2_t = c2_prev * forget2[t, :, di, :] + tmp * u2[t, :, di, :]
                c1_prev, c2_prev = c1_t, c2_t
                c1s[t,:,di,:], c2s[t,:,di,:]  = c1_t, c2_t

            c1_final.append(c1_t)
            c2_final.append(c2_t)

        return c1s, c2s, \
               torch.stack(c1_final, dim=1).view(batch, -1), \
               torch.stack(c2_final, dim=1).view(batch, -1)

    if semiring.type == 0:
        # plus times
        return rrnn_compute_cpu
    else:
        # otehrs
        return rrnn_semiring_compute_cpu

def RRNN_Unigram_Compute_CPU(d, k, semiring, bidirectional=False):

    def rrnn_compute_cpu(u, c_init=None):
        bidir = 2 if bidirectional else 1
        assert u.size(-1) == k
        length, batch = u.size(0), u.size(1)
        if c_init is None:
            assert False
        else:
            c_init = c_init.contiguous().view(batch, bidir, d)

        u, forget = u[..., 0], u[..., 1]

        c_final = []
        cs = Variable(u.data.new(length, batch, bidir, d))

        for di in range(bidir):
            if di == 0:
                time_seq = range(length)
            else:
                time_seq = range(length - 1, -1, -1)

            c_prev = c_init[:, di, :]
            for t in time_seq:
                c_t = c_prev * forget[t, :, di, :] + u[t, :, di, :]
                c_prev = c_t
                cs[t, :, di, :] = c_t
            c_final.append(c_t)

        return cs, torch.stack(c_final, dim=1).view(batch, -1)

    if semiring.type == 0:
        # plus times
        return rrnn_compute_cpu
    else:
        assert False


class RRNNCell(nn.Module):
    def __init__(self,
                 semiring,
                 n_in,
                 n_out,
                 pattern="bigram",
                 dropout=0.2,
                 rnn_dropout=0.2,
                 bidirectional=False,
                 use_tanh=1,
                 use_relu=0,
                 use_selu=0,
                 weight_norm=False,
                 index=-1,
                 use_output_gate=True,
                 use_rho=False,
                 rho_sum_to_one=False,
                 use_last_cs=False,
                 use_epsilon_steps=True):
        super(RRNNCell, self).__init__()
        #assert (n_out % 2) == 0
        self.semiring = semiring
        self.pattern = pattern
        self.n_in = n_in
        self.n_out = n_out
        self.rnn_dropout = rnn_dropout
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.bidir = 2 if self.bidirectional else 1
        self.weight_norm = weight_norm
        self.index = index
        self.activation_type = 0
        self.use_output_gate = use_output_gate  # borrowed from qrnn
        self.use_rho = use_rho
        self.rho_sum_to_one = rho_sum_to_one
        self.use_last_cs = use_last_cs
        self.use_epsilon_steps = use_epsilon_steps
        if use_tanh:
            self.activation_type = 1
        elif use_relu:
            self.activation_type = 2
        elif use_selu:
            self.activation_type = 3

        # basic: in1, in2, f1, f2
        # optional: output.

        if self.pattern == "bigram":
            self.k = 5 if self.use_output_gate else 4
        elif self.pattern == "unigram":
            self.k = 3 if self.use_output_gate else 2
        else:
            # it should be of the form "4-gram"
            # should probably implement epsilon stuff, as in bigram
            ngram = int(self.pattern.split("-")[0])
            self.k = 2 * (ngram)

        
        if self.pattern != "unigram" and self.pattern != "1-gram":
            if self.use_rho:
                self.bias_final = nn.Parameter(torch.Tensor(self.bidir*n_out*int(self.k/2)))
            if self.use_epsilon_steps:
                self.bias_eps = nn.Parameter(torch.Tensor(self.bidir*n_out))

        self.size_per_dir = n_out*self.k
        self.weight = nn.Parameter(torch.Tensor(
            n_in,
            self.size_per_dir*self.bidir
        ))
        self.bias = nn.Parameter(torch.Tensor(
            self.size_per_dir*self.bidir
        ))
        self.init_weights()


    def init_weights(self, rescale=True):
        val_range = (6.0 / (self.n_in + self.n_out)) ** 0.5
        self.weight.data.uniform_(-val_range, val_range)

        # initialize bias
        self.bias.data.zero_()

        if self.pattern != "unigram" and self.pattern != "1-gram":
            if self.use_rho:
                self.bias_final.data.zero_()
            if self.use_epsilon_steps:
                self.bias_eps.data.zero_()

        self.scale_x = 1
        if not rescale:
            return

        # re-scale weights in case there's dropout and / or layer normalization
        w_in = self.weight.data.view(self.n_in, -1, self.n_out, self.k)
        if self.rnn_dropout > 0:
            w_in.mul_((1 - self.rnn_dropout) ** 0.5)

        # re-parameterize when weight normalization is enabled
        if self.weight_norm:
            self.init_weight_norm()


    def init_weight_norm(self):
        weight_in = self.weight.data
        g = weight_in.norm(2, 0)
        self.gain_in = nn.Parameter(g)


    def apply_weight_norm(self, eps=0):
        wnorm = self.weight.norm(2, 0)  #, keepdim=True)
        return self.gain.expand_as(self.weight).mul(
            self.weight / (wnorm.expand_as(self.weight) + eps)
        )


    def calc_activation(self, x):
        if self.activation_type == 0:
            return x
        elif self.activation_type == 1:
            return x.tanh()
        elif self.activation_type == 2:
            return nn.functional.relu(x)
        else:
            assert False, "Activation type must be 0, 1, or 2, not {}".format(self.activation_type)

    def semiring_forward(self, input, init_hidden=None):
        assert input.dim() == 2 or input.dim() == 3
        assert not self.semiring.type == 0
        n_in, n_out = self.n_in, self.n_out
        length, batch = input.size(0), input.size(-2)
        bidir = self.bidir

        if init_hidden is None:
            size = (batch, n_out*bidir)
            c1_init = Variable(input.data.new(*size).zero_()) + Variable(self.semiring.zero(input.data, *size))
            c2_init = Variable(input.data.new(*size).zero_()) + Variable(self.semiring.zero(input.data, *size))
        else:
            assert (len(init_hidden) == 2)
            c1_init, c2_init = init_hidden

        if self.training and (self.rnn_dropout>0):
            mask = self.get_dropout_mask_((1, batch, n_in), self.rnn_dropout)
            x = input * mask.expand_as(input)
        else:
            x = input

        x_2d = x if x.dim() == 2 else x.contiguous().view(-1, n_in)

        weight_in = self.weight if not self.weight_norm else self.apply_weight_norm()
        u_ = x_2d.mm(weight_in)
        # reset is not passed to compute function
        u_ = u_.view(length, batch, bidir, n_out, self.k)

        bias = self.bias.view(self.n_bias, bidir, n_out)
        # basic: in1, in2, f1, f2
        # optional:  output.

        _, _, forget_bias1, forget_bias2 = bias[:4, ...]
        if self.use_output_gate:
            output_bias = bias[4, ...]
            output = (u_[..., 4] + output_bias).sigmoid()

        u = Variable(u_.data.new(length, batch, bidir, n_out, 4))

        forget1 = (u_[..., 2] + forget_bias1).sigmoid()
        forget2 = (u_[..., 3] + forget_bias2).sigmoid()
        if self.semiring.type == 1 or self.semiring.type == 2 or self.semiring.type == 3:
            # max_plus, max_times
            u[..., 2] = forget1.log()
            u[..., 3] = forget2.log()
            u[..., 0] = u_[..., 0]
            u[..., 1] = u_[..., 1]
        else:
            assert False

        if input.is_cuda:
            from rrnn_gpu import RRNN_Compute_GPU
            RRNN_Compute = RRNN_Compute_GPU(n_out, 4, self.semiring, self.bidirectional)
        else:
            RRNN_Compute = RRNN_Compute_CPU(n_out, 4, self.semiring, self.bidirectional)

        eps = self.bias_eps.view(bidir, n_out).sigmoid()
        if self.semiring.type == 1 or self.semiring.type == 2 or self.semiring.type == 3:
            eps = eps.log()
        c1s, c2s, c1_final, c2_final = RRNN_Compute(u, c1_init, c2_init, eps)

        rho = self.bias_final.view(bidir, n_out, 2).sigmoid() * 2
        if self.semiring.type == 1 or self.semiring.type == 2 or self.semiring.type == 3:
            rho = rho.log()
        cs = self.semiring.plus(
            self.semiring.times(c1s, rho[...,0]),
            self.semiring.times(c2s, rho[...,1])
        )

        if self.use_output_gate:
            gcs = self.calc_activation(output * cs.view(length, batch, bidir, n_out))
        else:
            gcs = self.calc_activation(cs).view(length, batch, bidir, n_out)
        return gcs.view(length, batch, -1), c1_final, c2_final


    def real_bigram_forward(self, input, init_hidden=None):
        assert input.dim() == 2 or input.dim() == 3
        n_in, n_out = self.n_in, self.n_out
        length, batch = input.size(0), input.size(-2)
        bidir = self.bidir
        if init_hidden is None:
            size = (batch, n_out * bidir)
            c1_init = Variable(input.data.new(*size).zero_())
            c2_init = Variable(input.data.new(*size).zero_())
        else:
            assert (len(init_hidden) == 2)
            c1_init, c2_init, = init_hidden

        if self.training and (self.rnn_dropout>0):
            mask = self.get_dropout_mask_((1, batch, n_in), self.rnn_dropout)
            x = input * mask.expand_as(input)
        else:
            x = input

        x_2d = x if x.dim() == 2 else x.contiguous().view(-1, n_in)

        weight_in = self.weight if not self.weight_norm else self.apply_weight_norm()
        u_ = x_2d.mm(weight_in)
        u_ = u_.view(length, batch, bidir, n_out, self.k)


        # basic: in1, in2, f1, f2
        # optional: output.
        bias = self.bias.view(self.k, bidir, n_out)

        _, _, forget_bias1, forget_bias2 = bias[:4, ...]
        if self.use_output_gate:
            output_bias = bias[4, ...]
            output = (u_[..., 4] + output_bias).sigmoid()

        u = Variable(u_.data.new(length, batch, bidir, n_out, 4))

        u[..., 2] = (u_[..., 2] + forget_bias1).sigmoid()   # forget 1
        u[..., 3] = (u_[..., 3] + forget_bias2).sigmoid()   # forget 2

        u[..., 0] = u_[..., 0] * (1. - u[..., 2])  # input 1
        u[..., 1] = u_[..., 1] * (1. - u[..., 3])  # input 2
        
        if input.is_cuda:
            from rrnn_gpu import RRNN_Bigram_Compute_GPU
            RRNN_Compute = RRNN_Bigram_Compute_GPU(n_out, 4, self.semiring, self.bidirectional)
        else:
            RRNN_Compute = RRNN_Bigram_Compute_CPU(n_out, 4, self.semiring, self.bidirectional)

        if self.use_epsilon_steps:
            eps = self.bias_eps.view(bidir, n_out).sigmoid()
        else:
            eps = None

        c1s, c2s, c1_final, c2_final= RRNN_Compute(u, c1_init, c2_init, eps)

        if self.use_rho:
            rho = self.bias_final.view(bidir, n_out, 2).sigmoid()
            cs = c1s * rho[...,0] + c2s * rho[...,1]
        else:
            cs = c1s + c2s

        if self.use_output_gate:
            gcs = self.calc_activation(output*cs)
        else:
            gcs = self.calc_activation(cs)

        return gcs.view(length, batch, -1), c1_final, c2_final

    def real_unigram_forward(self, input, init_hidden=None):
        assert input.dim() == 2 or input.dim() == 3
        n_in, n_out = self.n_in, self.n_out
        length, batch = input.size(0), input.size(-2)
        bidir = self.bidir
        if init_hidden is None:
            size = (batch, n_out * bidir)
            c_init = Variable(input.data.new(*size).zero_())
        else:
            c_init = init_hidden

        if self.training and (self.rnn_dropout>0):
            mask = self.get_dropout_mask_((1, batch, n_in), self.rnn_dropout)
            x = input * mask.expand_as(input)
        else:
            x = input

        x_2d = x if x.dim() == 2 else x.contiguous().view(-1, n_in)

        weight_in = self.weight if not self.weight_norm else self.apply_weight_norm()
        u_ = x_2d.mm(weight_in)
        u_ = u_.view(length, batch, bidir, n_out, self.k)


        # basic: in, f
        # optional: output.
        bias = self.bias.view(self.k, bidir, n_out)

        _, forget_bias = bias[:2, ...]
        if self.use_output_gate:
            output_bias = bias[3, ...]
            output = (u_[..., 3] + output_bias).sigmoid()

        u = Variable(u_.data.new(length, batch, bidir, n_out, 2))

        u[..., 1] = (u_[..., 1] + forget_bias).sigmoid()   # forget
        u[..., 0] = u_[..., 0] * (1. - u[..., 1])  # input


        if input.is_cuda:
            from rrnn_gpu import RRNN_Compute_GPU
            RRNN_Compute = RRNN_Unigram_Compute_GPU(n_out, 2, self.semiring, self.bidirectional)
        else:
            RRNN_Compute = RRNN_Unigram_Compute_CPU(n_out, 2, self.semiring, self.bidirectional)

        cs, c_final = RRNN_Compute(u, c_init)

        if self.use_output_gate:
            gcs = self.calc_activation(output*cs)
        else:
            gcs = self.calc_activation(cs)

        return gcs.view(length, batch, -1), c_final

    def real_ngram_forward(self, input, init_hidden=None, keep_trace=False):
        assert input.dim() == 3
        n_in, n_out = self.n_in, self.n_out
        length, batch = input.size(0), input.size(-2)
        bidir = self.bidir

        if init_hidden is None:
            size = (batch, n_out * bidir)
            cs_init = []
            for i in range(int(self.k/2)):
                cs_init.append(Variable(input.data.new(*size).zero_()))

        else:
            assert False, "NOT IMPLEMENTED!"
            assert (len(init_hidden) == 2)
            c1_init, c2_init, = init_hidden

        if self.training and (self.rnn_dropout>0):
            mask = self.get_dropout_mask_((1, batch, n_in), self.rnn_dropout)
            x = input * mask.expand_as(input)
        else:
            x = input

        x_2d = x if x.dim() == 2 else x.contiguous().view(-1, n_in)

        weight_in = self.weight if not self.weight_norm else self.apply_weight_norm()
        u_ = x_2d.mm(weight_in)
        u_ = u_.view(length, batch, bidir, n_out, self.k)

        # optional: output.
        bias = self.bias.view(self.k, bidir, n_out)

        u = Variable(u_.data.new(length, batch, bidir, n_out, self.k))

        for i in range(int(self.k/2),self.k):
            forget_bias = bias[i, ...]
            u[..., i] = (u_[..., i] + forget_bias).sigmoid()   # forget 

        for i in range(0, int(self.k/2)):
            u[..., i] = u_[..., i] * (1. - u[..., i + int(self.k/2)])  # input
            
        if input.is_cuda:

            if self.k == 8:

                from rrnn_gpu import RRNN_4gram_Compute_GPU
                RRNN_Compute_GPU = RRNN_4gram_Compute_GPU(n_out, self.k, self.semiring, self.bidirectional)
                c1s, c2s, c3s, c4s, last_c1, last_c2, last_c3, last_c4 = RRNN_Compute_GPU(u, cs_init[0], cs_init[1], cs_init[2], cs_init[3])
                css = [c1s, c2s, c3s, c4s]
                cs_final = [last_c1, last_c2, last_c3, last_c4]

            elif self.k == 6:
                from rrnn_gpu import RRNN_3gram_Compute_GPU
                RRNN_Compute_GPU = RRNN_3gram_Compute_GPU(n_out, self.k, self.semiring, self.bidirectional)
                c1s, c2s, c3s, last_c1, last_c2, last_c3 = RRNN_Compute_GPU(u, cs_init[0], cs_init[1], cs_init[2])
                css = [c1s, c2s, c3s]
                cs_final = [last_c1, last_c2, last_c3]

            elif self.k == 4:
                from rrnn_gpu import RRNN_2gram_Compute_GPU
                RRNN_Compute_GPU = RRNN_2gram_Compute_GPU(n_out, self.k, self.semiring, self.bidirectional)
                c1s, c2s, last_c1, last_c2 = RRNN_Compute_GPU(u, cs_init[0], cs_init[1])
                css = [c1s, c2s]
                cs_final = [last_c1, last_c2]

            elif self.k == 2:
                from rrnn_gpu import RRNN_1gram_Compute_GPU
                RRNN_Compute_GPU = RRNN_1gram_Compute_GPU(n_out, self.k, self.semiring, self.bidirectional)
                c1s, last_c1 = RRNN_Compute_GPU(u, cs_init[0])
                css = [c1s]
                cs_final = [last_c1]

            else:
                assert False, "custom cuda kernel only implemented for 1,2,3,4-gram models"                
        else:
            RRNN_Compute = RRNN_Ngram_Compute_CPU(n_out, self.k, self.semiring, self.bidirectional)
            css, cs_final, traces  = RRNN_Compute(u, cs_init, eps=None, keep_trace=keep_trace)



        # instead of using \rho to weight the sum, we can give uniform weight. this might be
        # more interpretable, as the \rhos might counteract the regularization terms
        if self.use_rho:
            if self.rho_sum_to_one:
                sm = nn.Softmax(dim=2)
                rho = sm(self.bias_final.view(bidir, n_out, int(self.k/2)))
            else:
                rho = self.bias_final.view(bidir, n_out, int(self.k/2)).sigmoid()
            css_times_rho = []
            for i in range(len(css)):
                css_times_rho.append(css[i] * rho[...,i])

            cs = sum(css_times_rho)
        else:
            if self.use_last_cs:
                cs = css[-1]
            else:
                cs = sum(css)

        if self.use_output_gate:
            assert False, "THIS HASN'T BEEN IMPLEMENTED YET!"
            gcs = self.calc_activation(output*cs)
        else:
            gcs = self.calc_activation(cs)

        return gcs.view(length, batch, -1), cs_final, traces

    
    def forward(self, input, init_hidden=None, keep_trace=False):

        if self.semiring.type == 0:
            # plus times
            if self.pattern == "bigram":
                return self.real_bigram_forward(input=input, init_hidden=init_hidden)
            elif self.pattern == "unigram":
                return self.real_unigram_forward(input=input, init_hidden=init_hidden)
            else:
                # it should be of the form "4-gram"
                return self.real_ngram_forward(input=input, init_hidden=init_hidden, keep_trace=keep_trace)
                
        else:
            assert False, "not implemented yet."
            return self.semiring_forward(input=input, init_hidden=init_hidden)

    def get_dropout_mask_(self, size, p, rescale=True):
        w = self.weight.data
        if rescale:
            return Variable(w.new(*size).bernoulli_(1-p).div_(1-p))
        else:
            return Variable(w.new(*size).bernoulli_(1-p))


class RRNNLayer(nn.Module):
    def __init__(self,
                 semiring,
                 n_in,
                 n_out,
                 pattern,
                 dropout=0.2,
                 rnn_dropout=0.2,
                 bidirectional=False,
                 use_tanh=1,
                 use_relu=0,
                 use_selu=0,
                 weight_norm=False,
                 index=-1,
                 use_output_gate=True,
                 use_rho=False,
                 rho_sum_to_one=False,
                 use_last_cs=False,
                 use_epsilon_steps=True):
        super(RRNNLayer, self).__init__()

        self.cells = nn.ModuleList()
        
        assert len(pattern) == len(n_out)
        num_cells = len(pattern)
        for i in range(num_cells):
            if n_out[i] > 0:
                one_cell = RRNNCell(
                    semiring=semiring,
                    n_in=n_in,
                    n_out=n_out[i],
                    pattern=pattern[i],
                    dropout=dropout,
                    rnn_dropout=rnn_dropout,
                    bidirectional=bidirectional,
                    use_tanh=use_tanh,
                    use_relu=use_relu,
                    use_selu=use_selu,
                    weight_norm=weight_norm,
                    index=index,
                    use_output_gate=use_output_gate,
                    use_rho=use_rho,
                    rho_sum_to_one=rho_sum_to_one,
                    use_last_cs=use_last_cs,
                    use_epsilon_steps=use_epsilon_steps
                )
                self.cells.append(one_cell)

    def init_weights(self):
        for cell in self.cells:
            cell.init_weights()
            
    def forward(self, input, init_hidden=None, keep_trace=False):
        #import pdb; pdb.set_trace()
        all_traces = []
        gcs, cs_final, traces = self.cells[0](input, init_hidden, keep_trace)
        all_traces.append(traces)
        for i, cell in enumerate(self.cells):
            if i == 0:
                continue
            else:
                gcs_cur, _, traces = cell(input, init_hidden, keep_trace)
                all_traces.append(traces)
                gcs = torch.cat((gcs, gcs_cur), 2)
                #for j in range(len(cs_final)):
                #    cs_final[j] = torch.cat((cs_final[j], cs_final_cur[j]), 1)
                #cs_final = torch.cat(cs_final, cs_final_cur)

        return gcs, None, all_traces
    
        
class RRNN(nn.Module):
    def __init__(self,
                 semiring,
                 input_size,
                 hidden_size,
                 num_layers,
                 pattern="bigram",
                 dropout=0.2,
                 rnn_dropout=0.2,
                 bidirectional=False,
                 use_tanh=1,
                 use_relu=0,
                 use_selu=0,
                 weight_norm=False,
                 layer_norm=False,
                 use_output_gate=True,
                 use_rho=False,
                 rho_sum_to_one=False,
                 use_last_cs=False,
                 use_epsilon_steps=True):
        super(RRNN, self).__init__()
        assert not bidirectional
        self.semiring = semiring
        self.input_size = input_size
        self.hidden_size = [int(one_size) for one_size in hidden_size.split(",")]
        self.num_layers = num_layers
        self.pattern = [one_pattern for one_pattern in pattern.split(",")]
        self.dropout = dropout
        self.rnn_dropout = rnn_dropout
        self.rnn_lst = nn.ModuleList()
        self.ln_lst = nn.ModuleList()
        self.bidirectional = bidirectional
        self.use_layer_norm = layer_norm
        self.use_wieght_norm = weight_norm
        #self.out_size = hidden_size * 2 if bidirectional else hidden_size

        assert len(self.hidden_size) == len(self.pattern), "each n-gram must have an output size."

        if use_tanh + use_relu + use_selu > 1:
            sys.stderr.write("\nWARNING: More than one activation enabled in RRNN"
                " (tanh: {}  relu: {}  selu: {})\n".format(use_tanh, use_relu, use_selu)
            )
            
        for i in range(num_layers):
            l = RRNNLayer(
                semiring=semiring,
                n_in=self.input_size if i == 0 else sum(self.hidden_size),
                n_out=self.hidden_size,
                pattern=self.pattern,
                dropout=dropout if i+1 != num_layers else 0.,
                rnn_dropout=rnn_dropout,
                bidirectional=bidirectional,
                use_tanh=use_tanh,
                use_relu=use_relu,
                use_selu=use_selu,
                weight_norm=weight_norm,
                index=i+1,
                use_output_gate=use_output_gate,
                use_rho=use_rho,
                rho_sum_to_one=rho_sum_to_one,
                use_last_cs=use_last_cs,
                use_epsilon_steps=use_epsilon_steps
            )
            self.rnn_lst.append(l)
            if layer_norm:
                self.ln_lst.append(LayerNorm(self.hidden_size))

    def init_weights(self):
        for l in self.rnn_lst:
            l.init_weights()

    def unigram_forward(self, input, init_hidden=None, return_hidden=True):
        assert input.dim() == 3  # (len, batch, n_in)
        if init_hidden is None:
            init_hidden = [None for _ in range(self.num_layers)]
        else:
            for c in init_hidden:
                assert c.dim() == 2
            init_hidden = [c.squeeze(0) for c in
                    init_hidden.chunk(self.num_layers, 0)]

        prevx = input
        lstc = []
        for i, rnn in enumerate(self.rnn_lst):
            h, c = rnn(prevx, init_hidden[i])
            prevx = self.ln_lst[i](h) if self.use_layer_norm else h
            lstc.append(c)

        if return_hidden:
            return prevx, torch.stack(lstc)
        else:
            return prevx

    def bigram_forward(self, input, init_hidden=None, return_hidden=True):

        assert input.dim() == 3  # (len, batch, n_in)
        if init_hidden is None:
            init_hidden = [None for _ in range(self.num_layers)]
        else:
            for c in init_hidden:
                assert c.dim() == 3
            init_hidden = [(c1.squeeze(0), c2.squeeze(0))
                           for c1, c2 in zip(
                    init_hidden[0].chunk(self.num_layers, 0),
                    init_hidden[1].chunk(self.num_layers, 0)
                )]

        prevx = input
        lstc1, lstc2 = [], []
        for i, rnn in enumerate(self.rnn_lst):
            h, c1, c2 = rnn(prevx, init_hidden[i])
            prevx = self.ln_lst[i](h) if self.use_layer_norm else h
            lstc1.append(c1)
            lstc2.append(c2)

        if return_hidden:
            return prevx, (torch.stack(lstc1), torch.stack(lstc2))
        else:
            return prevx



    def ngram_forward(self, input, init_hidden=None, return_hidden=True, keep_trace=False):
        assert input.dim() == 3  # (len, batch, n_in)
        if init_hidden is None:
            init_hidden = [None for _ in range(self.num_layers)]
        else:
            assert False, "THIS IS NOT IMPLEMENTED, I DON'T THINK IT'S NECESSARY FOR CLASSIFICATION"
            for c in init_hidden:
                assert c.dim() == int(self.k/2)
            init_hidden = [(c1.squeeze(0), c2.squeeze(0))
                           for c1, c2 in zip(
                    init_hidden[0].chunk(self.num_layers, 0),
                    init_hidden[1].chunk(self.num_layers, 0)
                )]
            

        prevx = input
        # ngram used to be a parameter to this method.
        #lstcs = [[] for i in range(ngram)]

        first_traces = None
        for i, rnn in enumerate(self.rnn_lst):
            h, cs, traces = rnn(prevx, init_hidden[i], keep_trace)
            if i == 0 and keep_trace:
                first_traces = traces

            #for j in range(len(cs)):
            #    lstcs[j].append(cs[j])
            prevx = self.ln_lst[i](h) if self.use_layer_norm else h

        #stacked_lstcs = [torch.stack(lstcs[i]) for i in range(len(lstcs))]
        stacked_lstcs = None
            
        if return_hidden:
            return prevx, stacked_lstcs, first_traces
        else:
            return prevx
                      
    def forward(self, input, init_hidden=None, return_hidden=True, keep_trace=False):
        if self.pattern == "unigram":
            return self.unigram_forward(input, init_hidden, return_hidden)
        elif self.pattern == "bigram":
            return self.bigram_forward(input, init_hidden, return_hidden)
        else:
            # it should be of the form "4-gram"
            #ngram = int(self.pattern.split("-")[0])
            return self.ngram_forward(input, init_hidden, return_hidden, keep_trace=keep_trace)


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(features), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
