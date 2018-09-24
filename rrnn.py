import sys
import torch
import torch.nn as nn
from torch.autograd import Variable

def RRNN_Compute_CPU(d, k, semiring, bidirectional=False):
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
                tmp = eps[di, :] + c1_prev
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


class RRNNCell(nn.Module):
    def __init__(self,
                 semiring,
                 n_in,
                 n_out,
                 dropout=0.2,
                 rnn_dropout=0.2,
                 bidirectional=False,
                 use_tanh=1,
                 use_relu=0,
                 use_selu=0,
                 weight_norm=False,
                 index=-1,
                 use_output_gate=True):
        super(RRNNCell, self).__init__()
        assert (n_out % 2) == 0
        self.semiring = semiring
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
        if use_tanh:
            self.activation_type = 1
        elif use_relu:
            self.activation_type = 2
        elif use_selu:
            self.activation_type = 3

        # basic: in1, in2, f1, f2
        # optional: output.
        self.k = 5 if self.use_output_gate else 4
        self.n_bias = 5 if self.use_output_gate else 4
        self.size_per_dir = n_out*self.k

        self.weight = nn.Parameter(torch.Tensor(
            n_in,
            self.size_per_dir*self.bidir
        ))
        self.bias = nn.Parameter(torch.Tensor(
            n_out*self.n_bias*self.bidir
        ))
        self.bias_eps = nn.Parameter(torch.Tensor(self.bidir*n_out))
        self.bias_final = nn.Parameter(torch.Tensor(self.bidir*n_out*2))
        self.init_weights()

    def init_weights(self, rescale=True):
        val_range = (6.0 / (self.n_in + self.n_out)) ** 0.5
        self.weight.data.uniform_(-val_range, val_range)

        # initialize bias
        self.bias.data.zero_()
        self.bias_eps.data.zero_()
        self.bias_final.data.zero_()
        n_out = self.n_out

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

    def real_forward(self, input, init_hidden=None):
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
        bias = self.bias.view(self.n_bias, bidir, n_out)

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
            from rrnn_gpu import RRNN_Compute_GPU
            RRNN_Compute = RRNN_Compute_GPU(n_out, 4, self.semiring, self.bidirectional)
        else:
            RRNN_Compute = RRNN_Compute_CPU(n_out, 4, self.semiring, self.bidirectional)

        eps = self.bias_eps.view(bidir, n_out).sigmoid()
        c1s, c2s, c1_final, c2_final= RRNN_Compute(u, c1_init, c2_init, eps)

        rho = self.bias_final.view(bidir, n_out, 2).sigmoid()
        cs = c1s * rho[...,0] + c2s * rho[...,1]

        if self.use_output_gate:
            gcs = self.calc_activation(output*cs)
        else:
            gcs = self.calc_activation(cs)

        return gcs.view(length, batch, -1), c1_final, c2_final

    def forward(self, input, init_hidden=None):
        if self.semiring.type == 0:
            # plus times
            return self.real_forward(input=input, init_hidden=init_hidden)
        else:
            return self.semiring_forward(input=input, init_hidden=init_hidden)

    def get_dropout_mask_(self, size, p, rescale=True):
        w = self.weight.data
        if rescale:
            return Variable(w.new(*size).bernoulli_(1-p).div_(1-p))
        else:
            return Variable(w.new(*size).bernoulli_(1-p))


class RRNN(nn.Module):
    def __init__(self,
                 semiring,
                 input_size,
                 hidden_size,
                 num_layers=2,
                 dropout=0.2,
                 rnn_dropout=0.2,
                 bidirectional=False,
                 use_tanh=1,
                 use_relu=0,
                 use_selu=0,
                 weight_norm=False,
                 layer_norm=False,
                 use_output_gate=True):
        super(RRNN, self).__init__()
        assert not bidirectional
        self.semiring = semiring
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn_dropout = rnn_dropout
        self.rnn_lst = nn.ModuleList()
        self.ln_lst = nn.ModuleList()
        self.bidirectional = bidirectional
        self.use_layer_norm = layer_norm
        self.use_wieght_norm = weight_norm
        self.out_size = hidden_size * 2 if bidirectional else hidden_size

        if use_tanh + use_relu + use_selu > 1:
            sys.stderr.write("\nWARNING: More than one activation enabled in RRNN"
                " (tanh: {}  relu: {}  selu: {})\n".format(use_tanh, use_relu, use_selu)
            )

        for i in range(num_layers):
            l = RRNNCell(
                semiring=semiring,
                n_in=self.input_size if i == 0 else self.out_size,
                n_out=self.hidden_size,
                dropout=dropout if i+1 != num_layers else 0.,
                rnn_dropout=rnn_dropout,
                bidirectional=bidirectional,
                use_tanh=use_tanh,
                use_relu=use_relu,
                use_selu=use_selu,
                weight_norm=weight_norm,
                index=i+1,
                use_output_gate=use_output_gate
            )
            self.rnn_lst.append(l)
            if layer_norm:
                self.ln_lst.append(LayerNorm(self.hidden_size))

    def init_weights(self):
        for l in self.rnn_lst:
            l.init_weights()

    def forward(self, input, init_hidden=None, return_hidden=True):
        assert input.dim() == 3 # (len, batch, n_in)
        if init_hidden is None:
            init_hidden = [None for _ in range(self.num_layers)]
        else:
            for c in init_hidden:
                assert c.dim() == 3
            init_hidden = [(c1.squeeze(0), c2.squeeze(0))
                           for c1,c2 in zip(
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
