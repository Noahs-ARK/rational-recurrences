import torch
from torch.autograd import Function
from collections import namedtuple
from pynvrtc.compiler import Program
from cupy.cuda import function
import numpy as np
from cuda.utils import *
from cuda.bigram_rrnn import *
from cuda.bigram_rrnn_semiring import *
from cuda.unigram_rrnn import *
from cuda.fourgram_rrnn import *
from cuda.threegram_rrnn import *
from cuda.twogram_rrnn import *
from cuda.onegram_rrnn import *


class RRNN_Unigram_Compute_GPU(Function):

    #_RRNN_PROG = Program((UTIL + UNIGRAM_RRNN).encode("utf-8"), "rrnn_prog.cu".encode())
    #_RRNN_PTX = _RRNN_PROG.compile()
    #_DEVICE2FUNC = {}


    def __init__(self, d_out, k, semiring, bidirectional=False):
        super(RRNN_Unigram_Compute_GPU, self).__init__()
        self.semiring = semiring
        self.d_out = d_out
        self.k = k
        self.bidirectional = bidirectional
        assert not bidirectional


    def compile_functions(self):
        device = torch.cuda.current_device()
        print ("RRNN loaded for gpu {}".format(device))
        mod = function.Module()
        mod.load(bytes(self._RRNN_PTX.encode()))

        if self.semiring.type == 0:
            fwd_func = mod.get_function("rrnn_fwd")
            bwd_func = mod.get_function("rrnn_bwd")
            Stream = namedtuple("Stream", ["ptr"])
            current_stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
            self._DEVICE2FUNC[device] = (
                current_stream, fwd_func, bwd_func,
            )
            return current_stream, fwd_func, bwd_func
        else:
            fwd_func = mod.get_function("rrnn_semiring_fwd")
            bwd_func = mod.get_function("rrnn_semiring_bwd")
            Stream = namedtuple("Stream", ["ptr"])
            current_stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
            self._DEVICE2FUNC[device] = (
                current_stream, fwd_func, bwd_func
            )
            return current_stream, fwd_func, bwd_func


    def get_functions(self):
        res = self._DEVICE2FUNC.get(torch.cuda.current_device(), None)
        return res if res else self.compile_functions()


    def forward(self, u, c_init=None):
        bidir = 2 if self.bidirectional else 1
        assert u.size(-1) == self.k
        length, batch = u.size(0), u.size(1)
        dim = self.d_out
        ncols = batch*dim*bidir
        thread_per_block = min(512, ncols)
        num_block = (ncols-1)//thread_per_block+1
        if c_init is None:
            assert False

        size = (length, batch, bidir, dim)
        cs = u.new(*size)
        stream, fwd_func, _ = self.get_functions()
        FUNC = fwd_func
        FUNC(args=[
            u.contiguous().data_ptr(),
            c_init.contiguous().data_ptr(),
            np.int32(length),
            np.int32(batch),
            np.int32(dim),
            np.int32(self.k),
            cs.data_ptr(),
            np.int32(self.semiring.type)],
            block = (thread_per_block,1,1), grid = (num_block,1,1),
            stream=stream
        )
        self.save_for_backward(u, c_init)
        self.intermediate_cs = cs
        if self.bidirectional:
            last_c = torch.cat((cs[-1,:,0,:], cs[0,:,1,:]), dim=1)
        else:
            last_c = cs[-1,...].view(batch, -1)
        return cs, last_c


    def backward(self, grad_cs, grad_last_c):
        bidir = 2 if self.bidirectional else 1
        u, c_init = self.saved_tensors
        cs = self.intermediate_cs
        length, batch = u.size(0), u.size(1)
        dim = self.d_out
        ncols = batch*dim*bidir
        thread_per_block = min(512, ncols)
        num_block = (ncols-1)//thread_per_block+1

        if c_init is None:
            assert False
        # init_ = x.new(ncols).zero_() if init is None else init
        grad_u = u.new(*u.size())
        grad_init_c = u.new(batch, dim*bidir)
        stream, _, bwd_func = self.get_functions()
        FUNC = bwd_func

        FUNC(args=[
            u.contiguous().data_ptr(),
            c_init.contiguous().data_ptr(),
            cs.data_ptr(),
            grad_cs.data_ptr(),
            grad_last_c.contiguous().data_ptr(),
            np.int32(length),
            np.int32(batch),
            np.int32(dim),
            np.int32(self.k),
            grad_u.data_ptr(),
            grad_init_c.data_ptr(),
            np.int32(self.semiring.type)],
            block = (thread_per_block,1,1), grid = (num_block,1,1),
            stream=stream
        )

        return grad_u, grad_init_c


class RRNN_Bigram_Compute_GPU(Function):

    _RRNN_PROG = Program((UTIL + BIGRAM_RRNN + BIGRAM_RRNN_SEMIRING).encode("utf-8"), "rrnn_prog.cu".encode())
    _RRNN_PTX = _RRNN_PROG.compile()
    _DEVICE2FUNC = {}


    def __init__(self, d_out, k, semiring, bidirectional=False):
        super(RRNN_Bigram_Compute_GPU, self).__init__()
        self.semiring = semiring
        self.d_out = d_out
        self.k = k
        self.bidirectional = bidirectional
        assert not bidirectional


    def compile_functions(self):
        device = torch.cuda.current_device()
        print ("RRNN loaded for gpu {}".format(device))
        mod = function.Module()
        mod.load(bytes(self._RRNN_PTX.encode()))

        if self.semiring.type == 0:
            fwd_func = mod.get_function("rrnn_fwd")
            bwd_func = mod.get_function("rrnn_bwd")
            Stream = namedtuple("Stream", ["ptr"])
            current_stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
            self._DEVICE2FUNC[device] = (
                current_stream, fwd_func, bwd_func,
            )
            return current_stream, fwd_func, bwd_func
        else:
            fwd_func = mod.get_function("rrnn_semiring_fwd")
            bwd_func = mod.get_function("rrnn_semiring_bwd")
            Stream = namedtuple("Stream", ["ptr"])
            current_stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
            self._DEVICE2FUNC[device] = (
                current_stream, fwd_func, bwd_func
            )
            return current_stream, fwd_func, bwd_func


    def get_functions(self):
        res = self._DEVICE2FUNC.get(torch.cuda.current_device(), None)
        return res if res else self.compile_functions()


    def forward(self, u, c1_init=None, c2_init=None, eps=None):
        bidir = 2 if self.bidirectional else 1
        assert u.size(-1) == self.k
        length, batch = u.size(0), u.size(1)
        dim = self.d_out
        ncols = batch*dim*bidir
        thread_per_block = min(512, ncols)
        num_block = (ncols-1)//thread_per_block+1
        if c1_init is None:
            assert False

        size = (length, batch, bidir, dim)
        c1s = u.new(*size)
        c2s = u.new(*size)
        stream, fwd_func, _ = self.get_functions()
        FUNC = fwd_func
        FUNC(args=[
            u.contiguous().data_ptr(),
            eps.contiguous().data_ptr(),
            c1_init.contiguous().data_ptr(),
            c2_init.contiguous().data_ptr(),
            np.int32(length),
            np.int32(batch),
            np.int32(dim),
            np.int32(self.k),
            c1s.data_ptr(),
            c2s.data_ptr(),
            np.int32(self.semiring.type)],
            block = (thread_per_block,1,1), grid = (num_block,1,1),
            stream=stream
        )
        self.save_for_backward(u, eps, c1_init, c2_init)
        self.intermediate_c1s, self.intermediate_c2s = c1s, c2s
        if self.bidirectional:
            last_c1, last_c2 \
                = torch.cat((c1s[-1,:,0,:], c1s[0,:,1,:]), dim=1), \
                  torch.cat((c2s[-1,:,0,:], c2s[0,:,1,:]), dim=1)
        else:
            last_c1 = c1s[-1,...].view(batch, -1)
            last_c2 = c2s[-1,...].view(batch, -1)
        return c1s, c2s, last_c1, last_c2


    def backward(self, grad_c1s, grad_c2s, grad_last_c1, grad_last_c2):
        bidir = 2 if self.bidirectional else 1
        u, eps, c1_init, c2_init = self.saved_tensors
        c1s, c2s = self.intermediate_c1s, self.intermediate_c2s
        length, batch = u.size(0), u.size(1)
        dim = self.d_out
        ncols = batch*dim*bidir
        thread_per_block = min(512, ncols)
        num_block = (ncols-1)//thread_per_block+1

        if c1_init is None:
            assert False
        # init_ = x.new(ncols).zero_() if init is None else init
        grad_u = u.new(*u.size())
        grad_eps = eps.new(*eps.size())
        grad_init_c1 = u.new(batch, dim*bidir)
        grad_init_c2 = u.new(batch, dim*bidir)
        stream, _, bwd_func = self.get_functions()
        FUNC = bwd_func

        FUNC(args=[
            u.contiguous().data_ptr(),
            eps.contiguous().data_ptr(),
            c1_init.contiguous().data_ptr(),
            c2_init.contiguous().data_ptr(),
            c1s.data_ptr(),
            c2s.data_ptr(),
            grad_c1s.data_ptr(),
            grad_c2s.data_ptr(),
            grad_last_c1.contiguous().data_ptr(),
            grad_last_c2.contiguous().data_ptr(),
            np.int32(length),
            np.int32(batch),
            np.int32(dim),
            np.int32(self.k),
            grad_u.data_ptr(),
            grad_eps.data_ptr(),
            grad_init_c1.data_ptr(),
            grad_init_c2.data_ptr(),
            np.int32(self.semiring.type)],
            block = (thread_per_block,1,1), grid = (num_block,1,1),
            stream=stream
        )

        return grad_u, grad_init_c1, grad_init_c2, grad_eps


class RRNN_1gram_Compute_GPU(Function):

    _RRNN_PROG = Program((UTIL + ONEGRAM_RRNN).encode("utf-8"), "rrnn_prog.cu".encode())
    _RRNN_PTX = _RRNN_PROG.compile()
    _DEVICE2FUNC = {}


    def __init__(self, d_out, k, semiring, bidirectional=False):
        super(RRNN_1gram_Compute_GPU, self).__init__()
        self.semiring = semiring
        self.d_out = d_out
        self.k = k
        self.bidirectional = bidirectional
        assert not bidirectional


    def compile_functions(self):
        device = torch.cuda.current_device()
        print ("RRNN loaded for gpu {}".format(device))
        mod = function.Module()
        mod.load(bytes(self._RRNN_PTX.encode()))

        if self.semiring.type == 0:
            fwd_func = mod.get_function("rrnn_fwd")
            bwd_func = mod.get_function("rrnn_bwd")
            Stream = namedtuple("Stream", ["ptr"])
            current_stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
            self._DEVICE2FUNC[device] = (
                current_stream, fwd_func, bwd_func,
            )
            return current_stream, fwd_func, bwd_func
        else:
            assert False, "other semirings are not currently implemented."

    def get_functions(self):
        res = self._DEVICE2FUNC.get(torch.cuda.current_device(), None)
        return res if res else self.compile_functions()


    def forward(self, u, c1_init=None):
        bidir = 2 if self.bidirectional else 1
        assert u.size(-1) == self.k
        length, batch = u.size(0), u.size(1)
        dim = self.d_out
        ncols = batch*dim*bidir
        thread_per_block = min(512, ncols)
        num_block = (ncols-1)//thread_per_block+1
        if c1_init is None:
            assert False

        size = (length, batch, bidir, dim)
        c1s = u.new(*size)
        stream, fwd_func, _ = self.get_functions()
        FUNC = fwd_func
        FUNC(args=[
            u.contiguous().data_ptr(),
            c1_init.contiguous().data_ptr(),
            np.int32(length),
            np.int32(batch),
            np.int32(dim),
            np.int32(self.k),
            c1s.data_ptr(),
            np.int32(self.semiring.type)],
            block = (thread_per_block,1,1), grid = (num_block,1,1),
            stream=stream
        )
        self.save_for_backward(u, c1_init)
        self.intermediate_c1s = c1s

        if self.bidirectional:
            assert False, "bidirectionality isn't implemented yet"
        else:
            last_c1 = c1s[-1,...].view(batch, -1)
        return c1s, last_c1


    def backward(self, grad_c1s, grad_last_c1):
        bidir = 2 if self.bidirectional else 1
        u, c1_init = self.saved_tensors
        c1s = self.intermediate_c1s
        length, batch = u.size(0), u.size(1)
        dim = self.d_out
        ncols = batch*dim*bidir
        thread_per_block = min(512, ncols)
        num_block = (ncols-1)//thread_per_block+1

        if c1_init is None:
            assert False
        # init_ = x.new(ncols).zero_() if init is None else init
        grad_u = u.new(*u.size())
        grad_init_c1 = u.new(batch, dim*bidir)
        stream, _, bwd_func = self.get_functions()
        FUNC = bwd_func

        FUNC(args=[
            u.contiguous().data_ptr(),
            c1_init.contiguous().data_ptr(),
            c1s.data_ptr(),
            grad_c1s.data_ptr(),
            grad_last_c1.contiguous().data_ptr(),
            np.int32(length),
            np.int32(batch),
            np.int32(dim),
            np.int32(self.k),
            grad_u.data_ptr(),
            grad_init_c1.data_ptr(),
            np.int32(self.semiring.type)],
            block = (thread_per_block,1,1), grid = (num_block,1,1),
            stream=stream
        )

        return grad_u, grad_init_c1

    
class RRNN_2gram_Compute_GPU(Function):

    _RRNN_PROG = Program((UTIL + TWOGRAM_RRNN).encode("utf-8"), "rrnn_prog.cu".encode())
    _RRNN_PTX = _RRNN_PROG.compile()
    _DEVICE2FUNC = {}


    def __init__(self, d_out, k, semiring, bidirectional=False):
        super(RRNN_2gram_Compute_GPU, self).__init__()
        self.semiring = semiring
        self.d_out = d_out
        self.k = k
        self.bidirectional = bidirectional
        assert not bidirectional


    def compile_functions(self):
        device = torch.cuda.current_device()
        print ("RRNN loaded for gpu {}".format(device))
        mod = function.Module()
        mod.load(bytes(self._RRNN_PTX.encode()))

        if self.semiring.type == 0:
            fwd_func = mod.get_function("rrnn_fwd")
            bwd_func = mod.get_function("rrnn_bwd")
            Stream = namedtuple("Stream", ["ptr"])
            current_stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
            self._DEVICE2FUNC[device] = (
                current_stream, fwd_func, bwd_func,
            )
            return current_stream, fwd_func, bwd_func
        else:
            assert False, "other semirings are not currently implemented."

    def get_functions(self):
        res = self._DEVICE2FUNC.get(torch.cuda.current_device(), None)
        return res if res else self.compile_functions()


    def forward(self, u, c1_init=None, c2_init=None):
        bidir = 2 if self.bidirectional else 1
        assert u.size(-1) == self.k
        length, batch = u.size(0), u.size(1)
        dim = self.d_out
        ncols = batch*dim*bidir
        thread_per_block = min(512, ncols)
        num_block = (ncols-1)//thread_per_block+1
        if c1_init is None:
            assert False

        size = (length, batch, bidir, dim)
        c1s = u.new(*size)
        c2s = u.new(*size)
        stream, fwd_func, _ = self.get_functions()
        FUNC = fwd_func
        FUNC(args=[
            u.contiguous().data_ptr(),
            c1_init.contiguous().data_ptr(),
            c2_init.contiguous().data_ptr(),
            np.int32(length),
            np.int32(batch),
            np.int32(dim),
            np.int32(self.k),
            c1s.data_ptr(),
            c2s.data_ptr(),
            np.int32(self.semiring.type)],
            block = (thread_per_block,1,1), grid = (num_block,1,1),
            stream=stream
        )
        self.save_for_backward(u, c1_init, c2_init)
        self.intermediate_c1s, self.intermediate_c2s = c1s, c2s

        if self.bidirectional:
            assert False, "bidirectionality isn't implemented yet"
        else:
            last_c1 = c1s[-1,...].view(batch, -1)
            last_c2 = c2s[-1,...].view(batch, -1)
        return c1s, c2s, last_c1, last_c2


    def backward(self, grad_c1s, grad_c2s, grad_last_c1, grad_last_c2):
        bidir = 2 if self.bidirectional else 1
        u, c1_init, c2_init = self.saved_tensors
        c1s, c2s = self.intermediate_c1s, self.intermediate_c2s
        length, batch = u.size(0), u.size(1)
        dim = self.d_out
        ncols = batch*dim*bidir
        thread_per_block = min(512, ncols)
        num_block = (ncols-1)//thread_per_block+1

        if c1_init is None:
            assert False
        # init_ = x.new(ncols).zero_() if init is None else init
        grad_u = u.new(*u.size())
        grad_init_c1 = u.new(batch, dim*bidir)
        grad_init_c2 = u.new(batch, dim*bidir)
        stream, _, bwd_func = self.get_functions()
        FUNC = bwd_func

        FUNC(args=[
            u.contiguous().data_ptr(),
            c1_init.contiguous().data_ptr(),
            c2_init.contiguous().data_ptr(),
            c1s.data_ptr(),
            c2s.data_ptr(),
            grad_c1s.data_ptr(),
            grad_c2s.data_ptr(),
            grad_last_c1.contiguous().data_ptr(),
            grad_last_c2.contiguous().data_ptr(),
            np.int32(length),
            np.int32(batch),
            np.int32(dim),
            np.int32(self.k),
            grad_u.data_ptr(),
            grad_init_c1.data_ptr(),
            grad_init_c2.data_ptr(),
            np.int32(self.semiring.type)],
            block = (thread_per_block,1,1), grid = (num_block,1,1),
            stream=stream
        )

        return grad_u, grad_init_c1, grad_init_c2

class RRNN_3gram_Compute_GPU(Function):

    _RRNN_PROG = Program((UTIL + THREEGRAM_RRNN).encode("utf-8"), "rrnn_prog.cu".encode())
    _RRNN_PTX = _RRNN_PROG.compile()
    _DEVICE2FUNC = {}


    def __init__(self, d_out, k, semiring, bidirectional=False):
        super(RRNN_3gram_Compute_GPU, self).__init__()
        self.semiring = semiring
        self.d_out = d_out
        self.k = k
        self.bidirectional = bidirectional
        assert not bidirectional


    def compile_functions(self):
        device = torch.cuda.current_device()
        print ("RRNN loaded for gpu {}".format(device))
        mod = function.Module()
        mod.load(bytes(self._RRNN_PTX.encode()))

        if self.semiring.type == 0:
            fwd_func = mod.get_function("rrnn_fwd")
            bwd_func = mod.get_function("rrnn_bwd")
            Stream = namedtuple("Stream", ["ptr"])
            current_stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
            self._DEVICE2FUNC[device] = (
                current_stream, fwd_func, bwd_func,
            )
            return current_stream, fwd_func, bwd_func
        else:
            assert False, "other semirings are not currently implemented."

    def get_functions(self):
        res = self._DEVICE2FUNC.get(torch.cuda.current_device(), None)
        return res if res else self.compile_functions()


    def forward(self, u, c1_init=None, c2_init=None, c3_init=None):
        bidir = 2 if self.bidirectional else 1
        assert u.size(-1) == self.k
        length, batch = u.size(0), u.size(1)
        dim = self.d_out
        ncols = batch*dim*bidir
        thread_per_block = min(512, ncols)
        num_block = (ncols-1)//thread_per_block+1
        if c1_init is None:
            assert False

        size = (length, batch, bidir, dim)
        c1s = u.new(*size)
        c2s = u.new(*size)
        c3s = u.new(*size)
        stream, fwd_func, _ = self.get_functions()
        FUNC = fwd_func
        FUNC(args=[
            u.contiguous().data_ptr(),
            c1_init.contiguous().data_ptr(),
            c2_init.contiguous().data_ptr(),
            c3_init.contiguous().data_ptr(),
            np.int32(length),
            np.int32(batch),
            np.int32(dim),
            np.int32(self.k),
            c1s.data_ptr(),
            c2s.data_ptr(),
            c3s.data_ptr(),
            np.int32(self.semiring.type)],
            block = (thread_per_block,1,1), grid = (num_block,1,1),
            stream=stream
        )
        self.save_for_backward(u, c1_init, c2_init, c3_init)
        self.intermediate_c1s, self.intermediate_c2s, self.intermediate_c3s = c1s, c2s, c3s

        if self.bidirectional:
            assert False, "bidirectionality isn't implemented yet"
        else:
            last_c1 = c1s[-1,...].view(batch, -1)
            last_c2 = c2s[-1,...].view(batch, -1)
            last_c3 = c3s[-1,...].view(batch, -1)
        return c1s, c2s, c3s, last_c1, last_c2, last_c3


    def backward(self, grad_c1s, grad_c2s, grad_c3s, grad_last_c1, grad_last_c2, grad_last_c3):
        bidir = 2 if self.bidirectional else 1
        u, c1_init, c2_init, c3_init = self.saved_tensors
        c1s, c2s, c3s = self.intermediate_c1s, self.intermediate_c2s, self.intermediate_c3s
        length, batch = u.size(0), u.size(1)
        dim = self.d_out
        ncols = batch*dim*bidir
        thread_per_block = min(512, ncols)
        num_block = (ncols-1)//thread_per_block+1

        if c1_init is None:
            assert False
        # init_ = x.new(ncols).zero_() if init is None else init
        grad_u = u.new(*u.size())
        grad_init_c1 = u.new(batch, dim*bidir)
        grad_init_c2 = u.new(batch, dim*bidir)
        grad_init_c3 = u.new(batch, dim*bidir)
        stream, _, bwd_func = self.get_functions()
        FUNC = bwd_func

        FUNC(args=[
            u.contiguous().data_ptr(),
            c1_init.contiguous().data_ptr(),
            c2_init.contiguous().data_ptr(),
            c3_init.contiguous().data_ptr(),
            c1s.data_ptr(),
            c2s.data_ptr(),
            c3s.data_ptr(),
            grad_c1s.data_ptr(),
            grad_c2s.data_ptr(),
            grad_c3s.data_ptr(),
            grad_last_c1.contiguous().data_ptr(),
            grad_last_c2.contiguous().data_ptr(),
            grad_last_c3.contiguous().data_ptr(),
            np.int32(length),
            np.int32(batch),
            np.int32(dim),
            np.int32(self.k),
            grad_u.data_ptr(),
            grad_init_c1.data_ptr(),
            grad_init_c2.data_ptr(),
            grad_init_c3.data_ptr(),
            np.int32(self.semiring.type)],
            block = (thread_per_block,1,1), grid = (num_block,1,1),
            stream=stream
        )

        return grad_u, grad_init_c1, grad_init_c2, grad_init_c3


class RRNN_4gram_Compute_GPU(Function):

    _RRNN_PROG = Program((UTIL + FOURGRAM_RRNN).encode("utf-8"), "rrnn_prog.cu".encode())
    _RRNN_PTX = _RRNN_PROG.compile()
    _DEVICE2FUNC = {}


    def __init__(self, d_out, k, semiring, bidirectional=False):
        super(RRNN_4gram_Compute_GPU, self).__init__()
        self.semiring = semiring
        self.d_out = d_out
        self.k = k
        self.bidirectional = bidirectional
        assert not bidirectional


    def compile_functions(self):
        device = torch.cuda.current_device()
        print ("RRNN loaded for gpu {}".format(device))
        mod = function.Module()
        mod.load(bytes(self._RRNN_PTX.encode()))

        if self.semiring.type == 0:
            fwd_func = mod.get_function("rrnn_fwd")
            bwd_func = mod.get_function("rrnn_bwd")
            Stream = namedtuple("Stream", ["ptr"])
            current_stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
            self._DEVICE2FUNC[device] = (
                current_stream, fwd_func, bwd_func,
            )
            return current_stream, fwd_func, bwd_func
        else:
            assert False, "other semirings are not currently implemented."

    def get_functions(self):
        res = self._DEVICE2FUNC.get(torch.cuda.current_device(), None)
        return res if res else self.compile_functions()


    def forward(self, u, c1_init=None, c2_init=None, c3_init=None, c4_init=None):
        bidir = 2 if self.bidirectional else 1
        assert u.size(-1) == self.k
        length, batch = u.size(0), u.size(1)
        dim = self.d_out
        ncols = batch*dim*bidir
        thread_per_block = min(512, ncols)
        num_block = (ncols-1)//thread_per_block+1
        if c1_init is None:
            assert False

        size = (length, batch, bidir, dim)
        c1s = u.new(*size)
        c2s = u.new(*size)
        c3s = u.new(*size)
        c4s = u.new(*size)
        stream, fwd_func, _ = self.get_functions()
        FUNC = fwd_func
        FUNC(args=[
            u.contiguous().data_ptr(),
            c1_init.contiguous().data_ptr(),
            c2_init.contiguous().data_ptr(),
            c3_init.contiguous().data_ptr(),
            c4_init.contiguous().data_ptr(),
            np.int32(length),
            np.int32(batch),
            np.int32(dim),
            np.int32(self.k),
            c1s.data_ptr(),
            c2s.data_ptr(),
            c3s.data_ptr(),
            c4s.data_ptr(),
            np.int32(self.semiring.type)],
            block = (thread_per_block,1,1), grid = (num_block,1,1),
            stream=stream
        )
        self.save_for_backward(u, c1_init, c2_init, c3_init, c4_init)
        self.intermediate_c1s, self.intermediate_c2s = c1s, c2s
        self.intermediate_c3s, self.intermediate_c4s = c3s, c4s
        if self.bidirectional:
            assert False, "bidirectionality isn't implemented yet"
        else:
            last_c1 = c1s[-1,...].view(batch, -1)
            last_c2 = c2s[-1,...].view(batch, -1)
            last_c3 = c3s[-1,...].view(batch, -1)
            last_c4 = c4s[-1,...].view(batch, -1)
        return c1s, c2s, c3s, c4s, last_c1, last_c2, last_c3, last_c4


    def backward(self, grad_c1s, grad_c2s, grad_c3s, grad_c4s, grad_last_c1, grad_last_c2, grad_last_c3, grad_last_c4):
        bidir = 2 if self.bidirectional else 1
        u, c1_init, c2_init, c3_init, c4_init = self.saved_tensors
        c1s, c2s, c3s, c4s = self.intermediate_c1s, self.intermediate_c2s, self.intermediate_c3s, self.intermediate_c4s
        length, batch = u.size(0), u.size(1)
        dim = self.d_out
        ncols = batch*dim*bidir
        thread_per_block = min(512, ncols)
        num_block = (ncols-1)//thread_per_block+1

        if c1_init is None:
            assert False
        # init_ = x.new(ncols).zero_() if init is None else init
        grad_u = u.new(*u.size())
        grad_init_c1 = u.new(batch, dim*bidir)
        grad_init_c2 = u.new(batch, dim*bidir)
        grad_init_c3 = u.new(batch, dim*bidir)
        grad_init_c4 = u.new(batch, dim*bidir)
        stream, _, bwd_func = self.get_functions()
        FUNC = bwd_func

        FUNC(args=[
            u.contiguous().data_ptr(),
            c1_init.contiguous().data_ptr(),
            c2_init.contiguous().data_ptr(),
            c3_init.contiguous().data_ptr(),
            c4_init.contiguous().data_ptr(),
            c1s.data_ptr(),
            c2s.data_ptr(),
            c3s.data_ptr(),
            c4s.data_ptr(),
            grad_c1s.data_ptr(),
            grad_c2s.data_ptr(),
            grad_c3s.data_ptr(),
            grad_c4s.data_ptr(),
            grad_last_c1.contiguous().data_ptr(),
            grad_last_c2.contiguous().data_ptr(),
            grad_last_c3.contiguous().data_ptr(),
            grad_last_c4.contiguous().data_ptr(),
            np.int32(length),
            np.int32(batch),
            np.int32(dim),
            np.int32(self.k),
            grad_u.data_ptr(),
            grad_init_c1.data_ptr(),
            grad_init_c2.data_ptr(),
            grad_init_c3.data_ptr(),
            grad_init_c4.data_ptr(),
            np.int32(self.semiring.type)],
            block = (thread_per_block,1,1), grid = (num_block,1,1),
            stream=stream
        )

        return grad_u, grad_init_c1, grad_init_c2, grad_init_c3, grad_init_c4
