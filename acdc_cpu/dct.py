
import os

import theano
from theano import gof

class DCT(gof.COp):
    __props__ = ()

    func_file = "dct.c"
    func_name = "APPLY_SPECIFIC(DCT_updateOutput)"

    def __init__(self):
        super(DCT, self).__init__(self.func_file, self.func_name)

    def make_node(self, x):
        x_ = theano.tensor.as_tensor_variable(x)
        return gof.Apply(self, [x_], [x_.type()])

    def grad(self, inputs, cost_grad):
        grad = cost_grad[0]
        return [IDCT()(grad)]

    def c_libraries(self):
        return ['fftw3']

    def c_headers(self):
        return ['fftw.hpp']

    def c_header_dirs(self):
        return [os.path.dirname(os.path.abspath(__file__))]

class IDCT(DCT):
    __props__ = ()

    func_file = "dct_inverse.c"
    func_name = "APPLY_SPECIFIC(IDCT_updateOutput)"

    def __init__(self):
        super(IDCT, self).__init__()
        super(DCT, self).__init__(self.func_file, self.func_name)

    def grad(self, inputs, cost_grad):
        grad = cost_grad[0]
        return [DCT()(grad)]
