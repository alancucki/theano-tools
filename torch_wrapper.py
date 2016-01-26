#
# Poor man's wrapper for Torch calls.
# Communicates with NumPy through the filesystem.
#

import os, sys
import subprocess

import numpy as np

TORCH_BIN_PATH = '' # XXX Specify path to the 'th' executable.

def apply_fun(M, fun):

    if TORCH_BIN_PATH == '':
        raise ValueError('Please specify path to Torch bin directory.')

    out_fpath_np = '/tmp/.acdc_tmp_np.csv'
    out_fpath_th = '/tmp/.acdc_tmp_th.csv'
    path = os.path.dirname(os.path.abspath(__file__))

    np.savetxt(out_fpath_np, M, delimiter=',')

    FNULL = open(os.devnull, 'w')
    subprocess.check_call([os.path.join(path, 'csv2t7.sh'),
                          out_fpath_np, out_fpath_th],
                          stdout=FNULL, stderr=subprocess.STDOUT)

    env = os.environ.copy()
    env["PATH"] = env["PATH"]+":"+TORCH_BIN_PATH
    print 'th', 'torch_wrapper.lua', out_fpath_th, fun

    subprocess.check_call(['th', 'torch_wrapper.lua', out_fpath_th, fun],
                          env=env, stdout=FNULL, stderr=subprocess.STDOUT)

    M = np.loadtxt(out_fpath_th, delimiter=',')
    return M

if __name__ == '__main__':

    M = np.random.normal(size=(3,3))
    print M
    print '----------'
    print apply_fun(M, 'dct')
