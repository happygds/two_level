import os
import torch
from torch.utils.ffi import create_extension
import subprocess

cmd = 'source deactivate && cd src/cuda && nvcc -c -o roi1d_pooling.cu.o roi1d_pooling_kernel.cu -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_52'
print(cmd)
subprocess.call(cmd, shell=True)
print('\n')

# sources = ['src/roi1d_pooling.c']
# headers = ['src/roi1d_pooling.h']
sources = []
headers = []
defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/roi1d_pooling_cuda.c']
    headers += ['src/roi1d_pooling_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)
extra_objects = ['src/cuda/roi1d_pooling.cu.o']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ffi = create_extension(
    '_ext.roi1d_pooling',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects
)

if __name__ == '__main__':
    ffi.build()
