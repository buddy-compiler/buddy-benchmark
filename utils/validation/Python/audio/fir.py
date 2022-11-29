from cffi import FFI
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import os
import sys
import time
import math
import random
from .audio_test import AudioTest
import utils.audio_format as af


class FIRTest(AudioTest):
    """FIR test class.
         Test params:
            - file: path to the audio file to be filtered.
            - fconf: filter configurations.
            - lib: path to the library file.

    """
    default_param = {"file": "../../../benchmarks/AudioProcessing/Audios/NASA_Mars.wav",
                     "fconf": ('kaiser', 4.0),
                     "lib": "../../../build/Utils/validation/AudioProcessing/libCWrapper"}

    def __init__(self, test_name, test_type, test_params=default_param):
        super(FIRTest, self).__init__(test_name, test_type, test_params)
        print(os.getcwd())
        self.params = test_params

    def run(self):
        self.run_file_test()

    def run_file_test(self):
        ffi = FFI()
        ffi.cdef('''
            float* fir(float* input, float* kernel, float* output, int inputSize, int kernelSize, int outputSize);
        ''')
        C = ffi.dlopen(self.params['lib'])

        sample_rate, sp_nasa = sp.io.wavfile.read(self.params['file'])

        firfilt = sp.signal.firwin(
            10, 0.1, window=self.params['fconf'], pass_zero='lowpass', scale=False).astype(np.float32)

        sp_nasa = af.pcm2float(sp_nasa, dtype='float32')

        grpdelay = sp.signal.group_delay((firfilt, 1.0))
        delay = round(np.mean(grpdelay[1]), 2)
        print(f"grpdelay: {delay}")
        # buddy fir filtering
        input = ffi.cast("float *", ffi.from_buffer(sp_nasa))
        kernel = ffi.cast("float *", firfilt.ctypes.data)
        output = ffi.new("float[]", sp_nasa.size)

        out = C.fir(input, kernel, output,
                    sp_nasa.size, firfilt.size, sp_nasa.size)

        out = ffi.unpack(out, sp_nasa.size)

        # scipy fir filtering
        out_sp = sp.signal.lfilter(firfilt, 1, sp_nasa)

        # numpy fir filtering
        out_np = np.convolve(sp_nasa, firfilt, mode='full')

        diff = math.floor(2*delay)
        for i in range(sp_nasa.size-diff):
            if (out[i] - out_sp[i+diff] > 0.0001):
                print(f"scipy and buddy are different at {i}")
                print("check failed.")
                sys.exit(1)
        print("check successful.")
