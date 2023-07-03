#    audio_file.py - Audio file class for audio validation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This File is for the AudioFile class.

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
import utils.lib_path as lp

class AudioFileTest(AudioTest):
    """AudioFile test class.
         Test params:
            - file: path to the audio file to be filtered.
            - savefile: path to the test file to be saved.
            - fconf: AudioFile configurations.
            - lib: path to the library file.

    """
    default_param = {"file": "../../benchmarks/AudioProcessing/Audios/NASA_Mars.wav",
                     "savefile": "./NASA_Mars_save.wav",
                     "lib": "../../build/validation/AudioProcessing/libAudioValidationLib"}

    def __init__(self, test_name, test_type, test_params=default_param):
        super(AudioFileTest, self).__init__(test_name, test_type, test_params)
        self.params = test_params

    def run(self):
        print(f"{self.test_name} started.")
        self.run_file_test()

    def run_file_test(self):
        ffi = FFI()
        ffi.cdef('''
            float* AudioRead(char* file, char* dest);
        ''')
        lib_path = lp.format_lib_path(self.params['lib'])
        C = ffi.dlopen(lib_path)

        # Read audio file using scipy
        sample_rate, sp_nasa = sp.io.wavfile.read(self.params['file'])
        sp_nasa = af.pcm2float(sp_nasa, dtype='float32')

        # Read audio file using AudioFile
        c_nasa = C.AudioRead(self.params['file'].encode(
            'utf-8'), self.params['savefile'].encode('utf-8'))

        for i in range(sp_nasa.size):
            if (sp_nasa[i] - c_nasa[i] > 0.0001):
                print(f"scipy and AudioFile are different at {i}")
                print("check failed.")
                sys.exit(1)
        print(f"{self.test_name} reading check successful.")

        new_sample_rate, new_sp_nasa = sp.io.wavfile.read(
            self.params['savefile'])
        new_sp_nasa = af.pcm2float(new_sp_nasa, dtype='float32')
        for i in range(sp_nasa.size):
            if (sp_nasa[i] - new_sp_nasa[i] > 0.0001):
                print(f"scipy and saved audio are different at {i}")
                print("check failed.")
                sys.exit(1)
        print(f"file written successfully at {self.params['savefile']}")
        print(f"{self.test_name} writing check successful.")
