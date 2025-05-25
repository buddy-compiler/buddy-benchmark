#    audio_test.py - Audio test base class.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This File is the base class for all audio tests.
# This file is part of the Validation framework for the Audio part of the buddy-mlir project.

# How to extend:
#   1. Add a new class that inherits from AudioTest.
#   2. Implement the run_file_test or/and run_random_test method.
#   3. Add the new class to the list of tests in the main function.


class AudioTest(object):
    """Audio test class."""

    def __init__(self, test_name, test_type, test_params):
        self.test_name = test_name
        self.test_type = test_type
        self.test_params = test_params

    def run(self):
        """Run the test."""
        if self.test_type == 'file':
            self.run_file_test()
        elif self.test_type == 'random':
            self.run_random_test()
        else:
            raise ValueError('Unknown test type: %s' % self.test_type)

    def run_file_test(self):
        pass

    def run_random_test(self):
        pass
