# Audio Test file.
# This file is part of the Validation framework for the Audio part of the buddy-mlir project.

# How to extend:
# 1. Add a new class that inherits from AudioTest.
# 2. Implement the run_file_test or/and run_random_test method.
# 3. Add the new class to the list of tests in the main function.

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
