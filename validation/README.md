# Correctness Checking Framework

## Python based correctness checking

## Environment Setup

Please build the "AudioValidationLib" target in CMake.
It would generate a dynamic library for CFFI to use.

Please install required packages by using:
```
pip install -r requirements.txt
```

It is recommended that you use a virtual environment to install the required packages.
Virtualenv or conda are recommended.

Target "CWrapper" should be built before using python code to test.
It would generate a dynamic library for CFFI to use.

If your build path is not 
```
build
```
Then the path should be configured before invoking the python script.

### Execution
```
python main.py
```

to test all the test cases in the test folder.
For configuration and example, please refer to each test files.

### How to add a test case
There is no strict rule for adding a test case.
The test case should be a python file with a class inherited from AudioTest.
You would need to modify CWrapper.cpp to add new function wrappers for the new test case.
The class should have a method named "run" which will be invoked by the main.py.
