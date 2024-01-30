# Third-Party Dependencies

Version of third-party performance comparison items:

| Comparison Item  | Version | Link |
| -------------- | ------------- | ------------- |
| Eigen  | 3.4.0 | [Homepage](https://eigen.tuxfamily.org/index.php?title=Main_Page), [Code](https://gitlab.com/libeigen/eigen/-/tree/master) |

## TVM

To install TVM, please refer to the official documentation ([Install TVM from source](https://tvm.apache.org/docs/install/from_source.html#install-from-source)). 
Here are some key steps：

### Clone Repository and Configure LLVM

```
$ git clone --recursive https://github.com/apache/tvm tvm
$ cd tvm
$ mkdir build
$ cp cmake/config.cmake build
```

Edit `build/config.cmake` to specify the LLVM configuration.

```
set(USE_LLVM ON)
```

### Build TVM

```
$ cd build
$ cmake .. -G Ninja
$ ninja
```

### Install TVM package and dependencies

Enter your virtual environment and set the environment variable:

```
export TVM_HOME=/path/to/tvm
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
```

Install Python dependencies

```
$ pip3 install --user numpy decorator attrs typing-extensions psutil scipy tornado psutil 'xgboost>=1.1.0' cloudpickle
```
