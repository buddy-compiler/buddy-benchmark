# Convolution Comparison

## Generate comparison build files

```
$ cd buddy-benchmark
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DBUILD_COMPARISONS=ON 
```

## Instructions for installing dependencies for comparison examples. 

 - Create and activate a separate virtual environment 

```
sudo apt install python3.8-venv
python3 -m venv /path/to/new/virtual/environment
source /path/to/new/virtual/environment/bin/activate
```

 - Install dependencies

```
pip install tensorflow
pip install opencv-python
pip install torch
pip install onnx
pip install onnxruntime
```

> **_NOTE:_**  In order to run boost_gil_conv2d.cpp, one has to install the Boost distribution and 
> its related dependencies. Instructions regarding Boost installation can be found here : https://www.boost.org/doc/libs/1_77_0/more/getting_started/unix-variants.html
> More information regarding external dependencies can be found here : https://github.com/boostorg/gil/blob/develop/CONTRIBUTING.md#install-dependencies

## Run Comparison

Please make sure TensorFlow, PyTorch, onnx, onnxruntime are installed in your environment.

- TensorFlow

```
$ cd buddy-benchmark/comparison/
$ python3 tf-conv2d.py
```

- PyTorch

```
$ cd buddy-benchmark/comparison/
$ python3 pytorch-conv2d.py
```

- ONNX Runtime

```
$ cd buddy-benchmark/comparison/
$ python3 gen-conv-models.py
$ python3 onnxruntime-conv2d.py
```

- Boost GIL

```
$ cd buddy-benchmark/build/
$ ninja boost_gil_conv2d
$ ./boost_gil_conv2d <input_image_path> <output_image_name>
```

Ex. `./boost_gil_conv2d ../../benchmarks/ImageProcessing/Images/gil_sample.png gil_output.png`
