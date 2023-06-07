# Equivalence Testing

Equivalence testing is to evaluate the similarity of image or audio results. 
Before conducting testing, please follow the instructions below to configure the virtual environment.

## Configure Virtual Environment

### Create Virtual Environment

```
$ python -m venv EquivalenceTest.venv
```

### Managing Packages with pip

```
$ pip install pillow
$ pip install numpy
$ pip install sys
```

## Run Equivalence Tests

### Image Equivalence Testing

```
$ source EquivalenceTest.venv/bin/activate
$ python compareImg.py <PATH/TO/Image1> <PATH/TO/Image2>
$ deactivate
```

Example: 
```
$ python compareImg.py $benchmark_bin_dir/BuddyResize_length_500_500_BI.png $benchmark_bin_dir/OpenCVResize_length_500_500_BI.png
```


### Audio Equivalence Testing

```
```