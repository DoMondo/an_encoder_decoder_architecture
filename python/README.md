# Python version

The version in this directory allows to prototype new ideas quickly, analyze intermediate results or better understand the code. It contains the four algorithms that are compared in the paper:

 * PDRT 2
 * PDRT 32
 * PS DRT
 * MDD DRT


## Preparing the environment
Within your python environment, run:
```shell
pip install -r requirements.txt
```

## Running the python code
If you do not pass any flag, the program will compute the images contained in `../examples`. The results will be in `./out`.

```shell
python main.py
```
Alternatively, you can pass the flag `--path` to specify a different directory containing a set of images.
```shell
python main.py --path <CUSTOM_PATH>
```

## Running the Halide code from python
By using the `-a, --use-halide` flag, you can use the Halide/C++ implementation which gives much faster execution times. 
For this, you need to have generated the library within the C++/Halide project (`../cpp`). For this, first see [../cpp/README.md: Building the dynamic library](../cpp/README.md).

Once generated, you can simply call the python program like this:

```shell
python main.py -a
```

## Running a webcam example
For a quick demonstration, you can use the webcam as an input of the algorithm. Again, for this you'll need to have generated the dynamic library. Note that although the algorithm runs with an input of 1024x1024, the OpenCV `VideoCapture` is returning images of 640x480, that are then cropped and resized to 1024x1024.
```shell
python camera.py
```
