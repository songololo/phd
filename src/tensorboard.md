# Tensorflow

## Install from source

Download bazel and tensorflow per official instructions. Run `./configure`:
```bash
/usr/local/bin/python3
/usr/local/lib/python3.7/site-packages
n
n
n
n
-mavx -mavx2 -mfma -msse4.2 (it may work to simply call defaults per -march-native in combination with --config=opt flag below...)
n
n
```

Then build with following flags:
(upper option may be more canonical, assume it invokes options listed in `./configure` instead of having to relist per below)

```bash
bazel build -c opt --config=opt //tensorflow/tools/pip_package:build_pip_package
vs.
bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-msse4.2 //tensorflow/tools/pip_package:build_pip_package
```
Then:
```bash
mkdir ./tmp
./bazel-bin/tensorflow/tools/pip_package/build_pip_package ./tensorflow_pkg
pip install ./tensorflow_pkg/tensorflow-2.2.0-cp37-cp37m-macosx_10_15_x86_64.whl
```


## Tensorboard

### install globally
```bash
pip3 install tensorflow
```

### launch
- do from console, not pycharm's console context:
```bash
cd /Users/gareth/dev/github/songololo/phd/src/temp_logs
tensorboard --logdir=./
```
- use the same log directory from `keras` `TensorBoard` callback, but be aware of different context:
```python
from keras.callbacks import TensorBoard

tensor_board = TensorBoard(log_dir='./tensorboard_logs/')

autoencoder.fit(
    ...
    tensor_board,
    ...
    )
```