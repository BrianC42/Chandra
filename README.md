# Chandra

1/4/2019
Set up Slack for Chandra and subscribed to this repository

How to set up your environment As of 7/6/2018
=============================================

1. Python - 3.6 (3.7 now offical but not supported by Tensorflow)
2. TensorFlow (with GPU support - not demonstrated yet)
3. CUDA Toolkit 9.0 (run time - select custom implementation)
	Add %CUDA_PATH%\bin to path
4. cuDNN (download, unzip and copy to %CUDA_PATH%)

	After steps 1 to 4 the NVIDIA CUDA extras (in the install directory) and the following python code should work
	>>> import tensorflow as tf
	>>> hello = tf.constant('Hello, TensorFlow!')
	>>> sess = tf.Session()
	>>> print(sess.run(hello))

5. Install Python packages
5.1 Numpy
5.2 Keras
5.3 matplotlib
5.4 pandas
5.5 quamdl
5.6 configparser

6. Create appdata\local\AI Projects\data.ini
[QUANDL]
key = xxxxxx
[DEVDATA]
dir = d:\brian\AI Projects\quandl
[LSTM]
dir = d:\brian\AI Projects\lstm
model = lstm.h5
plot = lstm.png
log = lstm_log
[TDAMERITRADE]
version = 0.1
source = XXXX
userid = xxxxxxx
password = xxxxxxx

7. install graphviz (www.graphviz.org) and set path to include bin directory

1/16/2018 Added call to plot_model Environment prerequisites install graphviz install pyplot install graphviz executable from https://graphviz.gitlab.io/_pages/Download/Download_windows.html Set path to include the graphviz executable
