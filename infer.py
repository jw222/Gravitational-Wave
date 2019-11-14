import sys
import os
import argparse
from noiser import *
from net import *
from helpers import *

keyStr = 'data'
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# parsing argument
parser = argparse.ArgumentParser(description='GW code')
parser.add_argument('--model', dest='model_file', type=str, default='../model/model1W/True_R1Wnoise.ckpt',
                    help='the file of the model')
parser.add_argument('--infer', dest='infer_file', type=str, default='data/oneSecondTestWhiten.h5',
                    help='the file of the testing data')
parser.add_argument('--name', dest='name', type=str, default='1on1',
                    help='infer on model')
parser.add_argument('--file', dest='file', type=bool, default=False,
                    help='whether cast output to a file')
parser.add_argument('--noise', dest='real_noise', type=bool, default=True,
                    help='whether add real noise or generated noise')
args = parser.parse_args()

model_path = args.model_file
infer_path = args.infer_file
test_num = args.name
real_noise = args.real_noise

f_infer = h5py.File(infer_path, "r")
LENGTH = f_infer[keyStr].shape[1]
# check nan
if np.isnan(f_infer[keyStr]).any():
    print("nan present in training data. Exiting...")
    sys.exit()

if args.file:
    stdoutOrigin = sys.stdout
    sys.stdout = open("testOut" + test_num + ".txt", "w")

tf.logging.set_verbosity(tf.logging.ERROR)

input_data = tf.placeholder(tf.float32, [None, None, 1])
input_label = tf.placeholder(tf.int32, [None, 2])
trainable = tf.placeholder(tf.bool)

# loss function operations
predictions = WaveNet(input_data, trainable)
loss = tf.losses.mean_squared_error(input_label, predictions)

# initialization
init = tf.global_variables_initializer()
saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(init)
loss_hist = []
val_loss = []
saver.restore(sess, model_path)

snrArr = np.linspace(5.0, 0.1, 50)
plot(sess, snrArr, f_infer, 'infer' + test_num + '0.0-1.0s')
plot(sess, snrArr, f_infer, 'infer' + test_num + '0.7-0.9s', shift=[int(LENGTH * 0.7), int(LENGTH * 0.9)])
plot(sess, snrArr, f_infer, 'infer' + test_num + '0.5-1.0s', shift=[int(LENGTH * 0.5), int(LENGTH * 1.0)])

snrArr = np.array([5.0, 3.0, 2.0, 1.5, 1.0, 0.7, 0.5, 0.3, 0.2, 0.1])
timeStamps = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
gradual(sess, snrArr, f_infer, 'infer' + test_num + '-', timeStamps)

plot(sess, snrArr, f_infer, test_num + 'zeroInput', shift=[0, 0])
window(sess, snrArr, f_infer, 'window' + test_num + '-', 1024)
