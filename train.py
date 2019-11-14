import matplotlib.pyplot as plt
import matplotlib.colors as colors
import datetime
import sys
import os
import argparse
from net import *
from batch import *
from helpers import *

keyStr = 'data'
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# parsing argument
parser = argparse.ArgumentParser(description='GW code')
parser.add_argument('--train', dest='train_file', type=str, default='data/oneSecondTrainWhiten.h5',
                    help='the file of the training data')
parser.add_argument('--test', dest='test_file', type=str, default='data/oneSecondTestWhiten.h5',
                    help='the file of the testing data')
parser.add_argument('--name', dest='test_num', type=str, default='1',
                    help='test number')
parser.add_argument('--file', dest='file', type=bool, default=False,
                    help='whether cast output to a file')
parser.add_argument('--snr_step', dest='snr_step', type=int, default=5,
                    help='how many steps does each snr train')
parser.add_argument('--noise', dest='real_noise', type=bool, default=True,
                    help='whether add real noise or generated noise')
args = parser.parse_args()

train_path = args.train_file
test_path = args.test_file
test_num = args.test_num
snr_step = args.snr_step
real_noise = args.real_noise

f_train = h5py.File(train_path, "r")
f_test = h5py.File(test_path, "r")
NUM_DATA = f_train[keyStr].shape[0]
LENGTH = f_train[keyStr].shape[1]

tf.logging.set_verbosity(tf.logging.ERROR)
# check nan
if np.isnan(f_train[keyStr]).any():
    print("nan present in training data. Exiting...")
    sys.exit()

if args.file:
    stdoutOrigin = sys.stdout
    sys.stdout = open("testOut" + test_num + ".txt", "w")

input_data = tf.placeholder(tf.float32, [None, None, 1])
input_label = tf.placeholder(tf.int32, [None, 2])
trainable = tf.placeholder(tf.bool)

# loss function operations
predictions = WaveNet(input_data, trainable)
loss = tf.losses.mean_squared_error(input_label, predictions)

# train operation
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(learning_rate=0.001,
                                           global_step=global_step,
                                           decay_steps=NUM_DATA // 64,
                                           decay_rate=0.96,
                                           staircase=True)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
train_op = optimizer.minimize(
    loss=loss,
    global_step=global_step)

# initialization
init = tf.global_variables_initializer()
saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(init)
loss_hist = []
val_loss = []
# saver.restore(sess, "../model/False_R1noise.ckpt")

start = datetime.datetime.now()
batch_size = 64
rate = 0.001
# len(snr) is 50
low = [0.6, 0.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.1, 0.1]
snrs = [5.0, 4.0, 3.0, 2.0, 1.7, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7] + [lows for lows in low for i in range(3)]
num_epoch = int(snr_step * len(snrs))
for i in range(num_epoch):
    snr = snrs[i // snr_step]
    train_data, train_label = get_batch(f_train, batch_size, length=LENGTH, real_noise=real_noise, snr=snr)
    for j in range(len(train_data)):
        cur_data = train_data[j]
        cur_label = train_label[j]
        _, loss_val = sess.run([train_op, loss],
                               feed_dict={input_data: cur_data,
                                          input_label: cur_label,
                                          trainable: True})
        loss_hist.append(loss_val)
        if j % 10 == 0:
            print('loss: ' + str(loss_hist[-1]))

    val_data, val_label = get_val(f_test, batch_size, length=LENGTH, real_noise=real_noise, snr=snr)
    validation = sess.run(loss, feed_dict={input_data: val_data, input_label: val_label, trainable: False})
    val_loss.append(validation)
    print('iter num: ' + str(i) + ' snr: ' + str(snr) + ' loss: ' + str(loss_hist[-1]) + ' val_loss: ' + str(
        val_loss[-1]))

end = datetime.datetime.now()
print('time: ' + str(end - start))

# save model
save_path = saver.save(sess, '../model/' + str(real_noise) + '_R' + test_num + 'noise.ckpt')
print("Model saved in path: %s" % save_path)
step = NUM_DATA // batch_size
axis = np.arange(step - 1, len(loss_hist), step)
plt.figure()
plt.plot(loss_hist)
plt.scatter(axis, val_loss, c='red')
plt.legend(['train_loss', 'val_loss'], loc=1)
plt.title('loss history--total time: ' + str(end - start))
plt.xlabel('epochs')
plt.ylabel('loss')
plt.savefig(test_num + 'testLoss.png')

# testing
snrArr = np.linspace(5.0, 0.1, 50)
plot(sess, snrArr, f_test, test_num + '0.0-1.0s')
plot(sess, snrArr, f_test, test_num + '0.7-0.9s', shift=[int(LENGTH * 0.7), int(LENGTH * 0.9)])
plot(sess, snrArr, f_test, test_num + '0.5-1.0s', shift=[int(LENGTH * 0.5), int(LENGTH * 1.0)])

snrArr = np.array([5.0, 3.0, 2.0, 1.5, 1.0, 0.7, 0.5, 0.3, 0.2, 0.1])
timeStamps = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
gradual(sess, snrArr, f_test, 'test' + test_num + '-', timeStamps)

plot(sess, snrArr, f_test, test_num + 'zeroInput', shift=[0, 0])
