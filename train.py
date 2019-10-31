import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys
import os
import argparse
from Net import *
from Batch import *

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# parsing argument
parser = argparse.ArgumentParser(description='GW code')
parser.add_argument('--train', dest='train_file', type=str, default='data/oneSecondTrain.h5',
                    help='the file of the training data')
parser.add_argument('--test', dest='test_file', type=str, default='data/oneSecondTest.h5',
                    help='the file of the testing data')
parser.add_argument('--test_num', dest='test_num', type=str, default='1',
                    help='test number')
parser.add_argument('--file', dest='file', type=bool, default=False,
                    help='whether cast output to a file')
parser.add_argument('--snr_step', dest='snr_step', type=int, default=10,
                    help='how many steps does each snr train')
parser.add_argument('--noise', dest='real_noise', type=bool, default=True,
                    help='whether add real noise or generated noise')
args = parser.parse_args()

train_path = args.train_file
test_path = args.test_file
test_num = args.test_num
snr_step = args.snr_step
real_noise = args.real_noise

if args.file:
    stdoutOrigin = sys.stdout
    sys.stdout = open("testOut" + test_num + ".txt", "w")

# check nan
f_train = h5py.File(train_path, "r")
f_test = h5py.File(test_path, "r")
NUM_DATA = f_train['data'].shape[0]
assert NUM_DATA == 9840
LENGTH = f_train['data'].shape[1]

tf.logging.set_verbosity(tf.logging.ERROR)
if np.isnan(f_train['data']).any():
    print("nan present in training data. Exiting...")
    sys.exit()

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
# len(snr) = 50
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
    # if i % 500 == 0:
    #   saver.save(sess, '../model/'+str(real_noise)+'_Rnoise.ckpt', global_step=i)

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


def plot(currSess, currSNR, f, fig, shift=None):
    def showplot(predict, name):
        testLabel = np.asarray(f['m1m2'])
        error1 = [abs(predict.T[0][n] - testLabel.T[0][n]) / testLabel.T[0][n] for n in range(len(testLabel))]
        error2 = [abs(predict.T[1][n] - testLabel.T[1][n]) / testLabel.T[1][n] for n in range(len(testLabel))]
        plt.figure(figsize=(18, 20))
        cm = plt.cm.get_cmap('seismic')
        plt.subplot(211)
        sc = plt.scatter(testLabel.T[0], testLabel.T[1], c=error1, vmin=0.0025, vmax=0.75,
                         cmap=cm, norm=colors.LogNorm(vmin=np.amin(error1), vmax=np.amax(error1)))
        plt.colorbar(sc)
        plt.xlabel('m1 mass')
        plt.ylabel('m2 mass')
        plt.title(name)
        plt.subplot(212)
        sc = plt.scatter(testLabel.T[0], testLabel.T[1], c=error2, vmin=0.0025, vmax=0.75,
                         cmap=cm, norm=colors.LogNorm(vmin=np.amin(error2), vmax=np.amax(error2)))
        plt.colorbar(sc)
        plt.xlabel('m1 mass')
        plt.ylabel('m2 mass')
        plt.title(name)
        plt.savefig(name + '.png')

    # testing without shift
    currStart = 0
    currEnd = LENGTH
    print("\n\nshift is: ", shift)
    noise = Noiser(LENGTH)
    m1s = []
    m2s = []
    for i in range(len(currSNR)):
        pred = []
        for idx in range(len(f['data'])):
            test_data = f['data'][idx][currStart:currEnd].reshape(1, currEnd - currStart)
            test_data = noise.add_shift(test_data)
            if shift is not None:
                test_data[0][:shift[0]] = 0
                test_data[0][shift[1]:] = 0
            if real_noise is False:
                test_data = noise.add_noise(input=test_data, SNR=currSNR[i])
            else:
                test_data = noise.add_real_noise(input=test_data, SNR=currSNR[i])
            test_data = test_data.reshape(1, currEnd - currStart, 1)
            test_label = f['m1m2'][idx].reshape(1, 2)

            pred.append(currSess.run(predictions,
                                     feed_dict={input_data: test_data, input_label: test_label, trainable: False})[0])
        pred = np.asarray(pred)
        test_label = np.asarray(f['m1m2'])
        m1 = np.mean(np.divide(abs(pred.T[0] - test_label.T[0]), test_label.T[0]))
        m2 = np.mean(np.divide(abs(pred.T[1] - test_label.T[1]), test_label.T[1]))
        m1s.append(m1)
        m2s.append(m2)
        print('SNR: ' + str(currSNR[i]) + ' -- m1: ' + "{0:.5%}".format(m1) + ' m2: ' + "{0:.5%}".format(m2))
        # if i % 51 == 0:
        #   showplot(pred,'testSNR--'+fig+str(snrs[i]))

    m1s = np.asarray(m1s)
    m2s = np.asarray(m2s)
    plt.figure()
    plt.plot(np.flip(currSNR, 0), np.flip(m1s * 100, 0))
    plt.plot(np.flip(currSNR, 0), np.flip(m2s * 100, 0))
    plt.legend(['m1', 'm2'], loc=1)
    plt.xlabel('SNR')
    plt.ylabel('Relative Error')
    plt.title('RE with SNR')
    plt.grid(True)
    plt.savefig(fig + '.png')


def gradual(currSess, currSnr, f, fig, times):
    noise = Noiser(LENGTH)
    for i in range(len(currSnr)):
        print("\n\nsnr is: ", currSnr[i])
        m1s = []
        m2s = []
        for stop in times:
            pred = []
            stop = int(stop * LENGTH)
            print("\n\nstop is: ", stop)
            for idx in range(len(f['data'])):
                test_data = f['data'][idx].reshape(1, LENGTH)
                test_data = noise.add_shift(test_data)
                test_data[0][stop:] = 0
                if real_noise is False:
                    test_data = noise.add_noise(input=test_data, SNR=currSnr[i])
                else:
                    test_data = noise.add_real_noise(input=test_data, SNR=currSnr[i])
                test_data = test_data.reshape(1, LENGTH, 1)
                test_label = f['m1m2'][idx].reshape(1, 2)
                pred.append(
                    currSess.run(predictions,
                                 feed_dict={input_data: test_data, input_label: test_label, trainable: False})[0])

            pred = np.asarray(pred)
            test_label = np.asarray(f['m1m2'])
            m1 = np.mean(np.divide(abs(pred.T[0] - test_label.T[0]), test_label.T[0]))
            m2 = np.mean(np.divide(abs(pred.T[1] - test_label.T[1]), test_label.T[1]))
            m1s.append(m1)
            m2s.append(m2)
            print('SNR: ' + str(currSnr[i]) + ' -- m1: ' + "{0:.5%}".format(m1) + ' m2: ' + "{0:.5%}".format(m2))
        m1s = np.asarray(m1s)
        m2s = np.asarray(m2s)
        plt.figure()
        plt.plot(times, m1s * 100)
        plt.plot(times, m2s * 100)
        plt.legend(['m1', 'm2'], loc=1)
        plt.xlabel('timeStamps in seconds')
        plt.ylabel('Relative Error')
        plt.title('RE with input length')
        plt.grid(True)
        plt.savefig(fig + str(currSnr[i]) + '.png')


snrArr = np.linspace(5.0, 0.1, 50)
plot(sess, snrArr, f_test, test_num + '0.0-1.0s')
plot(sess, snrArr, f_test, test_num + '0.7-0.9s', shift=[int(LENGTH * 0.7), int(LENGTH * 0.9)])
plot(sess, snrArr, f_test, test_num + '0.5-1.0s', shift=[int(LENGTH * 0.5), int(LENGTH * 1.0)])

snrArr = np.array([5.0, 3.0, 2.0, 1.5, 1.0, 0.7, 0.5, 0.3, 0.2, 0.1])
timeStamps = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
gradual(sess, snrArr, f_test, 'test' + test_num + '-', timeStamps)

plot(sess, snrArr, f_test, test_num + 'zeroInput', shift=[0, 0])
