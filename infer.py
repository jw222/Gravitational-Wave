import sys
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from noiser import *
from net import *

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


def getError(currSess, snr, f, length, shift=None):
    noise = Noiser(length)
    pred = []
    for j in range(len(f[keyStr])):
        test_data = f[keyStr][j].reshape(1, length)
        test_data = noise.add_shift(test_data)
        if shift is not None:
            test_data[0][:shift[0]] = 0
            test_data[0][shift[1]:] = 0
            # test_data[0] = test_data[0][shift[0]:shift[1]]
        if real_noise is False:
            test_data = noise.add_noise(input=test_data, SNR=snr)
        else:
            test_data = noise.add_real_noise(input=test_data, SNR=snr)
        test_data = test_data.reshape(1, length, 1)
        test_label = f['m1m2'][j].reshape(1, 2)
        pred.append(
            currSess.run(predictions,
                         feed_dict={input_data: test_data, input_label: test_label, trainable: False})[0])

    pred = np.asarray(pred)
    test_label = np.asarray(f['m1m2'])
    m1 = np.mean(np.divide(abs(pred.T[0] - test_label.T[0]), test_label.T[0]))
    m2 = np.mean(np.divide(abs(pred.T[1] - test_label.T[1]), test_label.T[1]))
    return m1, m2, pred


def triPlot(predict, name, f):
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


def plot(currSess, snrs, f, fig, shift=None):
    # testing without shift
    length = len(f[keyStr][0])
    print("\nshift is: ", shift)
    m1s = []
    m2s = []
    for snr in snrs:
        m1, m2, pred = getError(currSess, snr, f, length, shift)
        m1s.append(m1)
        m2s.append(m2)
        print('SNR: ' + str(snr) + ' -- m1: ' + "{0:.5%}".format(m1) + ' m2: ' + "{0:.5%}".format(m2))
        if 0.79 < snr < 0.81:
            triPlot(pred, fig + 'triSNR-' + str(snr), f)

    m1s = np.asarray(m1s)
    m2s = np.asarray(m2s)
    plt.figure()
    plt.plot(np.flip(snrs, 0), np.flip(m1s * 100, 0))
    plt.plot(np.flip(snrs, 0), np.flip(m2s * 100, 0))
    plt.legend(['m1', 'm2'], loc=1)
    plt.xlabel('SNR')
    plt.ylabel('Relative Error')
    plt.title('RE with SNR')
    plt.grid(True)
    plt.savefig(fig + '.png')


def gradual(currSess, snrs, f, fig, times):
    length = len(f[keyStr][0])
    num_secs = length // 8192
    for snr in snrs:
        print("\n\nsnr is: ", snr)
        m1s = []
        m2s = []
        for stop in times:
            shift = [0, int(stop * length)]
            m1, m2, _ = getError(currSess, snr, f, length, shift)
            m1s.append(m1)
            m2s.append(m2)
            print('stop: ' + str(shift) + ' -- m1: ' + "{0:.5%}".format(m1) + ' m2: ' + "{0:.5%}".format(m2))
        m1s = np.asarray(m1s)
        m2s = np.asarray(m2s)

        plt.figure()
        plt.plot(times * num_secs, m1s * 100)
        plt.plot(times * num_secs, m2s * 100)
        plt.legend(['m1', 'm2'], loc=1)
        plt.xlabel('timeStamps in seconds')
        plt.ylabel('Relative Error')
        plt.title('RE with end time')
        plt.grid(True)
        plt.savefig(fig + str(snr) + 'gradual.png')


def window(currSess, snrs, f, fig, step):
    length = len(f[keyStr][0])
    num_shift = (length - 8192) // step + 1
    for snr in snrs:
        print("\n\nsnr is: ", snr)
        m1s = []
        m2s = []
        for i in range(num_shift):
            shift = [i * step, i * step + 8192]
            print("\nwindow is: ", shift)
            m1, m2, _ = getError(currSess, snr, f, length, shift)
            m1s.append(m1)
            m2s.append(m2)
            print('window: ' + str(shift) + ' -- m1: ' + "{0:.5%}".format(m1) + ' m2: ' + "{0:.5%}".format(m2))
        m1s = np.asarray(m1s)
        m2s = np.asarray(m2s)

        x_axis = np.arange(0, length - 8192 + 1, step, dtype=np.float64)
        x_axis /= 8192.
        plt.figure()
        plt.plot(x_axis, m1s * 100)
        plt.plot(x_axis, m2s * 100)
        plt.legend(['m1', 'm2'], loc=1)
        plt.xlabel('start of window')
        plt.ylabel('Relative Error')
        plt.title('RE with window')
        plt.grid(True)
        plt.savefig(fig + str(snr) + 'window.png')


snrArr = np.linspace(5.0, 0.1, 50)
window(sess, snrArr, f_infer, 'window' + test_num + '-', 8192)
plot(sess, snrArr, f_infer, 'infer' + test_num + '0.0-1.0s')
plot(sess, snrArr, f_infer, 'infer' + test_num + '0.7-0.9s', shift=[int(LENGTH * 0.7), int(LENGTH * 0.9)])
plot(sess, snrArr, f_infer, 'infer' + test_num + '0.5-1.0s', shift=[int(LENGTH * 0.5), int(LENGTH * 1.0)])

snrArr = np.array([5.0, 3.0, 2.0, 1.5, 1.0, 0.7, 0.5, 0.3, 0.2, 0.1])
timeStamps = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
gradual(sess, snrArr, f_infer, 'infer' + test_num + '-', timeStamps)

plot(sess, snrArr, f_infer, test_num + 'zeroInput', shift=[0, 0])
