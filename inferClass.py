import matplotlib.pyplot as plt
import datetime
import sys
import os
import argparse
from net import *
from batch import *

keyStr = 'data'
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# parsing argument
parser = argparse.ArgumentParser(description='GW code')
parser.add_argument('--model', dest='model', type=str, default='../model/1Classifier.ckpt',
                    help='the file of the model')
parser.add_argument('--test', dest='test_file', type=str, default='data/oneSecondTestWhiten.h5',
                    help='the file of the testing data')
parser.add_argument('--name', dest='test_num', type=str, default='1',
                    help='test number')
parser.add_argument('--file', dest='file', type=bool, default=False,
                    help='whether cast output to a file')
parser.add_argument('--noise', dest='real_noise', type=bool, default=True,
                    help='whether add real noise or generated noise')
args = parser.parse_args()

model_path = args.model
test_path = args.test_file
test_num = args.test_num
real_noise = args.real_noise

f_test = h5py.File(test_path, "r")
NUM_DATA = f_test[keyStr].shape[0]
LENGTH = f_test[keyStr].shape[1]

tf.logging.set_verbosity(tf.logging.ERROR)
# check nan
if np.isnan(f_test[keyStr]).any():
    print("nan present in training data. Exiting...")
    sys.exit()

if args.file:
    stdoutOrigin = sys.stdout
    sys.stdout = open("testOut" + test_num + ".txt", "w")

input_data = tf.placeholder(tf.float32, [None, None, 1])
input_label = tf.placeholder(tf.int32, [None, 2])
trainable = tf.placeholder(tf.bool)

# loss function operations
predictions = Classifier(input_data, trainable)
loss = tf.losses.sigmoid_cross_entropy(input_label, predictions)

# train operation
global_step = tf.Variable(0, trainable=False)
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
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
saver.restore(sess, model_path)


def compute_accuracy(currSess, currSNR, f, length, shift):
    pred = []
    labels = []
    new_length = shift[1] - shift[0]
    noise = Noiser(new_length)
    for j in range(len(f[keyStr])):
        temp_test = f[keyStr][j].reshape(1, length)
        temp_test = noise.add_shift(temp_test)
        test_data = np.array(temp_test[0][shift[0]:shift[1]]).reshape(1, new_length)
        if real_noise is False:
            test_data = noise.add_noise(input=test_data, SNR=currSNR)
        else:
            test_data = noise.add_real_noise(input=test_data, SNR=currSNR)
        test_data = test_data.reshape(1, new_length, 1)
        labels.append(1)
        curr = currSess.run(predictions, feed_dict={input_data: test_data, trainable: False})[0]
        pred.append(np.argmax(curr))

    # use same number of noise as signal
    for _ in range(len(f[keyStr])):
        test_data = np.zeros(new_length).reshape(1, new_length)
        if real_noise is False:
            test_data = noise.add_noise(input=test_data, SNR=currSNR)
        else:
            test_data = noise.add_real_noise(input=test_data, SNR=currSNR)
        test_data = test_data.reshape(1, new_length, 1)
        labels.append(0)
        curr = currSess.run(predictions, feed_dict={input_data: test_data, trainable: False})[0]
        pred.append(np.argmax(curr))

    pred = np.asarray(pred)
    accuracies = np.mean(pred == labels)
    totalPos = np.sum(labels[i] == 1 for i in range(len(labels)))
    totalNeg = np.sum(labels[i] == 0 for i in range(len(labels)))
    tp = np.sum([pred[i] == labels[i] and labels[i] == 1 for i in range(len(labels))]) / totalPos
    fp = np.sum([pred[i] != labels[i] and labels[i] == 0 for i in range(len(labels))]) / totalNeg
    return accuracies, tp, fp


snrArr = np.linspace(5.0, 0.1, 50)
acc = []
sen = []
fa = []
for snr in snrArr:
    accuracy, sensitivity, false_alarm = compute_accuracy(sess, snr, f_test, length=LENGTH, shift=[0, LENGTH])
    print(f"Entire input with snr: {snr}\n accuracy: {accuracy}, sensitivity: {sensitivity}, false alarm rate: {false_alarm}")
    acc.append(accuracy)
    sen.append(sensitivity)
    fa.append(false_alarm)
plt.figure()
plt.plot(np.flip(snrArr, 0), np.flip(acc, 0))
plt.plot(np.flip(snrArr, 0), np.flip(sen, 0))
plt.plot(np.flip(snrArr, 0), np.flip(fa, 0))
plt.legend(['accuracy', 'sensitivity', 'false_alarm'])
plt.xlabel('SNR')
plt.ylabel('Accuracy')
plt.title('Accuracy with SNR')
plt.grid(True)
plt.savefig(test_num + 'OverallAccuracy.png')

snrArr = np.array([5.0, 3.0, 2.0, 1.5, 1.0, 0.7, 0.5, 0.3, 0.2, 0.1])
timeStamps = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
test_files = ['data/oneSecondTestWhiten.h5',
              'data/twoSecondTestWhiten.h5',
              'data/fourSecondTestWhiten.h5',
              'data/sixSecondTestWhiten.h5',
              'data/eightSecondTestWhiten.h5']
num_secs = LENGTH // 8192
for i in len(test_files):
    f_test = h5py.File(file_name, "r")
    for snr in snrArr:
        acc = []
        sen = []
        fa = []
        print(f"Current snr is: {snr}")
        for stop in timeStamps:
            currShift = [0, int(stop * LENGTH)]
            accuracy, sensitivity, false_alarm = compute_accuracy(sess, snr, f_test, length=LENGTH, shift=currShift)
            print(f"Entire input with stop: {currShift}\n accuracy: {accuracy}, sensitivity: {sensitivity}, false alarm rate: {false_alarm}")
            acc.append(accuracy)
            sen.append(sensitivity)
            fa.append(false_alarm)
        plt.figure()
        plt.plot(timeStamps * num_secs, acc)
        plt.plot(timeStamps * num_secs, sen)
        plt.plot(timeStamps * num_secs, fa)
        plt.legend(['accuracy', 'sensitivity', 'false_alarm'])
        plt.xlabel('timeStamps in seconds')
        plt.ylabel('Accuracy')
        plt.title('Accuracy with end time')
        plt.grid(True)
        plt.savefig(test_num + 'lengthIDX(' + str(i) + ')' + str(snr) + '-GradualClassify.png')
