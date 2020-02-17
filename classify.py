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
parser.add_argument('--train', dest='train_file', type=str, default='data/fourSecondTrainWhitenH.h5',
                    help='the file of the training data')
parser.add_argument('--test', dest='test_file', type=str, default='data/fourSecondTestWhitenH.h5',
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

start = datetime.datetime.now()
batch_size = 64
rate = 0.001
# len(snr) is 50
low = [0.6, 0.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.1, 0.1]
snrs = [5.0, 4.0, 3.0, 2.0, 1.7, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7] + [lows for lows in low for i in range(3)]
num_epoch = int(snr_step * len(snrs))
for i in range(num_epoch):
    snr = snrs[i // snr_step]
    train_data, train_label = get_classify_batch(f_train, batch_size, length=LENGTH, real_noise=real_noise, snr=snr)
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

    val_data, val_label = get_classify_val(f_test, batch_size, length=LENGTH, real_noise=real_noise, snr=snr)
    validation = sess.run(loss, feed_dict={input_data: val_data, input_label: val_label, trainable: False})
    val_loss.append(validation)
    print('iter num: ' + str(i) + ' snr: ' + str(snr) + ' loss: ' + str(loss_hist[-1]) + ' val_loss: ' + str(
        val_loss[-1]))

end = datetime.datetime.now()
print('time: ' + str(end - start))

# save model
save_path = saver.save(sess, '../model/' + test_num + 'Classifier.ckpt')
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


def compute_accuracy(currSess, currSNR, f, length, shift):
    pred = []
    labels = []
    new_length = shift[1] - shift[0]
    noise = Noiser(new_length)
    for j in range(len(f[keyStr])//2):
        temp_test = f[keyStr][j*2].reshape(1, length)
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
    for _ in range(len(f[keyStr])//2):
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
              #'data/twoSecondTestWhiten.h5',
              'data/fourSecondTestWhitenH.h5']
for i in range(len(test_files)):
    f_test = h5py.File(test_files[i], "r")
    LENGTH = f_test[keyStr].shape[1]
    num_secs = LENGTH // 8192
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
        plt.savefig(test_num + 'lengthIDX(' + str(2**i) + ')' + str(snr) + '-GradualClassify.png')
