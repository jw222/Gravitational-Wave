import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.interpolate import interp1d
import matplotlib.mlab as mlab
import readligo as rl
from net import *
from batch import *


def whiten(strainW, interp_psd, dt):
    Nt = len(strainW)
    freqsW = np.fft.rfftfreq(Nt, dt)

    # whitening: transform to freq domain, divide by asd, then transform back,
    # taking care to get normalization right.
    hf = np.fft.rfft(strainW)
    white_hf = hf / (np.sqrt(interp_psd(freqsW) / dt / 2.))
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht


# set model parameters
model_path = '../model/lessNClassifier.ckpt'
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

# sliding window data
fs = 8192
NFFT = 1 * fs
window = 8192 * 4
dt = 1. / 8192.
crops = [(50, 1.5), (50, 1.6), (50, 1.7), (50, 1.8), (50, 1.9), (50, 2.0),
         (100, 1.5), (100, 1.6), (100, 1.7), (100, 1.8), (100, 1.9), (100, 2.0),
         (200, 1.5), (200, 1.6), (200, 1.7), (200, 1.8), (200, 1.9), (200, 2.0),]
# 245760 is the time when the event happens: 245760 / 2 = 122880
event_time = 122880
signals = ['150914.hdf5', '151012.hdf5', '151226.hdf5']
for step, ratio in crops:
    for file in signals:
        exist = []
        confidence = []
        x_axis = []
        file_path = 'data/' + file
        strain, time, chan_dict = rl.loaddata(file_path, 'H1')
        # calculate psd
        strain = [strain[i] for i in range(0, len(strain), 2)]
        Pxx_H1, freqs = mlab.psd(strain, Fs=fs, NFFT=NFFT)
        psd_H1 = interp1d(freqs, Pxx_H1)

        # sliding window method
        for start in range(0, len(strain) - window, 1024):
            curr_strain = strain[start:start + window]
            curr_strain = whiten(curr_strain, psd_H1, dt)
            curr_strain = (curr_strain - np.mean(curr_strain)) / np.std(curr_strain)
            limit = np.amax(test[int(len(curr_strain) / 2) - 500:int(len(curr_strain) / 2) + 500]) * ratio
            crop = step
            for i in range(step, int(len(curr_strain) / 2), step):
                if max(np.amax(curr_strain[i - step:i]), -np.amin(curr_strain[i - step:i])) < limit:
                    break
                crop = i
            curr_strain = curr_strain[crop:-crop]
            curr_strain = np.array(curr_strain).reshape(1, window - 2 * crop, 1)
            result = sess.run(predictions, feed_dict={input_data: curr_strain, trainable: False})[0]
            pred = True if np.argmax(result) == 1 else False
            exist.append(pred)
            confidence.append(abs(result[1] - result[0]))
            x_axis.append((start + window / 2 - event_time) / fs)
            print(f"start time: {(start - event_time) / fs} ----- end time: {(start + window - event_time) / fs}")
            print(f"existence of signal: {pred}, result array: {result}")

        # plot result graph
        plt.figure()
        plt.plot(x_axis, exist)
        plt.xlabel('center time')
        plt.ylabel('signal existence')
        plt.title('signal existence with time')
        plt.savefig(file[:-5] + '-' + str(step) + '-' + str(ratio) + '-existence.png')

        # plot confidence graph
        plt.figure()
        plt.plot(x_axis, confidence)
        plt.xlabel('center time')
        plt.ylabel('confidence')
        plt.title('confidence with time')
        plt.grid(True)
        plt.savefig(file[:-5] + '-' + str(step) + '-' + str(ratio) + '-confidence.png')
