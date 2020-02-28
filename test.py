import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.interpolate import interp1d
import matplotlib.mlab as mlab
import readligo as rl
import argparse
import pickle
from net import *
from batch import *

parser = argparse.ArgumentParser(description='GW code')
parser.add_argument('--model', dest='model_path', type=str, default='../model/new1Classifier.ckpt',
                    help='model for testing')

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
args = parser.parse_args()
model_path = args.model_path
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
window = 8192*4
crop = 8192
dt = 1. / 8192.
crops = [(8192*2, 4096), (8192*4, 8192+4096), (8192*4, 8192), (8192*8, 8192*2), (8192*8, 8192*3)]
# 245760 for 32 seconds
event_time = 122880
signals = ['150914.hdf5', '151012.hdf5', '170104.hdf5', '170608.hdf5']
for window, crop in crops:
    for file in signals:
        exist = []
        confidence = []
        x_axis = []
        file_path = '../' + file
        strain, time, chan_dict = rl.loaddata(file_path, 'H1')
        # calculate psd
        strain = [strain[i] for i in range(0, len(strain), 2)]
        with open("freqs-"+file[:6], 'rb') as fh:
            freqs = pickle.load(fh)
        with open("pxxh-"+file[:6], 'rb') as fh:
            Pxx_H1 = pickle.load(fh)
            psd_H1 = interp1d(freqs, Pxx_H1)

        # sliding window method
        for start in range(0, len(strain) - window, 1024):
            curr_strain = strain[start:start + window]
            curr_strain = whiten(curr_strain, psd_H1, dt)
            '''
            limit = np.amax(curr_strain[int(len(curr_strain) / 2) - 500:int(len(curr_strain) / 2) + 500]) * ratio
            crop = step
            for i in range(step, int(len(curr_strain) / 2), step):
                if max(np.amax(curr_strain[i - step:i]), -np.amin(curr_strain[i - step:i])) < limit:
                    break
                crop = i
            '''
            curr_strain = curr_strain[crop:-crop]
            curr_strain = (curr_strain - np.mean(curr_strain)) / np.std(curr_strain)
            curr_strain = np.array(curr_strain).reshape(1, window - 2 * crop, 1)
            result = sess.run(predictions, feed_dict={input_data: curr_strain, trainable: False})[0]
            pred = True if np.argmax(result) == 1 else False
            exist.append(pred)
            confidence.append(abs(result[1] - result[0]))
            x_axis.append((start + window / 2 - event_time) / fs)
            print(f"start time: {(start - event_time + crop) / fs} ----- end time: {(start + window - event_time - crop) / fs}")
            print(f"existence of signal: {pred}, result array: {result}")

        # plot result graph
        plt.figure()
        plt.plot(x_axis, exist)
        plt.xlabel('center time')
        plt.ylabel('signal existence')
        plt.title('signal existence with time')
        plt.savefig(file[:6] + '-' + str(window/8192) + '-' + str(crop/8192) + '-existence.png')

        # plot confidence graph
        plt.figure()
        plt.plot(x_axis, confidence)
        plt.xlabel('center time')
        plt.ylabel('confidence')
        plt.title('confidence with time')
        plt.grid(True)
        plt.savefig(file[:6] + '-' + str(window/8192) + '-' + str(crop/8192) + '-confidence.png')
