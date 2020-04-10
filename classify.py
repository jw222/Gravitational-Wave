import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import tensorflow as tf
import numpy as np
import readligo as rl
import sys
import os
import argparse
import pickle
import h5py
from scipy.interpolate import interp1d
from scipy.signal import decimate
from scipy.signal import find_peaks
from net import *
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def whiten(strainW, interp_psd, dt=1./8192.):
    Nt = len(strainW)
    freqsW = np.fft.rfftfreq(Nt, dt)
    hf = np.fft.rfft(strainW)
    white_hf = hf / (np.sqrt(interp_psd(freqsW)/dt/2.))
    white_ht = np.fft.irfft(white_hf, n=Nt)
    white_ht[:100] = 0
    white_ht[-100:] = 0
    return white_ht


def parseTrainInput():
    # parsing argument
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--train', dest='train_file', type=str, default='data/TrainEOB_q-1-10-0.02_Flat.h5',
                        help='the file of the training data')
    parser.add_argument('--test', dest='test_file', type=str, default='data/TestEOB_q-1-10-0.02_Flat.h5',
                        help='the file of the testing data')
    parser.add_argument('--event', dest='event', type=str, default='150914',
                        help='the event we want to train on')
    parser.add_argument('--channel', dest='channel', type=str, default='H',
                        help='the psd that calculated from that event')
    parser.add_argument('--freq', dest='freq', type=int, default=8,
                        help='frequency used')
    parser.add_argument('--num_filters', dest='num_filters', type=int, default=256,
                        help='number of filters used in residual blocks')
    parser.add_argument('--num_residuals', dest='num_residuals', type=int, default=6,
                        help='number of residual blocks')
    parser.add_argument('--blank_ratio', dest='blank_ratio', type=float, default=0.5,
                        help='ratio of blank signal used for training and testing')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=1e-4,
                        help='learning rate for training')
    parser.add_argument('--batch', dest='batch_size', type=int, default=64,
                        help='batch size for training')
    parser.add_argument('--epoch', dest='epoch', type=int, default=10000,
                        help='epochs fro training')
    parser.add_argument('--output', dest='output', type=str, default='test',
                        help='output file name')
    parser.add_argument('--save_file', dest='save_file', type=bool, default=False,
                        help='whether cast output to a file')
    return parser.parse_args()


class Accuracy(tf.keras.metrics.Metric):
    def __init__(self, name='accuracy', threshold=0.5, num_points=4096):
        super(Accuracy, self).__init__(name=name)
        self.accuracy = self.add_weight(name='acc', initializer='zeros', dtype=tf.float32)
        self.counter = self.add_weight(name='count', initializer='zeros', dtype=tf.float32)
        self.threshold = self.add_weight(name='thres', initializer='zeros', dtype=tf.float32)
        self.num_points = self.add_weight(name='points', initializer='zeros', dtype=tf.int32)
        self.threshold.assign(threshold)
        self.num_points.assign(num_points)

    def _category(self, pred):
        pred = tf.cast(tf.map_fn(lambda x: x > self.threshold, pred, dtype=tf.bool), tf.int32)
        return tf.reduce_sum(pred) > self.num_points

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(tf.map_fn(self._category, y_pred, dtype=tf.bool), tf.int32)
        y_true = tf.cast(tf.reduce_max(y_true, axis=1), tf.int32)
        values = tf.reduce_mean(tf.cast(y_true == y_pred, tf.float32))
        new_acc = (self.counter * self.accuracy + values) / (self.counter + 1)
        self.counter.assign_add(1.0)
        self.accuracy.assign(new_acc)

    def result(self):
        return self.accuracy

    def reset_states(self):
        self.accuracy.assign(0.)
        self.counter.assign(0.)


class FalseAlarm(tf.keras.metrics.Metric):
    def __init__(self, name='false_alarm', threshold=0.5, num_points=4096):
        super(FalseAlarm, self).__init__(name=name)
        self.fp = self.add_weight(name='fp', initializer='zeros', dtype=tf.float32)
        self.totalN = self.add_weight(name='totalN', initializer='zeros', dtype=tf.float32)
        self.threshold = self.add_weight(name='thres', initializer='zeros', dtype=tf.float32)
        self.num_points = self.add_weight(name='points', initializer='zeros', dtype=tf.int32)
        self.threshold.assign(threshold)
        self.num_points.assign(num_points)

    def _category(self, pred):
        pred = tf.cast(tf.map_fn(lambda x: x > self.threshold, pred, dtype=tf.bool), tf.int32)
        return tf.reduce_sum(pred) > self.num_points

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(tf.map_fn(self._category, y_pred, dtype=tf.bool), tf.int32)
        y_true = tf.cast(tf.reduce_max(y_true, axis=1), tf.int32)
        fp = tf.cast(tf.reduce_sum(tf.cast(
            tf.map_fn(lambda x: tf.logical_and(x[0] != x[1], x[1] == 0),
                      tf.stack([y_pred, y_true], axis=1), dtype=tf.bool), tf.int32)), tf.float32)
        self.fp.assign_add(fp)
        totalN = tf.cast(tf.reduce_sum(tf.cast(
            tf.map_fn(lambda x: x == 0, y_true, dtype=tf.bool), tf.int32)), tf.float32)
        self.totalN.assign_add(totalN)

    def result(self):
        return self.fp / (self.totalN + 1e-5)

    def reset_states(self):
        self.fp.assign(0.)
        self.totalN.assign(0.)


class Sensitivity(tf.keras.metrics.Metric):
    def __init__(self, name='sensitivity', threshold=0.5, num_points=4096):
        super(Sensitivity, self).__init__(name=name)
        self.tp = self.add_weight(name='tp', initializer='zeros', dtype=tf.float32)
        self.totalP = self.add_weight(name='totalP', initializer='zeros', dtype=tf.float32)
        self.threshold = self.add_weight(name='thres', initializer='zeros', dtype=tf.float32)
        self.num_points = self.add_weight(name='points', initializer='zeros', dtype=tf.int32)
        self.threshold.assign(threshold)
        self.num_points.assign(num_points)

    def _category(self, pred):
        pred = tf.cast(tf.map_fn(lambda x: x > self.threshold, pred, dtype=tf.bool), tf.int32)
        return tf.reduce_sum(pred) > self.num_points

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(tf.map_fn(self._category, y_pred, dtype=tf.bool), tf.int32)
        y_true = tf.cast(tf.reduce_max(y_true, axis=1), tf.int32)
        tp = tf.cast(tf.reduce_sum(tf.cast(
            tf.map_fn(lambda x: tf.logical_and(x[0] == x[1], x[1] == 1),
                      tf.stack([y_pred, y_true], axis=1), dtype=tf.bool), tf.int32)), tf.float32)
        self.tp.assign_add(tp)
        totalP = tf.cast(tf.reduce_sum(tf.cast(
            tf.map_fn(lambda x: x == 1, y_true, dtype=tf.bool), tf.int32)), tf.float32)
        self.totalP.assign_add(totalP)

    def result(self):
        return self.tp / (self.totalP + 1e-5)

    def reset_states(self):
        self.tp.assign(0.)
        self.totalP.assign(0.)


def generator(file_name, file_prefix, ratio, pSNR=None):
    file_name = file_name.decode('utf-8')
    file_prefix = file_prefix.decode('utf-8')
    with open('psd/' + file_prefix + 'freqs', 'rb') as fh:
        freqs = pickle.load(fh)
    with open('psd/' + file_prefix + 'pxx', 'rb') as fh:
        pxx = pickle.load(fh)
        interp_psd = interp1d(freqs, pxx)

    f_noise = h5py.File('data/' + file_prefix + 'Noise.h5', 'r')
    f_train = h5py.File(file_name, 'r')
    for waveform in f_train['Input'][:2]:
        waveform = decimate(waveform, 2)[-8192:]
        if np.random.uniform() < ratio:
            # blank waveform
            whitened = np.zeros(8192)
            target = np.zeros(8192)
        else:
            whitened = whiten(waveform, interp_psd)
            shiftInt = np.random.randint(-4000, 1000)
            whitened = np.roll(whitened, shiftInt)
            if shiftInt >= 0:
                whitened[:shiftInt] = 0
            else:
                whitened[shiftInt:] = 0
            whitened /= np.amax(np.absolute(whitened))
            merger = np.argmax(whitened)
            target = np.zeros(8192)
            target[max(shiftInt, 0):merger] = 1.0
        noiseInt = np.random.randint(20, len(f_noise['noise']) - 8192)
        noise = f_noise['noise'][noiseInt:noiseInt+8192]
        if pSNR is None:
            pSNR = np.random.uniform(0.5, 1.5)
        noise /= np.std(noise)
        output = whitened * pSNR + noise
        output /= np.std(output)
        output = output.reshape(8192, 1)
        yield output, target


def fileCheck(fileStr):
    if os.path.exists('psd/' + fileStr + 'pxx') and \
            os.path.exists('psd/' + fileStr + 'freqs') and \
            os.path.exists('data/' + fileStr + 'Noise.h5'):
        return

    strain, time, chan_dict = rl.loaddata('raw/' + fileStr[:-1] + 'prev.hdf5', fileStr[-2] + '1')
    strain = decimate(strain, 2)
    fs = 8192
    pxx, freqs = mlab.psd(strain, Fs=fs, NFFT=fs)
    with open('psd/' + fileStr + 'freqs', 'wb') as fn:
        pickle.dump(freqs, fn)
    with open('psd/' + fileStr + 'pxx', 'wb') as fn:
        pickle.dump(pxx, fn)
    interp_psd = interp1d(freqs, pxx)
    whitened = whiten(strain, interp_psd)
    whitened = whitened[100000:-100000]
    with h5py.File('data/' + fileStr + 'Noise.h5', 'w') as fn:
        fn['noise'] = whitened


def testReal(trained_model, event, crop, step, output):
    # strain data preprocessing
    strain, time, chan_dict = rl.loaddata('raw/' + event + '-4096.hdf5', event[-1] + '1')
    strain = decimate(strain, 2)
    fs = 8192
    pxx, freqs = mlab.psd(strain, Fs=fs, NFFT=fs)
    interp_psd = interp1d(freqs, pxx)
    whitened = whiten(strain, interp_psd)
    whitened /= np.std(whitened)
    whitened = whitened[crop:-crop]

    # prepare prediction dataset
    x_predict = []
    for start in range(0, len(whitened)-8192, step):
        x_predict.append(whitened[start:start+8192])
    x_predict = np.asarray(x_predict)
    x_predict = np.expand_dims(x_predict, axis=-1)

    # get prediction
    y_temp = trained_model.predict(x_predict)
    y_predict = np.zeros(len(whitened))
    for start in range(0, len(whitened)-8192, step):
        y_predict[start:start+8192] += y_temp[start//step] * (step/8192.)
    y_predict = np.convolve(y_predict, np.ones(100)) / 100
    plt.figure(figsize=(30, 30))
    counter = 1
    for threshold in np.arange(0.1, 1.0, 0.1):
        peaks, _ = find_peaks(y_predict, height=threshold, distance=8192)
        plt.subplot('33'+str(counter))
        plt.plot(y_predict)
        plt.plot(peaks, y_predict[peaks], 'x')
        plt.plot(np.repeat(threshold, len(whitened)), "--", color='gray')
        plt.xlabel('time step')
        plt.ylabel('probability')
        plt.title('threshold-' + threshold)
        counter += 1
    plt.savefig(output + '-testReal.png')


if __name__ == '__main__':
    args = parseTrainInput()
    prefix = args.event + args.channel + str(args.freq)
    fileCheck(prefix)

    if args.save_file:
        stdoutOrigin = sys.stdout
        sys.stdout = open(args.output + '.txt', 'w')

    train_dataset = tf.data.Dataset.from_generator(generator,
                                                   (tf.float64, tf.float64),
                                                   ((8192, 1), 8192),
                                                   (args.train_file, prefix, args.blank_ratio))
    train_dataset = train_dataset.repeat(2).shuffle(buffer_size=1024).batch(args.batch_size)
    validation_dataset = tf.data.Dataset.from_generator(generator,
                                                        (tf.float64, tf.float64),
                                                        ((8192, 1), 8192),
                                                        (args.test_file, prefix, args.blank_ratio))
    validation_dataset = validation_dataset.shuffle(buffer_size=1024).batch(args.batch_size)

    log_dir = './logs/' + args.output + '/'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    model = WaveNet(args.num_residuals, args.num_filters)
    tf.keras.backend.clear_session()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[Accuracy(), FalseAlarm(), Sensitivity()])
    history = model.fit(train_dataset, epochs=args.epoch,
                        validation_data=validation_dataset, validation_freq=100,
                        callbacks=[tensorboard_callback])

    model.reset_metrics()
    model_path = 'model/' + args.output
    model.save_weights(model_path, save_format='tf')

    plt.figure(figsize=(30, 12))
    plt.title('loss history')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(history.history['loss'])
    plt.plot(np.arange(0, args.epoch, 100), history.history['val_loss'])
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['false_alarm'])
    plt.plot(history.history['sensitivity'])
    plt.legend(['loss', 'val_loss', 'accuracy', 'false_alarm', 'sensitivity'])
    plt.savefig(args.output + '-loss.png')

    # test overall accuracy
    result = []
    snrArr = np.array([3, 2.5, 2, 1.5, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    snrArr = np.flip(snrArr)
    for snr in snrArr:
        test_dataset = tf.data.Dataset.from_generator(generator,
                                                      (tf.float64, tf.float64),
                                                      ((8192, 1), 8192),
                                                      (args.test_file, prefix, args.blank_ratio, snr))
        test_dataset = test_dataset.repeat(2).batch(args.batch_size)
        curr = model.evaluate(test_dataset)
        result.append(curr)
        break
    result = np.asarray(result).T
    plt.figure(figsize=(30, 12))
    plt.title('evaluation v.s. peak snr')
    plt.xlabel('peak snr')
    plt.ylabel('percentage')
    plt.plot(snrArr, result[1])
    plt.plot(snrArr, result[2])
    plt.plot(snrArr, result[3])
    plt.legend(['accuracy', 'false_alarm', 'sensitivity'])
    plt.savefig(args.output+'-evaluation.png')

    # test real signal detection
    testReal(model, prefix[:-1], 8192*10, 4096, args.output)
