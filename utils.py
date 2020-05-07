import numpy as np
import readligo as rl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import os
import h5py
import pickle
from scipy.signal import decimate
from scipy.signal import find_peaks
from scipy.interpolate import interp1d


def train_step(optimizer, train_model, input, target):
    with tf.GradientTape() as tape:
        predictions = train_model(input)
        loss = tf.reduce_mean(
            tf.keras.losses.MeanSquaredError(
                target, predictions))
    grads = tape.gradient(loss, train_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, train_model.trainable_variables))

    return loss


def whiten(strainW, interp_psd, dt=1./8192.):
    Nt = len(strainW)
    freqsW = np.fft.rfftfreq(Nt, dt)
    hf = np.fft.rfft(strainW)
    white_hf = hf / (np.sqrt(interp_psd(freqsW)/dt/2.))
    white_ht = np.fft.irfft(white_hf, n=Nt)
    white_ht[:100] = 0
    white_ht[-100:] = 0
    return white_ht


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


def category(prediction, threshold):
    processed = np.convolve(prediction, np.ones(1024)) / 1024
    return np.greater(processed, np.full(len(processed), threshold)).any()


def testReal(trained_model, event, crop, step, output, length, twoChan=False):
    # strain data preprocessing
    if length is not 4096 and length is not 32:
        length = 4096
    num_predict = 0
    if not twoChan:
        strain, time, chan_dict = rl.loaddata('raw/' + event + '-' + str(length) + '.hdf5', event[-1] + '1')
        strain = decimate(strain, 2)
        fs = 8192
        pxx, freqs = mlab.psd(strain, Fs=fs, NFFT=fs)
        interp_psd = interp1d(freqs, pxx)
        whitened = whiten(strain, interp_psd)
        whitened = whitened[crop:-crop]
        whitened /= np.std(whitened)
        num_predict = len(whitened)

        # prepare prediction dataset
        x_predict = []
        for start in range(0, len(whitened)-8192, step):
            waveform = whitened[start:start+8192]
            waveform /= np.std(waveform)
            x_predict.append(waveform)
        x_predict = np.asarray(x_predict)
        x_predict = np.expand_dims(x_predict, axis=-1)
    else:
        strain_H, time_H, chan_dict_H = rl.loaddata('raw/' + event + 'H-' + str(length) + '.hdf5', 'H1')
        strain_L, time_L, chan_dict_L = rl.loaddata('raw/' + event + 'L-' + str(length) + '.hdf5', 'L1')
        strain_H = decimate(strain_H, 2)
        strain_L = decimate(strain_L, 2)
        fs = 8192
        pxx_H, freqs_H = mlab.psd(strain_H, Fs=fs, NFFT=fs)
        interp_psd_H = interp1d(freqs_H, pxx_H)
        pxx_L, freqs_L = mlab.psd(strain_L, Fs=fs, NFFT=fs)
        interp_psd_L = interp1d(freqs_L, pxx_L)
        whitened_H = whiten(strain_H, interp_psd_H)
        whitened_H = whitened_H[crop:-crop]
        whitened_H /= np.std(whitened_H)
        whitened_L = whiten(strain_L, interp_psd_L)
        whitened_L = whitened_L[crop:-crop]
        whitened_L /= np.std(whitened_L)

        # prepare prediction dataset
        x_predict = []
        if len(whitened_H) < len(whitened_L):
            whitened_L = whitened_L[:len(whitened_H)]
        elif len(whitened_H) > len(whitened_L):
            whitened_H = whitened_H[:len(whitened_L)]
        num_predict = len(whitened_H)
        for start in range(0, len(whitened_H) - 8192, step):
            waveform_H = whitened_H[start:start + 8192]
            waveform_H /= np.std(waveform_H)
            waveform_L = whitened_L[start:start + 8192]
            waveform_L /= np.std(waveform_L)
            waveform = np.stack([waveform_H, waveform_L], axis=-1)
            x_predict.append(waveform)
        x_predict = np.asarray(x_predict)

    # get prediction
    y_temp = trained_model.predict(x_predict)
    y_predict = np.zeros(num_predict)
    for start in range(0, num_predict-8192, step):
        y_predict[start:start+8192] += y_temp[start//step] * (step/8192.)
    y_predict = np.convolve(y_predict, np.ones(1024)) / 1024
    plt.figure(figsize=(30, 30))
    counter = 1
    total_peaks = {}
    for threshold in [0.5, 0.6, 0.7, 0.8]:
        peaks, _ = find_peaks(y_predict, height=threshold, distance=8192)
        plt.subplot(2, 2, counter)
        plt.plot(y_predict)
        plt.plot(peaks, y_predict[peaks], 'x')
        plt.plot(np.repeat(threshold, len(y_predict)), "--", color='gray')
        plt.axvline(x=len(y_predict)/2, color='r')
        plt.xlabel('time step')
        plt.ylabel('probability')
        plt.title('threshold-' + str(threshold))
        counter += 1
        total_peaks[threshold] = peaks
    plt.savefig(output + '-testReal.png')
    total_peaks['middle'] = len(y_predict)/2
    return total_peaks
