import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import readligo as rl
from scipy.interpolate import interp1d
import matplotlib.mlab as mlab
import datetime
import sys
import os
import argparse
import h5py
import pickle
from net import Classifier
from noiser import Noiser
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def parseTestInput():
    # parsing argument
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model', dest='model_path', type=str, default='../model/B64L1Ra58Classifier.ckpt',
                        help='the file of the model')
    parser.add_argument('--test', dest='test_file', type=str, default='data/150914H8S1Test.h5',
                        help='the file of the testing data')
    parser.add_argument('--noise', dest='noise_file', type=str, default='data/150914H8Noise.hdf5',
                        help='the noise file for training')
    parser.add_argument('--name', dest='output_file', type=str, default='1',
                        help='test number')
    parser.add_argument('--keyStr', dest='keyStr', type=str, default='data',
                        help='key to access hdf5 file')
    parser.add_argument('--freq', dest='freq', type=int, default=8,
                        help='sample rate in kHz')
    parser.add_argument('--file', dest='file', type=bool, default=False,
                        help='whether cast output to a file')
    parser.add_argument('--noiseType', dest='noiseType', type=bool, default=True,
                        help='whether add real noise or generated noise')
    parser.add_argument('--testOverall', dest='testOverall', type=bool, default=False,
                        help='whether to test overall accuracy')
    parser.add_argument('--testGradual', dest='testGradual', type=bool, default=False,
                        help='whether to test with gradual input')
    parser.add_argument('--testReal', dest='testReal', type=bool, default=False,
                        help='whether to test on real signal')
    return parser.parse_args()


class Inference(object):
    def __init__(self, model_path, test_file, noise_file, freq, noiseType, outputName, keyStr='data'):
        # structure initialization
        self.input_data = tf.placeholder(tf.float32, [None, None, 1])
        self.input_label = tf.placeholder(tf.int32, [None, 2])
        self.trainable = tf.placeholder(tf.bool)
        self.predictions = Classifier(self.input_data, self.trainable)

        # load model
        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        saver.restore(self.sess, model_path)

        # class initialization
        self.test_fp = h5py.File(test_file, 'r')
        self.noise_file = noise_file
        self.length = self.test_fp[keyStr].shape[1]
        self.freq = freq
        self.noiseType = noiseType
        self.outputName = outputName
        self.keyStr = keyStr
        self.event = test_file[5:11]

        # check nan
        if np.isnan(self.test_fp[keyStr]).any():
            print("nan present in testing data. Exiting...")
            sys.exit(1)

    def close(self):
        self.sess.close()

    def _whiten(self, strainW, interp_psd, dt):
        Nt = len(strainW)
        freqsW = np.fft.rfftfreq(Nt, dt)

        hf = np.fft.rfft(strainW)
        white_hf = hf / (np.sqrt(interp_psd(freqsW) / dt / 2.))
        white_ht = np.fft.irfft(white_hf, n=Nt)
        return white_ht

    def _compute_accuracy(self, currSNR, length, shift):
        pred = []
        labels = []
        new_length = shift[1] - shift[0]
        noise = Noiser(self.noise_file, new_length)
        for j in range(len(self.test_fp[self.keyStr])//2):
            temp_test = self.test_fp[self.keyStr][j*2].reshape(1, self.length)
            temp_test = noise.add_shift(temp_test)
            test_data = np.array(temp_test[0][shift[0]:shift[1]]).reshape(1, new_length)
            if self.noiseType is False:
                test_data = noise.add_noise(input=test_data, SNR=currSNR)
            else:
                test_data = noise.add_real_noise(input=test_data, SNR=currSNR)
            test_data = test_data.reshape(1, new_length, 1)
            labels.append(1)
            curr = self.sess.run(self.predictions, feed_dict={self.input_data: test_data, self.trainable: False})[0]
            pred.append(np.argmax(curr))

        # use same number of noise as signal
        for _ in range(len(self.test_fp[self.keyStr])//2):
            test_data = np.zeros(new_length).reshape(1, new_length)
            if self.noiseType is False:
                test_data = noise.add_noise(input=test_data, SNR=currSNR)
            else:
                test_data = noise.add_real_noise(input=test_data, SNR=currSNR)
            test_data = test_data.reshape(1, new_length, 1)
            labels.append(0)
            curr = self.sess.run(self.predictions, feed_dict={self.input_data: test_data, self.trainable: False})[0]
            pred.append(np.argmax(curr))

        pred = np.asarray(pred)
        accuracies = np.mean(pred == labels)
        totalPos = np.sum(labels[i] == 1 for i in range(len(labels)))
        totalNeg = np.sum(labels[i] == 0 for i in range(len(labels)))
        tp = np.sum([pred[i] == labels[i] and labels[i] == 1 for i in range(len(labels))]) / totalPos
        fp = np.sum([pred[i] != labels[i] and labels[i] == 0 for i in range(len(labels))]) / totalNeg
        return accuracies, tp, fp
        
    def overall_accuracy(self):
        print("Testing accuracy with entire input")
        snrArr = np.linspace(5.0, 0.1, 50)
        acc = []
        sen = []
        fa = []
        for snr in snrArr:
            accuracy, sensitivity, false_alarm = self._compute_accuracy(snr, length=self.length, shift=[0, self.length])
            print(f"snr: {snr}\n accuracy: {accuracy}, sensitivity: {sensitivity}, false alarm rate: {false_alarm}")
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
        plt.savefig(self.outputName + '-OverallAccuracy.png')

    def gradual_accuracy(self):
        print("Testing with gradual input")
        snrArr = np.array([5.0, 3.0, 2.0, 1.5, 1.0, 0.7, 0.5, 0.3, 0.2, 0.1])
        timeStamps = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        num_secs = self.length // self.freq
        for snr in snrArr:
            acc = []
            sen = []
            fa = []
            print(f"Current snr is: {snr}")
            for stop in timeStamps:
                currShift = [0, int(stop * self.length)]
                accuracy, sensitivity, false_alarm = self._compute_accuracy(snr, length=self.length, shift=currShift)
                print(f"stop: {currShift}\n accuracy: {accuracy}, sensitivity: {sensitivity}, false alarm rate: {false_alarm}")
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
            plt.savefig(self.outputName + '-' + str(snr) + '-GradualClassify.png')

    def real_accuracy(self):
        psdType = 'H1'
        # sliding window data
        fs = self.freq * 1024
        crops = [(fs*4, int(fs*1.5)), (fs*8, fs*2)]
        # 245760 for 32 seconds
        event_time = 245760//(16//self.freq)
        signals = [self.event+'.hdf5']

        for file in signals:
            file_path = 'raw/' + file
            strain, time, chan_dict = rl.loaddata(file_path, psdType)

            # calculate psd
            if self.freq != 16:
                strain = [strain[i] for i in range(0, len(strain), int(16/self.freq))]
            with open('psd/'+file[:6]+psdType[0]+str(self.freq)+'freqs', 'rb') as fh:
                freqs = pickle.load(fh)
            with open('psd/'+file[:6]+psdType[0]+str(self.freq)+'pxx', 'rb') as fh:
                pxx = pickle.load(fh)
                psd = interp1d(freqs, pxx)

            for window, crop in crops:
                exist = []
                confidence = []
                x_axis = []
                
                # sliding window method
                for start in range(0, len(strain) - window, 1024):
                    curr_strain = strain[start:start + window]
                    curr_strain = self._whiten(curr_strain, psd, 1./float(fs))
                    curr_strain = curr_strain[crop:-crop]
                    curr_strain = (curr_strain - np.mean(curr_strain)) / np.std(curr_strain)
                    curr_strain = np.array(curr_strain).reshape(1, window - 2 * crop, 1)
                    result = self.sess.run(self.predictions, feed_dict={self.input_data: curr_strain, self.trainable: False})[0]
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
                plt.savefig(file[:6] + '-' + str(window / fs) + '-' + str(crop / fs) + '-existence.png')

                # plot confidence graph
                plt.figure()
                plt.plot(x_axis, confidence)
                plt.xlabel('center time')
                plt.ylabel('confidence')
                plt.title('confidence with time')
                plt.grid(True)
                plt.savefig(file[:6] + '-' + str(window / fs) + '-' + str(crop / fs) + '-confidence.png')


if __name__ == '__main__':
    args = parseTestInput()

    if args.file:
        stdoutOrigin = sys.stdout
        sys.stdout = open("testOut" + args.output_file + ".txt", "w")
    tf.logging.set_verbosity(tf.logging.ERROR)

    infer = Inference(args.model_path, args.test_file, args.noise_file, args.freq, args.noiseType, args.output_file)
    if args.testOverall:
        infer.overall_accuracy()
    if args.testGradual:
        infer.gradual_accuracy()
    if args.testReal:
        infer.real_accuracy()
    infer.close()