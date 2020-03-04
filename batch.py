from noiser import Noiser
import numpy as np
import h5py
import sys


class Batch(object):
    def __init__(self, train_file, val_file, noise_file, batch_size, real_noise, keyStr='data'):
        """
        :param train_file: path of the training file data
        :param val_file: path of the testing file data
        :param batch_size: user specified batch size
        :param real_noise: True to add real noise and False to add generated noise
        :param keyStr: key to get hdf5 file data. Default is 'data'
        :return: None
        """

        self.train_fp = h5py.File(train_file, 'r')
        self.val_fp = h5py.File(val_file, 'r')
        self.num_train = self.train_fp[keyStr].shape[0]
        self.length = self.train_fp[keyStr].shape[1]
        assert self.train_fp[keyStr].shape[1] == self.val_fp[keyStr].shape[1]
        self.noiser = Noiser(noise_file, self.length)
        self.batch_size = batch_size
        self.real_noise = real_noise
        self.keyStr = keyStr

        # check nan
        if np.isnan(self.train_fp[keyStr]).any():
            print("nan present in training data. Exiting...")
            sys.exit(1)
        if np.isnan(self.val_fp[keyStr]).any():
            print("nan present in validation data. Exiting...")
            sys.exit(1)

    def get_train_batch(self, snrMin):
        """
        :param snrMin: the minimum snr value for the current batch
        :return: multiple batches for training
        """

        # return array initialization
        batch = []
        label = []

        # initialization
        idx = np.arange(self.num_train)
        np.random.shuffle(idx)
        num_batch = self.num_train // self.batch_size
        high = 3.0
        if snrMin > high:
            high = snrMin
        snrArr = np.random.uniform(low=snrMin, high=high, size=num_batch)

        # loop to get batch
        for i in range(num_batch):
            zero = []
            signal = []
            cur_label = []
            counter = 0
            for j in range(self.batch_size):
                # change here for best performance
                if counter < self.batch_size*5//8:
                    # ratio of pure noise
                    zero.append(np.zeros(self.length))
                    cur_label.append([1, 0])
                    counter += 1
                    continue
                signal.append(self.train_fp[self.keyStr][idx[self.batch_size * i + j]])
                cur_label.append([0, 1])

            # add noise and shift to input data
            if self.real_noise is False:
                if len(signal) is not 0:
                    signal = self.noiser.add_noise(input=signal, SNR=snrArr[i])
                if len(zero) is not 0:
                    zero = self.noiser.add_noise(input=zero, SNR=snrArr[i])
            else:
                if len(signal) is not 0:
                    signal = self.noiser.add_real_noise(input=signal, SNR=snrArr[i])
                if len(zero) is not 0:
                    zero = self.noiser.add_real_noise(input=zero, SNR=snrArr[i])
            cur_batch = np.array(list(zero) + list(signal))
            cur_batch = self.noiser.add_shift(cur_batch)

            # shuffle input data
            idxCurr = np.arange(self.batch_size)
            np.random.shuffle(idxCurr)
            cur_batch = [cur_batch[a] for a in idxCurr]
            cur_label = [cur_label[a] for a in idxCurr]
            batch.append(cur_batch)
            label.append(cur_label)

        # reshape
        batch = np.asarray(batch).reshape(num_batch, self.batch_size, self.length, 1)
        label = np.asarray(label)
        return batch, label


    def get_val_batch(self, snr):
        """
        :param snr: the snr value for the current batch
        :return: single batch for validation
        """

        signal = []
        zero = []
        label = []

        # initialization
        idx = np.random.choice(self.val_fp[self.keyStr].shape[0], self.batch_size, replace=False)

        # loop to get batch
        counter = 0
        for i in range(self.batch_size):
            # use half and half during testing
            if counter < self.batch_size // 2:
                zero.append(np.zeros(self.length))
                label.append([1, 0])
                counter += 1
                continue
            signal.append(self.val_fp[self.keyStr][idx[i]])
            label.append([0, 1])

        # add noise and shift
        snrArr = np.random.uniform(low=snr, high=snr, size=1)[0]
        if self.real_noise is False:
            signal = self.noiser.add_noise(input=signal, SNR=snrArr)
            zero = self.noiser.add_noise(input=zero, SNR=snrArr)
        else:
            signal = self.noiser.add_real_noise(input=signal, SNR=snrArr)
            zero = self.noiser.add_real_noise(input=zero, SNR=snrArr)
        batch = np.array(list(zero) + list(signal))
        batch = self.noiser.add_shift(batch)

        # reshape
        batch = np.asarray(batch).reshape(self.batch_size, self.length, 1)
        label = np.asarray(label)
        return batch, label
