import h5py
import numpy as np
import random


class Noiser(object):
    def __init__(self, file_path, length):
        self.real_noise = h5py.File(file_path, 'r')
        self.length = length

    def shift_single_wave(self, input):
        shift_index = np.random.randint(int(10), int(input.shape[0] * 0.06), size=1)[0]
        left_right_index = np.random.randint(2, size=1)[0]
        if left_right_index == 1:  # shift to the right
            return np.hstack((np.zeros(shift_index), input[:-shift_index]))
        elif left_right_index == 0:
            return np.hstack((input[shift_index:], np.zeros(shift_index)))
        else:
            assert left_right_index == 1 or left_right_index == 0

    def add_noise(self, input, SNR):
        tmp_out = np.asarray(input) / np.asarray(input).max(axis=-1)[np.newaxis].transpose()
        if not np.isnan(tmp_out[0][0]):
            output = tmp_out * SNR.astype(float) + np.random.normal(0., 1., np.asarray(input).shape)
            stds = np.std(output, axis=1)[np.newaxis].transpose()

            return output / stds
        else:
            # this is for adding noise to zeros template
            return np.random.normal(0., 1., np.asarray(input).shape)

    def add_shift(self, input):
        input = np.asarray(input)
        return np.asarray(list(map(self.shift_single_wave, input)))

    def add_real_noise(self, input, SNR):
        ramint = random.randint(20, len(self.real_noise['data']) - self.length)
        noise_data = self.real_noise['data'][ramint:(ramint + self.length)]
        noise_data -= np.mean(noise_data)
        noise_data_ = noise_data / np.std(noise_data)
        noise_data_ = np.tile(noise_data_[np.newaxis], [np.asarray(input).shape[0], 1])
        tmp_out = np.asarray(input) / np.asarray(input).max(axis=-1)[np.newaxis].transpose()

        if not np.isnan(tmp_out[0][0]):
            output = tmp_out * SNR.astype(float)
            output_ = output + noise_data_
            means_ = np.mean(output_, axis=1)[np.newaxis].transpose()
            stds_ = np.std(output_, axis=1)[np.newaxis].transpose()
            return (output_ - means_) / stds_
        else:
            # this is for adding noise to zeros template
            return noise_data_
