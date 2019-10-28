from Noiser import *

FACTOR = 1.0
NUM_DATA = 9840


def get_batch(f, k, length, real_noise=False, snr=None, shift=None):
    """
    :param f: hdf5 file object to get file
    :param k: batch size
    :param length: length of output starting from 0
    :param real_noise: whether to add real noise or generated noise
    :param snr: lowest snr value for current batch
    :param shift: whether zero out some portion of output
    :return: output batch, output label
    """

    # return array initialization
    batch = []
    label = []

    # constant initialization
    idx = np.arange(NUM_DATA)
    np.random.shuffle(idx)
    noise = Noiser(length)
    num_batch = NUM_DATA // k
    high = 3.0

    # initialize snr values
    snrArr = np.zeros(num_batch)
    if snr is not None:
        if snr > high:
            high = snr
        snrArr = np.random.uniform(low=snr, high=high, size=num_batch)

    # loop to get batch
    for i in range(num_batch):
        cur_batch = []
        cur_label = []
        for j in range(k):
            cur_batch.append(f['data'][idx[k * i + j]][:length])
            cur_label.append(f['m1m2'][idx[k * i + j]])
        cur_batch = noise.add_shift(cur_batch)
        if shift is not None:
            cur_batch.T[:shift[0]] = 0
            cur_batch.T[shift[1]:] = 0
        if snr is not None:
            if real_noise is False:
                cur_batch = noise.add_noise(input=cur_batch, SNR=snrArr[i])
            else:
                cur_batch = noise.add_real_noise(input=cur_batch, SNR=snrArr[i])
        batch.append(cur_batch)
        label.append(cur_label)

    # reshape
    batch = np.asarray(batch).reshape(num_batch, k, length, 1)
    label = np.asarray(label) / FACTOR
    return batch, label


def get_val(f, k, length, real_noise=False, snr=None, shift=None):
    """
    :param f: hdf5 file object to get file
    :param k: batch size
    :param length: length of output starting from 0
    :param real_noise: whether to add real noise or generated noise
    :param snr: lowest snr value for current batch
    :param shift: whether zero out some portion of output
    :return: output batch, output label
    """

    batch = []
    label = []

    # constant initialization
    idx = np.random.choice(f['data'].shape[0], k, replace=False)
    noise = Noiser(length)

    # loop to get batch
    for i in range(k):
        batch.append(f['data'][idx[i]][:length])
        label.append(f['m1m2'][idx[i]])
    if shift is not None:
        batch.T[:shift[0]] = 0
        batch.T[shift[1]:] = 0
    if snr is not None:
        snrArr = np.random.uniform(low=snr, high=snr, size=1)[0]
        if real_noise is False:
            batch = noise.add_noise(input=batch, SNR=snrArr)
        else:
            batch = noise.add_real_noise(input=batch, SNR=snrArr)
    batch = noise.add_shift(batch)
    batch = np.asarray(batch).reshape(k, length, 1)
    label = np.asarray(label) / FACTOR
    return batch, label
