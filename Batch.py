import numpy as np
import h5py
from Noiser import Noiser
FACTOR = 1.0

NUM_DATA = 9840
def get_batch(f,k,length,real_noise=False,SNR=None,shift=None,blankRatio=0.0): 
    batch = []
    label = []
    idx = np.arange(NUM_DATA)
    np.random.shuffle(idx)
    noise = Noiser(length)

    num_batch = NUM_DATA//k
    blankNum = int(blankRatio * k)
    snr = np.zeros(num_batch)
    high = 3.0
    if SNR is not None and SNR > high:
        high = SNR
    if SNR is not None:
        snr = np.random.uniform(low=SNR,high=high,size=num_batch)
    for i in range(num_batch):
        cur_batch = []
        cur_label = []
        for j in range(k):
            if j < blankNum:
                cur_batch.append(np.zeros(length))
                cur_label.append([0,0])
                continue
            cur_batch.append(f['data'][idx[k*i+j]][:length])
            cur_label.append(f['m1m2'][idx[k*i+j]])
        cur_batch = noise.add_shift(cur_batch)
        if shift is not None:
            cur_batch.T[:shift[0]] = 0
            cur_batch.T[shift[1]:] = 0
        if SNR is not None:
            if real_noise is False:
                cur_batch = noise.add_noise(input=cur_batch, SNR=snr[i])
            else:
                cur_batch = noise.add_real_noise(input=cur_batch, SNR=snr[i])
        batch.append(cur_batch)
        label.append(cur_label) 
        
    batch = np.asarray(batch).reshape(num_batch,k,length,1)
    label = np.asarray(label)/FACTOR
    return batch, label

def get_val(f,k,length,real_noise=False,SNR=None,shift=None):
    batch = []
    label = []
    idx = np.random.choice(f['data'].shape[0], k, replace=False)
    noise = Noiser(length)

    for i in range(k):
        batch.append(f['data'][idx[i]][:length])
        label.append(f['m1m2'][idx[i]])
    if shift is not None:
        batch.T[:shift[0]] = 0
        batch.T[shift[1]:] = 0
    if SNR is not None:
        snr = np.random.uniform(low=SNR,high=SNR,size=1)[0]            
        if real_noise is False:
            batch = noise.add_noise(input=batch, SNR=snr)
        else:
            batch = noise.add_real_noise(input=batch, SNR=snr)
    batch = noise.add_shift(batch)
    batch = np.asarray(batch).reshape(k,length,1)
    label = np.asarray(label)/FACTOR
    return batch, label
