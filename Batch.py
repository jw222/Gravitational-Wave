import numpy as np
import h5py
from Noiser import Noiser

f_train = h5py.File("data/TrainEOB_q-1-10-0.02_ProperWhitenZ.h5", "r")
f_test = h5py.File("data/TestEOB_q-1-10-0.02_ProperWhitenZ.h5", "r")

NUM_DATA = f_train['WhitenedSignals'].shape[0]
LENGTH = f_train['WhitenedSignals'].shape[1]
def get_batch(f,k,length=LENGTH,real_noise=False,SNR=None): 
	batch = []
	label = []
	idx = np.arange(NUM_DATA)
	np.random.shuffle(idx)
	noise = Noiser()

	num_batch = NUM_DATA//k
	high = 3.0
	if SNR is not None and SNR > high:
		high = SNR
	snr = np.random.uniform(low=SNR,high=high,size=num_batch)
	for i in range(num_batch):
		cur_batch = []
		cur_label = []
		for j in range(k):
			cur_batch.append(f['WhitenedSignals'][idx[k*i+j]][:length])
			cur_label.append(f['m1m2'][idx[k*i+j]])
		cur_batch = noise.add_shift(cur_batch)
		if SNR is not None:
			if real_noise is False:
				cur_batch = noise.add_noise(input=cur_batch, SNR=snr[i])
			else:
				cur_batch = noise.add_real_noise(input=cur_batch, SNR=snr[i])
		batch.append(cur_batch)
		label.append(cur_label) 
		
	batch = np.asarray(batch).reshape(num_batch,k,length,1)
	label = np.asarray(label)
	return batch, label

def get_val(f,k,length=LENGTH,real_noise=False,SNR=None):
	batch = []
	label = []
	idx = np.random.choice(f_test['WhitenedSignals'].shape[0], k, replace=False)
	noise = Noiser()

	for i in range(k):
		batch.append(f['WhitenedSignals'][idx[i]][:length])
		label.append(f['m1m2'][idx[i]])

	if SNR is not None:
		snr = np.random.uniform(low=SNR,high=SNR,size=1)[0]            
		print(snr)
		if real_noise is False:
			batch = noise.add_noise(input=batch, SNR=snr)
		else:
			batch = noise.add_real_noise(input=batch, SNR=snr)
	batch = noise.add_shift(batch)
	batch = np.asarray(batch).reshape(k,length,1)
	label = np.asarray(label)
	return batch, label
