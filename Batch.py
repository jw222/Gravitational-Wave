import numpy as np
import h5py

f_train = h5py.File("TrainEOB_q-1-10-0.02_ProperWhitenZ.h5", "r")
f_test = h5py.File("TestEOB_q-1-10-0.02_ProperWhitenZ.h5", "r")

NUM_DATA = f_train['WhitenedSignals'].shape[0]
LENGTH = int(f_train['WhitenedSignals'].shape[1])
def get_batch(f,k,length=LENGTH): 
    batch = []
    label = []
    idx = np.arange(NUM_DATA)
    np.random.shuffle(idx)

    num_batch = NUM_DATA//k
    for i in range(num_batch):
        cur_batch = []
        cur_label = []
        for j in range(k):
            pad = np.random.choice(LENGTH, 1)[0]
            cur_batch.append(np.pad(f['WhitenedSignals'][idx[k*i+j]][:length], 
                                    (0,pad), 'constant', constant_values=0)[pad:])
            cur_label.append(f['m1m2'][idx[k*i+j]])
        batch.append(cur_batch)
        label.append(cur_label) 
        
    batch = np.asarray(batch).reshape(num_batch,k,length,1)
    label = np.asarray(label)
    return batch, label

def get_test(f,k,length=LENGTH):
    batch = []
    label = []
    idx = np.random.choice(f_test['WhitenedSignals'].shape[0], k, replace=False)

    for i in range(k):
        pad = np.random.choice(LENGTH, 1)[0]
        batch.append(np.pad(f['WhitenedSignals'][idx[i]][:length], 
                                    (0,pad), 'constant', constant_values=0)[pad:])
        label.append(f['m1m2'][idx[i]])

    batch = np.asarray(batch).reshape(k,length,1)
    label = np.asarray(label)
    return batch, label

