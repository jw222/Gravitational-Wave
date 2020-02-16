import numpy as np
from scipy.interpolate import interp1d
import matplotlib.mlab as mlab
import readligo as rl
import h5py

save_path = 'data/noiseWhitened.hdf5'

def whiten_wrapper(wave, dt):
    fn_H1 = 'data/150914-4096.hdf5'
    strain_H1, time_H1, chan_dict_H1 = rl.loaddata(fn_H1, 'H1')
    conditioned = [strain_H1[i] for i in range(0,len(strain_H1),2)]
    
    fs = 8192
    NFFT = 1*fs
    Pxx_H1, freqs = mlab.psd(conditioned, Fs = fs, NFFT = NFFT)
    psd_H1 = interp1d(freqs, Pxx_H1)

    # function to whiten data
    def whiten(strainW, interp_psd, dt):
        Nt = len(strainW)
        freqsW = np.fft.rfftfreq(Nt, dt)

        # whitening: transform to freq domain, divide by asd, then transform back, 
        # taking care to get normalization right.
        hf = np.fft.rfft(strainW)
        white_hf = hf / (np.sqrt(interp_psd(freqsW) /dt/2.))
        white_ht = np.fft.irfft(white_hf, n=Nt)
        return white_ht

    # now whiten the data from H1 and L1, and also the NR template:
    wave_whiten = whiten(wave, psd_H1, dt)
    wave_whiten = (wave_whiten - np.mean(wave_whiten)) / np.std(wave_whiten)

    return wave_whiten

strain_test, time_test, chan_dict_test = rl.loaddata("data/150914-4096.hdf5", 'H1')
strain_test = [strain_test[i] for i in range(0,len(strain_test),2)]
strain_test = whiten_wrapper(strain_test, 1./8192.)
strain_test = strain_test[10000:-10000]

f = h5py.File(save_path, 'w')
f['Dataset1'] = res_array
f.close()