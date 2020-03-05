import numpy as np
from scipy.interpolate import interp1d
import matplotlib.mlab as mlab
import readligo as rl
import h5py
import argparse
import sys
import pickle


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate psd and noise')
    parser.add_argument('--file', dest='file_path', type=str, default='raw/150914-4096.hdf5',
                        help='file path of raw data')
    parser.add_argument('--freq', dest='freq', type=int, default=8,
                        help='frequency needed')
    parser.add_argument('--type', dest='detector', type=str, default='H1',
                        help='data from which detector. Either H1 or L1')
    parser.add_argument('--save', dest='save_prefix', type=str, default='150914',
                        help='file save path')
    parser.add_argument('--noise', dest='noise', type=bool, default=True,
                        help='whether to generate the noise file')
    args = parser.parse_args()

    fp = args.file_path
    freq = args.freq
    ifo = args.detector
    prefix = args.save_prefix

    # argument check
    if 16 % freq != 0:
        print('Frequency not valid. data is downsampled from 16kHz')
        sys.exit(1)
    if ifo != 'H1' and ifo != 'L1':
        print('Type has to be either H1 or L1')
        sys.exit(1)

    strain, time, chan_dict = rl.loaddata(fp, ifo)
    if freq != 16:
        # downsample
        strain = [strain[i] for i in range(0, len(strain), int(16/freq))]

    # whitening signal
    fs = 1024*freq
    pxx, freqs = mlab.psd(strain, Fs=fs, NFFT=fs)
    if args.noise:
        psd = interp1d(freqs, pxx)
        wave_whiten = whiten(strain, psd, 1./float(freq*1024))
        wave_whiten = (wave_whiten - np.mean(wave_whiten)) / np.std(wave_whiten)
        # crop out high edges
        wave_whiten = wave_whiten[20000:-20000]

        # save noise to file
        f = h5py.File('data/'+prefix+ifo[0]+str(freq)+'Noise.hdf5', 'w')
        f['data'] = wave_whiten
        f.close()

    # save frequency and psd to file
    with open('psd/'+prefix+ifo[0]+str(freq)+'freqs', 'wb') as fn:
        pickle.dump(freqs, fn)
    with open('psd/'+prefix+ifo[0]+str(freq)+'pxx', 'wb') as fn:
        pickle.dump(pxx, fn)
