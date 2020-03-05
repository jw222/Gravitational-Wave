import numpy as np
import h5py
import traceback
import sys
from task_pool import *
from waveform import Waveform

# change here
num_secs = 1.
freq = 8
psdType = 'H1'
event = '150914'
pxxStr = 'psd/'+event+psdType[0]+str(freq)+'pxx'
freqsStr = 'psd/'+event+psdType[0]+str(freq)+'freqs'

def work(waveform):
    """Wrapper for generating waveform from a Waveform object."""
    try:
        waveform.gen_waveform(num_secs=num_secs, delta_t=1./float(1024*freq), pxxStr=pxxStr, freqsStr=freqsStr)
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        waveform.failed = True
    return waveform


class Batch:
    """
    A group of Waveform objects to simulate in parallel.
    
    Note
    ----
    MPI rank 0 (the parent of ``exec_manager``) has different behaviors from
    the other ranks. Some data, like the full list of Waveform objects and 
    their simulation results, have only one copy in the parent rank. They can
    be invalid in other ranks, but they must still be defined to avoid 
    NameError. Therefore, many of them are set to their actual values in the
    parent and None in the children. Similarly, return values from different
    ranks can also be different. The result from the parent should be taken.

    Attributes
    ----------
    exec_manager : object
        MPITaskPool object for parallelism. Derived from Sibo Wang's code in
        Dr. Guan's lab at the Dept of NRES, UIUC.
    configs : list
        A list of tuples containing the configuration info for the waveforms.
    waveforms : list
        A list of Waveform objects in the same order as ``configs``.
    
    """
    def __init__(self, configs, exec_manager=None):
        self.configs = configs
        if exec_manager is None:
            exec_manager = MPITaskPool()
        self.exec_manager = exec_manager
        if self.exec_manager.is_parent():
            self.waveforms = [Waveform(*config) for config in configs]
        else:
            self.waveforms = None

    def run_all(self, save_path=None, log_freq=1):
        """
        Run all simulations in parallel.

        Parameters
        ----------
        save_path : str, optional
            The path to which the resultant HDF5 dataset should be stored. If 
            not specified, the results will not be serialized.
        log_freq : int, optional
            Misnomer. Should really be "log_interval". A log message will be
            written, by ``self.exec_manager``, every this many tasks are done.
            The default value is 1.

        Returns
        -------
        ndarray if rank == 0; otherwise 0
            In MPI rank 0, the 2d array of shape 
            (num_waveforms, num_time_steps) will be returned. In all other 
            ranks, None will be returned. See Note.

        """
        res_wfs = self.exec_manager.run(self.waveforms, work, log_freq=1)
        if self.exec_manager.is_parent():
            res_array = np.empty((len(self.configs), int(num_secs*1024*freq)), dtype='float32')
            res_array[:, :] = np.nan
            for i, wf in enumerate(res_wfs):
                if not wf.failed:
                    res_array[i, :] = wf.waveform
            if save_path is not None:
                config_flattened = [[x[0], x[1], *x[2], *x[3]] 
                                        for x in self.configs]
                f = h5py.File(save_path, 'w')
                f['data'] = res_array
                f['configs'] = np.array(config_flattened, dtype='float32')
                f['m1m2'] = np.array(config_flattened, dtype='float32').T[:2].T
                f.close()
        else:
            res_array = None
        return res_array

