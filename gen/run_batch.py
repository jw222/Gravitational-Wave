import pickle
from batch import Batch
from argparse import ArgumentParser


help_strs = {
    'desc': 'A wrapper script to generate a group of waveforms in parallel.',
    'config': 'path to pickle file containing the configurations of the '
              'waveforms to generate',
    'out': 'path to which the resulting HDF5 dataset should be stored'
}

parser = ArgumentParser(description=help_strs['desc'])
parser.add_argument('--config_file', type=str, required=True,
                    help=help_strs['config'])
parser.add_argument('--out_file', type=str, required=True, 
                    help=help_strs['out'])
args = parser.parse_args()

with open(args.config_file, 'rb') as fh:
    configs = pickle.load(fh)

batch = Batch(configs)
batch.run_all(save_path=args.out_file)

