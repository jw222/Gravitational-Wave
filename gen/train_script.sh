#!/bin/sh

module load python/3
module load openmpi
mpirun -n 10 python3 ~/code/gen/run_batch.py \
	--config_file ~/code/gen/trainConfig.txt \
	--out_file ~/code/data/150914H8S1Train.h5