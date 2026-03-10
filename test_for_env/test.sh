#!/bin/bash
#PBS -N GEARS
#PBS -l select=1:ncpus=4:ngpus=1:mem=400G
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o GEARS.log
#PBS -P 31010032
#PBS -q normal

cd $PBS_O_WORKDIR

module load miniforge3/24.3.0
conda activate GEARS
module load cuda/11.8.0
module load cudnn/11-8.9.7.29

start=$(date +%s)

python test.py

end=$(date +%s)

echo "Elapsed: $((end-start)) s"