#!/bin/bash
#PBS -N beyasian_para_search
#PBS -l select=1:ncpus=4:ngpus=1:mem=400G
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o beyasian_para_search.log
#PBS -P 31010032
#PBS -q normal

cd $PBS_O_WORKDIR

module load miniforge3/24.3.0
conda activate GEARS
module load cuda/11.8.0
module load cudnn/11-8.9.7.29
export WANDB_MODE=offline
export WANDB_DIR="/data/projects/31010032/xuanhong/GEARS/downstreak"
mkdir -p "$WANDB_DIR"

start=$(date +%s)

python adjust_hyperparameter.py \
  --study_name para_search_hyperparameters \
  --storage sqlite:////data/projects/31010032/xuanhong/GEARS/downstreak/optuna/gears_hpo.db \
  --save_path bys_para \
  --worker_mode para

end=$(date +%s)

echo "Elapsed: $((end-start)) s"