#!/bin/bash

for s in 1 2 3 4 5; do
  qsub -N "GEARS_${s}" -o "GEARS_${s}.log" -j oe <<PBS_EOF
#!/bin/bash
#PBS -l select=1:ncpus=4:ngpus=1:mem=400G
#PBS -l walltime=24:00:00
#PBS -P 31010032
#PBS -q normal

cd "\$PBS_O_WORKDIR"

module load miniforge3/24.3.0
conda activate GEARS
module load cuda/11.8.0
module load cudnn/11-8.9.7.29

export WANDB_MODE=offline
export WANDB_DIR="/data/projects/31010032/xuanhong/GEARS/model_fig_2/dixit/no-perturb/wandb/seed_${s}"
mkdir -p "\$WANDB_DIR"

start=\$(date +%s)
mkdir module_${s}
cd module_${s}

python /data/projects/31010032/xuanhong/GEARS/model_fig_2/dixit/no-perturb/fig2_train.py --seed ${s} --device 0 --dataset dixit --model no_perturb
end=\$(date +%s)

echo "Elapsed: \$((end-start)) s"
PBS_EOF
done


