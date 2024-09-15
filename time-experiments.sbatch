#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=14
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=time-experiments
module purge
singularity exec \
    --overlay /scratch/vp724/benchmark/benchmark.ext3:ro \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    /bin/bash -c "source /ext3/env.sh; python /scratch/vp724/xAI-Ranking-Benchmark/notebooks/0.3-time-experiment.py"
