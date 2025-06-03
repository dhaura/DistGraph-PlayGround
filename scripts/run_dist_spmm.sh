#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=30:00
#SBATCH --nodes=2
#SBATCH --tasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --constraint=cpu
#SBATCH --output=%j.log

export OMP_NUM_THREADS=16

export COMBLAS_ROOT=~/repos/PASSIONLab/CombBLAS
export DISTGRAPH_ROOT=~/repos/HipGraph/forks/DistGraph
module load intel

srun -n 4 build/spmm_demo -input datasets/mnist.mtx -dataset mnist -alpha 0.5 -beta 0.5
