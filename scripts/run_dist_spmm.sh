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

srun -n 2 build/spmm_demo -input-sparse datasets/spmm/sp_mat_1600.mtx -input-dense datasets/spmm/dense_mat_1600_128.csv -dataset spmm_sample_dist -output out_dense_mat -alpha 0.5 -beta 0.5
