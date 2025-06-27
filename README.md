# DistGraph PlayGround

This repository provides a practical tutorial and demonstration of the core functionality offered by the [DistGraph](https://github.com/HipGraph/DistGraph.git) library — a high-performance distributed framework for large-scale graph processing. It provides a rich set of utilities and abstractions for efficiently handling distributed graph data structures and algorithms.

## Software/ Tool Requirements
```bash
GCC version >= 4.9 (13.2)
OpenMP version >= 4.5
ComBLAS - https://github.com/PASSIONLab/CombBLAS 
CMake version = 3.17
intel/2023.2.0
cray-mpich/8.1.28
```

## Setup
1. Clone DistGraph repository.
```bash
git clone https://github.com/HipGraph/DistGraph.git
```

2. Build DistGraph library using the instructions in [its guide](https://github.com/HipGraph/DistGraph/blob/master/README.MD#compile).

3. Export the root of the Distgraph library as follows.
```bash
export DISTGRAPH_ROOT=<path to DistGraph root>
```

4. Now clone the DistGraph-Playground repository.
```bash
git clone https://github.com/dhaura/DistGraph-PlayGround.git
```

5. To compile the demo code, execute the following commands.
```bash
cd DistGraph
mkdir build
cd build
cmake -DCMAKE_CXX_FLAGS="-fopenmp"  ..
make all
```

## SpMM
The usage of SpMM functionality is illustrated in [spmm_demo.cpp](./spmm_demo.cpp).

- To perform Sparse Matrix–Dense Matrix Multiplication (SpMM) using DistGraph, you can initialize an instance of the `distblas::algo::SpMM` class. This requires two main inputs:
  - A sparse matrix in the `distblas::core::SpMat` format
    - The Distgraph library itself provides a sparse matrix reader which can read a sparse matrix in matrix market format (.mtx) and convert it into the `distblas::core::SpMat` format. Hence, it can be used as below.
    ```cpp
    auto reader = unique_ptr<distblas::io::ParallelIO>(new distblas::io::ParallelIO());
    reader.get()->parallel_read_MM<INDEX_TYPE, int, VALUE_TYPE>(sparse_input_file, shared_sparseMat.get(), false, false);
    ```

  - A dense matrix in the `distblas::core::DenseMat` format
    - A utility function is provided in this demo repository to read a dense matrix from a CSV file and convert it into the required distblas::core::DenseMat format. You can find this function in the [utilities.cpp](./utility/utilities.cpp) under the name `read_dense_csv()`.

- In addition to these matrices, you’ll also need to provide:
  - A `distblas::net::Process3DGrid` object, which serves as the communication grid among distributed processes
    ```cpp
    auto grid = std::unique_ptr<distblas::net::Process3DGrid>(new distblas::net::Process3DGrid(world_size, 1, 1, 1));
    ```
  - Two scalar values: alpha and beta, which control the linear combination in the SpMM operation (i.e., C = alpha * A * B + beta * C)

- Once the sparse and dense input matrices are set up—and an additional dense matrix is allocated to store the output (`dense_mat_output`)—the SpMM operation can be invoked as follows:
    ```cpp
    std::make_unique<distblas::algo::SpMM<INDEX_TYPE, VALUE_TYPE>>(
        grid.get(), 
        shared_sparseMat.get(), 
        dense_mat.get(), 
        dense_mat_output.get(), 
        alpha, 
        beta
    );
    ```
- This creates and executes an SpMM operation and the result is stored in `dense_mat_output`.
- Finally, the output can be written back into a text file by executing the following function on the output matrix.
    ```cpp
    dense_mat_output.get()->print_matrix();
    ```

### Single Node Execution
- The final demo setup can be found in [spmm_demo.cpp](./spmm_demo.cpp). You can run it using the following command, which will execute the SpMM operation on small sample sparse and dense matrices available in the [datasets/spmm](./datasets/spmm/) directory using a single node (shared memory manner). 
    ```bash
    mpirun -n 1 build/spmm_demo -input-sparse datasets/spmm/sp_mat_4.mtx -input-dense datasets/spmm/dense_mat_4_3.csv -dataset spmm_sample -output out_dense_mat -alpha 0.5 -beta 0.5
    ```
    > **Note**: Number of shared memeory threads can be spcified by exporting the following environment variable.
    > - ex: export OMP_NUM_THREADS=16

- If you're using NERSC Perlmutter, you can execute the demo by submitting the job script, [run_spmm.sh](./scripts/run_spmm.sh) through the SLURM workload manager.
    ```bash
    sbatch scripts/run_spmm.sh
    ```

### Multi Node Distributed Execution

- To run this in a distributed manner across multiple nodes, you can use the following command. This will execute the SpMM operation on larger sparse and dense matrices available in the [datasets/spmm](./datasets/spmm/) directory:
    ```bash
    mpirun -n 2 build/spmm_demo -input-sparse datasets/spmm/sp_mat_1600.mtx -input-dense datasets/spmm/dense_mat_1600_128.csv -dataset spmm_sample -output out_dense_mat -alpha 0.5 -beta 0.5
    ```
    > **Note**: Number of nodes are specified by `-n` tag and in this example 2 nodes are utilized.
- If you're using NERSC Perlmutter, you can execute the distributed demo by submitting the job script, [run_spmm.sh](./scripts/run_dist_spmm.sh) through the SLURM workload manager.
    ```bash
    sbatch scripts/run_dist_spmm.sh
    ```

> **Note**: You can also generate custom sparse and dense matrices of any size using the utility scripts available in the [scripts/python](./scripts/python/) directory ([dense_matrix_generator.py](./scripts/python/dense_matrix_generator.py) and [sparse_matrix_generator.py](./scripts/python/sparse_matrix_generator.py)). These scripts allow you to create random input data tailored to your experiments, making it easy to test the SpMM functionality at different scales.

