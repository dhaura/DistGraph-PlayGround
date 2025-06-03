#include "algo/spmm/spmm.hpp"
#include "core/common.h"
#include "core/sparse_mat.hpp"
#include "core/dense_mat.hpp"
#include "io/parrallel_IO.hpp"
#include "net/data_comm.hpp"

int main(int argc, char **argv)
{

    const int dimension = 128;
    int alpha = 0.5;
    int beta = 0.5;
    std::string input_file = "";
    std::string output_file = "";
    std::string dataset_name = "";

    for (int p = 0; p < argc; p++)
    {
        if (strcmp(argv[p], "-input") == 0)
        {
            input_file = argv[p + 1];
        }
        else if (strcmp(argv[p], "-output") == 0)
        {
            output_file = argv[p + 1];
        }
        else if (strcmp(argv[p], "-alpha") == 0)
        {
            alpha = atof(argv[p + 1]);
        }
        else if (strcmp(argv[p], "-beta") == 0)
        {
            beta = atof(argv[p + 1]);
        }
        else if (strcmp(argv[p], "-dataset") == 0)
        {
            dataset_name = argv[p + 1];
        }
    }

    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Initialize MPI DataTypes.
    initialize_mpi_datatypes<VALUE_TYPE, sp_tuple_max_dim>();

    // Initialize the communication grid.
    auto grid = std::unique_ptr<distblas::net::Process3DGrid>(
        new distblas::net::Process3DGrid(world_size, 1, 1, 1));

    auto shared_sparseMat = shared_ptr<distblas::core::SpMat<INDEX_TYPE, VALUE_TYPE>>(
        new distblas::core::SpMat<INDEX_TYPE, VALUE_TYPE>(grid.get()));

    // Initialize the distributed file reader.
    auto reader = unique_ptr<distblas::io::ParallelIO>(new distblas::io::ParallelIO());
    reader.get()->parallel_read_MM<INDEX_TYPE, int, VALUE_TYPE>(input_file, shared_sparseMat.get(), true);

    auto localARows = divide_and_round_up(shared_sparseMat.get()->gRows, grid.get()->col_world_size);
    auto localBRows = divide_and_round_up(shared_sparseMat.get()->gCols, grid.get()->col_world_size);

    shared_sparseMat.get()->batch_size = localARows;
    shared_sparseMat.get()->proc_row_width = localARows;
    shared_sparseMat.get()->proc_col_width = localBRows;

    auto dense_mat = std::make_unique<DenseMat<INDEX_TYPE, VALUE_TYPE>>(grid.get(), shared_sparseMat.get()->proc_row_width, dimension);
    auto dense_mat_output = std::make_unique<DenseMat<INDEX_TYPE, VALUE_TYPE>>(grid.get(), shared_sparseMat.get()->proc_row_width, dimension);
    auto spmm = std::make_unique<distblas::algo::SpMM<INDEX_TYPE, VALUE_TYPE>>(
        grid.get(), shared_sparseMat.get(), dense_mat.get(), dense_mat_output.get(), alpha, beta);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        std::cout << "SpMM calulation is completed." << std::endl;
    }

    MPI_Finalize();
    return 0;
}
