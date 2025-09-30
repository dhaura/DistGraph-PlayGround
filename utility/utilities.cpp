#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <memory>
#include "core/dense_mat.hpp"

VALUE_TYPE *read_dense_csv(const std::string &filename, INDEX_TYPE &rows, INDEX_TYPE &cols)
{

    std::ifstream file(filename);
    std::string line;
    std::vector<VALUE_TYPE> values;

    rows = 0;
    cols = -1;

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string cell;
        INDEX_TYPE col_count = 0;

        while (std::getline(ss, cell, ','))
        {
            values.push_back(std::stod(cell));
            col_count++;
        }

        if (cols == -1)
            cols = col_count;
        else if (cols != col_count)
        {
            throw std::runtime_error("Inconsistent number of columns in CSV.");
        }

        rows++;
    }

    VALUE_TYPE *data = new VALUE_TYPE[values.size()];
    std::copy(values.begin(), values.end(), data);
    return data;
}
