/*
*   Copyright Jouko Kalmari 2017
*
*   This file is part of HHFFT.
*
*   HHFFT is free software: you can redistribute it and/or modify
*   it under the terms of the GNU Lesser General Public License as published by
*   the Free Software Foundation, either version 3 of the License, or
*   (at your option) any later version.
*
*   HHFFT is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
*   GNU Lesser General Public License for more details.
*
*   You should have received a copy of the GNU Lesser General Public License
*   along with HHFFT. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef HHFFT_1D_D_H
#define HHFFT_1D_D_H

#include "architecture.h"
#include "step_info.h"
#include "aligned_arrays.h"

#include <vector>
#include <array>

namespace hhfft
{

// This class is responsible of making a plan on how to calculate the the FFT and calls the proper functions
class HHFFT_1D_D
{
public:
    HHFFT_1D_D(size_t n);

    // Copying currently not allowed. Data and pointers must be copied properly when implemented!
    HHFFT_1D_D(const HHFFT_1D_D &other) = delete;
    HHFFT_1D_D& operator=(const HHFFT_1D_D &other) = delete;

    // FFT with complex inputs and outputs
    void fft(const double *in, double *out);

    // IFFT with complex inputs and outputs
    void ifft(const double *in, double *out);

    // Allocate aligned array that contains enough space for the complex input and output data
    double* allocate_memory();

    // Free memory
    static void free_memory(double *data);

    static void print_complex_vector(const double *data, size_t n);

private:

    // Dimension of the vector (Number of complex numbers)
    size_t n;

    // Twiddle factors for each level
    std::vector<AlignedVector<double>> twiddle_factors;

    // Table that is used in the beginning to reorder the data.
    std::vector<uint32_t> reorder_table;

    // Some algorithms might need extra space that is allocated at the beginning
    size_t temp_data_size = 0;

    // The actual fft plan is a sequence of individual steps
    std::vector<StepInfoD> forward_steps;
    std::vector<StepInfoD> inverse_steps;
};

}

#endif // HHFFT_1D_D_H
