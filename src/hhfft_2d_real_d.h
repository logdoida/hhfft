/*
*   Copyright Jouko Kalmari 2017-2018
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

#ifndef HHFFT_2D_REAL_D_H
#define HHFFT_2D_REAL_D_H

#include "architecture.h"
#include "step_info.h"
#include "aligned_arrays.h"

#include <vector>
#include <array>
#include <memory>

namespace hhfft
{

// This class is responsible of making a plan on how to calculate the the FFT and calls the proper functions
class HHFFT_2D_REAL_D
{
public:
    HHFFT_2D_REAL_D(size_t n, size_t m, InstructionSet instruction_set = InstructionSet::automatic);

    // Copying currently not allowed. Data and pointers must be copied properly when implemented!
    HHFFT_2D_REAL_D(const HHFFT_2D_REAL_D &other) = delete;
    HHFFT_2D_REAL_D& operator=(const HHFFT_2D_REAL_D &other) = delete;

    // FFT with complex inputs and outputs
    void fft(const double *in, double *out) const;

    // IFFT with complex inputs and outputs
    void ifft(const double *in, double *out) const;

    // Calculates convolution of fourier transformed inputs
    void convolution(const double *in1, const double *in2, double *out) const;

    // Allocate aligned array that contains enough space for the complex input and output data
    double* allocate_memory() const;

    // Free memory
    static void free_memory(double *data);

    static void print_complex_matrix(const double *data, size_t n, size_t m);
    static void print_real_matrix(const double *data, size_t n, size_t m);

private:

    void set_radix_raders(size_t radix, StepInfoD &step, InstructionSet instruction_set);

    void plan_odd(InstructionSet instruction_set);
    void plan_even(InstructionSet instruction_set);

    // Used for cases when n=1 or m=1
    void plan_vector(size_t n, InstructionSet instruction_set, bool is_column);

    // Dimensions of the matrix (Number of complex numbers)
    size_t n; // Number of rows
    size_t m; // Number of columns

    // Twiddle factors for each level
    std::vector<AlignedVector<double>> twiddle_factors_rows, twiddle_factors_columns;

    // Table that is used in the beginning to reorder the data.
    std::vector<uint32_t> reorder_table_rows, reorder_table_columns;

    // Table that is used to reorder the data in-place.
    std::vector<uint32_t> reorder_table_in_place_rows;

    // Reorder table used in the ifft for odd sizes
    std::vector<uint32_t> reorder_table_ifft_odd_rows;

    // Some algorithms might need extra space that is allocated at the beginning
    size_t temp_data_size = 0;

    // The actual fft plan is a sequence of individual steps
    std::vector<StepInfoD> forward_steps;
    std::vector<StepInfoD> inverse_steps;

    // On some levels Rader's algorithm might be needed
    std::vector<std::unique_ptr<RadersD>> raders;

    // This is a pointer to a function that performs the convolution
    void (*convolution_function)(const double *, const double *, double *, size_t n);
};

}

#endif // HHFFT_2D_D_H
