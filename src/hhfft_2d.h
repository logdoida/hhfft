/*
*   Copyright Jouko Kalmari 2017-2019
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

#ifndef HHFFT_2D_H
#define HHFFT_2D_H

#include "architecture.h"
#include "step_info.h"
#include "aligned_arrays.h"
#include "raders.h"

#include <vector>
#include <array>
#include <memory>

namespace hhfft
{

// This class is responsible of making a plan on how to calculate the the FFT and calls the proper functions
template<typename T>
class HHFFT_2D
{
public:
    HHFFT_2D(size_t n, size_t m, InstructionSet instruction_set = InstructionSet::automatic);

    // Copying currently not allowed. Data and pointers must be copied properly when implemented!
    HHFFT_2D(const HHFFT_2D &other) = delete;
    HHFFT_2D& operator=(const HHFFT_2D &other) = delete;

    // FFT with complex inputs and outputs
    void fft(const T *in, T *out) const;

    // IFFT with complex inputs and outputs
    void ifft(const T *in, T *out) const;

    // Calculates convolution of fourier transformed inputs
    void convolution(const T *in1, const T *in2, T *out) const;

    // Allocate aligned array that contains enough space for the complex input and output data
    T* allocate_memory() const;

    // Return size as number complex values.
    std::array<size_t, 2> get_size() const;

    // Return strides as number complex values.
    std::array<size_t, 2> get_stride() const;

    // Free memory
    static void free_memory(T *data);

    static void print_complex_matrix(const T *data, size_t n, size_t m);

private:

    void set_radix_raders(size_t radix, StepInfo<T> &step, InstructionSet instruction_set);

    // Used for cases when n=1 or m=1
    void plan_vector(size_t n, InstructionSet instruction_set);

    // Dimensions of the matrix (Number of complex numbers)
    size_t n; // Number of rows
    size_t m; // Number of columns

    // Twiddle factors for each level
    std::vector<AlignedVector<T>> twiddle_factors_rows, twiddle_factors_columns;

    // Table that is used in the beginning to reorder the data.
    std::vector<uint32_t> reorder_table_rows, reorder_table_columns;

    // Some algorithms might need extra space that is allocated at the beginning
    size_t temp_data_size = 0;

    // The actual fft plan is a sequence of individual steps
    std::vector<StepInfo<T>> forward_steps;
    std::vector<StepInfo<T>> inverse_steps;

    // On some levels Rader's algorithm might be needed
    std::vector<std::unique_ptr<RadersGeneric<T>>> raders;

    // This is a pointer to a function that performs the convolution
    void (*convolution_function)(const T *, const T *, T *, size_t n);
};

typedef HHFFT_2D<double> HHFFT_2D_D;
typedef HHFFT_2D<float> HHFFT_2D_F;

}

#endif // HHFFT_2D_H
