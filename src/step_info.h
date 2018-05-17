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

#ifndef HHFFT_STEP_INFO
#define HHFFT_STEP_INFO

#include <cstddef>
#include <stdint.h>

namespace hhfft
{

enum StepDataType {data_in = 0, data_out = 1, temp_data = 2};

// StrideX -> stride = X
// StrideXN -> stride divisible by X
// StrideN -> stride something else
enum SizeType{Size1, Size2, Size4, Size2N, Size4N, SizeN};

template<typename T> struct StepInfo
{
    // This is a pointer to a function that performs some operation to data
    void (*step_function)(const T *, T *, StepInfo &);

    // This constants are used inside the function
    size_t radix = 1;
    size_t stride = 1;
    size_t repeats = 1;
    size_t size = 1;

    // True if fft is done, false if ifft
    bool forward = true;

    // Used in some step in ifft. Equal to 1/N
    T norm_factor = 1.0;

    // Twiddle factors or reorder table might be used in function
    T *twiddle_factors = nullptr;
    uint32_t *reorder_table = nullptr;    
    uint32_t *reorder_table_inplace = nullptr;
    size_t reorder_table_inplace_size = 0;

    // These tell what data is used and where does it start
    size_t start_index_in = 0;
    size_t start_index_out = 0;
    StepDataType data_type_in = data_out;
    StepDataType data_type_out = data_out;
};

typedef StepInfo<double> StepInfoD;

}

#endif // HHFFT_STEP_INFO
