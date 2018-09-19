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

#ifndef HHFFT_RADERS_D_H
#define HHFFT_RADERS_D_H

#include "architecture.h"
#include "step_info.h"
#include "aligned_arrays.h"

#include <vector>
#include <array>

namespace hhfft
{

// This class is responsible of making a plan on how to calculate the the FFT and calls the proper functions
class RadersD
{    
public:
    RadersD(size_t n, InstructionSet instruction_set = InstructionSet::automatic);

    // Copying not allowed
    RadersD(const RadersD &other) = delete;
    RadersD& operator=(const RadersD &other) = delete;

    // FFT with complex inputs and outputs
    void fft(double *data) const;

    // IFFT with complex inputs and outputs
    void ifft(double *data) const;

    // Allocate aligned array that contains enough space for the complex input and output data
    double* allocate_memory(size_t scale = 1) const;

    // Free memory
    static void free_memory(double *data);

    // Dimension of the vector (Number of complex numbers)
    size_t n_org, n, n_data_size;

    // Tables that are used in the beginning and end to reorder the data.
    std::vector<uint32_t> reorder_table_raders_inverse;
    std::vector<uint32_t> reorder_table_raders_inverse2;

    AlignedVector<double> fft_b;

    double scale;

private:

    void calculate_fft_b(const std::vector<uint32_t> &reorder_table_inverse, const std::vector<uint32_t> &reorder_table_raders);

    // Size that is allocated from memory
    size_t n_bytes_aligned;

    // Twiddle factors for each level
    std::vector<AlignedVector<double>> twiddle_factors;

    std::vector<uint32_t> reorder_table_inverted;
    //std::vector<uint32_t> reorder_table_raders;   // TODO if this is needed only during initialization, it can be removed from here
    //std::vector<uint32_t> reorder_table_inverse;   // TODO if this is needed only during initialization, it can be removed from here

    // The actual fft plan is a sequence of individual steps
    std::vector<StepInfoD> forward_steps;
    std::vector<StepInfoD> inverse_steps;    
};

}

#endif // HHFFT_RADERS_D_H
