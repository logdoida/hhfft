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

#include <iostream>
#include <cmath>
#include <algorithm>
#include <assert.h>

#include "architecture.h"
#include "utilities.h"

#include "hhfft_1d_d.h"
#include "1d_complex/hhfft_1d_complex_d.h"

using namespace hhfft;
using hhfft::HHFFT_1D_D;

double* HHFFT_1D_D::allocate_memory()
{
    return (double *) allocate_aligned_memory(2*n*sizeof(double));
}

void HHFFT_1D_D::free_memory(double *data)
{
    free(data);
}

// Does the planning step
HHFFT_1D_D::HHFFT_1D_D(size_t n, InstructionSet instruction_set)
{
    this->n = n;

    // This limitation comes from using uint32 in reorder table
    if (n >= (1ul << 32ul))
    {
        throw(std::runtime_error("HHFFT error: maximum size for the fft size is 2^32 - 1!"));
    }

    // Define instruction set if needed
    if (instruction_set == InstructionSet::automatic)
    {
        instruction_set = hhfft::get_best_instruction_set();
    }

    // Set the convolution function
    convolution_function = HHFFT_1D_Complex_D_set_convolution_function(instruction_set);

    // For small problems, it is better to use a single level function
    StepInfoD step_info_fft, step_info_ifft;
    HHFFT_1D_Complex_D_set_small_function(step_info_fft, n, true, instruction_set);
    HHFFT_1D_Complex_D_set_small_function(step_info_ifft, n, false, instruction_set);
    if (step_info_fft.step_function && step_info_ifft.step_function)
    {
        forward_steps.push_back(step_info_fft);
        inverse_steps.push_back(step_info_ifft);
        return;
    }

    // Calculate factorization
    std::vector<size_t> N = calculate_factorization(n);

    // TESTING print factorization    
    //for (size_t i = 0; i < N.size(); i++)  { std::cout << N[i] << " ";} std::cout << std::endl;

    // First calculate the reorder table
    reorder_table = calculate_reorder_table(N);

    // Then in-place version of the reorder table
    reorder_table_in_place = calculate_reorder_table_in_place(reorder_table);

    // TESTING print reorder tables
    //std::cout << "reorder = " << std::endl;
    //for (auto r: reorder_table)  { std::cout << r << " ";} std::cout << std::endl;
    //std::cout << "reorder_table_in_place = " << std::endl;
    //for (auto r: reorder_table_in_place)  { std::cout << r << " ";} std::cout << std::endl;

    // Calculate twiddle factors
    // NOTE that a portion of these are always one and they could be removed to decrease memory requirements.
    twiddle_factors.push_back(AlignedVector<double>()); // No twiddle factors are needed before the first fft-level
    for (size_t i = 1; i < N.size(); i++)
    {     

        AlignedVector<double> w = calculate_twiddle_factors_DIT(i, N);
        twiddle_factors.push_back(w);

        //print_complex_vector(w.data(), w.size()/2); // TESTING
    }

    // DIT
    // Put first fft step combined with reordering
    {
        hhfft::StepInfoD step;
        step.radix = N[0];
        step.stride = 1;
        step.repeats = n / step.radix;
        step.data_type_in = hhfft::StepDataType::data_in;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.reorder_table = reorder_table.data();
        step.reorder_table_inplace = reorder_table_in_place.data(); // It is possible that data_in = data_out!
        step.reorder_table_inplace_size = reorder_table_in_place.size();
        step.norm_factor = 1.0;
        HHFFT_1D_Complex_D_set_function(step, instruction_set);
        forward_steps.push_back(step);
    }

    // then put rest fft steps combined with twiddle factor
    for (size_t i = 1; i < N.size(); i++)
    {
        hhfft::StepInfoD step;
        hhfft::StepInfoD &step_prev = forward_steps.back();
        step.radix = N[i];
        step.stride = step_prev.stride * step_prev.radix;
        step.repeats = step_prev.repeats / step.radix;
        step.data_type_in = hhfft::StepDataType::data_out;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.twiddle_factors = twiddle_factors[i].data();
        HHFFT_1D_Complex_D_set_function(step, instruction_set);
        forward_steps.push_back(step);
    }

    // Make the inverse steps. They are otherwise the same, but different version of function is called    
    for (size_t i = 0; i < forward_steps.size(); i++)
    {
        auto step = forward_steps[i];
        step.forward = false;        

        // Scaling is be done in reordering step (first or last step)
        if (i == 0)
        {
             step.norm_factor = 1.0/(double(n));
        }

        HHFFT_1D_Complex_D_set_function(step, instruction_set);
        inverse_steps.push_back(step);
    }    
}

void HHFFT_1D_D::fft(const double *in, double *out)
{
    // If there is just one step, run it directly
    if (forward_steps.size() == 1)
    {
        forward_steps[0].step_function(in,out,forward_steps[0]);
        return;
    }

    // Allocate some extra space if needed    
    hhfft::AlignedVector<double> temp_data(temp_data_size);

    // Put all possible input/output data sources here
    const double *data_in[3] = {in, out, temp_data.data()};
    double *data_out[3] = {nullptr, out, temp_data.data()};

    // Run all the steps
    for (auto &step: forward_steps)
    {
        step.step_function(data_in[step.data_type_in] + step.start_index_in, data_out[step.data_type_out] + step.start_index_out, step);

        // TESTING print
        //print_complex_vector(data_out[step.data_type_out], n);
    }
}

void HHFFT_1D_D::ifft(const double *in, double *out)
{    
    // If there is just one step, run it directly
    if (inverse_steps.size() == 1)
    {
        inverse_steps[0].step_function(in,out,inverse_steps[0]);
        return;
    }

    // Allocate some extra space if needed    
    hhfft::AlignedVector<double> temp_data(temp_data_size);

    // Put all possible input/output data sources here
    const double *data_in[3] = {in, out, temp_data.data()};
    double *data_out[3] = {nullptr, out, temp_data.data()};

    // Run all the steps
    for (auto &step: inverse_steps)
    {
        step.step_function(data_in[step.data_type_in] + step.start_index_in, data_out[step.data_type_out] + step.start_index_out, step);

        // TESTING print
        //print_complex_vector(data_out[step.data_type_out], n);
    }
}


// Calculates convolution in Fourier space
void HHFFT_1D_D::convolution(const double *in1, const double *in2, double *out)
{
    convolution_function(in1, in2, out, n);
}

// Prints contents of a 1d-vector that has n complex numbers (2*n doubles)
void HHFFT_1D_D::print_complex_vector(const double *data, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        double real = data[2*i];
        double imag = data[2*i+1];
        if (imag >= 0.0)
            std::cout << real << "+" << imag << "i  ";
        else
            std::cout << real << imag << "i  ";
    }

    std::cout << std::endl;
}
