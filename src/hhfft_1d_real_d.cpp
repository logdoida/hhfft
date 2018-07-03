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

#include "hhfft_1d_real_d.h"
#include "1d_complex/hhfft_1d_complex_d.h"
#include "1d_real/hhfft_1d_real_setter_d.h"

using namespace hhfft;
using hhfft::HHFFT_1D_REAL_D;

double* HHFFT_1D_REAL_D::allocate_memory()
{
    // For real data only n doubles are needed
    // For complex data 2*((n/2)+1) doubles are needed
    size_t size = 2*((n/2)+1)*sizeof(double);

    // Allow the allocation of some extra space for larger arrays
    bool allocate_extra = size >= 64;

    return (double *) allocate_aligned_memory(size, allocate_extra);
}

void HHFFT_1D_REAL_D::free_memory(double *data)
{
    free(data);
}


// Does the planning step
HHFFT_1D_REAL_D::HHFFT_1D_REAL_D(size_t n, InstructionSet instruction_set)
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
    HHFFT_1D_Real_D_set_small_function(step_info_fft, n, true, instruction_set);
    HHFFT_1D_Real_D_set_small_function(step_info_ifft, n, false, instruction_set);
    if (step_info_fft.step_function && step_info_ifft.step_function)
    {
        forward_steps.push_back(step_info_fft);
        inverse_steps.push_back(step_info_ifft);
        return;
    }

    if (n%2 == 0)
    {
        plan_even(instruction_set);
    } else
    {
        plan_odd(instruction_set);
    }    
}

void HHFFT_1D_REAL_D::plan_odd(InstructionSet instruction_set)
{
    //throw(std::runtime_error("HHFFT error: odd n fft size not supported!"));

    // Calculate factorization
    std::vector<size_t> N = calculate_factorization(n);

    // TESTING print factorization
    //for (size_t i = 0; i < N.size(); i++)  { std::cout << N[i] << " ";} std::cout << std::endl;

    // First calculate the reorder table
    reorder_table = calculate_reorder_table(N);

    // Calculate reorder table for inverse fft
    reorder_table_ifft_odd = calculate_reorder_table_ifft_odd(reorder_table, N);

    // TESTING print reorder tables
    //std::cout << "reorder = " << std::endl;
    //for (auto r: reorder_table)  { std::cout << r << " ";} std::cout << std::endl;
    //std::cout << "reorder_table_ifft_odd = " << std::endl;
    //for (auto r: reorder_table_ifft_odd)  { std::cout << r << " ";} std::cout << std::endl;

    // Calculate twiddle factors
    twiddle_factors.push_back(AlignedVector<double>()); // No twiddle factors are needed before the first fft-level
    for (size_t i = 1; i < N.size(); i++)
    {
        AlignedVector<double> w = calculate_twiddle_factors_DIT(i, N);
        twiddle_factors.push_back(w);

        //print_complex_vector(w.data(), w.size()/2);
    }

    ///////// FFT /////////////
    // Put first fft step combined with reordering
    {
        hhfft::StepInfoD step;
        step.radix = N[0];
        step.stride = 1;
        step.repeats = n / step.radix;
        step.data_type_in = hhfft::StepDataType::data_in;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.reorder_table = reorder_table.data();        
        step.start_index_out = 1; // This way there is no need to move data when it is ready
        HHFFT_1D_Real_D_odd_set_function(step, instruction_set);
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
        step.start_index_in = 1;
        step.start_index_out = 1;
        HHFFT_1D_Real_D_odd_set_function(step, instruction_set);
        forward_steps.push_back(step);
    }


    ///////// IFFT /////////////
    // Put first fft step combined with reordering
    {
        hhfft::StepInfoD step;
        step.radix = N[0];
        step.stride = 1;
        step.repeats = (n / step.radix + 1)/2;
        step.data_type_in = hhfft::StepDataType::data_in;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.reorder_table = reorder_table_ifft_odd.data();
        step.norm_factor = 1.0/(double(n));
        step.forward = false;
        HHFFT_1D_Real_D_odd_set_function(step, instruction_set);
        inverse_steps.push_back(step);
    }

    // then put rest fft steps combined with twiddle factor
    for (size_t i = 1; i < N.size(); i++)
    {
        hhfft::StepInfoD step;
        hhfft::StepInfoD &step_prev = inverse_steps.back();
        step.radix = N[i];
        step.stride = step_prev.stride * step_prev.radix;
        step.repeats = ((2*step_prev.repeats - 1) / step.radix + 1)/2;
        step.data_type_in = hhfft::StepDataType::data_out;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.twiddle_factors = twiddle_factors[i].data();
        step.forward = false;
        HHFFT_1D_Real_D_odd_set_function(step, instruction_set);
        inverse_steps.push_back(step);
    }

}

void HHFFT_1D_REAL_D::plan_even(InstructionSet instruction_set)
{
    // FFT is done using complex fft of size n/2
    size_t n_complex = n/2;

    // Calculate factorization
    std::vector<size_t> N = calculate_factorization(n_complex);

    // TESTING print factorization
    //for (size_t i = 0; i < N.size(); i++)  { std::cout << N[i] << " ";} std::cout << std::endl;

    // First calculate the reorder table
    reorder_table = calculate_reorder_table(N);

    // Then calculate inverse reorder table
    reorder_table_inverse = calculate_inverse_reorder_table(reorder_table);

    // Then in-place version of the reorder table
    reorder_table_in_place = calculate_reorder_table_in_place(reorder_table);

    // TESTING print reorder tables
    //std::cout << "reorder = " << std::endl;
    //for (auto r: reorder_table)  { std::cout << r << " ";} std::cout << std::endl;
    //std::cout << "reorder_inverse = " << std::endl;
    //for (auto r: reorder_table_inverse)  { std::cout << r << " ";} std::cout << std::endl;
    //std::cout << "reorder_table_in_place = " << std::endl;
    //for (auto r: reorder_table_in_place)  { std::cout << r << " ";} std::cout << std::endl;

    // Add packing factors
    AlignedVector<double> packing_factors = hhfft::calculate_packing_factors(n);
    twiddle_factors.push_back(packing_factors);

    // Calculate twiddle factors
    // NOTE that a portion of these are always one and they could be removed to decrease memory requirements.
    for (size_t i = 1; i < N.size(); i++)
    {
        AlignedVector<double> w = calculate_twiddle_factors_DIT(i, N);
        twiddle_factors.push_back(w);

        //print_complex_vector(w.data(), w.size()/2);
    }

    ///////// FFT /////////////

    // Put first fft step combined with reordering
    {
        hhfft::StepInfoD step;
        step.radix = N[0];
        step.stride = 1;
        step.repeats = n_complex / step.radix;
        step.data_type_in = hhfft::StepDataType::data_in;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.reorder_table = reorder_table.data();        
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

    // Add complex-to-complex-packed step to fft as the last step
    {
        hhfft::StepInfo<double> step;
        step.repeats = n_complex;
        step.twiddle_factors = twiddle_factors[0].data();
        step.data_type_in = hhfft::StepDataType::data_out;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.norm_factor = 1.0;
        step.forward = true;
        HHFFT_1D_Real_D_set_complex_to_complex_packed_function(step, instruction_set);
        forward_steps.push_back(step);
    }

    ///////// IFFT /////////////

    // Combined complex-packed-to-complex, reordering    
    {
        hhfft::StepInfoD step;
        step.repeats = n_complex;
        step.data_type_in = hhfft::StepDataType::data_in;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.twiddle_factors = twiddle_factors[0].data();
        step.reorder_table = reorder_table_inverse.data();
        step.forward = false;
        step.norm_factor = 1.0/(double(n_complex));
        HHFFT_1D_Real_D_set_complex_to_complex_packed_function(step, instruction_set);
        inverse_steps.push_back(step);
    }

    // Put first ifft step
    {
        hhfft::StepInfoD step;
        step.radix = N[0];
        step.stride = 1;
        step.repeats = n_complex / step.radix;
        step.data_type_in = hhfft::StepDataType::data_out;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.reorder_table = nullptr;
        step.forward = false;
        HHFFT_1D_Complex_D_set_function(step, instruction_set);
        inverse_steps.push_back(step);
    }

    // then put rest ifft steps combined with twiddle factor
    for (size_t i = 1; i < N.size(); i++)
    {
        hhfft::StepInfoD step;
        hhfft::StepInfoD &step_prev = inverse_steps.back();
        step.radix = N[i];
        step.stride = step_prev.stride * step_prev.radix;
        step.repeats = step_prev.repeats / step.radix;
        step.data_type_in = hhfft::StepDataType::data_out;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.twiddle_factors = twiddle_factors[i].data();
        step.forward = false;
        HHFFT_1D_Complex_D_set_function(step, instruction_set);
        inverse_steps.push_back(step);
    }
}



void HHFFT_1D_REAL_D::fft(const double *in, double *out)
{
    // If there is just one step, run it directly
    if (forward_steps.size() == 1)
    {
        forward_steps[0].step_function(in,out,forward_steps[0]);
        return;
    }

    // If transform is made in-place, copy input to a temporary variable
    hhfft::AlignedVector<double> temp_data_in;
    if (in == out)
    {
        size_t nn = n;
        temp_data_in.resize(nn);
        std::copy(in, in + nn, temp_data_in.data());
        in = temp_data_in.data();
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
        //print_complex_vector(data_out[step.data_type_out], n/2 + 1);
    }

    // On odd FFTs the first real value needs to be moved one position
    if ((n & 1) == 1)
    {
        out[0] = out[1];
        out[1] = 0;
    }
}

void HHFFT_1D_REAL_D::ifft(const double *in, double *out)
{
    // If there is just one step, run it directly
    if (inverse_steps.size() == 1)
    {
        inverse_steps[0].step_function(in,out,inverse_steps[0]);
        return;
    }

    // If transform is made in-place, copy input to a temporary variable
    hhfft::AlignedVector<double> temp_data_in;
    if (in == out)
    {
        size_t nn = 2*((n/2)+1);
        temp_data_in.resize(nn);
        std::copy(in, in + nn, temp_data_in.data());
        in = temp_data_in.data();
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
        //print_complex_vector(data_out[step.data_type_out], n/2 + 1);
    }    
}

// Calculates convolution in Fourier space
void HHFFT_1D_REAL_D::convolution(const double *in1, const double *in2, double *out)
{
    convolution_function(in1, in2, out, n/2 + 1);
}

// Prints contents of a 1d-vector that has n numbers (n doubles)
void HHFFT_1D_REAL_D::print_real_vector(const double *data, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        std::cout << data[i] << " ";
    }

    std::cout << std::endl;
}

// Prints contents of a 1d-vector that contains packed complex numbers (n doubles)
void HHFFT_1D_REAL_D::print_complex_packed_vector(const double *data, size_t n)
{
    size_t i1 = 1;
    if (n%2 == 0)
    {
        i1 = 2;
    }
    std::cout << data[0];
    for (size_t i = i1; i < n; i+=2)
    {
        double real = data[i];
        double imag = data[i+1];
        if (imag < 0.0)
            std::cout << " " << real << imag << "i";
        else
            std::cout << " " << real << "+" << imag << "i";
    }
    if (n%2 == 0)
    {
        std::cout << " " << data[1];
    }

}

// Prints contents of a 1d-vector that has n complex numbers (2*n doubles)
void HHFFT_1D_REAL_D::print_complex_vector(const double *data, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        double real = data[2*i];
        double imag = data[2*i+1];
        if (imag < 0.0)
            std::cout << real << imag << "i  ";
        else
            std::cout << real << "+" << imag << "i  ";
    }

    std::cout << std::endl;
}
