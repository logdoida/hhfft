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
#include "1d_complex/hhfft_1d_complex_f.h"

using namespace hhfft;
using hhfft::HHFFT_1D;

template<typename T> T* HHFFT_1D<T>::allocate_memory() const
{
    return (T *) allocate_aligned_memory(2*n*sizeof(T));
}

template<typename T> void HHFFT_1D<T>::free_memory(T *data)
{
    free(data);
}

template<typename T> void HHFFT_1D<T>::set_radix_raders(size_t radix, StepInfo<T> &step, InstructionSet instruction_set)
{
    if (radix > 8)
    {
        // Use Rader's algorithm instead
        raders.push_back(std::unique_ptr<RadersGeneric<T>>(new RadersGeneric<T>(radix, instruction_set)));
        step.raders = raders.back().get();        
    }
}

// Specialized template functions that call the proper setter function
template<typename T> static void complex_set_function(StepInfo<T> &step_info, hhfft::InstructionSet instruction_set);
template<> void complex_set_function<double>(StepInfo<double> &step_info, hhfft::InstructionSet instruction_set)
{
    HHFFT_1D_Complex_D_set_function(step_info, instruction_set);
}
template<> void complex_set_function<float>(StepInfo<float> &step_info, hhfft::InstructionSet instruction_set)
{    
    HHFFT_1D_Complex_F_set_function(step_info, instruction_set);
}

template<typename T> static std::vector<size_t> complex_set_small_function(StepInfo<T> &step_info, size_t n, bool forward, hhfft::InstructionSet instruction_set);
template<> std::vector<size_t> complex_set_small_function<double>(StepInfo<double> &step_info, size_t n, bool forward, hhfft::InstructionSet instruction_set)
{
    return HHFFT_1D_Complex_D_set_small_function(step_info, n, forward, instruction_set);
}
template<> std::vector<size_t> complex_set_small_function<float>(StepInfo<float> &step_info, size_t n, bool forward, hhfft::InstructionSet instruction_set)
{    
    return  HHFFT_1D_Complex_F_set_small_function(step_info, n, forward, instruction_set);
}

template<typename T> static void complex_set_1level_raders_function(StepInfo<T> &step_info, bool forward, hhfft::InstructionSet instruction_set);
template<> void complex_set_1level_raders_function<double>(StepInfo<double> &step_info, bool forward, hhfft::InstructionSet instruction_set)
{
    HHFFT_1D_Complex_D_set_1level_raders_function(step_info, forward, instruction_set);
}
template<> void complex_set_1level_raders_function<float>(StepInfo<float> &step_info, bool forward, hhfft::InstructionSet instruction_set)
{    
    HHFFT_1D_Complex_F_set_1level_raders_function(step_info, forward, instruction_set);
}

template<typename T> void (*set_convolution_function(hhfft::InstructionSet instruction_set))(const T *, const T *, T *, size_t);
template<> void (*set_convolution_function<double>(hhfft::InstructionSet instruction_set))(const double *, const double *, double *, size_t)
{
    return HHFFT_1D_Complex_D_set_convolution_function(instruction_set);
}
template<> void (*set_convolution_function<float>(hhfft::InstructionSet instruction_set))(const float *, const float *, float *, size_t)
{    
    return HHFFT_1D_Complex_F_set_convolution_function(instruction_set);
}

// Does the planning step
template<typename T> HHFFT_1D<T>::HHFFT_1D(size_t n, InstructionSet instruction_set)
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
    convolution_function = set_convolution_function<T>(instruction_set);

    // For small problems, it is better to use a single level function
    StepInfo<T> step_info_fft, step_info_ifft;
    std::vector<size_t> N_small = complex_set_small_function(step_info_fft, n, true, instruction_set);
    complex_set_small_function(step_info_ifft, n, false, instruction_set);
    if (step_info_fft.step_function && step_info_ifft.step_function)
    {
        // twiddle factors are needed if two level function is used
        if (N_small.size() > 1)
        {
            twiddle_factors.push_back(calculate_twiddle_factors_DIT<T>(1, N_small));
            step_info_fft.twiddle_factors = twiddle_factors[0].data();
            step_info_ifft.twiddle_factors = twiddle_factors[0].data();
        }
        forward_steps.push_back(step_info_fft);
        inverse_steps.push_back(step_info_ifft);
        return;
    }

    // Calculate factorization
    std::vector<size_t> N = calculate_factorization(n);

    // If Raders one level function is possible, do it here
    if (N.size() == 1)
    {
        StepInfo<T> step_info_fft, step_info_ifft;
        complex_set_1level_raders_function(step_info_fft, true, instruction_set);
        complex_set_1level_raders_function(step_info_ifft, false, instruction_set);
        step_info_fft.radix = n;
        step_info_ifft.radix = n;
        set_radix_raders(n, step_info_fft, instruction_set);
        set_radix_raders(n, step_info_ifft, instruction_set);
        forward_steps.push_back(step_info_fft);
        inverse_steps.push_back(step_info_ifft);
        return;
    }

    // TESTING print factorization    
    //for (size_t i = 0; i < N.size(); i++)  { std::cout << N[i] << " ";} std::cout << std::endl;

    // First calculate the reorder table
    reorder_table = calculate_reorder_table(N);

    // Add extra values to the end for ifft reordering
    append_reorder_table(reorder_table, n/N.back());

    // TESTING print reorder tables
    //std::cout << "reorder = " << std::endl;
    //for (auto r: reorder_table)  { std::cout << r << " ";} std::cout << std::endl;

    // Calculate twiddle factors
    // NOTE that a portion of these are always one and they could be removed to decrease memory requirements.
    twiddle_factors.push_back(AlignedVector<T>()); // No twiddle factors are needed before the first fft-level
    for (size_t i = 1; i < N.size(); i++)
    {     
        AlignedVector<T> w = calculate_twiddle_factors_DIT<T>(i, N);
        twiddle_factors.push_back(w);        
    }

    // DIT
    // Put first fft step combined with reordering
    {
        hhfft::StepInfo<T> step;
        set_radix_raders(N[0], step, instruction_set);
        step.stride = 1;
        step.radix = N[0];
        step.repeats = n / step.radix;
        step.data_type_in = hhfft::StepDataType::data_in;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.reorder_table = reorder_table.data();
        step.reorder_table_size = reorder_table.size();
        step.norm_factor = 1.0;
        complex_set_function(step, instruction_set);
        forward_steps.push_back(step);
    }

    // then put rest fft steps combined with twiddle factor
    for (size_t i = 1; i < N.size(); i++)
    {
        hhfft::StepInfo<T> step;
        hhfft::StepInfo<T> &step_prev = forward_steps.back();
        set_radix_raders(N[i], step, instruction_set);
        step.radix = N[i];
        step.stride = step_prev.stride * step_prev.radix;
        step.repeats = step_prev.repeats / step.radix;
        step.data_type_in = hhfft::StepDataType::data_out;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.twiddle_factors = twiddle_factors[i].data();
        complex_set_function(step, instruction_set);
        forward_steps.push_back(step);
    }

    // Make the inverse steps. They are otherwise the same, but different version of function is called    
    for (size_t i = 0; i < forward_steps.size(); i++)
    {
        auto step = forward_steps[i];

        // Scaling is be done in reordering step
        if (i == 0)
        {
             step.norm_factor = 1.0/(T(n));
             step.forward = false;
        }

        complex_set_function(step, instruction_set);
        inverse_steps.push_back(step);
    }    
}

template<typename T> void HHFFT_1D<T>::fft(const T *in, T *out) const
{    
    // If there is just one step, run it directly
    if (forward_steps.size() == 1)
    {
        forward_steps[0].step_function(in,out,forward_steps[0]);
        return;
    }

    // If transform is made in-place, copy input to a temporary variable
    hhfft::AlignedVector<T> temp_data_in;
    if (in == out)
    {
        size_t nn = 2*n;
        temp_data_in.resize(nn);
        std::copy(in, in + nn, temp_data_in.data());
        in = temp_data_in.data();
    }

    // Allocate some extra space if needed    
    hhfft::AlignedVector<T> temp_data(temp_data_size);

    // Put all possible input/output data sources here
    const T *data_in[3] = {in, out, temp_data.data()};
    T *data_out[3] = {nullptr, out, temp_data.data()};

    // Run all the steps
    for (auto &step: forward_steps)
    {        
        step.step_function(data_in[step.data_type_in] + step.start_index_in, data_out[step.data_type_out] + step.start_index_out, step);

        // TESTING print
        //print_complex_vector(data_out[step.data_type_out], n);
    }
}

template<typename T> void HHFFT_1D<T>::ifft(const T *in, T *out) const
{    
    // If there is just one step, run it directly
    if (inverse_steps.size() == 1)
    {
        inverse_steps[0].step_function(in,out,inverse_steps[0]);
        return;
    }

    // If transform is made in-place, copy input to a temporary variable
    hhfft::AlignedVector<T> temp_data_in;
    if (in == out)
    {
        size_t nn = 2*n;
        temp_data_in.resize(nn);
        std::copy(in, in + nn, temp_data_in.data());
        in = temp_data_in.data();
    }

    // Allocate some extra space if needed    
    hhfft::AlignedVector<T> temp_data(temp_data_size);

    // Put all possible input/output data sources here
    const T *data_in[3] = {in, out, temp_data.data()};
    T *data_out[3] = {nullptr, out, temp_data.data()};

    // Run all the steps
    for (auto &step: inverse_steps)
    {
        step.step_function(data_in[step.data_type_in] + step.start_index_in, data_out[step.data_type_out] + step.start_index_out, step);

        // TESTING print
        //print_complex_vector(data_out[step.data_type_out], n);
    }
}


// Calculates convolution in Fourier space
template<typename T> void HHFFT_1D<T>::convolution(const T *in1, const T *in2, T *out) const
{
    convolution_function(in1, in2, out, n);
}

// Prints contents of a 1d-vector that has n complex numbers (2*n number)
template<typename T> void HHFFT_1D<T>::print_complex_vector(const T *data, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        T real = data[2*i];
        T imag = data[2*i+1];
        if (imag >= 0.0)
            std::cout << real << "+" << imag << "i  ";
        else
            std::cout << real << imag << "i  ";
    }

    std::cout << std::endl;
}

// Explicitly instantiate double and float versions
template class HHFFT_1D<double>;
template class HHFFT_1D<float>;
