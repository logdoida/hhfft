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

#include <iostream>
#include <cmath>
#include <algorithm>
#include <assert.h>

#include "architecture.h"
#include "utilities.h"

#include "hhfft_1d_real.h"
#include "1d_complex/hhfft_1d_complex_setter.h"
#include "1d_real/hhfft_1d_real_setter.h"

using namespace hhfft;
using hhfft::HHFFT_1D_REAL;

template<typename T> T* HHFFT_1D_REAL<T>::allocate_memory() const
{
    // For real data only n Ts are needed
    // For complex data 2*((n/2)+1) Ts are needed
    size_t size = 2*((n/2)+1)*sizeof(T);

    // Allow the allocation of some extra space for larger arrays
    bool allocate_extra = size >= 64;

    return (T *) allocate_aligned_memory(size, allocate_extra);
}

template<typename T> void HHFFT_1D_REAL<T>::free_memory(T *data)
{
    free(data);
}

// Does the planning step
template<typename T> HHFFT_1D_REAL<T>::HHFFT_1D_REAL(size_t n, InstructionSet instruction_set)
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
    convolution_function = HHFFT_1D_Complex_set_convolution_function<T>(instruction_set);

    // For small problems, it is better to use a single level function
    StepInfo<T> step_info_fft, step_info_ifft;
    HHFFT_1D_Real_set_small_function<T>(step_info_fft, n, true, instruction_set);
    HHFFT_1D_Real_set_small_function<T>(step_info_ifft, n, false, instruction_set);
    if (step_info_fft.step_function && step_info_ifft.step_function)
    {
        // Data type in/out are set as they might be used in 2d real!
        step_info_fft.data_type_in = hhfft::StepDataType::data_in;
        step_info_fft.data_type_out = hhfft::StepDataType::data_out;
        step_info_ifft.data_type_in = hhfft::StepDataType::data_in;
        step_info_ifft.data_type_out = hhfft::StepDataType::data_out;
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

template<typename T> void HHFFT_1D_REAL<T>::set_radix_raders(size_t radix, StepInfo<T> &step, InstructionSet instruction_set)
{
    if (radix > 8)
    {
        // Use Rader's algorithm instead
        raders.push_back(std::unique_ptr<RadersGeneric<T>>(new RadersGeneric<T>(radix, instruction_set)));
        step.raders = raders.back().get();
    }
}

template<typename T> void HHFFT_1D_REAL<T>::plan_odd(InstructionSet instruction_set)
{
    //throw(std::runtime_error("HHFFT error: odd n fft size not supported!"));

    // Calculate factorization
    std::vector<size_t> N = calculate_factorization(n);

    // TESTING print factorization
    //for (size_t i = 0; i < N.size(); i++)  { std::cout << N[i] << " ";} std::cout << std::endl;

    // First calculate the reorder table
    reorder_table = calculate_reorder_table(N);

    // If Raders one level function is possible, do it here
    if (N.size() == 1)
    {
        StepInfo<T> step_info_fft, step_info_ifft;
        HHFFT_1D_Real_set_1level_raders_function<T>(step_info_fft, true, instruction_set);
        HHFFT_1D_Real_set_1level_raders_function<T>(step_info_ifft, false, instruction_set);
        step_info_fft.radix = n;
        step_info_ifft.radix = n;
        step_info_fft.data_type_in = hhfft::StepDataType::data_in;
        step_info_fft.data_type_out = hhfft::StepDataType::data_out;
        step_info_ifft.data_type_in = hhfft::StepDataType::data_in;
        step_info_ifft.data_type_out = hhfft::StepDataType::data_out;
        set_radix_raders(n, step_info_fft, instruction_set);
        set_radix_raders(n, step_info_ifft, instruction_set);
        forward_steps.push_back(step_info_fft);
        inverse_steps.push_back(step_info_ifft);
        return;
    }

    // Calculate reorder table for inverse fft
    reorder_table_ifft_odd = calculate_reorder_table_ifft_odd(reorder_table, N);

    // TESTING print reorder tables
    //std::cout << "reorder = " << std::endl;
    //for (auto r: reorder_table)  { std::cout << r << " ";} std::cout << std::endl;
    //std::cout << "reorder_table_ifft_odd = " << std::endl;
    //for (auto r: reorder_table_ifft_odd)  { std::cout << r << " ";} std::cout << std::endl;

    // Calculate twiddle factors
    twiddle_factors.push_back(AlignedVector<T>()); // No twiddle factors are needed before the first fft-level
    for (size_t i = 1; i < N.size(); i++)
    {
        AlignedVector<T> w = calculate_twiddle_factors_DIT<T>(i, N);
        twiddle_factors.push_back(w);

        //print_complex_vector(w.data(), w.size()/2);
    }

    ///////// FFT /////////////
    // Put first fft step combined with reordering
    {
        hhfft::StepInfo<T> step;
        set_radix_raders(N[0], step, instruction_set);
        step.radix = N[0];
        step.stride = 1;
        step.repeats = n / step.radix;
        step.data_type_in = hhfft::StepDataType::data_in;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.reorder_table = reorder_table.data();        
        step.start_index_out = 1; // This way there is no need to move data when it is ready
        HHFFT_1D_Real_odd_set_function<T>(step, instruction_set);
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
        step.start_index_in = 1;
        step.start_index_out = 1;
        HHFFT_1D_Real_odd_set_function<T>(step, instruction_set);
        forward_steps.push_back(step);
    }


    ///////// IFFT /////////////
    // Put first fft step combined with reordering
    {
        hhfft::StepInfo<T> step;
        set_radix_raders(N[0], step, instruction_set);
        step.radix = N[0];
        step.stride = 1;
        step.repeats = (n / step.radix + 1)/2;
        step.data_type_in = hhfft::StepDataType::data_in;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.reorder_table = reorder_table_ifft_odd.data();
        step.norm_factor = 1.0/(T(n));
        step.forward = false;
        HHFFT_1D_Real_odd_set_function<T>(step, instruction_set);
        inverse_steps.push_back(step);
    }

    // then put rest fft steps combined with twiddle factor
    for (size_t i = 1; i < N.size(); i++)
    {
        hhfft::StepInfo<T> step;
        hhfft::StepInfo<T> &step_prev = inverse_steps.back();
        set_radix_raders(N[i], step, instruction_set);
        step.radix = N[i];
        step.stride = step_prev.stride * step_prev.radix;
        step.repeats = ((2*step_prev.repeats - 1) / step.radix + 1)/2;
        step.data_type_in = hhfft::StepDataType::data_out;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.twiddle_factors = twiddle_factors[i].data();
        step.forward = false;
        HHFFT_1D_Real_odd_set_function<T>(step, instruction_set);
        inverse_steps.push_back(step);
    }

}

template<typename T> void HHFFT_1D_REAL<T>::plan_even(InstructionSet instruction_set)
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

    // TESTING print reorder tables
    //std::cout << "reorder = " << std::endl;
    //for (auto r: reorder_table)  { std::cout << r << " ";} std::cout << std::endl;
    //std::cout << "reorder_inverse = " << std::endl;
    //for (auto r: reorder_table_inverse)  { std::cout << r << " ";} std::cout << std::endl;
    //std::cout << "reorder_table_in_place = " << std::endl;
    //for (auto r: reorder_table_in_place)  { std::cout << r << " ";} std::cout << std::endl;

    // Add packing factors
    AlignedVector<T> packing_factors = hhfft::calculate_packing_factors<T>(n);
    twiddle_factors.push_back(packing_factors);

    // Calculate twiddle factors
    // NOTE that a portion of these are always one and they could be removed to decrease memory requirements.
    for (size_t i = 1; i < N.size(); i++)
    {
        AlignedVector<T> w = calculate_twiddle_factors_DIT<T>(i, N);
        twiddle_factors.push_back(w);

        //print_complex_vector(w.data(), w.size()/2);
    }

    ///////// FFT /////////////

    // Put first fft step combined with reordering
    {
        hhfft::StepInfo<T> step;
        set_radix_raders(N[0], step, instruction_set);
        step.radix = N[0];
        step.stride = 1;
        step.repeats = n_complex / step.radix;
        step.data_type_in = hhfft::StepDataType::data_in;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.reorder_table = reorder_table.data();        
        step.norm_factor = 1.0;
        HHFFT_1D_Complex_set_function<T>(step, instruction_set);
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
        HHFFT_1D_Complex_set_function<T>(step, instruction_set);
        forward_steps.push_back(step);
    }

    // Add complex-to-complex-packed step to fft as the last step
    {
        hhfft::StepInfo<T> step;
        step.repeats = n_complex;
        step.twiddle_factors = twiddle_factors[0].data();
        step.data_type_in = hhfft::StepDataType::data_out;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.norm_factor = 1.0;
        step.forward = true;
        HHFFT_1D_Real_set_complex_to_complex_packed_function<T>(step, instruction_set);
        forward_steps.push_back(step);
    }

    ///////// IFFT /////////////

    // Combined complex-packed-to-complex, reordering    
    {
        hhfft::StepInfo<T> step;
        step.repeats = n_complex;
        step.data_type_in = hhfft::StepDataType::data_in;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.twiddle_factors = twiddle_factors[0].data();
        step.reorder_table = reorder_table_inverse.data();
        step.forward = false;
        step.norm_factor = 1.0/(T(n_complex));
        HHFFT_1D_Real_set_complex_to_complex_packed_function<T>(step, instruction_set);
        inverse_steps.push_back(step);
    }

    // Put first ifft step
    {
        hhfft::StepInfo<T> step;
        set_radix_raders(N[0], step, instruction_set);
        step.radix = N[0];        
        step.stride = 1;
        step.repeats = n_complex / step.radix;
        step.data_type_in = hhfft::StepDataType::data_out;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.reorder_table = nullptr;        
        HHFFT_1D_Complex_set_function<T>(step, instruction_set);
        inverse_steps.push_back(step);
    }

    // then put rest ifft steps combined with twiddle factor
    for (size_t i = 1; i < N.size(); i++)
    {
        hhfft::StepInfo<T> step;
        hhfft::StepInfo<T> &step_prev = inverse_steps.back();
        set_radix_raders(N[i], step, instruction_set);
        step.radix = N[i];
        step.stride = step_prev.stride * step_prev.radix;
        step.repeats = step_prev.repeats / step.radix;
        step.data_type_in = hhfft::StepDataType::data_out;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.twiddle_factors = twiddle_factors[i].data();        
        HHFFT_1D_Complex_set_function<T>(step, instruction_set);
        inverse_steps.push_back(step);
    }
}



template<typename T> void HHFFT_1D_REAL<T>::fft(const T *in, T *out) const
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
        size_t nn = n;
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
        //print_complex_vector(data_out[step.data_type_out], n/2 + 1);
    }

    // On odd FFTs the first real value needs to be moved one position
    if ((n & 1) == 1)
    {
        out[0] = out[1];
        out[1] = 0;
    }
}

template<typename T> void HHFFT_1D_REAL<T>::ifft(const T *in, T *out) const
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
        size_t nn = 2*((n/2)+1);
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
        //print_complex_vector(data_out[step.data_type_out], n/2 + 1);
    }    
}

// Calculates convolution in Fourier space
template<typename T> void HHFFT_1D_REAL<T>::convolution(const T *in1, const T *in2, T *out) const
{
    convolution_function(in1, in2, out, n/2 + 1);
}

// Prints contents of a 1d-vector that has n numbers (n Ts)
template<typename T> void HHFFT_1D_REAL<T>::print_real_vector(const T *data, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        std::cout << data[i] << " ";
    }

    std::cout << std::endl;
}

// Prints contents of a 1d-vector that contains packed complex numbers (n Ts)
template<typename T> void HHFFT_1D_REAL<T>::print_complex_packed_vector(const T *data, size_t n)
{
    size_t i1 = 1;
    if (n%2 == 0)
    {
        i1 = 2;
    }
    std::cout << data[0];
    for (size_t i = i1; i < n; i+=2)
    {
        T real = data[i];
        T imag = data[i+1];
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

// Prints contents of a 1d-vector that has n complex numbers (2*n Ts)
template<typename T> void HHFFT_1D_REAL<T>::print_complex_vector(const T *data, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        T real = data[2*i];
        T imag = data[2*i+1];
        if (imag < 0.0)
            std::cout << real << imag << "i  ";
        else
            std::cout << real << "+" << imag << "i  ";
    }

    std::cout << std::endl;
}


// Explicitly instantiate double and float versions
template class hhfft::HHFFT_1D_REAL<double>;
template class hhfft::HHFFT_1D_REAL<float>;
