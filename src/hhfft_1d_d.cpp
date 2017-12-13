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

#include <iostream>
#include <cmath>
#include <algorithm>
#include <assert.h>

#include "architecture.h"
#include "utilities.h"
#include "hhfft_1d_d.h"
#include "hhfft_1d_plain_d.h"
#include "hhfft_1d_avx_d.h"

using namespace hhfft;
using hhfft::HHFFT_1D_D;

// Comment out if DIT should be used
#define USE_DIF

// Finds an efficient factorization (not necassery a prime factorization)
std::vector<size_t> calculate_factorization(size_t n)
{
    std::vector<size_t> factors;

#ifdef USE_DIF
    std::array<size_t, 5> radices = {7, 5, 3, 4, 2}; // DIF
    //std::array<size_t, 5> radices = {7, 5, 3, 2}; // DIF: TESTING use 2 instead of 4
#else
    // This list is the supported factorizations in order of preference
    std::array<size_t, 5> radices = {4, 2, 3, 5, 7}; // DIT
    //std::array<size_t, 5> radices = {2, 3, 5, 7}; // DIT: TESTING use 2 instead of 4
#endif

    while(n > 1)
    {
        bool radix_found = false;
        for (auto r: radices)
        {
            if(n%r == 0)
            {
                factors.push_back(r);
                n = n / r;
                radix_found = true;
                break;
            }
        }
        if (!radix_found)
        {
            // TODO these should be combined (?) and taken care of with other algorithms!
            throw(std::runtime_error("HHFFT error: size is not factorizable!"));
        }
    }

    // Reverse the order as last radix in the vector is actually used first    
    std::reverse(factors.begin(),factors.end());

    return factors;
}


void HHFFT_1D_set_function(StepInfoD &step_info, bool DIF)
{
    // TODO this should be done only once
    hhfft::CPUID_info info = hhfft::get_supported_instructions();

#ifdef HHFFT_COMPILED_WITH_AVX512F
    if (info.avx512f)
    {
        if (DIF)
        {
            // TODO add support for avx512f
            // HHFFT_1D_AVX512F_set_function_DIF(step_info);
            // return;
        } else
        {
            // TODO add support for avx512f
            // HHFFT_1D_AVX512F_set_function(step_info);
            // return;
        }
    }
#endif

#ifdef HHFFT_COMPILED_WITH_AVX
    if (info.avx)
    {
        if (DIF)
        {            
            HHFFT_1D_AVX_set_function_DIF(step_info);
            return;
        } else
        {
            HHFFT_1D_AVX_set_function(step_info);
            return;
        }
    }
#endif

    if (DIF)
    {
        HHFFT_1D_Plain_set_function_DIF(step_info);
    } else
    {
        HHFFT_1D_Plain_set_function(step_info);
    }
}

double* HHFFT_1D_D::allocate_memory()
{
    return (double *) allocate_aligned_memory(2*n*sizeof(double));
}

void HHFFT_1D_D::free_memory(double *data)
{
    free(data);
}

// Does the planning step
HHFFT_1D_D::HHFFT_1D_D(size_t n)
{
    this->n = n;

    // This limitation comes from using uint32 in reorder table
    if (n >= (1ul << 32ul))
    {
        throw(std::runtime_error("HHFFT error: maximum size for the fft size is 2^32 - 1!"));
    }

    if (n == 1)
    {
        // TODO add a support to small radices with a single pass dft
        throw(std::runtime_error("HHFFT error: fft size must be larger than 1!"));
    }

    // Calculate factorization
    std::vector<size_t> N = calculate_factorization(n);

    // TESTING print factorization
    for (size_t i = 0; i < N.size(); i++)  { std::cout << N[N.size() - i - 1] << " ";} std::cout << std::endl;

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

#ifdef USE_DIF
        // DIF
        AlignedVector<double> w = calculate_twiddle_factors_DIF(i, N);
#else
        // DIT
        AlignedVector<double> w = calculate_twiddle_factors_DIT(i, N);
#endif
        twiddle_factors.push_back(w);

        //print_complex_vector(w.data(), w.size()/2);
    }

#ifdef USE_DIF
    // DIF
    // Put first fft step    
    hhfft::StepInfoD step1;
    step1.radix = N[N.size() - 1];
    step1.stride = n / step1.radix;
    step1.repeats = 1;
    step1.data_type_in = hhfft::StepDataType::data_in;
    step1.data_type_out = hhfft::StepDataType::data_out;    
    HHFFT_1D_set_function(step1, true);
    forward_steps.push_back(step1);    

    // then put rest fft steps combined with twiddle factor
    for (size_t i = 1; i < N.size(); i++)
    {
        hhfft::StepInfoD step;
        hhfft::StepInfoD &step_prev = forward_steps.back();
        step.radix = N[N.size() - i - 1];
        step.stride = step_prev.stride / step.radix;
        step.repeats = step_prev.repeats * step_prev.radix;
        step.data_type_in = hhfft::StepDataType::data_out;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.twiddle_factors = twiddle_factors[i].data();
        HHFFT_1D_set_function(step, true);
        forward_steps.push_back(step);
    }

    // Last put reordering step (in-place)
    hhfft::StepInfoD step2;
    step2.data_type_in = hhfft::StepDataType::data_out;
    step2.data_type_out = hhfft::StepDataType::data_out;
    step2.reorder_table = reorder_table_in_place.data();
    step2.repeats = reorder_table_in_place.size();
    step2.stride = n;
    step2.norm_factor = 1.0/(double(n));
    HHFFT_1D_set_function(step2, true);
    forward_steps.push_back(step2);    

#else
    // DIT
    // Put reordering step
    hhfft::StepInfoD step1;
    step1.data_type_in = hhfft::StepDataType::data_in;
    step1.data_type_out = hhfft::StepDataType::data_out;
    step1.reorder_table = reorder_table.data();
    step1.repeats = reorder_table.size();
    step1.norm_factor = 1.0/(double(n));
    HHFFT_1D_set_function(step1, false);
    forward_steps.push_back(step1);

    // Put first fft step    
    hhfft::StepInfoD step2;
    step2.radix = N[N.size() - 1];
    step2.stride = 1;
    step2.repeats = n / step2.radix;
    step2.data_type_in = hhfft::StepDataType::data_out;
    step2.data_type_out = hhfft::StepDataType::data_out;
    HHFFT_1D_set_function(step2, false);
    forward_steps.push_back(step2);

    // then put rest fft steps combined with twiddle factor
    for (size_t i = 1; i < N.size(); i++)
    {
        hhfft::StepInfoD step;
        hhfft::StepInfoD &step_prev = forward_steps.back();
        step.radix = N[N.size() - i - 1];
        step.stride = step_prev.stride * step_prev.radix;
        step.repeats = step_prev.repeats / step.radix;
        step.data_type_in = hhfft::StepDataType::data_out;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.twiddle_factors = twiddle_factors[i].data();
        HHFFT_1D_set_function(step, false);
        forward_steps.push_back(step);
    }
#endif


    // Make the inverse steps. They are otherwise the same, but different version of function is called    
    for (auto step: forward_steps)
    {
        step.forward = false;
#ifdef USE_DIF
        HHFFT_1D_set_function(step,true); // DIF
#else
        HHFFT_1D_set_function(step,false); // DIT
#endif
        inverse_steps.push_back(step);
    }
}

void HHFFT_1D_D::fft(const double *in, double *out)
{
    // Allocate some extra space if needed
    std::vector<double> temp_data(temp_data_size);

    // Put all possible input/output data sources here
    const double *data_in[3] = {in, out, temp_data.data()};
    double *data_out[3] = {nullptr, out, temp_data.data()};

    for (auto &step: forward_steps)
    {
        step.step_function(data_in[step.data_type_in] + step.start_index_in, data_out[step.data_type_out] + step.start_index_out, step);
    }
}

void HHFFT_1D_D::ifft(const double *in, double *out)
{
    // Allocate some extra space if needed
    std::vector<double> temp_data(temp_data_size);

    // Put all possible input/output data sources here
    const double *data_in[3] = {in, out, temp_data.data()};
    double *data_out[3] = {nullptr, out, temp_data.data()};

    for (auto &step: inverse_steps)
    {
        step.step_function(data_in[step.data_type_in] + step.start_index_in, data_out[step.data_type_out] + step.start_index_out, step);
    }
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
