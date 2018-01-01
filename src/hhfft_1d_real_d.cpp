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
#include "hhfft_1d_real_d.h"
#include "hhfft_1d_plain_real_d.h"
#include "hhfft_1d_avx_real_d.h"

using namespace hhfft;
using hhfft::HHFFT_1D_REAL_D;

// Finds an efficient factorization (not necassery a prime factorization)
std::vector<size_t> calculate_factorization_real(size_t n)
{
    std::vector<size_t> factors;

    // This list is the supported factorizations in order of preference
    std::array<size_t, 5> radices = {4, 2, 3, 5, 7};
    //std::array<size_t, 5> radices = {2, 3, 5, 7}; // TESTING use 2 instead of 4

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

void HHFFT_1D_Real_set_function(StepInfoRealD &step_info)
{
    // TODO this should be done only once
    hhfft::CPUID_info info = hhfft::get_supported_instructions();

#ifdef HHFFT_COMPILED_WITH_AVX512F
    if (info.avx512f)
    {
       // TODO add support for avx512f
       // HHFFT_1D_AVX512F_REAL_set_function(step_info);
       // return;
    }
#endif

#ifdef HHFFT_COMPILED_WITH_AVX
    if (info.avx)
    {        
        HHFFT_1D_AVX_real_set_function(step_info);
        return;
    }
#endif

    HHFFT_1D_Plain_real_set_function(step_info);
}

double* HHFFT_1D_REAL_D::allocate_memory()
{
    return (double *) allocate_aligned_memory(n*sizeof(double));
}

void HHFFT_1D_REAL_D::free_memory(double *data)
{
    free(data);
}

// Does the planning step
HHFFT_1D_REAL_D::HHFFT_1D_REAL_D(size_t n)
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
    std::vector<size_t> N = calculate_factorization_real(n);

    // TESTING print factorization
    for (size_t i = 0; i < N.size(); i++)  { std::cout << N[N.size() - i - 1] << " ";} std::cout << std::endl;

    // Calculate the reorder table
    // TODO inplace reordering is to be supported
    reorder_table = calculate_reorder_table(N);

    // TESTING print reorder tables
    //std::cout << "reorder = " << std::endl;
    //for (auto r: reorder_table)  { std::cout << r << " ";} std::cout << std::endl;
    //std::cout << "reorder_table_in_place = " << std::endl;
    //for (auto r: reorder_table_in_place)  { std::cout << r << " ";} std::cout << std::endl;


    // Calculate twiddle factors
    twiddle_factors.push_back(AlignedVector<double>()); // No twiddle factors are needed before the first fft-level
    for (size_t i = 1; i < N.size(); i++)
    {
        // DIT
        AlignedVector<double> w = calculate_twiddle_factors_DIT(i, N);
        twiddle_factors.push_back(w);
        //print_complex_vector(w.data(), w.size()/2);
    }

    // DIT
    // Put first a reordering step
    // Put reordering step
    hhfft::StepInfoRealD step1;
    step1.data_type_in = hhfft::StepDataType::data_in;
    step1.data_type_out = hhfft::StepDataType::data_out;
    step1.reorder_table = reorder_table.data();
    step1.repeats = reorder_table.size();
    step1.norm_factor = 1.0/(double(n));
    HHFFT_1D_Real_set_function(step1);
    forward_steps.push_back(step1);

    // Put the first fft real step.    
    hhfft::StepInfoRealD step2;
    step2.radix = N[N.size() - 1];
    step2.stride = 1;
    step2.repeats = n / step2.radix;
    //step2.reorder_table = reorder_table.data(); // TODO reorder in the first step!
    step2.data_type_in = hhfft::StepDataType::data_out;
    step2.data_type_out = hhfft::StepDataType::data_out;    
    HHFFT_1D_Real_set_function(step2);
    forward_steps.push_back(step2);

    // then put rest dht steps combined with twiddle factor
    for (size_t i = 1; i < N.size(); i++)
    {
        hhfft::StepInfoRealD step;
        hhfft::StepInfoRealD &step_prev = forward_steps.back();
        step.radix = N[N.size() - i - 1];
        step.stride = step_prev.stride * step_prev.radix;
        step.repeats = step_prev.repeats / step.radix;
        step.data_type_in = hhfft::StepDataType::data_out;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.twiddle_factors = twiddle_factors[i].data();
        HHFFT_1D_Real_set_function(step);
        forward_steps.push_back(step);
    }


    // Make the inverse steps. They are otherwise the same, but different version of function might be called
    // TODO
    /*
    for (auto step: forward_steps)
    {
        step.forward = false;
        HHFFT_1D_Real_set_function(step);
        inverse_steps.push_back(step);
    }
    */
}

void HHFFT_1D_REAL_D::fft(const double *in, double *out)
{
    // Allocate some extra space if needed
    AlignedVector<double> temp_data(temp_data_size);

    // Put all possible input/output data sources here
    const double *data_in[3] = {in, out, temp_data.data()};
    double *data_out[3] = {nullptr, out, temp_data.data()};

    for (auto &step: forward_steps)
    {
        step.step_function(data_in[step.data_type_in] + step.start_index_in, data_out[step.data_type_out] + step.start_index_out, step);

        /*
        // TESTING print the result of the step
        for (size_t i = 0; i < step.repeats; i++)
        {
            std::cout << "[";
            print_complex_packed_vector(data_out[step.data_type_out] + i*this->n/step.repeats, this->n/step.repeats);
            std::cout << "] ";
        }
        std::cout << std::endl;
        */
    }
}

void HHFFT_1D_REAL_D::ifft(const double *in, double *out)
{
    // Allocate some extra space if needed
    AlignedVector<double> temp_data(temp_data_size);

    // Put all possible input/output data sources here
    const double *data_in[3] = {in, out, temp_data.data()};
    double *data_out[3] = {nullptr, out, temp_data.data()};

    for (auto &step: inverse_steps)
    {
        step.step_function(data_in[step.data_type_in] + step.start_index_in, data_out[step.data_type_out] + step.start_index_out, step);
    }
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
