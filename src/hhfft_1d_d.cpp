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

#include "architecture.h"
#include "hhfft_1d_d.h"
#include "hhfft_1d_plain_d.h"
#include "hhfft_1d_avx_d.h"

using namespace hhfft;
using hhfft::HHFFT_1D_D;

std::vector<size_t> index_to_n(size_t i, const std::vector<size_t> &N)
{
    size_t n_dim = N.size();
    std::vector<size_t> n(n_dim);

    size_t temp = i;
    for(size_t j = 0; j < n_dim; j++)
    {
        size_t jj = n_dim-j-1;
        n[jj] = temp % N[jj];
        temp = temp/N[jj];
    }
    return n;
}

std::vector<uint32_t> calculate_reorder_table(const std::vector<size_t> &N)
{
    size_t n_dim = N.size();

    // calculate a vector that contains [1 N1 N1*N2 N1*N2*N3...]
    std::vector<uint32_t> temp1(n_dim+1);
    temp1[0] = 1;
    for (size_t i = 1; i < n_dim+1; i++)
    {
        temp1[i] = temp1[i-1]*N[i-1];
    }

    size_t N_tot = temp1.back(); // N1*N2*...
    std::vector<uint32_t> reorder(N_tot);

    for(size_t i = 0; i < N_tot; i++)
    {
        auto n = index_to_n(i,N);
        uint32_t temp = 0;
        for(size_t j = 0; j < n_dim; j++)
        {
            temp = temp + n[j]*temp1[j];
        }
        reorder[i] = temp;
    }

    return reorder;
}

// Calculates twiddle factors for a given level
std::vector<double> calculate_twiddle_factors(size_t level, const std::vector<size_t> &N)
{
    size_t n_dim = N.size();

    // calculate the number of twiddle factors on this level
    size_t num = 1;
    for(size_t i = 0; i <= level; i++)
    {
        num = num*N[n_dim - i - 1];
    }
    std::vector<double> w(num*2);

    // calculate a vector that contains [1 N1 N1*N2 N1*N2*N3...]
    std::vector<uint32_t> temp1(n_dim+1);
    temp1[0] = 1;
    for (size_t i = 1; i < n_dim+1; i++)
    {
        temp1[i] = temp1[i-1]*N[i-1];
    }

    size_t N_tot = temp1.back(); // N1*N2*...

    // calculate a vector that contains [1 N(k) N(k)*N(k-1) N(k)*N(k-1)*N(k-2) ...]
    std::vector<uint32_t> temp2(n_dim);
    temp2[0] = 1;
    for (size_t i = 1; i < n_dim; i++)
    {
        temp2[i] = temp2[i-1]*N[n_dim-i];
    }

    for (size_t i = 0; i < num; i++)
    {
        auto n = index_to_n(i,N);
        double ww = 0;
        for (size_t j = 0; j < level; j++)
        {
            ww = ww + n[n_dim-level-1]*temp1[n_dim-level-1]*n[n_dim-j-1]*temp2[j];
        }
        w[2*i]   = cos(-2.0*M_PI*ww/N_tot);
        w[2*i+1] = sin(-2.0*M_PI*ww/N_tot);
    }

    return w;
}

// Finds an efficient factorization (not necassery a prime factorization)
std::vector<size_t> calculate_factorization(size_t n)
{
    std::vector<size_t> factors;

    // This list is the supported factorizations in order of preference
    std::array<size_t, 5> radices = {4, 2, 3, 5, 7};
    //std::array<size_t, 5> radices = {2, 4, 3, 5, 7}; // TESTING use 2 instead of 4

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
    // NOTE for DIF the order should not be reversed!
    std::reverse(factors.begin(),factors.end());

    return factors;
}


void HHFFT_1D_set_function(StepInfoD &step_info)
{
    // TODO this should be done only once
    hhfft::CPUID_info info = hhfft::get_supported_instructions();

#ifdef HHFFT_COMPILED_WITH_AVX512F
    if (info.avx512f)
    {
        // TODO add support for avx512f
        //HHFFT_1D_AVX512F_set_function(step_info);
        // return;
    }
#endif

#ifdef HHFFT_COMPILED_WITH_AVX
    if (info.avx)
    {
        HHFFT_1D_AVX_set_function(step_info);
        return;
    }
#endif

    HHFFT_1D_Plain_set_function(step_info);
}



// Does the planning step
HHFFT_1D_D::HHFFT_1D_D(size_t n)
{
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

    // Calculate twiddle factors
    // NOTE that a portion of these are always one and they could be removed to decrease memory requirements.
    twiddle_factors.push_back(std::vector<double>()); // No twiddle factors are needed before the first fft-level
    for (size_t i = 1; i < N.size(); i++)
    {
        // TODO allocating aligned memory for these could be beneficial
        std::vector<double> w = calculate_twiddle_factors(i, N);
        twiddle_factors.push_back(w);
        //print_complex_vector(w.data(), w.size()/2);
    }


    // Put reordering step    
    hhfft::StepInfoD step1;
    step1.data_type_in = hhfft::StepDataType::data_in;
    step1.data_type_out = hhfft::StepDataType::data_out;
    step1.reorder_table = reorder_table.data();
    step1.repeats = reorder_table.size();
    HHFFT_1D_set_function(step1);
    steps.push_back(step1);

    // Put first fft step
    hhfft::StepInfoD step2;
    step2.radix = N[N.size() - 1];
    step2.stride = 1;
    step2.repeats = n / step2.radix;
    step2.data_type_in = hhfft::StepDataType::data_out;
    step2.data_type_out = hhfft::StepDataType::data_out;    
    HHFFT_1D_set_function(step2);
    steps.push_back(step2);

    // then put rest fft steps combined with twiddle factor
    for (size_t i = 1; i < N.size(); i++)
    {
        hhfft::StepInfoD step;
        hhfft::StepInfoD &step_prev = steps.back();
        step.radix = N[N.size() - i - 1];
        step.stride = step_prev.stride * step_prev.radix;
        step.repeats = step_prev.repeats / step.radix;
        step.data_type_in = hhfft::StepDataType::data_out;
        step.data_type_out = hhfft::StepDataType::data_out;        
        step.twiddle_factors = twiddle_factors[i].data();
        HHFFT_1D_set_function(step);
        steps.push_back(step);
    }    
}

void HHFFT_1D_D::fft(const double *in, double *out)
{
    // Allocate some extra space if needed
    std::vector<double> temp_data(temp_data_size);

    // Put all possible input/output data sources here
    const double *data_in[3] = {in, out, temp_data.data()};
    double *data_out[3] = {nullptr, out, temp_data.data()};

    for (auto &step: steps)
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