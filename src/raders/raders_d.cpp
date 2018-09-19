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

#include "raders_d.h"
#include "1d_complex/hhfft_1d_complex_d.h"

using namespace hhfft;
using hhfft::RadersD;

// find the prime factorization of a number
// TODO there are more efficient methods too... Like having a table of all small prime numbers
std::vector<size_t> find_factors(size_t n)
{
    std::vector<size_t> factors;

    size_t k = 2;
    while (n > 1)
    {
        while ((n % k) == 0)
        {
            factors.push_back(k);
            n = n / k;
        }
        k++;
    }

    return factors;
}

// Checks if number is composite of small factors 2,3 or 5
bool has_small_factors(size_t n)
{
    std::array<size_t, 3> small_factors = {2,3,5};

    bool radix_found = true;
    while(radix_found)
    {
        radix_found = false;
        for (auto r: small_factors)
        {
            if(n%r == 0)
            {
                n = n / r;
                radix_found = true;
                break;
            }
        }
    }

    return n == 1;
}

// find all prime factors of a number
std::vector<size_t> find_unique_factors(size_t n)
{
    std::vector<size_t> factors = find_factors(n);
    std::vector<size_t> unique_factors;

    for (size_t f: factors)
    {
        if (unique_factors.size() == 0 || unique_factors.back() != f)
        {
            unique_factors.push_back(f);
        }
    }

    return unique_factors;
}

// calculates x^y mod m
size_t pow_mod(size_t x, size_t y, size_t m)
{
    size_t z = 1;

    // t = x^1 mod m
    size_t t = x % m;

    while (y > 0)
    {
        if (y%2 == 1)
        {
            z = (z*t) % m;
        }
        y = y/2;
        t = (t*t) % m;
    }

    return z;
}

// Finds a primitive root for a prime number
size_t find_primitive_root(size_t n)
{

    //Eulers totient function for prime numbers
    size_t s = n - 1;

    // find unique factors
    std::vector<size_t> p_all = find_unique_factors(s);

    // Try all possible primitive roots
    for (size_t m = 2; m < n; m++)
    {
        bool ok = true;

        // Check that all m^(s/p) != 1 mod n
        for (size_t p: p_all)
        {
            size_t ss = s/p;
            if (pow_mod(m,ss,n) == 1)
            {
                ok = false;
                break;
            }
        }

        // Correct one found!
        if(ok)
        {
            return m;
        }
    }

    // This should never happen
    return 0;
}

// Calculates the fft of other sequence "b"
void RadersD::calculate_fft_b(const std::vector<uint32_t> &reorder_table_inverse, const std::vector<uint32_t> &reorder_table_raders)
{    
    fft_b.resize(2*n);

    double re, im;
    for (size_t i = 0; i < n; i++)
    {
        size_t i1 = reorder_table_inverse[i];
        if (i < n_org - 2)
        {
            double val = reorder_table_raders[n_org - 3 - i];
            hhfft::calculate_exp_neg_2_pi_i(val, n_org, re, im);
            fft_b[2*i1 + 0] = re;
            fft_b[2*i1 + 1] = im;
        }
        if ((i > n - n_org + 1) && (i < n - 1))
        {
            size_t i3 = i - (n - n_org + 1);
            double val = reorder_table_raders[n_org - 3 - i3];
            hhfft::calculate_exp_neg_2_pi_i(val, n_org, re, im);
            fft_b[2*i1 + 0] = re;
            fft_b[2*i1 + 1] = im;
        }
        if ((i == n_org - 2) || (i == n - 1))
        {
            hhfft::calculate_exp_neg_2_pi_i(1.0, n_org, re, im);
            fft_b[2*i1 + 0] = re;
            fft_b[2*i1 + 1] = im;
        }
    }

    // FFT
    fft(fft_b.data());
}


double* RadersD::allocate_memory(size_t scale) const
{
    return (double *) allocate_aligned_memory(n_bytes_aligned * scale);
}

void RadersD::free_memory(double *data)
{
    free(data);
}

// Does the planning step
RadersD::RadersD(size_t n_org, InstructionSet instruction_set)
{
    // Raders should not be used for smaller radices
    if (n_org < 11)
    {
        throw(std::runtime_error("HHFFT error: Rader's algorithm used for too small problem"));
    }

    this->n_org = n_org;

    // Check if using n_org-1 is possible
    if (has_small_factors(n_org - 1))
    {
        n = n_org - 1;
    } else
    {
        // Find a power of two that is atleast 2*n_org-3
        n = 2;
        while (n < 2*n_org-3)
        {
            n = 2*n;
        }
    }

    scale = 1.0/n;

    // Extra space for two complex numbers is allocated
    n_bytes_aligned = calculate_aligned_size(2*(n+2)*sizeof(double));
    n_data_size = n_bytes_aligned/sizeof(double);

    // This limitation comes from using uint32 in reorder table
    if (n >= (1ul << 32ul))
    {
        throw(std::runtime_error("HHFFT error (Raders): maximum size for the fft size is 2^32 - 1!"));
    }

    // Find a primitive root
    size_t g = find_primitive_root(n_org);

    // Calculate factorization
    std::vector<size_t> N = calculate_factorization(n);

    // TESTING print factorization
    //for (size_t i = 0; i < N.size(); i++)  { std::cout << N[i] << " ";} std::cout << std::endl;

    // Calculate the reorder table
    std::vector<uint32_t> reorder_table = calculate_reorder_table(N);
    std::vector<uint32_t> reorder_table_inverse = calculate_inverse_reorder_table(reorder_table);

    // Calculate reorder table in place "inverted" as input is actually reordered instead of calling ifft
    std::vector<uint32_t> reorder_table_temp(n);
    for (size_t i = 1; i < n; i++)
    {
        reorder_table_temp[i] = n - reorder_table[i];
    }
    reorder_table_inverted = calculate_reorder_table_in_place(reorder_table_temp);

    // Crate reordering table needed for Raders
    std::vector<uint32_t> reorder_table_raders(n_org);
    for (size_t i = 0; i < reorder_table_raders.size() - 1; i++)
    {
        reorder_table_raders[i] = pow_mod(g, i + 1, n_org);
    }
    reorder_table_raders_inverse = calculate_inverse_reorder_table(reorder_table_raders);

    // Combine both reordering tables needed in set-function
    for (size_t i = 1; i < reorder_table_raders_inverse.size(); i++)
    {
        reorder_table_raders_inverse[i] = reorder_table_inverse[reorder_table_raders_inverse[i]];
    }

    // First input value is stored to the extra space allocated in the end
    reorder_table_raders_inverse[0] = n + 1;

    // Create reordering table needed in the end
    std::vector<uint32_t> reorder_table_raders_temp(n_org);
    for (size_t i = 0; i < reorder_table_raders_temp.size() - 1; i++)
    {
        reorder_table_raders_temp[i] = reorder_table_raders[(2*n_org - i - 5)%(n_org - 1)];
    }
    reorder_table_raders_inverse2 = calculate_inverse_reorder_table(reorder_table_raders_temp);
    reorder_table_raders_inverse2[0] = n;

    // TESTING
    /*
    std::cout << "n = " << n << std::endl;
    std::cout << "primitive root: " << g << std::endl;
    std::cout << "reorder_raders = " << std::endl;
    for (auto r: reorder_table_raders)  { std::cout << r << " ";} std::cout << std::endl;
    std::cout << "reorder_raders_inverse = " << std::endl;
    for (auto r: reorder_table_raders_inverse)  { std::cout << r << " ";} std::cout << std::endl;
    std::cout << "reorder = " << std::endl;
    for (auto r: reorder_table)  { std::cout << r << " ";} std::cout << std::endl;
    std::cout << "reorder_table_inverse = " << std::endl;
    for (auto r: reorder_table_inverse)  { std::cout << r << " ";} std::cout << std::endl;
    std::cout << "reorder_table_inverted = " << std::endl;
    for (auto r: reorder_table_inverted)  { std::cout << r << " ";} std::cout << std::endl;
    std::cout << "reorder_table_raders_inverse2 = " << std::endl;
    for (auto r: reorder_table_raders_inverse2)  { std::cout << r << " ";} std::cout << std::endl;
    */

    // Calculate twiddle factors
    // NOTE that a portion of these are always one and they could be removed to decrease memory requirements.
    twiddle_factors.push_back(AlignedVector<double>()); // No twiddle factors are needed before the first fft-level
    for (size_t i = 1; i < N.size(); i++)
    {     
        AlignedVector<double> w = calculate_twiddle_factors_DIT(i, N);
        twiddle_factors.push_back(w);
    }

    // Forward steps
    // Put first fft step
    {
        hhfft::StepInfoD step;
        step.radix = N[0];
        step.stride = 1;
        step.repeats = n / step.radix;
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
        step.twiddle_factors = twiddle_factors[i].data();
        HHFFT_1D_Complex_D_set_function(step, instruction_set);
        forward_steps.push_back(step);
    }

    // Inverse steps
    // Inverse steps are identical to forward steps, except that reordering inplace is done first
    if (reorder_table_inverted.size() > 0)
    {
        hhfft::StepInfoD step;
        step.reorder_table_inplace = reorder_table_inverted.data();
        step.reorder_table_inplace_size = reorder_table_inverted.size();
        HHFFT_1D_Complex_D_set_reorder_function(step, instruction_set);
        inverse_steps.push_back(step);
    }

    for (auto &step: forward_steps)
    {
        inverse_steps.push_back(step);
    }

    // Calculate fft(b)
    calculate_fft_b(reorder_table_inverse, reorder_table_raders);
}

//
void RadersD::fft(double *data) const
{    
    // Run all the steps
    for (auto &step: forward_steps)
    {
        step.step_function(data + step.start_index_in, data + step.start_index_out, step);

        // TESTING print
        //print_complex_vector(data_out[step.data_type_out], n);
    }
}

//
void RadersD::ifft(double *data) const
{
    // Run all the steps
    for (auto &step: inverse_steps)
    {
        step.step_function(data + step.start_index_in, data + step.start_index_out, step);

        // TESTING print
        //print_complex_vector(data_out[step.data_type_out], n);
    }
}
