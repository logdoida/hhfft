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

#include "utilities.h"
#include <array>
#include <algorithm>
#include <assert.h>

// calculates cos and -sin for a = 2*M_PI*a/b
template<typename T> inline void calculate_cos_sin(size_t a, size_t b, T &c, T &s)
{
    a = a%b;

    if (a == 0)
    {
        c = 1.0; s = 0.0;
    } else if (a*4 == b)
    {
        c = 0.0; s = -1.0;
    } else if (a*2 == b)
    {
        c = -1.0; s = 0.0;
    } else if (a*4 == b*3)
    {
        c = 0.0; s = 1.0;
    } else
    {
        T angle = 2.0*M_PI*a/b;
        c = cos(angle);
        s = -sin(angle);
    }
}

template<typename T> void hhfft::calculate_exp_neg_2_pi_i(size_t a, size_t b, T &re, T &im)
{
    calculate_cos_sin<T>(a,b,re,im);
}


// Calculates twiddle factors for a given level for DIT
template<typename T> hhfft::AlignedVector<T> hhfft::calculate_twiddle_factors_DIT(size_t level, const std::vector<size_t> &N)
{
    size_t n_dim = N.size();

    // calculate the number of twiddle factors on this level
    size_t num = 1;
    for(size_t i = 0; i <= level; i++)
    {
        num = num*N[i];
    }
    hhfft::AlignedVector<T> w(num*2);

    // calculate a vector that contains [1 N1 N1*N2 N1*N2*N3...]
    std::vector<uint32_t> temp1(n_dim+1);
    temp1[0] = 1;
    for (size_t i = 1; i < n_dim+1; i++)
    {
        temp1[i] = temp1[i-1]*N[n_dim-i];
    }

    size_t N_tot = temp1.back(); // N1*N2*...

    // calculate a vector that contains [1 N(k) N(k)*N(k-1) N(k)*N(k-1)*N(k-2) ...]
    std::vector<uint32_t> temp2(n_dim);
    temp2[0] = 1;
    for (size_t i = 1; i < n_dim; i++)
    {
        temp2[i] = temp2[i-1]*N[i-1];
    }

    for (size_t i = 0; i < num; i++)
    {
        auto n = hhfft::index_to_n(i,N);
        size_t ww = 0;
        for (size_t j = 0; j < level; j++)
        {
            ww = ww + n[level]*temp1[n_dim-level-1]*n[j]*temp2[j];
        }        
        calculate_cos_sin(ww, N_tot, w[2*i], w[2*i+1]);
    }

    return w;
}

std::vector<size_t> hhfft::index_to_n(size_t i, const std::vector<size_t> &N)
{
    size_t n_dim = N.size();
    std::vector<size_t> n(n_dim);

    size_t temp = i;
    for(size_t j = 0; j < n_dim; j++)
    {        
        n[j] = temp % N[j];
        temp = temp/N[j];
    }
    return n;
}

std::vector<uint32_t> hhfft::calculate_reorder_table(const std::vector<size_t> &N)
{
    size_t n_dim = N.size();

    // calculate a vector that contains [1 N1 N1*N2 N1*N2*N3...]
    std::vector<uint32_t> temp1(n_dim+1);
    temp1[0] = 1;
    for (size_t i = 1; i < n_dim+1; i++)
    {
        temp1[i] = temp1[i-1]*N[n_dim-i];
    }

    size_t N_tot = temp1.back(); // N1*N2*...
    std::vector<uint32_t> reorder(N_tot);

    for(size_t i = 0; i < N_tot; i++)
    {
        auto n = index_to_n(i,N);
        uint32_t temp = 0;
        for(size_t j = 0; j < n_dim; j++)
        {
            temp = temp + n[n_dim-j-1]*temp1[j];
        }
        reorder[i] = temp;
    }

    return reorder;
}

std::vector<uint32_t> hhfft::calculate_inverse_reorder_table(const std::vector<uint32_t> &reorder)
{
    size_t n = reorder.size();
    std::vector<uint32_t> reorder_inverse(n);

    for (size_t i = 0; i < n; i++)
    {
        reorder_inverse[reorder[i]] = i;
    }

    return reorder_inverse;
}

std::vector<uint32_t> hhfft::calculate_reorder_table_ifft_odd(const std::vector<uint32_t> &reorder, const std::vector<size_t> &N)
{
    size_t n_levels = N.size();
    std::vector<uint32_t> reorder_ifft;

    // Add first
    reorder_ifft.push_back(reorder[0]);
    size_t index = 1;
    size_t k = 1;

    // Loop over all levels
    for (size_t level = 0; level < n_levels; level++)
    {
        // New ones to be added (and skipped) on this level
        size_t n_new = k*(N[level] - 1)/2;

        // Add new ones
        for (size_t i = 0; i < n_new; i++)
        {
            reorder_ifft.push_back(reorder[index + i]);
        }

        index = index + 2*n_new;
        k = k * N[level];
    }

    return reorder_ifft;
}

std::vector<uint32_t> hhfft::calculate_reorder_table_in_place(const std::vector<uint32_t> &reorder)
{
    // Reordering by swapping
    size_t N_tot = reorder.size();
    std::vector<uint32_t> reorder_in_place;

    std::vector<uint32_t> indices(N_tot); // Current status of the indices
    for (size_t i = 0; i < N_tot; i++)
    {
        indices[i] = i;
    }

    // NOTE this could be done more efficiently with the help of another table pointing where each index can be found
    for (size_t i = 1; i < N_tot - 1; i++)
    {
        size_t i2 = std::find(indices.begin(), indices.end(), reorder[i]) - indices.begin();
        reorder_in_place.push_back(i2);
        std::swap(indices[i],indices[i2]);
    }

    // Last ones can be removed if there is no actual swapping needed
    size_t n_remove = 0;
    size_t n = reorder_in_place.size();
    while (n_remove < n && reorder_in_place[n - n_remove - 1] == n - n_remove)
        n_remove++;

    reorder_in_place.resize(n-n_remove);

    return reorder_in_place;   
}

void hhfft::append_reorder_table(std::vector<uint32_t> &reorder_table, size_t n_extra)
{
    size_t n = reorder_table.size();
    for (size_t i = 0; i < n_extra; i++)
    {
        reorder_table.push_back(n - reorder_table[n_extra - i -1]);
    }
    reorder_table.back() = 0;
}

// Finds an efficient factorization (not necassery a prime factorization)
std::vector<size_t> hhfft::calculate_factorization(size_t n)
{
    std::vector<size_t> factors;

    // This list is the supported factorizations in order of preference
    std::array<size_t, 7> radices = {6,8,4,2,3,5,7};
    //std::array<size_t, 4> radices = {2,3,5,7}; // for TESTING use 2 and 3 instead of 4 or 8 or 6
    //std::array<size_t, 6> radices = {6,4,2,3,5,7}; // for TESTING use 4 instead of 8

    bool radix_found = true;
    while(radix_found)
    {                
        // First try to use the small radices
        radix_found = false;
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
    }

    // Then start searching for larger ones
    size_t r = 11;
    while (n > 1)
    {
        while ((n % r) == 0)
        {
            factors.push_back(r);
            n = n / r;
        }
        r++;
    }

    return factors;
}

// Calculates packing factors used for converting complex packed to real
template <typename T> hhfft::AlignedVector<T> hhfft::calculate_packing_factors(size_t n)
{
    hhfft::AlignedVector<T> w(n/2 + 1);

    for (size_t i = 0; i < n/2; i+=2)
    {
        double c,s;
        calculate_cos_sin(2*i + n, 8*n, c, s);

        w[i+0] = T(-s*s);
        w[i+1] = T(s*c);
    }

    return w;
}

// Instantiations of the functions defined in this class
template hhfft::AlignedVector<float> hhfft::calculate_twiddle_factors_DIT<float>(size_t level, const std::vector<size_t> &N);
template hhfft::AlignedVector<double> hhfft::calculate_twiddle_factors_DIT<double>(size_t level, const std::vector<size_t> &N);

template void hhfft::calculate_exp_neg_2_pi_i<float>(size_t a, size_t b, float &re, float &im);
template void hhfft::calculate_exp_neg_2_pi_i<double>(size_t a, size_t b, double &re, double &im);

template hhfft::AlignedVector<float> hhfft::calculate_packing_factors<float>(size_t n);
template hhfft::AlignedVector<double> hhfft::calculate_packing_factors<double>(size_t n);

