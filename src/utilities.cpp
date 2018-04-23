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

#include "utilities.h"
#include <array>
#include <algorithm>
#include <assert.h>
#include <iostream> //TESTING

// Calculates twiddle factors for a given level for DIT
hhfft::AlignedVector<double> hhfft::calculate_twiddle_factors_DIT(size_t level, const std::vector<size_t> &N)
{
    size_t n_dim = N.size();

    // calculate the number of twiddle factors on this level
    size_t num = 1;
    for(size_t i = 0; i <= level; i++)
    {
        num = num*N[i];
    }
    hhfft::AlignedVector<double> w(num*2);

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

// Calculates twiddle factors for a given level for DIF
hhfft::AlignedVector<double> hhfft::calculate_twiddle_factors_DIF(size_t level, const std::vector<size_t> &N)
{
    size_t n_dim = N.size();

    hhfft::AlignedVector<double> w_temp = calculate_twiddle_factors_DIT(level, N);

    // Re-order twiddle factors
    std::vector<size_t> N_temp(level+1);
    for(size_t i = 0; i <= level; i++)
    {        
        N_temp[level - i] = N[i];
    }

    std::vector<uint32_t> reorder = hhfft::calculate_reorder_table(N_temp);

    size_t num = reorder.size();    
    assert (2*num == w_temp.size());

    hhfft::AlignedVector<double> w(2*num);
    for (size_t i = 0; i < num; i++)
    {
        size_t i2 = reorder[i];
        w[2*i + 0] = w_temp[2*i2 + 0];
        w[2*i + 1] = w_temp[2*i2 + 1];
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
        size_t jj = n_dim-j-1;
        n[jj] = temp % N[j];
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
            temp = temp + n[j]*temp1[j];
        }
        reorder[i] = temp;
    }

    return reorder;
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

    // TODO this could be done more efficiently with the help of another table pointing where each index can be found
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

    // Reordering in a "cycle"
    /*
    size_t N_tot = reorder.size();
    std::vector<uint32_t> reorder_in_place;
    std::vector<bool> covered(N_tot, false); // True for all indices that have been reordered

    size_t ind = 0;
    while (ind < N_tot)
    {
       // Skip indices that have already been covered and that do not need any reordering
       if (!covered[ind] && reorder[ind] != ind)
       {
           size_t ind1 = ind;
           reorder_in_place.push_back(0);  // First place is used as a temporary variable to store the one that is overwritten
           do {
               reorder_in_place.push_back(ind);
               covered[ind] = true;
               ind = reorder[ind];
           } while (ind != ind1);
       } else
       {
           ind++;
       }
    }

    // Also the last one is temporary variable
    if (reorder_in_place.size() > 0)
    {
        reorder_in_place.push_back(0);
    }

    return reorder_in_place;
    */
}

// Finds an efficient factorization (not necassery a prime factorization)
std::vector<size_t> hhfft::calculate_factorization(size_t n, bool use_dif)
{
    std::vector<size_t> factors;

    // This list is the supported factorizations in order of preference
    std::array<size_t, 5> radices = {4,2,3,5,7};
    //std::array<size_t, 4> radices = {2,3,5,7}; // for TESTING use 2 instead of 4

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

    if (use_dif)
    {
        // Reverse the order as last radix in the vector is actually used first
        std::reverse(factors.begin(),factors.end());
    }

    return factors;
}
