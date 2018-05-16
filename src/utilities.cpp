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


// calculates cos and -sin for a = 2*M_PI*a/b
inline void calculate_cos_sin(size_t a, size_t b, double &c, double &s)
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
        double angle = 2.0*M_PI*a/b;
        c = cos(angle);
        s = -sin(angle);
    }
}


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
std::vector<size_t> hhfft::calculate_factorization(size_t n)
{
    std::vector<size_t> factors;

    // This list is the supported factorizations in order of preference
    std::array<size_t, 6> radices = {8,4,2,3,5,7};
    //std::array<size_t, 5> radices = {4,2,3,5,7}; // for TESTING use 4 instead of 8
    //std::array<size_t, 4> radices = {2,3,5,7}; // for TESTING use 2 instead of 4 or 8

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

    return factors;
}

// Calculates packing factors used for converting complex packed to real
hhfft::AlignedVector<double> hhfft::calculate_packing_factors(size_t n)
{
    hhfft::AlignedVector<double> w(n);

    for (size_t i = 0; i < n/2; i+=2)
    {
        //double c = cos(0.25*M_PI*(2.0/n*i + 1.0));
        //double s = sin(0.25*M_PI*(2.0/n*i + 1.0));
        //w[i+0] = -s*s;
        //w[i+1] = -s*c;

        double c,s;
        calculate_cos_sin(2*i + n, 8*n, c, s);

        w[i+0] = -s*s;
        w[i+1] = s*c;
    }

    return w;
}
