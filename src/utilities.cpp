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
#include <algorithm>

std::vector<size_t> hhfft::index_to_n(size_t i, const std::vector<size_t> &N)
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

std::vector<uint32_t> hhfft::calculate_reorder_table(const std::vector<size_t> &N)
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
