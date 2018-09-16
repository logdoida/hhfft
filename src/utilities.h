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

#ifndef HHFFT_UTILITIES
#define HHFFT_UTILITIES

#include <vector>
#include <stddef.h>
#include <cstdint>
#include <aligned_arrays.h>

namespace hhfft
{
    AlignedVector<double> calculate_twiddle_factors_DIT(size_t level, const std::vector<size_t> &N);
    AlignedVector<double> calculate_packing_factors(size_t n);
    std::vector<size_t> index_to_n(size_t i, const std::vector<size_t> &N);
    std::vector<uint32_t> calculate_reorder_table(const std::vector<size_t> &N);
    std::vector<uint32_t> calculate_inverse_reorder_table(const std::vector<uint32_t> &reorder);
    std::vector<uint32_t> calculate_reorder_table_in_place(const std::vector<uint32_t> &reorder);
    std::vector<uint32_t> calculate_reorder_table_ifft_odd(const std::vector<uint32_t> &reorder, const std::vector<size_t> &N);
    std::vector<size_t> calculate_factorization(size_t n);
    void calculate_exp_neg_2_pi_i(size_t a, size_t b, double &re, double &im);
}

#endif // HHFFT_UTILITIES
