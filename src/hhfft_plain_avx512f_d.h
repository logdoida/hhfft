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

#ifndef HHFFT_PLAIN_AVX512F_D_H
#define HHFFT_PLAIN_AVX512F_D_H

#include "architecture.h"
#include <vector>
#include <array>

#include "hhfft_base.h"

namespace hhfft
{

// This class is can be used when there is AVX512F support. No hand optimization though

class HHFFT_Plain_AVX512F_D : public HHFFT_Base<double>
{
public:
    HHFFT_Plain_AVX512F_D(size_t n, size_t m);

    virtual void fft_real(const double *in, double *out);
    virtual void ifft_real(const double *in, double *out);
    virtual void convolution_real(const double *in1, const double *in2, double *out);

    static void print_real_matrix(const double *matrix, size_t n, size_t m);
    static void print_complex_matrix(const double *matrix, size_t n, size_t m);

private:

    std::vector<double> calculate_packing_table(size_t n);
    std::vector<double> calculate_factor_table(size_t n);
    std::vector<uint32_t> calculate_bit_reverse_table(size_t n, size_t n_bits);
    std::vector<std::array<uint32_t,2>> calculate_bit_reverse_table_inplace(std::vector<uint32_t> &table_in);

    // Dimensions of the matrix
    size_t n, m;

    // Look-up tables for both dimensions
    std::vector<double> packing_table;
    std::vector<double> factor_table_1;
    std::vector<double> factor_table_2;
    std::vector<uint32_t> bit_reverse_table_1;
    std::vector<uint32_t> bit_reverse_table_2;
    std::vector<std::array<uint32_t,2>>  bit_reverse_table_2_inplace;

};
}

#endif // HHFFT_PLAIN_AVX512F_D_H
