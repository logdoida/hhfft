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

// This header contains some constants and functions that are used elsewhere

#ifndef HHFFT_COMMON_D
#define HHFFT_COMMON_D

#include "hhfft_common.h"

static const double k5_d = cos(2.0*M_PI*1.0/7.0);
static const double k6_d = sin(2.0*M_PI*1.0/7.0);
static const double k7_d = -cos(2.0*M_PI*2.0/7.0);
static const double k8_d = sin(2.0*M_PI*2.0/7.0);
static const double k9_d = -cos(2.0*M_PI*3.0/7.0);
static const double k10_d = sin(2.0*M_PI*3.0/7.0);

static const double coeff_radix_7_cos_d[9] = {
    k5_d, -k7_d, -k9_d,
    -k7_d, -k9_d,  k5_d,
    -k9_d,  k5_d, -k7_d};

static const double coeff_radix_7_sin_d[9] = {
    -k6_d, -k8_d, -k10_d,
    -k8_d, k10_d, k6_d,
    -k10_d, k6_d, -k8_d};


// Packing tables to be used on small problems
#define packing_a_d(i,n) (0.5*cos(2.0*M_PI*(2*i+n)/(4*n))-0.5)
#define packing_b_d(i,n) (-0.5*sin(2.0*M_PI*(2*i+n)/(4*n)))
const double packing_table_2[2] = {-0.5, -0.5};
const double packing_table_4[2] = {-0.5, -0.5};
const double packing_table_6[4] = {-0.5, -0.5, packing_a_d(2,6), packing_b_d(2,6)};
const double packing_table_8[4] = {-0.5, -0.5, packing_a_d(2,8), packing_b_d(2,8)};
const double packing_table_10[6] = {-0.5, -0.5, packing_a_d(2,10), packing_b_d(2,10), packing_a_d(4,10), packing_b_d(4,10)};
const double packing_table_12[6] = {-0.5, -0.5, packing_a_d(2,12), packing_b_d(2,12), packing_a_d(4,12), packing_b_d(4,12)};
const double packing_table_14[8] = {-0.5, -0.5, packing_a_d(2,14), packing_b_d(2,14), packing_a_d(4,14), packing_b_d(4,14), packing_a_d(6,14), packing_b_d(6,14)};
const double packing_table_16[8] = {-0.5, -0.5, packing_a_d(2,16), packing_b_d(2,16), packing_a_d(4,16), packing_b_d(4,16), packing_a_d(6,16), packing_b_d(6,16)};

template<size_t n> static const double *get_packing_table()
{
    switch (n)
    {
        case 2:
            return packing_table_2;
        case 4:
            return packing_table_4;
        case 6:
            return packing_table_6;
        case 8:
            return packing_table_8;
        case 10:
            return packing_table_10;
        case 12:
            return packing_table_12;
        case 14:
            return packing_table_14;
        case 16:
            return packing_table_16;
        default:
            return nullptr;
    }

    return nullptr;
}

#endif
