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

static const double cos_1_7_d = cos(2.0*M_PI*1.0/7.0);
static const double sin_1_7_d = sin(2.0*M_PI*1.0/7.0);
static const double cos_2_7_d = cos(2.0*M_PI*2.0/7.0);
static const double sin_2_7_d = sin(2.0*M_PI*2.0/7.0);
static const double cos_3_7_d = cos(2.0*M_PI*3.0/7.0);
static const double sin_3_7_d = sin(2.0*M_PI*3.0/7.0);

static const double coeff_radix_7_cos_d[9] = {
    cos_1_7_d, cos_2_7_d, cos_3_7_d,
    cos_2_7_d, cos_3_7_d,  cos_1_7_d,
    cos_3_7_d,  cos_1_7_d, cos_2_7_d};

static const double coeff_radix_7_sin_d[9] = {
    -sin_1_7_d, -sin_2_7_d, -sin_3_7_d,
    -sin_2_7_d, sin_3_7_d, sin_1_7_d,
    -sin_3_7_d, sin_1_7_d, -sin_2_7_d};


static const double cos_1_11_d = cos(2.0*M_PI*1.0/11.0);
static const double sin_1_11_d = sin(2.0*M_PI*1.0/11.0);
static const double cos_2_11_d = cos(2.0*M_PI*2.0/11.0);
static const double sin_2_11_d = sin(2.0*M_PI*2.0/11.0);
static const double cos_3_11_d = cos(2.0*M_PI*3.0/11.0);
static const double sin_3_11_d = sin(2.0*M_PI*3.0/11.0);
static const double cos_4_11_d = cos(2.0*M_PI*4.0/11.0);
static const double sin_4_11_d = sin(2.0*M_PI*4.0/11.0);
static const double cos_5_11_d = cos(2.0*M_PI*5.0/11.0);
static const double sin_5_11_d = sin(2.0*M_PI*5.0/11.0);

static const double coeff_radix_11_cos_d[25] = {
    cos_1_11_d, cos_2_11_d, cos_3_11_d, cos_4_11_d, cos_5_11_d,
    cos_2_11_d, cos_4_11_d, cos_5_11_d, cos_3_11_d, cos_1_11_d,
    cos_3_11_d, cos_5_11_d, cos_2_11_d, cos_1_11_d, cos_4_11_d,
    cos_4_11_d, cos_3_11_d, cos_1_11_d, cos_5_11_d, cos_2_11_d,
    cos_5_11_d, cos_1_11_d, cos_4_11_d, cos_2_11_d, cos_3_11_d};
    
static const double coeff_radix_11_sin_d[25] = {
    -sin_1_11_d, -sin_2_11_d, -sin_3_11_d, -sin_4_11_d, -sin_5_11_d,
    -sin_2_11_d, -sin_4_11_d, sin_5_11_d, sin_3_11_d, sin_1_11_d,
    -sin_3_11_d, sin_5_11_d, sin_2_11_d, -sin_1_11_d, -sin_4_11_d,
    -sin_4_11_d, sin_3_11_d, -sin_1_11_d, -sin_5_11_d, sin_2_11_d,
    -sin_5_11_d, sin_1_11_d, -sin_4_11_d, sin_2_11_d, -sin_3_11_d};


static const double cos_1_13_d = cos(2.0*M_PI*1.0/13.0);
static const double sin_1_13_d = sin(2.0*M_PI*1.0/13.0);
static const double cos_2_13_d = cos(2.0*M_PI*2.0/13.0);
static const double sin_2_13_d = sin(2.0*M_PI*2.0/13.0);
static const double cos_3_13_d = cos(2.0*M_PI*3.0/13.0);
static const double sin_3_13_d = sin(2.0*M_PI*3.0/13.0);
static const double cos_4_13_d = cos(2.0*M_PI*4.0/13.0);
static const double sin_4_13_d = sin(2.0*M_PI*4.0/13.0);
static const double cos_5_13_d = cos(2.0*M_PI*5.0/13.0);
static const double sin_5_13_d = sin(2.0*M_PI*5.0/13.0);
static const double cos_6_13_d = cos(2.0*M_PI*6.0/13.0);
static const double sin_6_13_d = sin(2.0*M_PI*6.0/13.0);

static const double coeff_radix_13_cos_d[36] = {
    cos_1_13_d, cos_2_13_d, cos_3_13_d, cos_4_13_d, cos_5_13_d, cos_6_13_d,
    cos_2_13_d, cos_4_13_d, cos_6_13_d, cos_5_13_d, cos_3_13_d, cos_1_13_d,
    cos_3_13_d, cos_6_13_d, cos_4_13_d, cos_1_13_d, cos_2_13_d, cos_5_13_d,
    cos_4_13_d, cos_5_13_d, cos_1_13_d, cos_3_13_d, cos_6_13_d, cos_2_13_d,
    cos_5_13_d, cos_3_13_d, cos_2_13_d, cos_6_13_d, cos_1_13_d, cos_4_13_d,
    cos_6_13_d, cos_1_13_d, cos_5_13_d, cos_2_13_d, cos_4_13_d, cos_3_13_d};
    
static const double coeff_radix_13_sin_d[36] = {
    -sin_1_13_d, -sin_2_13_d, -sin_3_13_d, -sin_4_13_d, -sin_5_13_d, -sin_6_13_d,
    -sin_2_13_d, -sin_4_13_d, -sin_6_13_d, sin_5_13_d, sin_3_13_d, sin_1_13_d,
    -sin_3_13_d, -sin_6_13_d, sin_4_13_d, sin_1_13_d, -sin_2_13_d, -sin_5_13_d,
    -sin_4_13_d, sin_5_13_d, sin_1_13_d, -sin_3_13_d, sin_6_13_d, sin_2_13_d,
    -sin_5_13_d, sin_3_13_d, -sin_2_13_d, sin_6_13_d, sin_1_13_d, -sin_4_13_d,
    -sin_6_13_d, sin_1_13_d, -sin_5_13_d, sin_2_13_d, -sin_4_13_d, sin_3_13_d};

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
const double packing_table_22[12] = {-0.5, -0.5, packing_a_d(2,22), packing_b_d(2,22), packing_a_d(4,22), packing_b_d(4,22), packing_a_d(6,22), packing_b_d(6,22), packing_a_d(8,22), packing_b_d(8,22), packing_a_d(10,22), packing_b_d(10,22)};
const double packing_table_26[14] = {-0.5, -0.5, packing_a_d(2,26), packing_b_d(2,26), packing_a_d(4,26), packing_b_d(4,26), packing_a_d(6,26), packing_b_d(6,26), packing_a_d(8,26), packing_b_d(8,26), packing_a_d(10,26), packing_b_d(10,26), packing_a_d(12,26), packing_b_d(12,26)};

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
        case 22:
            return packing_table_22;
        case 26:
            return packing_table_26;
        default:
            return nullptr;
    }

    return nullptr;
}

#endif
