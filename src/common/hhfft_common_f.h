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

#ifndef HHFFT_COMMON_F
#define HHFFT_COMMON_F

#include "hhfft_common.h"

static const float cos_1_7_f = float(cos(2.0*M_PI*1.0/7.0));
static const float sin_1_7_f = float(sin(2.0*M_PI*1.0/7.0));
static const float cos_2_7_f = float(cos(2.0*M_PI*2.0/7.0));
static const float sin_2_7_f = float(sin(2.0*M_PI*2.0/7.0));
static const float cos_3_7_f = float(cos(2.0*M_PI*3.0/7.0));
static const float sin_3_7_f = float(sin(2.0*M_PI*3.0/7.0));

static const float coeff_radix_7_cos[9] = {
    cos_1_7_f, cos_2_7_f, cos_3_7_f,
    cos_2_7_f, cos_3_7_f,  cos_1_7_f,
    cos_3_7_f,  cos_1_7_f, cos_2_7_f};

static const float coeff_radix_7_sin[9] = {
    -sin_1_7_f, -sin_2_7_f, -sin_3_7_f,
    -sin_2_7_f, sin_3_7_f, sin_1_7_f,
    -sin_3_7_f, sin_1_7_f, -sin_2_7_f};

static const float cos_1_11_f = float(cos(2.0*M_PI*1.0/11.0));
static const float sin_1_11_f = float(sin(2.0*M_PI*1.0/11.0));
static const float cos_2_11_f = float(cos(2.0*M_PI*2.0/11.0));
static const float sin_2_11_f = float(sin(2.0*M_PI*2.0/11.0));
static const float cos_3_11_f = float(cos(2.0*M_PI*3.0/11.0));
static const float sin_3_11_f = float(sin(2.0*M_PI*3.0/11.0));
static const float cos_4_11_f = float(cos(2.0*M_PI*4.0/11.0));
static const float sin_4_11_f = float(sin(2.0*M_PI*4.0/11.0));
static const float cos_5_11_f = float(cos(2.0*M_PI*5.0/11.0));
static const float sin_5_11_f = float(sin(2.0*M_PI*5.0/11.0));

static const float coeff_radix_11_cos[25] = {
    cos_1_11_f, cos_2_11_f, cos_3_11_f, cos_4_11_f, cos_5_11_f,
    cos_2_11_f, cos_4_11_f, cos_5_11_f, cos_3_11_f, cos_1_11_f,
    cos_3_11_f, cos_5_11_f, cos_2_11_f, cos_1_11_f, cos_4_11_f,
    cos_4_11_f, cos_3_11_f, cos_1_11_f, cos_5_11_f, cos_2_11_f,
    cos_5_11_f, cos_1_11_f, cos_4_11_f, cos_2_11_f, cos_3_11_f};
    
static const float coeff_radix_11_sin[25] = {
    -sin_1_11_f, -sin_2_11_f, -sin_3_11_f, -sin_4_11_f, -sin_5_11_f,
    -sin_2_11_f, -sin_4_11_f, sin_5_11_f, sin_3_11_f, sin_1_11_f,
    -sin_3_11_f, sin_5_11_f, sin_2_11_f, -sin_1_11_f, -sin_4_11_f,
    -sin_4_11_f, sin_3_11_f, -sin_1_11_f, -sin_5_11_f, sin_2_11_f,
    -sin_5_11_f, sin_1_11_f, -sin_4_11_f, sin_2_11_f, -sin_3_11_f};

static const float cos_1_13_f = float(cos(2.0*M_PI*1.0/13.0));
static const float sin_1_13_f = float(sin(2.0*M_PI*1.0/13.0));
static const float cos_2_13_f = float(cos(2.0*M_PI*2.0/13.0));
static const float sin_2_13_f = float(sin(2.0*M_PI*2.0/13.0));
static const float cos_3_13_f = float(cos(2.0*M_PI*3.0/13.0));
static const float sin_3_13_f = float(sin(2.0*M_PI*3.0/13.0));
static const float cos_4_13_f = float(cos(2.0*M_PI*4.0/13.0));
static const float sin_4_13_f = float(sin(2.0*M_PI*4.0/13.0));
static const float cos_5_13_f = float(cos(2.0*M_PI*5.0/13.0));
static const float sin_5_13_f = float(sin(2.0*M_PI*5.0/13.0));
static const float cos_6_13_f = float(cos(2.0*M_PI*6.0/13.0));
static const float sin_6_13_f = float(sin(2.0*M_PI*6.0/13.0));

static const float coeff_radix_13_cos[36] = {
    cos_1_13_f, cos_2_13_f, cos_3_13_f, cos_4_13_f, cos_5_13_f, cos_6_13_f,
    cos_2_13_f, cos_4_13_f, cos_6_13_f, cos_5_13_f, cos_3_13_f, cos_1_13_f,
    cos_3_13_f, cos_6_13_f, cos_4_13_f, cos_1_13_f, cos_2_13_f, cos_5_13_f,
    cos_4_13_f, cos_5_13_f, cos_1_13_f, cos_3_13_f, cos_6_13_f, cos_2_13_f,
    cos_5_13_f, cos_3_13_f, cos_2_13_f, cos_6_13_f, cos_1_13_f, cos_4_13_f,
    cos_6_13_f, cos_1_13_f, cos_5_13_f, cos_2_13_f, cos_4_13_f, cos_3_13_f};
    
static const float coeff_radix_13_sin[36] = {
    -sin_1_13_f, -sin_2_13_f, -sin_3_13_f, -sin_4_13_f, -sin_5_13_f, -sin_6_13_f,
    -sin_2_13_f, -sin_4_13_f, -sin_6_13_f, sin_5_13_f, sin_3_13_f, sin_1_13_f,
    -sin_3_13_f, -sin_6_13_f, sin_4_13_f, sin_1_13_f, -sin_2_13_f, -sin_5_13_f,
    -sin_4_13_f, sin_5_13_f, sin_1_13_f, -sin_3_13_f, sin_6_13_f, sin_2_13_f,
    -sin_5_13_f, sin_3_13_f, -sin_2_13_f, sin_6_13_f, sin_1_13_f, -sin_4_13_f,
    -sin_6_13_f, sin_1_13_f, -sin_5_13_f, sin_2_13_f, -sin_4_13_f, sin_3_13_f};

// Packing tables to be used on small problems
#define packing_a(i,n) float(0.5*cos(2.0*M_PI*(2*i+n)/(4*n))-0.5)
#define packing_b(i,n) float(-0.5*sin(2.0*M_PI*(2*i+n)/(4*n)))
const float packing_table_2_f[2] = {-0.5, -0.5};
const float packing_table_4_f[2] = {-0.5, -0.5};
const float packing_table_6_f[4] = {-0.5, -0.5, packing_a(2,6), packing_b(2,6)};
const float packing_table_8_f[4] = {-0.5, -0.5, packing_a(2,8), packing_b(2,8)};
const float packing_table_10_f[6] = {-0.5, -0.5, packing_a(2,10), packing_b(2,10), packing_a(4,10), packing_b(4,10)};
const float packing_table_12_f[6] = {-0.5, -0.5, packing_a(2,12), packing_b(2,12), packing_a(4,12), packing_b(4,12)};
const float packing_table_14_f[8] = {-0.5, -0.5, packing_a(2,14), packing_b(2,14), packing_a(4,14), packing_b(4,14), packing_a(6,14), packing_b(6,14)};
const float packing_table_16_f[8] = {-0.5, -0.5, packing_a(2,16), packing_b(2,16), packing_a(4,16), packing_b(4,16), packing_a(6,16), packing_b(6,16)};
const float packing_table_22_f[12] = {-0.5, -0.5, packing_a(2,22), packing_b(2,22), packing_a(4,22), packing_b(4,22), packing_a(6,22), packing_b(6,22), packing_a(8,22), packing_b(8,22), packing_a(10,22), packing_b(10,22)};
const float packing_table_26_f[14] = {-0.5, -0.5, packing_a(2,26), packing_b(2,26), packing_a(4,26), packing_b(4,26), packing_a(6,26), packing_b(6,26), packing_a(8,26), packing_b(8,26), packing_a(10,26), packing_b(10,26), packing_a(12,26), packing_b(12,26)};

template<size_t n> static const float *get_packing_table_f()
{
    switch (n)
    {
        case 2:
            return packing_table_2_f;
        case 4:
            return packing_table_4_f;
        case 6:
            return packing_table_6_f;
        case 8:
            return packing_table_8_f;
        case 10:
            return packing_table_10_f;
        case 12:
            return packing_table_12_f;
        case 14:
            return packing_table_14_f;
        case 16:
            return packing_table_16_f;
        case 22:
            return packing_table_22_f;
        case 26:
            return packing_table_26_f;
        default:
            return nullptr;
    }

    return nullptr;
}

#endif
