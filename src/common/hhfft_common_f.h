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

// This header contains some constants and functions that are used elsewhere

#ifndef HHFFT_COMMON_F
#define HHFFT_COMMON_F

static const float k5 = cos(2.0*M_PI*1.0/7.0);
static const float k6 = sin(2.0*M_PI*1.0/7.0);
static const float k7 = -cos(2.0*M_PI*2.0/7.0);
static const float k8 = sin(2.0*M_PI*2.0/7.0);
static const float k9 = -cos(2.0*M_PI*3.0/7.0);
static const float k10 = sin(2.0*M_PI*3.0/7.0);

static const float coeff_radix_7_cos[9] = {
    k5, -k7, -k9,
    -k7, -k9,  k5,
    -k9,  k5, -k7};

static const float coeff_radix_7_sin[9] = {
    -k6, -k8, -k10,
    -k8, k10, k6,
    -k10, k6, -k8};


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
        default:
            return nullptr;
    }

    return nullptr;
}

// This function can help compiler to optimze the code
// TODO this function has already been defined in hhfft_common_d!
template<hhfft::SizeType size_type> inline size_t get_size(size_t size)
{
    if (size_type == hhfft::SizeType::Size1)
    {
        return 1;
    } else if (size_type == hhfft::SizeType::Size2)
    {
        return 2;
    } else if (size_type == hhfft::SizeType::Size4)
    {
        return 4;
    } else
    {
        return size;
    }    
}

#endif
