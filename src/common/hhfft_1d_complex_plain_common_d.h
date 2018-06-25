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

// This header contains some small functions that are used many times

static const double coeff_radix_2[8] = {
    1, 0, 1, 0,
    1, 0, -1, 0};

static const double k0 = 0.5*sqrt(3);
static const double coeff_radix_3[18] = {
    1, 0, 1, 0, 1, 0,
    1, 0, -0.5, -k0, -0.5, k0,
    1, 0, -0.5,  k0, -0.5, -k0};

static const double coeff_radix_4[32] = {
    1, 0,  1,  0,  1, 0,  1,  0,
    1, 0,  0, -1, -1, 0,  0,  1,
    1, 0, -1,  0,  1, 0, -1,  0,
    1, 0,  0,  1, -1, 0,  0, -1};

static const double k1 = cos(2.0*M_PI*1.0/5.0);
static const double k2 = sin(2.0*M_PI*1.0/5.0);
static const double k3 = -cos(2.0*M_PI*2.0/5.0);
static const double k4 = sin(2.0*M_PI*2.0/5.0);
static const double coeff_radix_5[50] = {
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0,  k1, -k2, -k3, -k4, -k3,  k4,  k1,  k2,
    1, 0, -k3, -k4,  k1,  k2,  k1, -k2, -k3,  k4,
    1, 0, -k3,  k4,  k1, -k2,  k1,  k2, -k3, -k4,
    1, 0,  k1,  k2, -k3,  k4, -k3, -k4,  k1, -k2};

static const double coeff_radix_6[72] = {
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 0.5, -k0, -0.5, -k0, -1, 0, -0.5, k0, 0.5, k0,
    1, 0, -0.5, -k0, -0.5, k0, 1, 0, -0.5, -k0, -0.5, k0,
    1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, -2.3887e-15,
    1, 0, -0.5, k0, -0.5, -k0, 1, 0, -0.5, k0, -0.5, -k0,
    1, 0, 0.5, k0, -0.5, k0, -1, 0, -0.5, -k0, 0.5, -k0,
};

static const double k5 = cos(2.0*M_PI*1.0/7.0);
static const double k6 = sin(2.0*M_PI*1.0/7.0);
static const double k7 = -cos(2.0*M_PI*2.0/7.0);
static const double k8 = sin(2.0*M_PI*2.0/7.0);
static const double k9 = -cos(2.0*M_PI*3.0/7.0);
static const double k10 = sin(2.0*M_PI*3.0/7.0);
static const double coeff_radix_7[98] = {
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0,  k5, -k6, -k7, -k8, -k9, -k10,-k9,  k10, -k7, k8,  k5,  k6,
    1, 0, -k7, -k8, -k9,  k10, k5,  k6,  k5, -k6, -k9, -k10, -k7, k8,
    1, 0, -k9, -k10, k5,  k6, -k7, -k8, -k7,  k8,  k5, -k6, -k9,  k10,
    1, 0, -k9,  k10, k5, -k6, -k7,  k8, -k7, -k8,  k5,  k6, -k9, -k10,
    1, 0, -k7,  k8, -k9, -k10, k5, -k6,  k5,  k6, -k9,  k10, -k7, -k8,
    1, 0,  k5,  k6, -k7,  k8, -k9,  k10,-k9, -k10, -k7, -k8, k5, -k6};

static const double k11 = sqrt(0.5);
static const double coeff_radix_8[128] = {
1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
1, 0, k11, -k11, 0, -1, -k11, -k11, -1, 0, -k11, k11, 0, 1, k11, k11,
1, 0, 0, -1, -1, 0, 0, 1, 1, 0, 0, -1, -1, 0, 0, 1,
1, 0, -k11, -k11, 0, 1, k11, -k11, -1, 0, k11, k11, 0, -1, -k11, k11,
1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0,
1, 0, -k11, k11, 0, -1, k11, k11, -1, 0, k11, -k11, 0, 1, -k11, -k11,
1, 0, 0, 1, -1, 0, 0, -1, 1, 0, 0, 1, -1, 0, 0, -1,
1, 0, k11, k11, 0, 1, -k11, k11, -1, 0, -k11, -k11, 0, -1, k11, -k11};


template<size_t radix, bool forward> inline void multiply_coeff(const double *x_in, double *x_out)
{
    const double *coeff = nullptr;

    if (radix == 2)
    {
        coeff = coeff_radix_2;
    } else if (radix == 3)
    {
        coeff = coeff_radix_3;
    } else if (radix == 4)
    {
        coeff = coeff_radix_4;
    } else if (radix == 5)
    {
        coeff = coeff_radix_5;
    } else if (radix == 6)
    {
        coeff = coeff_radix_6;
    } else if (radix == 7)
    {
        coeff = coeff_radix_7;
    } else if (radix == 8)
    {
        coeff = coeff_radix_8;
    }

    for (size_t i = 0; i < radix; i++)
    {
        x_out[2*i + 0] = 0;
        x_out[2*i + 1] = 0;
        for (size_t j = 0; j < radix; j++)
        {
            double a = coeff[2*radix*i + 2*j + 0];
            double b = forward ? coeff[2*radix*i + 2*j + 1]: -coeff[2*radix*i + 2*j + 1];
            x_out[2*i + 0] += a*x_in[2*j + 0] - b*x_in[2*j + 1];
            x_out[2*i + 1] += b*x_in[2*j + 0] + a*x_in[2*j + 1];
        }
    }
}

template<size_t radix, bool forward> inline void multiply_twiddle(const double *x_in, double *x_out, const double *twiddle_factors)
{
    // It is assumed that first twiddle factors are always (1 + 0i)
    x_out[0] = x_in[0];
    x_out[1] = x_in[1];

    // Read in the values used in this step and multiply them with twiddle factors
    for (size_t j = 1; j < radix; j++)
    {
        double x_r = x_in[2*j + 0];
        double x_i = x_in[2*j + 1];
        double w_r = twiddle_factors[2*j + 0];
        double w_i = twiddle_factors[2*j + 1];

        if (forward == 1)
        {
            x_out[2*j + 0] = w_r*x_r - w_i*x_i;
            x_out[2*j + 1] = w_i*x_r + w_r*x_i;
        } else
        {
            x_out[2*j + 0] =  w_r*x_r + w_i*x_i;
            x_out[2*j + 1] = -w_i*x_r + w_r*x_i;
        }
    }
}

template<size_t radix> void multiply_coeff_real_odd_forward(const double *x_in, double *x_out)
{
    // Use the normal function
    double x_temp[2*radix];
    multiply_coeff<radix,true>(x_in, x_temp);

    // And then apply some reordering and conjugation
    for (size_t i = 0; i < radix; i+=2)
    {
        x_out[2*i + 0] = x_temp[2*(i/2) + 0];
        x_out[2*i + 1] = x_temp[2*(i/2) + 1];
    }
    for (size_t i = 1; i < radix; i+=2)
    {
        x_out[2*i + 0] =  x_temp[2*(radix - (i+1)/2) + 0];
        x_out[2*i + 1] = -x_temp[2*(radix - (i+1)/2) + 1];
    }
}


// Packing tables to be used on small problems
#define packing_a(i,n) (0.5*cos(2.0*M_PI*(2*i+n)/(4*n))-0.5)
#define packing_b(i,n) (-0.5*sin(2.0*M_PI*(2*i+n)/(4*n)))
const double packing_table_2[2] = {-0.5, -0.5};
const double packing_table_4[2] = {-0.5, -0.5};
const double packing_table_6[4] = {-0.5, -0.5, packing_a(2,6), packing_b(2,6)};
const double packing_table_8[4] = {-0.5, -0.5, packing_a(2,8), packing_b(2,8)};
const double packing_table_10[6] = {-0.5, -0.5, packing_a(2,10), packing_b(2,10), packing_a(4,10), packing_b(4,10)};
const double packing_table_12[6] = {-0.5, -0.5, packing_a(2,12), packing_b(2,12), packing_a(4,12), packing_b(4,12)};
const double packing_table_14[8] = {-0.5, -0.5, packing_a(2,14), packing_b(2,14), packing_a(4,14), packing_b(4,14), packing_a(6,14), packing_b(6,14)};
const double packing_table_16[8] = {-0.5, -0.5, packing_a(2,16), packing_b(2,16), packing_a(4,16), packing_b(4,16), packing_a(6,16), packing_b(6,16)};

template<size_t n> const double *get_packing_table()
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
    }

    return nullptr;
}

// This function can help compiler to optimze the code
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
