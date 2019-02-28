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

// The Plain implementation mainly serves as a reference as sse2 is supported by all x64 cpus
// This header contains some small functions that are used many times

#ifndef HHFFT_COMMON_PLAIN_F
#define HHFFT_COMMON_PLAIN_F

#include "hhfft_common_f.h"

// contains a single complex number: [r i]
typedef struct
{
    float real, imag;
} ComplexF;

// addition, subtraction, multiplication
inline ComplexF operator+(ComplexF a, ComplexF b)
{
    ComplexF out = {a.real + b.real, a.imag + b.imag};
    return out;
}
inline ComplexF operator-(ComplexF a, ComplexF b)
{
    ComplexF out = {a.real - b.real, a.imag - b.imag};
    return out;
}
inline ComplexF operator*(ComplexF a, ComplexF b)
{
    ComplexF out = {a.real * b.real, a.imag * b.imag};
    return out;
}
inline ComplexF operator+=(ComplexF &a, ComplexF b)
{
    a.real += b.real;
    a.imag += b.imag;
    return a;
}
inline ComplexF operator-=(ComplexF &a, ComplexF b)
{
    a.real -= b.real;
    a.imag -= b.imag;
    return a;
}


// Read complex numbers
inline ComplexF load_F(float r, float i)
{
    ComplexF out;
    out.real = r;
    out.imag = i;
    return out;
}
inline ComplexF load_F(const float *v)
{
    return load_F(v[0], v[1]);
}
inline ComplexF broadcast32_F(float x)
{
    return load_F(x,x);
}
// Load only real part, imaginary part is set to zero
inline ComplexF load_real_F(const float *v)
{
    return load_F(v[0],0);
}

// Store
inline void store_F(ComplexF val, float &r, float &i)
{
    r = val.real;
    i = val.imag;
}
inline void store_F(ComplexF val, float *v)
{
    v[0] = val.real;
    v[1] = val.imag;
}
// Store only real part
inline void store_real_F(ComplexF val, float *v)
{
    v[0] = val.real;
}

// Changes signs of [x1 x2] using [s1 s2]. s should contain only 1.0 and -1.0
// NOTE that for sse2 and avx similar functions use 0.0 and -0.0 instead of 1.0 and -1.0
inline ComplexF change_sign_F(ComplexF x, ComplexF s)
{
    return load_F(x.real*s.real, x.imag*s.imag);
}

const ComplexF const1_F = load_F(1.0, -1.0);
const ComplexF const2_F = load_F(-1.0, 1.0);

// Multiplies complex numbers. If other of them changes more frequently, set it to b.
inline ComplexF mul_F(ComplexF a, ComplexF b)
{
    float real = a.real*b.real - a.imag*b.imag;
    float imag = a.real*b.imag + a.imag*b.real;

    return load_F(real, imag);
}

// Calculates a*conj(b)
inline ComplexF mul_conj_F(ComplexF a, ComplexF b)
{
    float real = a.real*b.real + a.imag*b.imag;
    float imag = a.imag*b.real - a.real*b.imag;

    return load_F(real, imag);
}

// Multiplies two complex numbers. The forward means a*b, inverse a*conj(b)
template<bool forward> inline ComplexF mul_w_F(ComplexF a, ComplexF b)
{
    if (forward)
    {
        return mul_F(a,b);
    } else
    {
        return mul_conj_F(a,b);
    }
}

// Complex conjugate
inline ComplexF conj_F(ComplexF x)
{
    return load_F(x.real, -x.imag);
}

// Multiply with i
inline ComplexF mul_i_F(ComplexF x)
{
    return load_F(-x.imag, x.real);
}


template<size_t radix, bool forward> inline __attribute__((always_inline)) void multiply_coeff_F(const ComplexF *x_in, ComplexF *x_out)
{
    // Implementation for radix = 1 (this is actually needed!)
    if (radix == 1)
    {
        x_out[0] = x_in[0];
        return;
    }

    // Implementation for radix = 2
    if (radix == 2)
    {
        x_out[0] = x_in[0] + x_in[1];
        x_out[1] = x_in[0] - x_in[1];
        return;
    }

    // Implementation for radix = 3
    if (radix == 3)
    {
        const ComplexF k0 = broadcast32_F(0.5);
        const ComplexF k1 = broadcast32_F(0.5*sqrt(3.0));
        ComplexF t0 = x_in[1] + x_in[2];
        ComplexF t1 = x_in[0] - k0*t0;
        ComplexF t2 = mul_i_F(k1*(x_in[1] - x_in[2]));

        x_out[0] = x_in[0] + t0;
        if (forward)
        {
            x_out[1] = t1 - t2;
            x_out[2] = t1 + t2;
        } else
        {
            x_out[1] = t1 + t2;
            x_out[2] = t1 - t2;
        }
        return;
    }

    // Implementation for radix = 4
    if (radix == 4)
    {
        ComplexF t0 = x_in[0] + x_in[2];
        ComplexF t1 = x_in[1] + x_in[3];
        ComplexF t2 = x_in[0] - x_in[2];
        ComplexF t3;
        if (forward)
            t3 = mul_i_F(x_in[3] - x_in[1]);
        else
            t3 = mul_i_F(x_in[1] - x_in[3]);

        x_out[0] = t0 + t1;
        x_out[1] = t2 + t3;
        x_out[2] = t0 - t1;
        x_out[3] = t2 - t3;
        return;
    }

    // Implementation for radix = 5
    if (radix == 5)
    {
        const ComplexF k1 = broadcast32_F(cos(2.0*M_PI*1.0/5.0));
        const ComplexF k2 = broadcast32_F(sin(2.0*M_PI*1.0/5.0));
        const ComplexF k3 = broadcast32_F(-cos(2.0*M_PI*2.0/5.0));
        const ComplexF k4 = broadcast32_F(sin(2.0*M_PI*2.0/5.0));

        ComplexF t0 = x_in[1] + x_in[4];
        ComplexF t1 = x_in[2] + x_in[3];
        ComplexF t2 = x_in[1] - x_in[4];
        ComplexF t3 = x_in[2] - x_in[3];
        ComplexF t4 = x_in[0] + k1*t0 - k3*t1;
        ComplexF t5 = x_in[0] + k1*t1 - k3*t0;
        ComplexF t6 = mul_i_F(k2*t2 + k4*t3);
        ComplexF t7 = mul_i_F(k4*t2 - k2*t3);

        x_out[0] = x_in[0] + t0 + t1;
        if (forward)
        {
            x_out[1] = t4 - t6;
            x_out[2] = t5 - t7;
            x_out[3] = t5 + t7;
            x_out[4] = t4 + t6;
        }
        else
        {
            x_out[1] = t4 + t6;
            x_out[2] = t5 + t7;
            x_out[3] = t5 - t7;
            x_out[4] = t4 - t6;
        }
        return;
    }

    // Implementation for radix = 6
    if (radix == 6)
    {
        const ComplexF k0 = broadcast32_F(0.5);
        ComplexF k1;
        if (forward)
            k1 = broadcast32_F(0.5*sqrt(3.0));
        else
            k1 = broadcast32_F(-0.5*sqrt(3.0));

        ComplexF t6 = x_in[2] + x_in[4];
        ComplexF t7 = x_in[1] + x_in[5];
        ComplexF t8 = x_in[0] - k0*t6;
        ComplexF t9 = x_in[3] - k0*t7;
        ComplexF t10 = mul_i_F(k1*(x_in[4] - x_in[2]));
        ComplexF t11 = mul_i_F(k1*(x_in[5] - x_in[1]));
        ComplexF t0 = x_in[0] + t6;
        ComplexF t1 = x_in[3] + t7;
        ComplexF t2 = t8 + t10;
        ComplexF t3 = t11 - t9;
        ComplexF t4 = t8 - t10;
        ComplexF t5 = t9 + t11;
        x_out[0] = t0 + t1;
        x_out[1] = t2 + t3;
        x_out[2] = t4 + t5;
        x_out[3] = t0 - t1;
        x_out[4] = t2 - t3;
        x_out[5] = t4 - t5;

        return;
    }

    // Implementation for radix = 8
    if (radix == 8)
    {
        const ComplexF k = broadcast32_F(sqrt(0.5));

        ComplexF t12 = x_in[1] + x_in[5];
        ComplexF t13 = x_in[3] + x_in[7];
        ComplexF t14 = x_in[1] - x_in[5];
        ComplexF t15 = x_in[7] - x_in[3];

        ComplexF t1 = t12 + t13;
        ComplexF t5;
        if (forward)
            t5 = mul_i_F(t13 - t12);
        else
            t5 = mul_i_F(t12 - t13);

        ComplexF t16 = k*(t14 + t15);
        ComplexF t17;
        if (forward)
            t17 = k*mul_i_F(t15 - t14);
        else
            t17 = k*mul_i_F(t14 - t15);
        ComplexF t3 = t16 + t17;
        ComplexF t7 = t17 - t16;

        ComplexF t8  = x_in[0] + x_in[4];
        ComplexF t9  = x_in[2] + x_in[6];
        ComplexF t10 = x_in[0] - x_in[4];
        ComplexF t11;
        if (forward)
            t11 = mul_i_F(x_in[2] - x_in[6]);
        else
            t11 = mul_i_F(x_in[6] - x_in[2]);
        ComplexF t0  = t8 + t9;
        ComplexF t4  = t8 - t9;
        ComplexF t2  = t10 - t11;
        ComplexF t6  = t10 + t11;

        x_out[0] = t0 + t1;
        x_out[1] = t2 + t3;
        x_out[2] = t4 + t5;
        x_out[3] = t6 + t7;
        x_out[4] = t0 - t1;
        x_out[5] = t2 - t3;
        x_out[6] = t4 - t5;
        x_out[7] = t6 - t7;
        return;
    }

    // Implementation for radix = 7
    if (radix == 7)
    {
        const float *coeff_cos = coeff_radix_7_cos;
        const float *coeff_sin = coeff_radix_7_sin;

        // Calculate sums and differences
        ComplexF sums[3];
        ComplexF diffs[3];
        for (size_t i = 0; i < 3; i++)
        {
            sums[i] = x_in[i+1] + x_in[radix - i - 1];
            diffs[i] = mul_i_F(x_in[radix - i - 1] - x_in[i+1]);
        }

        // Initialize all outputs with x_in[0]
        for (size_t i = 0; i < radix; i++)
        {
            x_out[i] = x_in[0];
        }

        // Calculate x_out[0]
        for (size_t i = 0; i < 3; i++)
        {
            x_out[0] += sums[i];
        }

        // use cos-coefficients
        for (size_t i = 0; i < 3; i++)
        {
            ComplexF x = load_F(0,0);
            for (size_t j = 0; j < 3; j++)
            {
                ComplexF coeff = broadcast32_F(coeff_cos[3*i + j]);
                x += coeff*sums[j];
            }
            x_out[i+1] += x;
            x_out[radix - i - 1] += x;
        }

        // use sin-coefficients
        for (size_t i = 0; i < 3; i++)
        {
            ComplexF x = load_F(0,0);
            for (size_t j = 0; j < 3; j++)
            {
                ComplexF coeff = broadcast32_F(coeff_sin[3*i + j]);
                x += coeff*diffs[j];
            }
            if (forward)
            {
                x_out[i+1] -= x;
                x_out[radix - i - 1] += x;
            } else
            {
                x_out[i+1] += x;
                x_out[radix - i - 1] -= x;
            }
        }

        return;
    }
}

template<size_t radix> inline __attribute__((always_inline)) void multiply_coeff_real_odd_forward_F(const ComplexF *x_in, ComplexF *x_out)
{
    // Use the normal function
    ComplexF x_temp[radix];
    multiply_coeff_F<radix,true>(x_in, x_temp);

    // And then apply some reordering and conjugation
    for (size_t i = 0; i < radix; i+=2)
    {
        x_out[i] = x_temp[i/2];
    }
    for (size_t i = 1; i < radix; i+=2)
    {
        x_out[i] = conj_F(x_temp[radix - (i+1)/2]);
    }
}

template<size_t radix, bool forward> inline __attribute__((always_inline)) void multiply_twiddle_F(const ComplexF *x_in, ComplexF *x_out, const ComplexF *twiddle_factors)
{
    // It is assumed that first twiddle factors are always (1 + 0i)
    x_out[0] = x_in[0];

    // Read in the values used in this step and multiply them with twiddle factors
    for (size_t j = 1; j < radix; j++)
    {
        ComplexF x = x_in[j];
        ComplexF w = twiddle_factors[j];

        x_out[j] = mul_w_F<forward>(x, w);
    }
}

// Multiplies the first values with twiddle factors and then conjugates them and saves as last values
template<size_t radix> inline __attribute__((always_inline)) void multiply_conj_twiddle_odd_F(const ComplexF *x_in, ComplexF *x_out, const ComplexF *twiddle_factors)
{
    // It is assumed that first twiddle factors are always (1 + 0i)
    x_out[0] = x_in[0];

    // multiply values with twiddle factors and conjugate them
    for (size_t j = 1; j <= radix/2; j++)
    {
        ComplexF x = x_in[j];

        // multiplication with twiddle factors is done first
        ComplexF w = twiddle_factors[j];
        ComplexF x_temp = mul_w_F<true>(x, w);

        x_out[j] = x_temp;
        x_out[radix-j] = conj_F(x_temp);
    }
}

#endif
