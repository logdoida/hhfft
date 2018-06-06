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

#define ENABLE_AVX
#include "hhfft_1d_complex_sse2_common_d.h"

////////////////////////////////////////// ComplexD2 ///////////////////////////////////////

// contains two complex numbers: [r1 i1 r2 i2]
typedef __m256d ComplexD2;

// Read two complex number
inline ComplexD2 load_D2(double r1, double i1, double r2, double i2)
{
    //return _mm256_set_pd(i2,r2,i1,r1); // Why this order?
    return _mm256_setr_pd(r1,i1,r2,i2); // Reversed. Why this order?
}
inline const ComplexD2 load_D2(const double *v)
{
    return _mm256_loadu_pd(v);
}

// Loads same complex number twice: [r i] -> [r i r i]
inline const ComplexD2 broadcast128_D2(const double *v)
{
    // TODO is this safe? Does alignment cause trouble?
    return _mm256_broadcast_pd((const __m128d*) v);
}

// Loads same number four times: [x] -> [x x x x]
inline const ComplexD2 broadcast64_D2(const double x)
{
    return _mm256_broadcast_sd(&x);
}

// Combines complex number from two separate memory locations: [a1 a2], [b1 b2] -> [a1 a2 b1 b2]
inline const ComplexD2 load_two_128_D2(const double *a, const double *b)
{
    // NOTE this should compile into two operations
    const ComplexD2 aa = _mm256_castpd128_pd256(load_D(a));
    const ComplexD bb = load_D(b);
    return _mm256_insertf128_pd (aa, bb, 1);
}

// Store a complex number
inline void store_D2(ComplexD2 val, double &r1, double &i1, double &r2, double &i2)
{
    double v[4];
    _mm256_storeu_pd(v, val);
    r1 = val[0]; i1 = val[1]; r2 = val[2]; i2 = val[3];
}
inline void store_D2(ComplexD2 val, double *v)
{
    _mm256_storeu_pd(v, val);
}

// Divides the complex numbers to two separate memory locations: [a1 a2 b1 b2] -> [a1 a2], [b1 b2]
inline void store_two_128_D2(ComplexD2 val, double *a, double *b)
{
    // NOTE this should compile into three operations
    ComplexD aa = _mm256_castpd256_pd128(val);
    store_D(aa, a);
    ComplexD bb = _mm256_extractf128_pd(val, 1);
    store_D(bb, b);
}

// Changes signs of [x1 x2 x3 x4] using [s1 s2 s3 s4]. s should contain only 0.0 and -0.0
// NOTE this seems to actually be bit slower than plain multiplication. Compare!
inline ComplexD2 change_sign_D2(ComplexD2 x, ComplexD2 s)
{
    return _mm256_xor_pd(x,s);
}

// It seems that it is more efficient to define the constant inside the functions
//static const ComplexD2 const1 = load(0.0, -0.0, 0.0, -0.0);

// Multiplies two packed complex numbers.
inline ComplexD2 __attribute__((always_inline)) mul_D2(ComplexD2 a, ComplexD2 b)
{
    ComplexD2 a2 = _mm256_permute_pd(a, 1 + 4);

    ComplexD2 t1 = _mm256_mul_pd(a, b);
    ComplexD2 t2 = _mm256_mul_pd(a2, b);

    ComplexD2 t3 = _mm256_unpacklo_pd(t1,t2);
    ComplexD2 t4 = _mm256_unpackhi_pd(t1,t2);

    return _mm256_addsub_pd(t3,t4);
}

// Multiplies two packed complex numbers. The forward means a*b, inverse a*conj(b)
template<bool forward> inline __attribute__((always_inline)) ComplexD2 mul_w_D2(ComplexD2 a, ComplexD2 b)
{        
    if (forward)
    {
        ComplexD2 a2 = _mm256_permute_pd(a, 1 + 4);

        ComplexD2 t1 = _mm256_mul_pd(a, b);
        ComplexD2 t2 = _mm256_mul_pd(a2, b);

        ComplexD2 t3 = _mm256_unpacklo_pd(t1,t2);
        ComplexD2 t4 = _mm256_unpackhi_pd(t1,t2);

        return _mm256_addsub_pd(t3,t4);
    } else
    {
        // NOTE this is slightly slower than forward version. Is there way to improve?
        const ComplexD2 const1 = load_D2(0.0, -0.0, 0.0, -0.0);

        ComplexD2 b2 = _mm256_permute_pd(change_sign_D2(b, const1), 1 + 4);

        ComplexD2 t1 = _mm256_mul_pd(a, b);
        ComplexD2 t2 = _mm256_mul_pd(a, b2);

        ComplexD2 y = _mm256_hadd_pd(t1, t2);
        return y;
    }
}

// Multiplies packed complex numbers with i
inline __attribute__((always_inline)) ComplexD2 mul_i_D2(ComplexD2 a)
{    
    const ComplexD2 const1 = load_D2(0.0, -0.0, 0.0, -0.0);

    ComplexD2 a1 = change_sign_D2(a, const1);
    ComplexD2 y = _mm256_permute_pd(a1, 1 + 4);
    return y;
}

template<size_t radix, bool forward> inline __attribute__((always_inline)) void multiply_coeff_D2(const ComplexD2 *x_in, ComplexD2 *x_out)
{
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
        const ComplexD2 k0 = broadcast64_D2(0.5);
        const ComplexD2 k1 = broadcast64_D2(0.5*sqrt(3.0));
        ComplexD2 t0 = x_in[1] + x_in[2];
        ComplexD2 t1 = x_in[0] - k0*t0;
        ComplexD2 t2 = mul_i_D2(k1*(x_in[1] - x_in[2]));

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
        ComplexD2 t0 = x_in[0] + x_in[2];
        ComplexD2 t1 = x_in[1] + x_in[3];
        ComplexD2 t2 = x_in[0] - x_in[2];
        ComplexD2 t3;
        if (forward)
            t3 = mul_i_D2(x_in[3] - x_in[1]);
        else
            t3 = mul_i_D2(x_in[1] - x_in[3]);

        x_out[0] = t0 + t1;
        x_out[1] = t2 + t3;
        x_out[2] = t0 - t1;
        x_out[3] = t2 - t3;
        return;
    }

    // Implementation for radix = 5
    if (radix == 5)
    {
        const ComplexD2 k1 = broadcast64_D2(cos(2.0*M_PI*1.0/5.0));
        const ComplexD2 k2 = broadcast64_D2(sin(2.0*M_PI*1.0/5.0));
        const ComplexD2 k3 = broadcast64_D2(-cos(2.0*M_PI*2.0/5.0));
        const ComplexD2 k4 = broadcast64_D2(sin(2.0*M_PI*2.0/5.0));

        ComplexD2 t0 = x_in[1] + x_in[4];
        ComplexD2 t1 = x_in[2] + x_in[3];
        ComplexD2 t2 = x_in[1] - x_in[4];
        ComplexD2 t3 = x_in[2] - x_in[3];
        ComplexD2 t4 = x_in[0] + k1*t0 - k3*t1;
        ComplexD2 t5 = x_in[0] + k1*t1 - k3*t0;
        ComplexD2 t6 = mul_i_D2(k2*t2 + k4*t3);
        ComplexD2 t7 = mul_i_D2(k4*t2 - k2*t3);

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
        const ComplexD2 k0 = broadcast64_D2(0.5);
        ComplexD2 k1;
        if (forward)
            k1 = broadcast64_D2(0.5*sqrt(3.0));
        else
            k1 = broadcast64_D2(-0.5*sqrt(3.0));

        ComplexD2 t6 = x_in[2] + x_in[4];
        ComplexD2 t7 = x_in[1] + x_in[5];
        ComplexD2 t8 = x_in[0] - k0*t6;
        ComplexD2 t9 = x_in[3] - k0*t7;
        ComplexD2 t10 = mul_i_D2(k1*(x_in[4] - x_in[2]));
        ComplexD2 t11 = mul_i_D2(k1*(x_in[5] - x_in[1]));
        ComplexD2 t0 = x_in[0] + t6;
        ComplexD2 t1 = x_in[3] + t7;
        ComplexD2 t2 = t8 + t10;
        ComplexD2 t3 = t11 - t9;
        ComplexD2 t4 = t8 - t10;
        ComplexD2 t5 = t9 + t11;
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
        const ComplexD2 k = broadcast64_D2(sqrt(0.5));

        ComplexD2 t12 = x_in[1] + x_in[5];
        ComplexD2 t13 = x_in[3] + x_in[7];
        ComplexD2 t14 = x_in[1] - x_in[5];
        ComplexD2 t15 = x_in[7] - x_in[3];

        ComplexD2 t1 = t12 + t13;
        ComplexD2 t5;
        if (forward)
            t5 = mul_i_D2(t13 - t12);
        else
            t5 = mul_i_D2(t12 - t13);

        ComplexD2 t16 = k*(t14 + t15);
        ComplexD2 t17;
        if (forward)
            t17 = k*mul_i_D2(t15 - t14);
        else
            t17 = k*mul_i_D2(t14 - t15);
        ComplexD2 t3 = t16 + t17;
        ComplexD2 t7 = t17 - t16;

        ComplexD2 t8  = x_in[0] + x_in[4];
        ComplexD2 t9  = x_in[2] + x_in[6];
        ComplexD2 t10 = x_in[0] - x_in[4];
        ComplexD2 t11;
        if (forward)
            t11 = mul_i_D2(x_in[2] - x_in[6]);
        else
            t11 = mul_i_D2(x_in[6] - x_in[2]);
        ComplexD2 t0  = t8 + t9;
        ComplexD2 t4  = t8 - t9;
        ComplexD2 t2  = t10 - t11;
        ComplexD2 t6  = t10 + t11;

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

    // Other radices
    const double *coeff = nullptr;

    if (radix == 7)
    {
        coeff = coeff_radix_7;
    }

    // First row is (1,0)
    x_out[0] = x_in[0];
    for (size_t i = 1; i < radix; i++)
    {
        x_out[0] = x_out[0] + x_in[i];
    }

    for (size_t i = 1; i < radix; i++)
    {
        x_out[i] = x_in[0]; // First column is always (1,0)
        for (size_t j = 1; j < radix; j++)
        {
            ComplexD2 w = broadcast128_D2(coeff + 2*radix*i + 2*j);

            x_out[i] = x_out[i] + mul_w_D2<forward>(x_in[j], w);
        }
    }
}


template<size_t radix, bool forward> inline __attribute__((always_inline)) void multiply_twiddle_D2(const ComplexD2 *x_in, ComplexD2 *x_out, const ComplexD2 *twiddle_factors)
{
    // It is assumed that first twiddle factors are always (1 + 0i)
    x_out[0] = x_in[0];

    // Read in the values used in this step and multiply them with twiddle factors
    for (size_t j = 1; j < radix; j++)
    {
        ComplexD2 x = x_in[j];
        ComplexD2 w = twiddle_factors[j];

        x_out[j] = mul_w_D2<forward>(x, w);
    }
}


////////////////////////////////////////// ComplexD4S ///////////////////////////////////////
// contains four complex numbers separated to real and imaginary parts : [r0 r2 r1 r3] & [i0 i2 i1 i3]
// this is mainly intended to be used with FMA instructions, but it might also sometimes be more efficient way also with normal avx
typedef struct
{
    __m256d real, imag;
} ComplexD4S;


// Read four complex numbers and reorder them to real and complex parts
inline const ComplexD4S load512s_D4S(const double *x)
{
    __m256d x0 = _mm256_loadu_pd(x);
    __m256d x1 = _mm256_loadu_pd(x + 4);

    ComplexD4S out;
    out.real = _mm256_unpacklo_pd(x0, x1);
    out.imag = _mm256_unpackhi_pd(x0, x1);

    return out;
}

// Store four complex numbers
inline void store_D4S(ComplexD4S val, double *v)
{
    // Reorder data back to correct form
    __m256d x0 = _mm256_unpacklo_pd(val.real, val.imag);
    __m256d x1 = _mm256_unpackhi_pd(val.real, val.imag);

    _mm256_storeu_pd(v, x0);
    _mm256_storeu_pd(v + 4, x1);
}

// Multiplies four complex numbers.
inline ComplexD4S mul_D4S(ComplexD4S a, ComplexD4S b)
{
    ComplexD4S out;
    out.real = a.real * b.real - a.imag*b.imag;
    out.imag = a.real * b.imag + a.imag*b.real;

    return out;
}

// Multiplies four complex numbers. Forward means a*b, inverse a*conj(b)
template<bool forward> inline void mul_w_D4S(ComplexD4S a, ComplexD4S b)
{
    ComplexD4S out;

    if (forward)
    {
        out.real = a.real * b.real - a.imag*b.imag;
        out.imag = a.real * b.imag + a.imag*b.real;

    } else
    {
        out.real = a.real * b.real + a.imag*b.imag;
        out.imag = a.real * b.imag - a.imag*b.real;
    }

    return out;
}

