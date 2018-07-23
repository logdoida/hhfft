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

// Load only real part, imaginary part is set to zero. [r1 r2] -> [r1 0 r2 0]
inline const ComplexD2 load_real_D2(const double *v)
{
    const ComplexD2 a = _mm256_castpd128_pd256(load_real_D(v));
    const ComplexD b = load_real_D(v + 1);
    return _mm256_insertf128_pd(a, b, 1);
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

// Store only real parts [r1 x r2 x] -> [r1 r2]
inline void store_real_D2(ComplexD2 val, double *v)
{
    ComplexD aa = _mm256_castpd256_pd128(val);
    store_real_D(aa, v);
    ComplexD bb = _mm256_extractf128_pd(val, 1);
    store_real_D(bb, v + 1);
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

// Complex conjugate
inline ComplexD2 conj_D2(ComplexD2 x)
{
    const ComplexD2 const1 = load_D2(0.0, -0.0, 0.0, -0.0);
    return change_sign_D2(x,const1);
}

template<size_t radix, bool forward> inline __attribute__((always_inline)) void multiply_coeff_D2(const ComplexD2 *x_in, ComplexD2 *x_out)
{
    // Implementation for radix = 1 (this might not be needed)
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

// Multiplies the first values with twiddle factors and then conjugates them and saves as last values
template<size_t radix> inline __attribute__((always_inline)) void multiply_conj_twiddle_odd_D2(const ComplexD2 *x_in, ComplexD2 *x_out, const ComplexD2 *twiddle_factors)
{
    // It is assumed that first twiddle factors are always (1 + 0i)
    x_out[0] = x_in[0];

    // multiply values with twiddle factors and conjugate them
    for (size_t j = 1; j <= radix/2; j++)
    {
        ComplexD2 x = x_in[j];

        // multiplication with conjugated twiddle factors is done first
        ComplexD2 w = twiddle_factors[j];
        ComplexD2 x_temp = mul_w_D2<false>(x, w);

        x_out[j] = x_temp;
        x_out[radix-j] = conj_D2(x_temp);
    }
}

////////////////////////////////////////// ComplexD4S ///////////////////////////////////////
// contains four complex numbers separated to real and imaginary parts : [r r r r] & [i i i i]
// this is mainly intended to be used with FMA instructions, but it might also sometimes be more efficient way also with normal avx
typedef struct
{
    __m256d real, imag;
} ComplexD4S;

// addition, subtraction, multiplication
inline ComplexD4S operator+(ComplexD4S a, ComplexD4S b)
{
    ComplexD4S out = {a.real + b.real, a.imag + b.imag};
    return out;
}
inline ComplexD4S operator-(ComplexD4S a, ComplexD4S b)
{
    ComplexD4S out = {a.real - b.real, a.imag - b.imag};
    return out;
}
inline ComplexD4S operator*(ComplexD4S a, ComplexD4S b)
{
    ComplexD4S out = {a.real * b.real, a.imag * b.imag};
    return out;
}

// Read complex numbers with real and imaginary parts from different memory locations
inline __attribute__((always_inline))  const ComplexD4S load_D4S(const double *r, const double *i)
{
    ComplexD4S out;
    out.real = _mm256_loadu_pd(r);
    out.imag = _mm256_loadu_pd(i);

    return out;
}

// Read four complex numbers and reorder them to real and complex parts [r0 r2 r1 r3] & [i0 i2 i1 i3]
inline __attribute__((always_inline))  const ComplexD4S load512s_D4S(const double *x)
{
    __m256d x0 = _mm256_loadu_pd(x);
    __m256d x1 = _mm256_loadu_pd(x + 4);

    ComplexD4S out;
    out.real = _mm256_unpacklo_pd(x0, x1);
    out.imag = _mm256_unpackhi_pd(x0, x1);

    return out;
}

// Read four complex numbers and reorder them to real and complex parts [r0 r1 r2 r3] & [i0 i1 i2 i3]
inline __attribute__((always_inline)) const ComplexD4S load512_D4S(const double *x)
{
    __m256d x0 = load_two_128_D2(x + 0, x + 4);
    __m256d x1 = load_two_128_D2(x + 2, x + 6);

    ComplexD4S out;
    out.real = _mm256_unpacklo_pd(x0, x1);
    out.imag = _mm256_unpackhi_pd(x0, x1);

    return out;
}

// Store four complex numbers [r0 r2 r1 r3] & [i0 i2 i1 i3]
inline __attribute__((always_inline)) void store512s_D4S(ComplexD4S val, double *v)
{
    // Reorder data back to correct form
    __m256d x0 = _mm256_unpacklo_pd(val.real, val.imag);
    __m256d x1 = _mm256_unpackhi_pd(val.real, val.imag);

    _mm256_storeu_pd(v, x0);
    _mm256_storeu_pd(v + 4, x1);
}

// Store four complex numbers with real and imaginary parts to separate locations
inline __attribute__((always_inline)) void store_D4S(ComplexD4S val, double *r, double *i)
{
    _mm256_storeu_pd(r, val.real);
    _mm256_storeu_pd(i, val.imag);
}

inline __attribute__((always_inline)) ComplexD4S broadcast64_D4S(double x)
{
    ComplexD4S out;
    out.real = broadcast64_D2(x);
    out.imag = out.real;
    return out;
}

// Multiplies four complex numbers.
inline __attribute__((always_inline)) ComplexD4S mul_D4S(ComplexD4S a, ComplexD4S b)
{
    ComplexD4S out;
    out.real = a.real * b.real - a.imag*b.imag;
    out.imag = a.real * b.imag + a.imag*b.real;

    return out;
}

// Multiplies four complex numbers. Forward means a*b, inverse a*conj(b)
template<bool forward> inline __attribute__((always_inline)) ComplexD4S mul_w_D4S(ComplexD4S a, ComplexD4S b)
{
    ComplexD4S out;

    if (forward)
    {
        out.real = a.real * b.real - a.imag*b.imag;
        out.imag = a.real * b.imag + a.imag*b.real;

    } else
    {
        out.real = a.real * b.real + a.imag * b.imag;
        out.imag = a.imag * b.real - a.real * b.imag;
    }

    return out;
}

inline __attribute__((always_inline)) ComplexD4S mul_i_D4S(ComplexD4S x)
{
    ComplexD4S out;
    out.real = -x.imag;
    out.imag = x.real;
    return out;
}

template<size_t radix, bool forward> inline __attribute__((always_inline)) void multiply_twiddle_D4S(const ComplexD4S *x_in, ComplexD4S *x_out, const ComplexD4S *twiddle_factors)
{
    // It is assumed that first twiddle factors are always (1 + 0i)
    x_out[0] = x_in[0];

    // Read in the values used in this step and multiply them with twiddle factors
    for (size_t j = 1; j < radix; j++)
    {
        ComplexD4S x = x_in[j];
        ComplexD4S w = twiddle_factors[j];

        x_out[j] = mul_w_D4S<forward>(x, w);
    }
}

template<size_t radix, bool forward> inline __attribute__((always_inline)) void multiply_coeff_D4S(const ComplexD4S *x_in, ComplexD4S *x_out)
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
        const ComplexD4S k0 = broadcast64_D4S(0.5);
        const ComplexD4S k1 = broadcast64_D4S(0.5*sqrt(3.0));
        ComplexD4S t0 = x_in[1] + x_in[2];
        ComplexD4S t1 = x_in[0] - k0*t0;
        ComplexD4S t2 = mul_i_D4S(k1*(x_in[1] - x_in[2]));

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
        ComplexD4S t0 = x_in[0] + x_in[2];
        ComplexD4S t1 = x_in[1] + x_in[3];
        ComplexD4S t2 = x_in[0] - x_in[2];
        ComplexD4S t3;
        if (forward)
            t3 = mul_i_D4S(x_in[3] - x_in[1]);
        else
            t3 = mul_i_D4S(x_in[1] - x_in[3]);

        x_out[0] = t0 + t1;
        x_out[1] = t2 + t3;
        x_out[2] = t0 - t1;
        x_out[3] = t2 - t3;
        return;
    }

    // Implementation for radix = 5
    if (radix == 5)
    {
        const ComplexD4S k1 = broadcast64_D4S(cos(2.0*M_PI*1.0/5.0));
        const ComplexD4S k2 = broadcast64_D4S(sin(2.0*M_PI*1.0/5.0));
        const ComplexD4S k3 = broadcast64_D4S(-cos(2.0*M_PI*2.0/5.0));
        const ComplexD4S k4 = broadcast64_D4S(sin(2.0*M_PI*2.0/5.0));

        ComplexD4S t0 = x_in[1] + x_in[4];
        ComplexD4S t1 = x_in[2] + x_in[3];
        ComplexD4S t2 = x_in[1] - x_in[4];
        ComplexD4S t3 = x_in[2] - x_in[3];
        ComplexD4S t4 = x_in[0] + k1*t0 - k3*t1;
        ComplexD4S t5 = x_in[0] + k1*t1 - k3*t0;
        ComplexD4S t6 = mul_i_D4S(k2*t2 + k4*t3);
        ComplexD4S t7 = mul_i_D4S(k4*t2 - k2*t3);

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
        const ComplexD4S k0 = broadcast64_D4S(0.5);
        ComplexD4S k1;
        if (forward)
            k1 = broadcast64_D4S(0.5*sqrt(3.0));
        else
            k1 = broadcast64_D4S(-0.5*sqrt(3.0));

        ComplexD4S t6 = x_in[2] + x_in[4];
        ComplexD4S t7 = x_in[1] + x_in[5];
        ComplexD4S t8 = x_in[0] - k0*t6;
        ComplexD4S t9 = x_in[3] - k0*t7;
        ComplexD4S t10 = mul_i_D4S(k1*(x_in[4] - x_in[2]));
        ComplexD4S t11 = mul_i_D4S(k1*(x_in[5] - x_in[1]));
        ComplexD4S t0 = x_in[0] + t6;
        ComplexD4S t1 = x_in[3] + t7;
        ComplexD4S t2 = t8 + t10;
        ComplexD4S t3 = t11 - t9;
        ComplexD4S t4 = t8 - t10;
        ComplexD4S t5 = t9 + t11;
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
        const ComplexD4S k = broadcast64_D4S(sqrt(0.5));

        ComplexD4S t12 = x_in[1] + x_in[5];
        ComplexD4S t13 = x_in[3] + x_in[7];
        ComplexD4S t14 = x_in[1] - x_in[5];
        ComplexD4S t15 = x_in[7] - x_in[3];

        ComplexD4S t1 = t12 + t13;
        ComplexD4S t5;
        if (forward)
            t5 = mul_i_D4S(t13 - t12);
        else
            t5 = mul_i_D4S(t12 - t13);

        ComplexD4S t16 = k*(t14 + t15);
        ComplexD4S t17;
        if (forward)
            t17 = k*mul_i_D4S(t15 - t14);
        else
            t17 = k*mul_i_D4S(t14 - t15);
        ComplexD4S t3 = t16 + t17;
        ComplexD4S t7 = t17 - t16;

        ComplexD4S t8  = x_in[0] + x_in[4];
        ComplexD4S t9  = x_in[2] + x_in[6];
        ComplexD4S t10 = x_in[0] - x_in[4];
        ComplexD4S t11;
        if (forward)
            t11 = mul_i_D4S(x_in[2] - x_in[6]);
        else
            t11 = mul_i_D4S(x_in[6] - x_in[2]);
        ComplexD4S t0  = t8 + t9;
        ComplexD4S t4  = t8 - t9;
        ComplexD4S t2  = t10 - t11;
        ComplexD4S t6  = t10 + t11;

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
            // TODO this should be a function
            ComplexD4S w;
            double re = coeff[2*radix*i + 2*j + 0];
            double im = coeff[2*radix*i + 2*j + 1];
            w.real = broadcast64_D2(re);
            w.imag = broadcast64_D2(im);

            x_out[i] = x_out[i] + mul_w_D4S<forward>(x_in[j], w);
        }
    }
}
