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
// if this is compiled using the -mavx flag, the outcome will be different
// using flag ENABLE_AVX, will also enable the usage of more efficient instructions

#include "hhfft_1d_complex_plain_common_d.h"
#include <immintrin.h>

// contains a single complex number: [r i]
typedef __m128d ComplexD;

// Read complex numbers
inline ComplexD load_D(double r, double i)
{
    return _mm_setr_pd(r,i);
}
inline const ComplexD load_D(const double *v)
{
    return _mm_loadu_pd(v);
}
inline ComplexD broadcast64_D(double x)
{
    return _mm_setr_pd(x,x);
}
// Load only real part, imaginary part is set to zero
inline const ComplexD load_real_D(const double *v)
{
    return _mm_load_sd(v);
}

// Store
inline void store_D(ComplexD val, double &r, double &i)
{
    double v[2];
    _mm_storeu_pd(v, val);
    r = val[0]; i = val[1];
}
inline void store_D(ComplexD val, double *v)
{
    _mm_storeu_pd(v, val);
}
// Store only real part
inline void store_real_D(ComplexD val, double *v)
{
    _mm_store_sd(v, val);
}

// Changes signs of [x1 x2] using [s1 s2]. s should contain only 0.0 and -0.0
// NOTE this seems to actually be bit slower than plain multiplication? Compare!
inline ComplexD change_sign_D(ComplexD x, ComplexD s)
{
    return _mm_xor_pd(x,s);
}

const ComplexD const1_128 = load_D(0.0, -0.0);
const ComplexD const2_128 = load_D(-0.0, 0.0);

// Multiplies complex numbers. If other of them changes more frequently, set it to b.
#ifdef ENABLE_AVX
// AVX-version is slightly more efficient
inline ComplexD mul_D(ComplexD a, ComplexD b)
{
    // If AVX-was supported...
    ComplexD a1 = change_sign_D(a, const1_128);
    ComplexD a2 = _mm_permute_pd(a, 1); // Not supported in sse2

    ComplexD t1 = _mm_mul_pd(a1, b);
    ComplexD t2 = _mm_mul_pd(a2, b);

    ComplexD y = _mm_hadd_pd(t1, t2); // Not supported in sse2
    return y;
}
#else
inline ComplexD mul_D(ComplexD a, ComplexD b)
{
    ComplexD a1 = change_sign_D(a, const1_128);
    ComplexD a2 = _mm_shuffle_pd(a, a, 1);

    ComplexD t1 = _mm_mul_pd(a1, b);
    ComplexD t2 = _mm_mul_pd(a2, b);

    ComplexD t3 = _mm_unpackhi_pd(t1, t2);
    ComplexD t4 = _mm_unpacklo_pd(t1, t2);

    ComplexD y = _mm_add_pd(t3, t4);

    return y;
}
#endif


// Calculates a*conj(b)
#ifdef ENABLE_AVX
// AVX-version is slightly more efficient
inline ComplexD mul_conj_D(ComplexD a, ComplexD b)
{
    // If AVX-was supported...
    ComplexD a1 = change_sign_D(a, const2_128);
    ComplexD b1 = _mm_permute_pd(b, 1); // Not supported in sse2

    ComplexD t1 = _mm_mul_pd(a, b);
    ComplexD t2 = _mm_mul_pd(a1, b1);

    ComplexD y = _mm_hadd_pd(t1, t2); // Not supported in sse2
    return y;
}
#else
inline ComplexD mul_conj_D(ComplexD a, ComplexD b)
{
    ComplexD a1 = change_sign_D(a, const2_128);
    ComplexD b1 = _mm_shuffle_pd(b, b, 1);

    ComplexD t1 = _mm_mul_pd(a, b);
    ComplexD t2 = _mm_mul_pd(a1, b1);

    ComplexD t3 = _mm_unpackhi_pd(t1, t2);
    ComplexD t4 = _mm_unpacklo_pd(t1, t2);

    ComplexD y = _mm_add_pd(t3, t4);

    return y;
}
#endif


// Multiplies two packed complex numbers. The forward means a*b, inverse a*conj(b)
template<bool forward> inline ComplexD mul_w_D(ComplexD a, ComplexD b)
{
    if (forward)
    {
        return mul_D(a,b);
    } else
    {
        return mul_conj_D(a,b);
    }
}

// Complex conjugate
inline ComplexD conj_D(ComplexD x)
{
    return change_sign_D(x,const1_128);
}

// Multiply with i
#ifdef ENABLE_AVX
inline ComplexD mul_i_D(ComplexD a)
{    
    ComplexD a1 = change_sign_D(a, const1_128);
    ComplexD y = _mm_permute_pd(a1, 1);
    return y;
}
#else
inline ComplexD mul_i_D(ComplexD a)
{
    ComplexD a1 = change_sign_D(a, const1_128);
    ComplexD y = _mm_shuffle_pd(a1, a1, 1);
    return y;
}
#endif


template<size_t radix, bool forward> inline __attribute__((always_inline)) void multiply_coeff_D(const ComplexD *x_in, ComplexD *x_out)
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
        const ComplexD k0 = broadcast64_D(0.5);
        const ComplexD k1 = broadcast64_D(0.5*sqrt(3.0));
        ComplexD t0 = x_in[1] + x_in[2];
        ComplexD t1 = x_in[0] - k0*t0;
        ComplexD t2 = mul_i_D(k1*(x_in[1] - x_in[2]));

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
        ComplexD t0 = x_in[0] + x_in[2];
        ComplexD t1 = x_in[1] + x_in[3];
        ComplexD t2 = x_in[0] - x_in[2];
        ComplexD t3;
        if (forward)
            t3 = mul_i_D(x_in[3] - x_in[1]);
        else
            t3 = mul_i_D(x_in[1] - x_in[3]);

        x_out[0] = t0 + t1;
        x_out[1] = t2 + t3;
        x_out[2] = t0 - t1;
        x_out[3] = t2 - t3;
        return;
    }

    // Implementation for radix = 5
    if (radix == 5)
    {
        const ComplexD k1 = broadcast64_D(cos(2.0*M_PI*1.0/5.0));
        const ComplexD k2 = broadcast64_D(sin(2.0*M_PI*1.0/5.0));
        const ComplexD k3 = broadcast64_D(-cos(2.0*M_PI*2.0/5.0));
        const ComplexD k4 = broadcast64_D(sin(2.0*M_PI*2.0/5.0));

        ComplexD t0 = x_in[1] + x_in[4];
        ComplexD t1 = x_in[2] + x_in[3];
        ComplexD t2 = x_in[1] - x_in[4];
        ComplexD t3 = x_in[2] - x_in[3];
        ComplexD t4 = x_in[0] + k1*t0 - k3*t1;
        ComplexD t5 = x_in[0] + k1*t1 - k3*t0;
        ComplexD t6 = mul_i_D(k2*t2 + k4*t3);
        ComplexD t7 = mul_i_D(k4*t2 - k2*t3);

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
        const ComplexD k0 = broadcast64_D(0.5);
        ComplexD k1;
        if (forward)
            k1 = broadcast64_D(0.5*sqrt(3.0));
        else
            k1 = broadcast64_D(-0.5*sqrt(3.0));

        ComplexD t6 = x_in[2] + x_in[4];
        ComplexD t7 = x_in[1] + x_in[5];
        ComplexD t8 = x_in[0] - k0*t6;
        ComplexD t9 = x_in[3] - k0*t7;
        ComplexD t10 = mul_i_D(k1*(x_in[4] - x_in[2]));
        ComplexD t11 = mul_i_D(k1*(x_in[5] - x_in[1]));
        ComplexD t0 = x_in[0] + t6;
        ComplexD t1 = x_in[3] + t7;
        ComplexD t2 = t8 + t10;
        ComplexD t3 = t11 - t9;
        ComplexD t4 = t8 - t10;
        ComplexD t5 = t9 + t11;
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
        const ComplexD k = broadcast64_D(sqrt(0.5));

        ComplexD t12 = x_in[1] + x_in[5];
        ComplexD t13 = x_in[3] + x_in[7];
        ComplexD t14 = x_in[1] - x_in[5];
        ComplexD t15 = x_in[7] - x_in[3];

        ComplexD t1 = t12 + t13;
        ComplexD t5;
        if (forward)
            t5 = mul_i_D(t13 - t12);
        else
            t5 = mul_i_D(t12 - t13);

        ComplexD t16 = k*(t14 + t15);
        ComplexD t17;
        if (forward)
            t17 = k*mul_i_D(t15 - t14);
        else
            t17 = k*mul_i_D(t14 - t15);
        ComplexD t3 = t16 + t17;
        ComplexD t7 = t17 - t16;

        ComplexD t8  = x_in[0] + x_in[4];
        ComplexD t9  = x_in[2] + x_in[6];
        ComplexD t10 = x_in[0] - x_in[4];
        ComplexD t11;
        if (forward)
            t11 = mul_i_D(x_in[2] - x_in[6]);
        else
            t11 = mul_i_D(x_in[6] - x_in[2]);
        ComplexD t0  = t8 + t9;
        ComplexD t4  = t8 - t9;
        ComplexD t2  = t10 - t11;
        ComplexD t6  = t10 + t11;

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
            ComplexD w = load_D(coeff + 2*radix*i + 2*j);

            x_out[i] = x_out[i] + mul_w_D<forward>(x_in[j], w);
        }
    }   
}

template<size_t radix> inline __attribute__((always_inline)) void multiply_coeff_real_odd_forward_D(const ComplexD *x_in, ComplexD *x_out)
{
    // Use the normal function
    ComplexD x_temp[radix];
    multiply_coeff_D<radix,true>(x_in, x_temp);

    // And then apply some reordering and conjugation
    for (size_t i = 0; i < radix; i+=2)
    {
        x_out[i] = x_temp[i/2];
    }
    for (size_t i = 1; i < radix; i+=2)
    {
        x_out[i] = conj_D(x_temp[radix - (i+1)/2]);
    }
}

template<size_t radix, bool forward> inline __attribute__((always_inline)) void multiply_twiddle_D(const ComplexD *x_in, ComplexD *x_out, const ComplexD *twiddle_factors)
{        
    // It is assumed that first twiddle factors are always (1 + 0i)
    x_out[0] = x_in[0];

    // Read in the values used in this step and multiply them with twiddle factors
    for (size_t j = 1; j < radix; j++)
    {
        ComplexD x = x_in[j];
        ComplexD w = twiddle_factors[j];

        x_out[j] = mul_w_D<forward>(x, w);
    }
}

// Multiplies the first values with twiddle factors and then conjugates them and saves as last values
template<size_t radix> inline __attribute__((always_inline)) void multiply_conj_twiddle_odd_D(const ComplexD *x_in, ComplexD *x_out, const ComplexD *twiddle_factors)
{
    // It is assumed that first twiddle factors are always (1 + 0i)
    x_out[0] = x_in[0];

    // multiply values with twiddle factors and conjugate them
    for (size_t j = 1; j <= radix/2; j++)
    {
        ComplexD x = x_in[j];

        // multiplication with conjugated twiddle factors is done first
        ComplexD w = twiddle_factors[j];
        ComplexD x_temp = mul_w_D<false>(x, w);

        x_out[j] = x_temp;
        x_out[radix-j] = conj_D(x_temp);
    }
}


////////////////////////////////////////// ComplexD2S ///////////////////////////////////////
// contains two complex numbers separated to real and imaginary parts : [r0 r1] & [i0 i1]
typedef struct
{
    __m128d real, imag;
} ComplexD2S;

// addition, subtraction, multiplication
inline ComplexD2S operator+(ComplexD2S a, ComplexD2S b)
{
    ComplexD2S out = {a.real + b.real, a.imag + b.imag};
    return out;
}
inline ComplexD2S operator-(ComplexD2S a, ComplexD2S b)
{
    ComplexD2S out = {a.real - b.real, a.imag - b.imag};
    return out;
}
inline ComplexD2S operator*(ComplexD2S a, ComplexD2S b)
{
    ComplexD2S out = {a.real * b.real, a.imag * b.imag};
    return out;
}

// Read complex numbers with real and imaginary parts from different memory locations
inline __attribute__((always_inline)) ComplexD2S load_D2S(const double *r, const double *i)
{
    ComplexD2S out;
    out.real = _mm_loadu_pd(r);
    out.imag = _mm_loadu_pd(i);

    return out;
}

// Read complex numbers [r0 c0 r1 c1] and reorder them to real and complex parts [r0 r1] & [i0 i1]
inline __attribute__((always_inline)) const ComplexD2S load256s_D2S(const double *x)
{
    __m128d x0 = _mm_loadu_pd(x);
    __m128d x1 = _mm_loadu_pd(x + 2);

    ComplexD2S out;
    out.real = _mm_unpacklo_pd(x0, x1);
    out.imag = _mm_unpackhi_pd(x0, x1);

    return out;
}

inline __attribute__((always_inline)) ComplexD2S broadcast64_D2S(double x)
{
    ComplexD2S out;
    out.real =_mm_setr_pd(x,x);
    out.imag = out.real;
    return out;
}

// Store two complex numbers with real and imaginary parts to separate locations
inline __attribute__((always_inline)) void store_D2S(ComplexD2S val, double *r, double *i)
{
    _mm_storeu_pd(r, val.real);
    _mm_storeu_pd(i, val.imag);
}

// Multiplies four complex numbers.
inline __attribute__((always_inline)) ComplexD2S mul_D2S(ComplexD2S a, ComplexD2S b)
{
    ComplexD2S out;
    out.real = a.real * b.real - a.imag*b.imag;
    out.imag = a.real * b.imag + a.imag*b.real;

    return out;
}

// Multiplies four complex numbers. Forward means a*b, inverse a*conj(b)
template<bool forward> inline __attribute__((always_inline)) ComplexD2S mul_w_D2S(ComplexD2S a, ComplexD2S b)
{
    ComplexD2S out;

    if (forward)
    {
        out.real = a.real * b.real - a.imag * b.imag;
        out.imag = a.real * b.imag + a.imag * b.real;

    } else
    {
        out.real = a.real * b.real + a.imag * b.imag;
        out.imag = a.imag * b.real - a.real * b.imag;
    }

    return out;
}

inline __attribute__((always_inline)) ComplexD2S mul_i_D2S(ComplexD2S x)
{
    ComplexD2S out;
    out.real = -x.imag;
    out.imag = x.real;
    return out;
}



template<size_t radix, bool forward> inline __attribute__((always_inline)) void multiply_twiddle_D2S(const ComplexD2S *x_in, ComplexD2S *x_out, const ComplexD2S *twiddle_factors)
{
    // It is assumed that first twiddle factors are always (1 + 0i)
    x_out[0] = x_in[0];

    // Read in the values used in this step and multiply them with twiddle factors
    for (size_t j = 1; j < radix; j++)
    {
        ComplexD2S x = x_in[j];
        ComplexD2S w = twiddle_factors[j];

        x_out[j] = mul_w_D2S<forward>(x, w);
    }
}

template<size_t radix, bool forward> inline __attribute__((always_inline)) void multiply_coeff_D2S(const ComplexD2S *x_in, ComplexD2S *x_out)
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
        const ComplexD2S k0 = broadcast64_D2S(0.5);
        const ComplexD2S k1 = broadcast64_D2S(0.5*sqrt(3.0));
        ComplexD2S t0 = x_in[1] + x_in[2];
        ComplexD2S t1 = x_in[0] - k0*t0;
        ComplexD2S t2 = mul_i_D2S(k1*(x_in[1] - x_in[2]));

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
        ComplexD2S t0 = x_in[0] + x_in[2];
        ComplexD2S t1 = x_in[1] + x_in[3];
        ComplexD2S t2 = x_in[0] - x_in[2];
        ComplexD2S t3;
        if (forward)
            t3 = mul_i_D2S(x_in[3] - x_in[1]);
        else
            t3 = mul_i_D2S(x_in[1] - x_in[3]);

        x_out[0] = t0 + t1;
        x_out[1] = t2 + t3;
        x_out[2] = t0 - t1;
        x_out[3] = t2 - t3;
        return;
    }

    // Implementation for radix = 5
    if (radix == 5)
    {
        const ComplexD2S k1 = broadcast64_D2S(cos(2.0*M_PI*1.0/5.0));
        const ComplexD2S k2 = broadcast64_D2S(sin(2.0*M_PI*1.0/5.0));
        const ComplexD2S k3 = broadcast64_D2S(-cos(2.0*M_PI*2.0/5.0));
        const ComplexD2S k4 = broadcast64_D2S(sin(2.0*M_PI*2.0/5.0));

        ComplexD2S t0 = x_in[1] + x_in[4];
        ComplexD2S t1 = x_in[2] + x_in[3];
        ComplexD2S t2 = x_in[1] - x_in[4];
        ComplexD2S t3 = x_in[2] - x_in[3];
        ComplexD2S t4 = x_in[0] + k1*t0 - k3*t1;
        ComplexD2S t5 = x_in[0] + k1*t1 - k3*t0;
        ComplexD2S t6 = mul_i_D2S(k2*t2 + k4*t3);
        ComplexD2S t7 = mul_i_D2S(k4*t2 - k2*t3);

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
        const ComplexD2S k0 = broadcast64_D2S(0.5);
        ComplexD2S k1;
        if (forward)
            k1 = broadcast64_D2S(0.5*sqrt(3.0));
        else
            k1 = broadcast64_D2S(-0.5*sqrt(3.0));

        ComplexD2S t6 = x_in[2] + x_in[4];
        ComplexD2S t7 = x_in[1] + x_in[5];
        ComplexD2S t8 = x_in[0] - k0*t6;
        ComplexD2S t9 = x_in[3] - k0*t7;
        ComplexD2S t10 = mul_i_D2S(k1*(x_in[4] - x_in[2]));
        ComplexD2S t11 = mul_i_D2S(k1*(x_in[5] - x_in[1]));
        ComplexD2S t0 = x_in[0] + t6;
        ComplexD2S t1 = x_in[3] + t7;
        ComplexD2S t2 = t8 + t10;
        ComplexD2S t3 = t11 - t9;
        ComplexD2S t4 = t8 - t10;
        ComplexD2S t5 = t9 + t11;
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
        const ComplexD2S k = broadcast64_D2S(sqrt(0.5));

        ComplexD2S t12 = x_in[1] + x_in[5];
        ComplexD2S t13 = x_in[3] + x_in[7];
        ComplexD2S t14 = x_in[1] - x_in[5];
        ComplexD2S t15 = x_in[7] - x_in[3];

        ComplexD2S t1 = t12 + t13;
        ComplexD2S t5;
        if (forward)
            t5 = mul_i_D2S(t13 - t12);
        else
            t5 = mul_i_D2S(t12 - t13);

        ComplexD2S t16 = k*(t14 + t15);
        ComplexD2S t17;
        if (forward)
            t17 = k*mul_i_D2S(t15 - t14);
        else
            t17 = k*mul_i_D2S(t14 - t15);
        ComplexD2S t3 = t16 + t17;
        ComplexD2S t7 = t17 - t16;

        ComplexD2S t8  = x_in[0] + x_in[4];
        ComplexD2S t9  = x_in[2] + x_in[6];
        ComplexD2S t10 = x_in[0] - x_in[4];
        ComplexD2S t11;
        if (forward)
            t11 = mul_i_D2S(x_in[2] - x_in[6]);
        else
            t11 = mul_i_D2S(x_in[6] - x_in[2]);
        ComplexD2S t0  = t8 + t9;
        ComplexD2S t4  = t8 - t9;
        ComplexD2S t2  = t10 - t11;
        ComplexD2S t6  = t10 + t11;

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
            ComplexD2S w;
            double re = coeff[2*radix*i + 2*j + 0];
            double im = coeff[2*radix*i + 2*j + 1];
            w.real = _mm_setr_pd(re,re);
            w.imag = _mm_setr_pd(im,im);

            x_out[i] = x_out[i] + mul_w_D2S<forward>(x_in[j], w);
        }
    }
}
