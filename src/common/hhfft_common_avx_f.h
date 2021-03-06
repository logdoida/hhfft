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

// This header contains some small functions that are used many times

#ifndef HHFFT_COMMON_AVX_F
#define HHFFT_COMMON_AVX_F

#include "hhfft_common_f.h"
#include <immintrin.h>


//////////////////////////////////////// ComplexF /////////////////////////////////////////

// contains a single complex number: [r i x x] (note the two extra padding values that can be anything)
typedef __m128 ComplexF;

// Read complex numbers
inline ComplexF load_F(float r, float i)
{    
    return _mm_setr_ps(r,i,0,0);
}
inline ComplexF load_F(const float *v)
{    
    ComplexF x = _mm_setzero_ps();
    // Note according to intriniscs guide, memory address does not need to be aligned
    return _mm_loadl_pi(x, (const __m64*) v);
}
inline ComplexF broadcast32_F(float x)
{
    return _mm_setr_ps(x,x,0,0);
}
// Load only real part, imaginary part is set to zero
inline ComplexF load_real_F(const float *v)
{
    return _mm_load_ss(v);
}

// Store
inline void store_F(ComplexF val, float &r, float &i)
{
    float v[2];
    // Note according to intriniscs guide, memory address does not need to be aligned
    _mm_storel_pi((__m64*) v, val);
    r = val[0]; i = val[1];
}
inline void store_F(ComplexF val, float *v)
{
    // It is assumed that there is no requirement for alignment, similar to _mm_loadl_pi
    _mm_storel_pi((__m64*) v, val);
}
// Store only real part
inline void store_real_F(ComplexF val, float *v)
{
    _mm_store_ss(v, val);
}

// Changes signs of [a1 a2 x x] using [s1 s2 x x]. s should contain only 0.0 and -0.0
inline ComplexF change_sign_F(ComplexF a, ComplexF s)
{
    return _mm_xor_ps(a,s);
}

const ComplexF const1_F = load_F(0.0f, -0.0f);
//const ComplexF const2_F = load_F(-0.0f, 0.0f);
const ComplexF const3_F = _mm_setr_ps(0.0f, -0.0f, 0.0f, 0.0f);
const ComplexF const4_F = _mm_setr_ps(0.0f, 0.0f, -0.0f, 0.0f);

// Multiplies complex numbers. If other of them changes more frequently, set it to b.
inline ComplexF mul_F(ComplexF a, ComplexF b)
{ 
    ComplexF a1 = _mm_permute_ps(a, 0*1 + 1*4 + 0*16 + 1*64);
    ComplexF b1 = _mm_permute_ps(b, 0*1 + 1*4 + 1*16 + 0*64);

    ComplexF t1 = change_sign_F(a1, const3_F);
    ComplexF t2 = _mm_mul_ps(t1, b1);

    ComplexF y = _mm_hadd_ps(t2, t2);
    return y;
}


// Calculates a*conj(b)
inline ComplexF mul_conj_F(ComplexF a, ComplexF b)
{
    ComplexF a1 = _mm_permute_ps(a, 0*1 + 1*4 + 0*16 + 1*64);
    ComplexF b1 = _mm_permute_ps(b, 0*1 + 1*4 + 1*16 + 0*64);

    ComplexF t1 = change_sign_F(a1, const4_F);
    ComplexF t2 = _mm_mul_ps(t1, b1);

    ComplexF y = _mm_hadd_ps(t2, t2);
    return y;
}

// Multiplies two packed complex numbers. The forward means a*b, inverse a*conj(b)
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
    return change_sign_F(x,const1_F);
}

// Multiply with i
inline ComplexF mul_i_F(ComplexF a)
{
    ComplexF a1 = change_sign_F(a, const1_F);
    ComplexF y = _mm_permute_ps(a1, 1);
    return y;
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
        const ComplexF k1 = broadcast32_F(float(0.5*sqrt(3.0)));
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
        const ComplexF k1 = broadcast32_F(float(cos(2.0*M_PI*1.0/5.0)));
        const ComplexF k2 = broadcast32_F(float(sin(2.0*M_PI*1.0/5.0)));
        const ComplexF k3 = broadcast32_F(float(-cos(2.0*M_PI*2.0/5.0)));
        const ComplexF k4 = broadcast32_F(float(sin(2.0*M_PI*2.0/5.0)));

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
            k1 = broadcast32_F(float(0.5*sqrt(3.0)));
        else
            k1 = broadcast32_F(float(-0.5*sqrt(3.0)));

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
        const ComplexF k = broadcast32_F(float(sqrt(0.5)));

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

//////////////////////////////////////// ComplexF2 /////////////////////////////////////////

// contains two complex number: [r i r i]
typedef __m128 ComplexF2;

// Read complex numbers
inline ComplexF2 load_F2(float r1, float i1, float r2, float i2)
{
    return _mm_setr_ps(r1,i1,r2,i2);
}
inline ComplexF2 load_F2(const float *v)
{
    return _mm_loadu_ps(v);
}

// Loads same complex number twice: [r i] -> [r i r i]
inline ComplexF2 broadcast64_F2(const float *v)
{
    ComplexF x = _mm_setzero_ps();
    x = _mm_loadl_pi(x, (const __m64*) v);
    return _mm_loadh_pi(x, (const __m64*) v);
}

// Loads same number four times: [x] -> [x x x x]
inline ComplexF2 broadcast32_F2(float x)
{
    return _mm_set1_ps(x);
}

// Load only real part, imaginary part is set to zero. [r1 r2] -> [r1 0 r2 0]
inline ComplexF2 load_real_F2(const float *v)
{
    ComplexF2 a = _mm_load_ss(v);
    ComplexF2 b = _mm_load_ss(v + 1);

    return _mm_shuffle_ps(a, b, 1*0 + 4*1 + 16*0 + 64*1);
}

// Combines complex number from two separate memory locations: [a1 a2 x x], [b1 b2 x x] -> [a1 a2 b1 b2]
inline ComplexF2 load_two_64_F2(const float *a, const float *b)
{
    ComplexF x = _mm_setzero_ps();
    x = _mm_loadl_pi(x, (const __m64*) a);
    return _mm_loadh_pi(x, (const __m64*) b);
}

// Combines two complex numbers
inline ComplexF2 combine_two_64_F2(ComplexF a, ComplexF b)
{
    return _mm_shuffle_ps(a, b, 0*1 + 1*4 + 0*16 + 1*64);
}

// Combines complex number from two separate real and imaginary memory locations: [r1 r2], [i1 i2] -> [r1 i1 r2 i2]
inline ComplexF2 load_real_imag_F2(const float *re, const float *im)
{
    ComplexF2 x = load_two_64_F2(re, im);
    return _mm_permute_ps(x, 0*1 + 2*4 + 1*16 + 3*64);
}

// Store
inline void store_F2(ComplexF2 val, float &r1, float &i1, float &r2, float &i2)
{
    float v[4];
    _mm_storeu_ps(v, val);
    r1 = val[0]; i1 = val[1]; r2 = val[2]; i2 = val[3];
}
inline void store_F2(ComplexF2 val, float *v)
{
    _mm_storeu_ps(v, val);
}

// Divides the complex numbers to two separate memory locations: [a1 a2 b1 b2] -> [a1 a2], [b1 b2]
inline void store_two_64_F2(ComplexF2 val, float *a, float *b)
{
    _mm_storel_pi((__m64*) a, val);
    _mm_storeh_pi((__m64*) b, val);
}

// Stores complex number to two separate real and imaginary memory locations: [r1 i1 r2 i2] -> [r1 r2], [i1 i2]
inline void store_real_imag_F2(ComplexF2 val, float *re, float *im)
{
    ComplexF2 x = _mm_permute_ps(val, 0*1 + 2*4 + 1*16 + 3*64);
    store_two_64_F2(x, re, im);
}

// Divides the complex numbers to two separate numbers: [a1 a2 b1 b2] -> [a1 a2 x x], [b1 b2 x x]
inline void divide_two_64_F2(ComplexF2 val, ComplexF &a, ComplexF &b)
{
    a = val;
    b = _mm_movehl_ps(val,val);
}

// Store only real parts [r1 x r2 x] -> [r1 r2]
inline void store_real_F2(ComplexF2 val, float *v)
{
    _mm_store_ss(v, val);    
    ComplexF2 a = _mm_permute_ps(val, 2);
    _mm_store_ss(v + 1, a);
}

// Changes signs of [a1 a2 a3 a4] using [s1 s2 s3 s4]. s should contain only 0.0 and -0.0
inline ComplexF2 change_sign_F2(ComplexF2 a, ComplexF2 s)
{
    return _mm_xor_ps(a,s);
}

const ComplexF2 const1_F2 = load_F2(0.0f, -0.0f, 0.0f, -0.0f);
//const ComplexF2 const2_F2 = load_F2(-0.0f, 0.0f, -0.0f, 0.0f);

// Multiplies complex numbers. If other of them changes more frequently, set it to b.
inline ComplexF2 mul_F2(ComplexF2 a, ComplexF2 b)
{    
    ComplexF2 a1 = _mm_permute_ps(a, 0*1 + 0*4 + 2*16 + 2*64);
    ComplexF2 a2 = _mm_permute_ps(a, 1*1 + 1*4 + 3*16 + 3*64);
    ComplexF2 b1 = _mm_permute_ps(b, 1*1 + 0*4 + 3*16 + 2*64);

    ComplexF2 t1 = _mm_mul_ps(a1, b);
    ComplexF2 t2 = _mm_mul_ps(a2, b1);

    return _mm_addsub_ps(t1,t2);
}

// Calculates a*conj(b)
inline ComplexF2 mul_conj_F2(ComplexF2 a, ComplexF2 b)
{
    ComplexF2 a1 = _mm_permute_ps(a, 0*1 + 0*4 + 2*16 + 2*64);
    ComplexF2 a2 = _mm_permute_ps(a, 1*1 + 1*4 + 3*16 + 3*64);
    ComplexF2 b1 = _mm_permute_ps(b, 1*1 + 0*4 + 3*16 + 2*64);

    ComplexF2 t1 = _mm_mul_ps(a1, b);
    ComplexF2 t2 = _mm_mul_ps(a2, b1);
    ComplexF2 t3 = change_sign_F2(t1, const1_F2);

    return _mm_add_ps(t2,t3);
}

// Multiplies two packed complex numbers. The forward means a*b, inverse a*conj(b)
template<bool forward> inline ComplexF2 mul_w_F2(ComplexF2 a, ComplexF2 b)
{
    if (forward)
    {
        return mul_F2(a,b);
    } else
    {
        return mul_conj_F2(a,b);
    }
}

// Complex conjugate
inline ComplexF2 conj_F2(ComplexF2 x)
{
    return change_sign_F2(x,const1_F2);
}

// Multiply with i
inline ComplexF2 mul_i_F2(ComplexF2 a)
{
    ComplexF2 a1 = change_sign_F2(a, const1_F2);
    ComplexF2 y = _mm_permute_ps(a1, 1*1 + 0*4 + 3*16 + 2*64);
    return y;
}

template<size_t radix, bool forward> inline __attribute__((always_inline)) void multiply_coeff_F2(const ComplexF2 *x_in, ComplexF2 *x_out)
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
        const ComplexF2 k0 = broadcast32_F2(0.5);
        const ComplexF2 k1 = broadcast32_F2(float(0.5*sqrt(3.0)));
        ComplexF2 t0 = x_in[1] + x_in[2];
        ComplexF2 t1 = x_in[0] - k0*t0;
        ComplexF2 t2 = mul_i_F2(k1*(x_in[1] - x_in[2]));

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
        ComplexF2 t0 = x_in[0] + x_in[2];
        ComplexF2 t1 = x_in[1] + x_in[3];
        ComplexF2 t2 = x_in[0] - x_in[2];
        ComplexF2 t3;
        if (forward)
            t3 = mul_i_F2(x_in[3] - x_in[1]);
        else
            t3 = mul_i_F2(x_in[1] - x_in[3]);

        x_out[0] = t0 + t1;
        x_out[1] = t2 + t3;
        x_out[2] = t0 - t1;
        x_out[3] = t2 - t3;
        return;
    }

    // Implementation for radix = 5
    if (radix == 5)
    {
        const ComplexF2 k1 = broadcast32_F2(float(cos(2.0*M_PI*1.0/5.0)));
        const ComplexF2 k2 = broadcast32_F2(float(sin(2.0*M_PI*1.0/5.0)));
        const ComplexF2 k3 = broadcast32_F2(float(-cos(2.0*M_PI*2.0/5.0)));
        const ComplexF2 k4 = broadcast32_F2(float(sin(2.0*M_PI*2.0/5.0)));

        ComplexF2 t0 = x_in[1] + x_in[4];
        ComplexF2 t1 = x_in[2] + x_in[3];
        ComplexF2 t2 = x_in[1] - x_in[4];
        ComplexF2 t3 = x_in[2] - x_in[3];
        ComplexF2 t4 = x_in[0] + k1*t0 - k3*t1;
        ComplexF2 t5 = x_in[0] + k1*t1 - k3*t0;
        ComplexF2 t6 = mul_i_F2(k2*t2 + k4*t3);
        ComplexF2 t7 = mul_i_F2(k4*t2 - k2*t3);

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
        const ComplexF2 k0 = broadcast32_F2(0.5);
        ComplexF2 k1;
        if (forward)
            k1 = broadcast32_F2(float(0.5*sqrt(3.0)));
        else
            k1 = broadcast32_F2(float(-0.5*sqrt(3.0)));

        ComplexF2 t6 = x_in[2] + x_in[4];
        ComplexF2 t7 = x_in[1] + x_in[5];
        ComplexF2 t8 = x_in[0] - k0*t6;
        ComplexF2 t9 = x_in[3] - k0*t7;
        ComplexF2 t10 = mul_i_F2(k1*(x_in[4] - x_in[2]));
        ComplexF2 t11 = mul_i_F2(k1*(x_in[5] - x_in[1]));
        ComplexF2 t0 = x_in[0] + t6;
        ComplexF2 t1 = x_in[3] + t7;
        ComplexF2 t2 = t8 + t10;
        ComplexF2 t3 = t11 - t9;
        ComplexF2 t4 = t8 - t10;
        ComplexF2 t5 = t9 + t11;
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
        const ComplexF2 k = broadcast32_F2(float(sqrt(0.5)));

        ComplexF2 t12 = x_in[1] + x_in[5];
        ComplexF2 t13 = x_in[3] + x_in[7];
        ComplexF2 t14 = x_in[1] - x_in[5];
        ComplexF2 t15 = x_in[7] - x_in[3];

        ComplexF2 t1 = t12 + t13;
        ComplexF2 t5;
        if (forward)
            t5 = mul_i_F2(t13 - t12);
        else
            t5 = mul_i_F2(t12 - t13);

        ComplexF2 t16 = k*(t14 + t15);
        ComplexF2 t17;
        if (forward)
            t17 = k*mul_i_F2(t15 - t14);
        else
            t17 = k*mul_i_F2(t14 - t15);
        ComplexF2 t3 = t16 + t17;
        ComplexF2 t7 = t17 - t16;

        ComplexF2 t8  = x_in[0] + x_in[4];
        ComplexF2 t9  = x_in[2] + x_in[6];
        ComplexF2 t10 = x_in[0] - x_in[4];
        ComplexF2 t11;
        if (forward)
            t11 = mul_i_F2(x_in[6] - x_in[2]);
        else
            t11 = mul_i_F2(x_in[2] - x_in[6]);
        ComplexF2 t0  = t8 + t9;
        ComplexF2 t4  = t8 - t9;
        ComplexF2 t2  = t10 + t11;
        ComplexF2 t6  = t10 - t11;

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
        ComplexF2 sums[3];
        ComplexF2 diffs[3];
        for (size_t i = 0; i < 3; i++)
        {
            sums[i] = x_in[i+1] + x_in[radix - i - 1];
            diffs[i] = mul_i_F2(x_in[radix - i - 1] - x_in[i+1]);
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
            ComplexF2 x = load_F(0,0);
            for (size_t j = 0; j < 3; j++)
            {
                ComplexF2 coeff = broadcast32_F2(coeff_cos[3*i + j]);
                x += coeff*sums[j];
            }
            x_out[i+1] += x;
            x_out[radix - i - 1] += x;
        }

        // use sin-coefficients
        for (size_t i = 0; i < 3; i++)
        {
            ComplexF2 x = load_F(0,0);
            for (size_t j = 0; j < 3; j++)
            {
                ComplexF2 coeff = broadcast32_F2(coeff_sin[3*i + j]);
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

template<size_t radix> inline __attribute__((always_inline)) void multiply_coeff_real_odd_forward_F2(const ComplexF2 *x_in, ComplexF2 *x_out)
{
    // Use the normal function
    ComplexF2 x_temp[radix];
    multiply_coeff_F2<radix,true>(x_in, x_temp);

    // And then apply some reordering and conjugation
    for (size_t i = 0; i < radix; i+=2)
    {
        x_out[i] = x_temp[i/2];
    }
    for (size_t i = 1; i < radix; i+=2)
    {
        x_out[i] = conj_F2(x_temp[radix - (i+1)/2]);
    }
}

template<size_t radix, bool forward> inline __attribute__((always_inline)) void multiply_twiddle_F2(const ComplexF2 *x_in, ComplexF2 *x_out, const ComplexF2 *twiddle_factors)
{
    // It is assumed that first twiddle factors are always (1 + 0i)
    x_out[0] = x_in[0];

    // Read in the values used in this step and multiply them with twiddle factors
    for (size_t j = 1; j < radix; j++)
    {
        ComplexF2 x = x_in[j];
        ComplexF2 w = twiddle_factors[j];

        x_out[j] = mul_w_F2<forward>(x, w);
    }
}

// Multiplies the first values with twiddle factors and then conjugates them and saves as last values
template<size_t radix> inline __attribute__((always_inline)) void multiply_conj_twiddle_odd_F2(const ComplexF2 *x_in, ComplexF2 *x_out, const ComplexF2 *twiddle_factors)
{
    // It is assumed that first twiddle factors are always (1 + 0i)
    x_out[0] = x_in[0];

    // multiply values with twiddle factors and conjugate them
    for (size_t j = 1; j <= radix/2; j++)
    {
        ComplexF2 x = x_in[j];

        // multiplication with twiddle factors is done first
        ComplexF2 w = twiddle_factors[j];
        ComplexF2 x_temp = mul_w_F2<true>(x, w);

        x_out[j] = x_temp;
        x_out[radix-j] = conj_F2(x_temp);
    }
}

//////////////////////////////////////// ComplexF4 /////////////////////////////////////////

// contains four complex number: [r i r i r i r i]
typedef __m256 ComplexF4;

// Read complex numbers
inline ComplexF4 load_F4(float r1, float i1, float r2, float i2, float r3, float i3, float r4, float i4)
{
    return _mm256_setr_ps(r1,i1,r2,i2,r3,i3,r4,i4);
}
inline ComplexF4 load_F4(const float *v)
{
    return _mm256_loadu_ps(v);
}

// Loads same complex number four times: [r i] -> [r i r i r i r i]
inline ComplexF4 broadcast64_F4(const float *v)
{
    __m128 x = _mm_setzero_ps();
    x = _mm_loadl_pi(x, (const __m64*) v);
    x = _mm_loadh_pi(x, (const __m64*) v);

    const ComplexF4 xx = _mm256_castps128_ps256(x);
    return _mm256_insertf128_ps (xx, x, 1);
}

// Loads same number four times: [x] -> [x x x x]
inline ComplexF4 broadcast32_F4(float x)
{
    return _mm256_set1_ps(x);
}

// Load only real part, imaginary part is set to zero. [r1 r2 r3 r4] -> [r1 0 r2 0 r3 0 r4 0]
inline ComplexF4 load_real_F4(const float *v)
{
    // TODO is there a faster way?
    ComplexF2 a = _mm_load_ss(v);
    ComplexF2 b = _mm_load_ss(v + 1);
    ComplexF2 c = _mm_load_ss(v + 2);
    ComplexF2 d = _mm_load_ss(v + 3);

    ComplexF2 ab = _mm_shuffle_ps(a, b, 1*0 + 4*1 + 16*0 + 64*1);
    ComplexF2 cd = _mm_shuffle_ps(c, d, 1*0 + 4*1 + 16*0 + 64*1);

    const ComplexF4 xx = _mm256_castps128_ps256(ab);
    return _mm256_insertf128_ps (xx, cd, 1);
}

// Combines complex number from two separate memory locations: [a1 a2 a3 a4], [b1 b2 b3 b4] -> [a1 a2 a3 a4 b1 b2 b3 b4]
inline ComplexF4 load_two_128_F4(const float *a, const float *b)
{
    ComplexF2 aa = _mm_loadu_ps(a);
    ComplexF2 bb = _mm_loadu_ps(b);

    const ComplexF4 xx = _mm256_castps128_ps256(aa);
    return _mm256_insertf128_ps (xx, bb, 1);
}

// Combines complex number from four separate memory locations: [a1 a2], [b1 b2], [c1 c2], [d1 d2] -> [a1 a2 b1 b2 c1 c2 c3 c4]
inline ComplexF4 load_four_64_F4(const float *a, const float *b, const float *c, const float *d)
{
    ComplexF2 ab = load_two_64_F2(a,b);        
    ComplexF2 cd = load_two_64_F2(c,d);
    const ComplexF4 xx = _mm256_castps128_ps256(ab);
    return _mm256_insertf128_ps (xx, cd, 1);
}

// Store
inline void store_F4(ComplexF4 val, float &r1, float &i1, float &r2, float &i2, float &r3, float &i3, float &r4, float &i4)
{
    float v[8];
    _mm256_storeu_ps(v, val);
    r1 = val[0]; i1 = val[1]; r2 = val[2]; i2 = val[3]; r3 = val[4]; i3 = val[5]; r4 = val[6]; i4 = val[7];
}
inline void store_F4(ComplexF4 val, float *v)
{
    _mm256_storeu_ps(v, val);
}

// Divides the complex numbers to two separate memory locations: [a1 a2 a3 a4 b1 b2 b3 b4] -> [a1 a2 a3 a4], [b1 b2 b3 b4]
inline void store_two_128_F4(ComplexF4 val, float *a, float *b)
{
    ComplexF2 aa = _mm256_castps256_ps128(val);
    store_F2(aa, a);
    ComplexF2 bb = _mm256_extractf128_ps(val, 1);
    store_F2(bb, b);
}

// Combines complex number from two separate real and imaginary memory locations: [r1 r2 r3 r4], [i1 i2 i3 i4] -> [r1 i1 r2 i2 r3 i3 r4 i4]
inline ComplexF4 load_real_imag_F4(const float *re, const float *im)
{
    ComplexF2 x0 = load_F2(re);
    ComplexF2 x1 = load_F2(im);
    ComplexF2 t0 = _mm_shuffle_ps(x0, x1, 0*1 + 1*4 + 0*16 + 1*64);
    ComplexF2 t1 = _mm_shuffle_ps(x0, x1, 2*1 + 3*4 + 2*16 + 3*64);

    ComplexF4 t3 = _mm256_castps128_ps256(t0);
    t3 = _mm256_insertf128_ps (t3, t1, 1);

    return  _mm256_permute_ps(t3, 0*1 + 2*4 + 1*16 + 3*64);
}

// Divides the complex numbers to four separate memory locations: [a1 a2 b1 b2 c1 c2 c3 c4] -> [a1 a2], [b1 b2], [c1 c2], [d1 d2]
inline void store_four_64_F4(ComplexF4 val, float *a, float *b, float *c, float *d)
{
    ComplexF2 ab = _mm256_castps256_ps128(val);
    store_two_64_F2(ab, a, b);
    ComplexF2 cd = _mm256_extractf128_ps(val, 1);
    store_two_64_F2(cd, c, d);
}

// Stores complex number to two separate real and imaginary memory locations: [r1 i1 r2 i2 r3 i3 r4 i4] -> [r1 r2 r3 r4], [i1 i2 i3 i4]
inline void store_real_imag_F4(ComplexF4 val, float *re, float *im)
{
    ComplexF4 t3 = _mm256_permute_ps(val, 0*1 + 2*4 + 1*16 + 3*64);
    ComplexF2 t0 = _mm256_castps256_ps128(t3);
    ComplexF2 t1 = _mm256_extractf128_ps(t3, 1);
    ComplexF2 x0 = _mm_shuffle_ps(t0, t1, 0*1 + 1*4 + 0*16 + 1*64);
    ComplexF2 x1 = _mm_shuffle_ps(t0, t1, 2*1 + 3*4 + 2*16 + 3*64);
    store_F2(x0, re);
    store_F2(x1, im);
}

// Divides the complex numbers to two separate numbers: [a1 a2 a3 a4 b1 b2 b3 b4] -> [a1 a2 a3 a4], [b1 b2 b3 b4]
inline void divide_two_128_F4(ComplexF4 val, ComplexF2 &a, ComplexF2 &b)
{
    a = _mm256_castps256_ps128(val);
    b = _mm256_extractf128_ps(val, 1);
}

// Store only real parts [r1 x r2 x r3 x r4 x] -> [r1 r2 r3 r4]
inline void store_real_F4(ComplexF4 val, float *v)
{
    ComplexF2 a = _mm256_castps256_ps128(val);
    ComplexF2 b = _mm256_extractf128_ps(val, 1);

    ComplexF2 ab = _mm_shuffle_ps(a, b, 0*1 + 2*4 + 0*16 + 2*64);
    store_F2(ab, v);
}

// Changes signs of [a1 a2 a3 a4 a5 a6 a7 a8] using [s1 s2 s3 s4 s5 s6 s7 s8]. s should contain only 0.0 and -0.0
inline ComplexF4 change_sign_F4(ComplexF4 a, ComplexF4 s)
{
    return _mm256_xor_ps(a,s);
}


const ComplexF4 const1_F4 = load_F4(0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f);
//const ComplexF4 const2_F4 = load_F4(-0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f);

// Multiplies complex numbers. If other of them changes more frequently, set it to b.
inline ComplexF4 mul_F4(ComplexF4 a, ComplexF4 b)
{
    ComplexF4 a1 = _mm256_permute_ps(a, 0*1 + 0*4 + 2*16 + 2*64);
    ComplexF4 a2 = _mm256_permute_ps(a, 1*1 + 1*4 + 3*16 + 3*64);
    ComplexF4 b1 = _mm256_permute_ps(b, 1*1 + 0*4 + 3*16 + 2*64);

    ComplexF4 t1 = _mm256_mul_ps(a1, b);
    ComplexF4 t2 = _mm256_mul_ps(a2, b1);

    return _mm256_addsub_ps(t1,t2);
}

// Calculates a*conj(b)
inline ComplexF4 mul_conj_F4(ComplexF4 a, ComplexF4 b)
{
    ComplexF4 a1 = _mm256_permute_ps(a, 0*1 + 0*4 + 2*16 + 2*64);
    ComplexF4 a2 = _mm256_permute_ps(a, 1*1 + 1*4 + 3*16 + 3*64);
    ComplexF4 b1 = _mm256_permute_ps(b, 1*1 + 0*4 + 3*16 + 2*64);

    ComplexF4 t1 = _mm256_mul_ps(a1, b);
    ComplexF4 t2 = _mm256_mul_ps(a2, b1);
    ComplexF4 t3 = change_sign_F4(t1, const1_F4);

    return _mm256_add_ps(t2,t3);
}

// Multiplies two packed complex numbers. The forward means a*b, inverse a*conj(b)
template<bool forward> inline ComplexF4 mul_w_F4(ComplexF4 a, ComplexF4 b)
{
    if (forward)
    {
        return mul_F4(a,b);
    } else
    {
        return mul_conj_F4(a,b);
    }
}

// Complex conjugate
inline ComplexF4 conj_F4(ComplexF4 x)
{
    return change_sign_F4(x,const1_F4);
}

// Multiply with i
inline ComplexF4 mul_i_F4(ComplexF4 a)
{
    ComplexF4 a1 = change_sign_F4(a, const1_F4);
    ComplexF4 y = _mm256_permute_ps(a1, 1*1 + 0*4 + 3*16 + 2*64);
    return y;
}

template<size_t radix, bool forward> inline __attribute__((always_inline)) void multiply_coeff_F4(const ComplexF4 *x_in, ComplexF4 *x_out)
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
        const ComplexF4 k0 = broadcast32_F4(0.5);
        const ComplexF4 k1 = broadcast32_F4(float(0.5*sqrt(3.0)));
        ComplexF4 t0 = x_in[1] + x_in[2];
        ComplexF4 t1 = x_in[0] - k0*t0;
        ComplexF4 t2 = mul_i_F4(k1*(x_in[1] - x_in[2]));

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
        ComplexF4 t0 = x_in[0] + x_in[2];
        ComplexF4 t1 = x_in[1] + x_in[3];
        ComplexF4 t2 = x_in[0] - x_in[2];
        ComplexF4 t3;
        if (forward)
            t3 = mul_i_F4(x_in[3] - x_in[1]);
        else
            t3 = mul_i_F4(x_in[1] - x_in[3]);

        x_out[0] = t0 + t1;
        x_out[1] = t2 + t3;
        x_out[2] = t0 - t1;
        x_out[3] = t2 - t3;
        return;
    }

    // Implementation for radix = 5
    if (radix == 5)
    {
        const ComplexF4 k1 = broadcast32_F4(float(cos(2.0*M_PI*1.0/5.0)));
        const ComplexF4 k2 = broadcast32_F4(float(sin(2.0*M_PI*1.0/5.0)));
        const ComplexF4 k3 = broadcast32_F4(float(-cos(2.0*M_PI*2.0/5.0)));
        const ComplexF4 k4 = broadcast32_F4(float(sin(2.0*M_PI*2.0/5.0)));

        ComplexF4 t0 = x_in[1] + x_in[4];
        ComplexF4 t1 = x_in[2] + x_in[3];
        ComplexF4 t2 = x_in[1] - x_in[4];
        ComplexF4 t3 = x_in[2] - x_in[3];
        ComplexF4 t4 = x_in[0] + k1*t0 - k3*t1;
        ComplexF4 t5 = x_in[0] + k1*t1 - k3*t0;
        ComplexF4 t6 = mul_i_F4(k2*t2 + k4*t3);
        ComplexF4 t7 = mul_i_F4(k4*t2 - k2*t3);

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
        const ComplexF4 k0 = broadcast32_F4(0.5);
        ComplexF4 k1;
        if (forward)
            k1 = broadcast32_F4(float(0.5*sqrt(3.0)));
        else
            k1 = broadcast32_F4(float(-0.5*sqrt(3.0)));

        ComplexF4 t6 = x_in[2] + x_in[4];
        ComplexF4 t7 = x_in[1] + x_in[5];
        ComplexF4 t8 = x_in[0] - k0*t6;
        ComplexF4 t9 = x_in[3] - k0*t7;
        ComplexF4 t10 = mul_i_F4(k1*(x_in[4] - x_in[2]));
        ComplexF4 t11 = mul_i_F4(k1*(x_in[5] - x_in[1]));
        ComplexF4 t0 = x_in[0] + t6;
        ComplexF4 t1 = x_in[3] + t7;
        ComplexF4 t2 = t8 + t10;
        ComplexF4 t3 = t11 - t9;
        ComplexF4 t4 = t8 - t10;
        ComplexF4 t5 = t9 + t11;
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
        const ComplexF4 k = broadcast32_F4(float(sqrt(0.5)));

        ComplexF4 t12 = x_in[1] + x_in[5];
        ComplexF4 t13 = x_in[3] + x_in[7];
        ComplexF4 t14 = x_in[1] - x_in[5];
        ComplexF4 t15 = x_in[7] - x_in[3];

        ComplexF4 t1 = t12 + t13;
        ComplexF4 t5;
        if (forward)
            t5 = mul_i_F4(t13 - t12);
        else
            t5 = mul_i_F4(t12 - t13);

        ComplexF4 t16 = k*(t14 + t15);
        ComplexF4 t17;
        if (forward)
            t17 = k*mul_i_F4(t15 - t14);
        else
            t17 = k*mul_i_F4(t14 - t15);
        ComplexF4 t3 = t16 + t17;
        ComplexF4 t7 = t17 - t16;

        ComplexF4 t8  = x_in[0] + x_in[4];
        ComplexF4 t9  = x_in[2] + x_in[6];
        ComplexF4 t10 = x_in[0] - x_in[4];
        ComplexF4 t11;
        if (forward)
            t11 = mul_i_F4(x_in[2] - x_in[6]);
        else
            t11 = mul_i_F4(x_in[6] - x_in[2]);
        ComplexF4 t0  = t8 + t9;
        ComplexF4 t4  = t8 - t9;
        ComplexF4 t2  = t10 - t11;
        ComplexF4 t6  = t10 + t11;

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
        ComplexF4 sums[3];
        ComplexF4 diffs[3];
        for (size_t i = 0; i < 3; i++)
        {
            sums[i] = x_in[i+1] + x_in[radix - i - 1];
            diffs[i] = mul_i_F4(x_in[radix - i - 1] - x_in[i+1]);
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
            ComplexF4 x = _mm256_setzero_ps();
            for (size_t j = 0; j < 3; j++)
            {
                ComplexF4 coeff = broadcast32_F4(coeff_cos[3*i + j]);
                x += coeff*sums[j];
            }
            x_out[i+1] += x;
            x_out[radix - i - 1] += x;
        }

        // use sin-coefficients
        for (size_t i = 0; i < 3; i++)
        {
            ComplexF4 x = _mm256_setzero_ps();
            for (size_t j = 0; j < 3; j++)
            {
                ComplexF4 coeff = broadcast32_F4(coeff_sin[3*i + j]);
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

template<size_t radix> inline __attribute__((always_inline)) void multiply_coeff_real_odd_forward_F4(const ComplexF4 *x_in, ComplexF4 *x_out)
{
    // Use the normal function
    ComplexF4 x_temp[radix];
    multiply_coeff_F4<radix,true>(x_in, x_temp);

    // And then apply some reordering and conjugation
    for (size_t i = 0; i < radix; i+=2)
    {
        x_out[i] = x_temp[i/2];
    }
    for (size_t i = 1; i < radix; i+=2)
    {
        x_out[i] = conj_F4(x_temp[radix - (i+1)/2]);
    }
}

template<size_t radix, bool forward> inline __attribute__((always_inline)) void multiply_twiddle_F4(const ComplexF4 *x_in, ComplexF4 *x_out, const ComplexF4 *twiddle_factors)
{
    // It is assumed that first twiddle factors are always (1 + 0i)
    x_out[0] = x_in[0];

    // Read in the values used in this step and multiply them with twiddle factors
    for (size_t j = 1; j < radix; j++)
    {
        ComplexF4 x = x_in[j];
        ComplexF4 w = twiddle_factors[j];

        x_out[j] = mul_w_F4<forward>(x, w);
    }
}

// Multiplies the first values with twiddle factors and then conjugates them and saves as last values
template<size_t radix> inline __attribute__((always_inline)) void multiply_conj_twiddle_odd_F4(const ComplexF4 *x_in, ComplexF4 *x_out, const ComplexF4 *twiddle_factors)
{
    // It is assumed that first twiddle factors are always (1 + 0i)
    x_out[0] = x_in[0];

    // multiply values with twiddle factors and conjugate them
    for (size_t j = 1; j <= radix/2; j++)
    {
        ComplexF4 x = x_in[j];

        // multiplication with twiddle factors is done first
        ComplexF4 w = twiddle_factors[j];
        ComplexF4 x_temp = mul_w_F4<true>(x, w);

        x_out[j] = x_temp;
        x_out[radix-j] = conj_F4(x_temp);
    }
}

// Transposes 4x4 or 8x4 complex numbers
template<size_t radix> inline __attribute__((always_inline)) void transpose_F4(const ComplexF4 *x_in, ComplexF4 *x_out)
{
    if (radix == 4)
    {
        ComplexF4 temp[4];
        temp[0] = _mm256_permute2f128_ps(x_in[0], x_in[2], 0*1 + 2*16);
        temp[1] = _mm256_permute2f128_ps(x_in[0], x_in[2], 1*1 + 3*16);
        temp[2] = _mm256_permute2f128_ps(x_in[1], x_in[3], 0*1 + 2*16);
        temp[3] = _mm256_permute2f128_ps(x_in[1], x_in[3], 1*1 + 3*16);

        x_out[0] = _mm256_shuffle_ps(temp[0], temp[2], 0*1 + 1*4 + 0*16 + 1*64);
        x_out[1] = _mm256_shuffle_ps(temp[0], temp[2], 2*1 + 3*4 + 2*16 + 3*64);
        x_out[2] = _mm256_shuffle_ps(temp[1], temp[3], 0*1 + 1*4 + 0*16 + 1*64);
        x_out[3] = _mm256_shuffle_ps(temp[1], temp[3], 2*1 + 3*4 + 2*16 + 3*64);
    }

    if (radix == 8)
    {
        ComplexF4 temp[4];
        temp[0] = _mm256_permute2f128_ps(x_in[0], x_in[2], 0*1 + 2*16);
        temp[1] = _mm256_permute2f128_ps(x_in[0], x_in[2], 1*1 + 3*16);
        temp[2] = _mm256_permute2f128_ps(x_in[1], x_in[3], 0*1 + 2*16);
        temp[3] = _mm256_permute2f128_ps(x_in[1], x_in[3], 1*1 + 3*16);

        x_out[0] = _mm256_shuffle_ps(temp[0], temp[2], 0*1 + 1*4 + 0*16 + 1*64);
        x_out[2] = _mm256_shuffle_ps(temp[0], temp[2], 2*1 + 3*4 + 2*16 + 3*64);
        x_out[4] = _mm256_shuffle_ps(temp[1], temp[3], 0*1 + 1*4 + 0*16 + 1*64);
        x_out[6] = _mm256_shuffle_ps(temp[1], temp[3], 2*1 + 3*4 + 2*16 + 3*64);

        temp[0] = _mm256_permute2f128_ps(x_in[4], x_in[6], 0*1 + 2*16);
        temp[1] = _mm256_permute2f128_ps(x_in[4], x_in[6], 1*1 + 3*16);
        temp[2] = _mm256_permute2f128_ps(x_in[5], x_in[7], 0*1 + 2*16);
        temp[3] = _mm256_permute2f128_ps(x_in[5], x_in[7], 1*1 + 3*16);

        x_out[1] = _mm256_shuffle_ps(temp[0], temp[2], 0*1 + 1*4 + 0*16 + 1*64);
        x_out[3] = _mm256_shuffle_ps(temp[0], temp[2], 2*1 + 3*4 + 2*16 + 3*64);
        x_out[5] = _mm256_shuffle_ps(temp[1], temp[3], 0*1 + 1*4 + 0*16 + 1*64);
        x_out[7] = _mm256_shuffle_ps(temp[1], temp[3], 2*1 + 3*4 + 2*16 + 3*64);
    }
}

#endif
