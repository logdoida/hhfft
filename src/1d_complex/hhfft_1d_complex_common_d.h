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
// Flags ENABLE_COMPLEX_D (SSE2 or AVX or AVX512), ENABLE_COMPLEX_D2 (AVX or AVX512), ENABLE_AVX512 (AVX512) will enable and disable certain features

#include <immintrin.h>
#include <iostream> //For testing...

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
    1, 0, k1, -k2, -k3, -k4, -k3, k4, k1, k2,
    1, 0, -k3, -k4, k1, k2, k1, -k2, -k3, k4,
    1, 0, -k3, k4, k1, -k2, k1, k2, -k3, -k4,
    1, 0, k1, k2, -k3, k4, -k3, -k4, k1, -k2};

static const double k5 = cos(2.0*M_PI*1.0/7.0);
static const double k6 = sin(2.0*M_PI*1.0/7.0);
static const double k7 = -cos(2.0*M_PI*2.0/7.0);
static const double k8 = sin(2.0*M_PI*2.0/7.0);
static const double k9 = -cos(2.0*M_PI*3.0/7.0);
static const double k10 = sin(2.0*M_PI*3.0/7.0);
static const double coeff_radix_7[98] = {
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, k5, -k6, -k7, -k8, -k9, -k10, -k9, k10, -k7, k8, k5, k6,
    1, 0, -k7, -k8, -k9, k10, k5, k6, k5, -k6, -k9, -k10, -k7, k8,
    1, 0, -k9, -k10, k5, k6, -k7, -k8, -k7, k8, k5, -k6, -k9, k10,
    1, 0, -k9, k10, k5, -k6, -k7, k8, -k7, -k8, k5, k6, -k9, -k10,
    1, 0, -k7, k8, -k9, -k10, k5, -k6, k5, k6, -k9, k10, -k7, -k8,
    1, 0, k5, k6, -k7, k8, -k9, k10, -k9, -k10, -k7, -k8, k5, -k6};


//////////////////////////// ComplexD /////////////////////////////////////////
#ifdef ENABLE_COMPLEX_D

// contains a single complex number: [r i]
typedef __m128d ComplexD;

// Read complex numbers
inline ComplexD load(double r, double i)
{
    return _mm_setr_pd(r,i);
}
inline const ComplexD load128(const double *v)
{
    return _mm_loadu_pd(v);
}

// Store
inline void store(ComplexD val, double &r, double &i)
{
    double v[2];
    _mm_storeu_pd(v, val);
    r = val[0]; i = val[1];
}
inline void store(ComplexD val, double *v)
{
    _mm_storeu_pd(v, val);
}

// Changes signs of [x1 x2] using [s1 s2]. s should contain only 0.0 and -0.0
// NOTE this seems to actually be bit slower than plain multiplication? Compare!
inline ComplexD change_sign(ComplexD x, ComplexD s)
{
    return _mm_xor_pd(x,s);
}

const ComplexD const1_128 = load(0.0, -0.0);
const ComplexD const2_128 = load(-0.0, 0.0);

// Multiplies complex numbers. If other of them changes more frequently, set it to b.
#ifdef ENABLE_COMPLEX_D2
// AVX-version is slightly more efficient
inline ComplexD mul(ComplexD a, ComplexD b)
{
    // If AVX-was supported...
    ComplexD a1 = change_sign(a, const1_128);
    ComplexD a2 = _mm_permute_pd(a, 1); // Not supported in sse2

    ComplexD t1 = _mm_mul_pd(a1, b);
    ComplexD t2 = _mm_mul_pd(a2, b);

    ComplexD y = _mm_hadd_pd(t1, t2); // Not supported in sse2
    return y;
}
#else
inline ComplexD mul(ComplexD a, ComplexD b)
{
    ComplexD a1 = change_sign(a, const1_128);
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
#ifdef ENABLE_COMPLEX_D2
// AVX-version is slightly more efficient
inline ComplexD mul_conj(ComplexD a, ComplexD b)
{
    // If AVX-was supported...
    ComplexD a1 = change_sign(a, const2_128);
    ComplexD b1 = _mm_permute_pd(b, 1); // Not supported in sse2

    ComplexD t1 = _mm_mul_pd(a, b);
    ComplexD t2 = _mm_mul_pd(a1, b1);

    ComplexD y = _mm_hadd_pd(t1, t2); // Not supported in sse2
    return y;
}
#else
inline ComplexD mul_conj(ComplexD a, ComplexD b)
{
    ComplexD a1 = change_sign(a, const2_128);
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
template<bool forward> inline ComplexD mul_w(ComplexD a, ComplexD b)
{
    if (forward)
    {
        return mul(a,b);
    } else
    {
        return mul_conj(a,b);
    }
}

// Complex conjugate
inline ComplexD conj(ComplexD x)
{
    return change_sign(x,const1_128);
}

// Multiply with i
inline ComplexD mul_i(ComplexD a)
{
    //const ComplexD const1 = load(0.0, -0.0);
    ComplexD a1 = change_sign(a, const1_128);
    ComplexD y = _mm_permute_pd(a1, 1);
    return y;
}

// For testing
inline std::ostream& operator<<(std::ostream& os, const ComplexD &x)
{
    double v[2];
    store(x, v);
    os << v[0] << ", " << v[1];
    return os;
}

#endif


//////////////////////////// ComplexD2 ////////////////////////////////////////
#ifdef ENABLE_COMPLEX_D2

// contains two complex numbers: [r1 i1 r2 i2]
typedef __m256d ComplexD2;

// Read two complex number
inline ComplexD2 load(double r1, double i1, double r2, double i2)
{
    //return _mm256_set_pd(i2,r2,i1,r1); // Why this order?
    return _mm256_setr_pd(r1,i1,r2,i2); // Reversed. Why this order?
}
inline const ComplexD2 load(const double *v)
{
    return _mm256_loadu_pd(v);
}

// Loads same complex number twice: [r i] -> [r i r i]
inline const ComplexD2 broadcast128(const double *v)
{
    // TODO is this safe? Does alignment cause trouble?
    return _mm256_broadcast_pd((const __m128d*) v);
}

// Store a complex number
inline void store(ComplexD2 val, double &r1, double &i1, double &r2, double &i2)
{
    double v[4];
    _mm256_storeu_pd(v, val);
    r1 = val[0]; i1 = val[1]; r2 = val[2]; i2 = val[3];
}
inline void store(ComplexD2 val, double *v)
{
    _mm256_storeu_pd(v, val);
}

// Changes signs of [x1 x2 x3 x4] using [s1 s2 s3 s4]. s should contain only 0.0 and -0.0
// NOTE this seems to actually be bit slower than plain multiplication. Compare!
inline ComplexD2 change_sign(ComplexD2 x, ComplexD2 s)
{
    return _mm256_xor_pd(x,s);
}

// Defining the constants here or within the functions can have some difference. Compare!
static const ComplexD2 const1 = load(0.0, -0.0, 0.0, -0.0);

// Multiplies two packed complex numbers. If other of them changes more frequently, set it to b.
inline ComplexD2 mul(ComplexD2 a, ComplexD2 b)
{
    ComplexD2 a1 = change_sign(a, const1);
    ComplexD2 a2 = _mm256_permute_pd(a, 1 + 4);

    ComplexD2 t1 = _mm256_mul_pd(a1, b);
    ComplexD2 t2 = _mm256_mul_pd(a2, b);

    ComplexD2 y = _mm256_hadd_pd(t1, t2);
    return y;
}

// Multiplies two packed complex numbers. The forward means a*b, inverse a*conj(b)
template<bool forward> inline ComplexD2 mul_w(ComplexD2 a, ComplexD2 b)
{
    ComplexD2 b1, b2;
    if (forward)
    {
        b1 = change_sign(b, const1);
        b2 = _mm256_permute_pd(b, 1 + 4);
    } else
    {
        b1 = b;
        b2 = _mm256_permute_pd(change_sign(b, const1), 1 + 4);
    }

    ComplexD2 t1 = _mm256_mul_pd(a, b1);
    ComplexD2 t2 = _mm256_mul_pd(a, b2);

    ComplexD2 y = _mm256_hadd_pd(t1, t2);
    return y;
}

// Multiplies packed complex numbers with i
inline ComplexD2 mul_i(ComplexD2 a)
{
    //const ComplexD2 const1 = load(0.0, -0.0, 0.0, -0.0);

    ComplexD2 a1 = change_sign(a, const1);
    ComplexD2 y = _mm256_permute_pd(a1, 1 + 4);
    return y;
}

// For testing
inline std::ostream& operator<<(std::ostream& os, const ComplexD2 &x)
{
    double v[4];
    store(x, v);
    os << v[0] << ", " << v[1] << ", " << v[2] << ", " << v[3];
    return os;
}


#endif


//////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_COMPLEX_D

template<size_t radix, bool forward> inline void multiply_coeff(const ComplexD *x_in, ComplexD *x_out)
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
    } else if (radix == 7)
    {
        coeff = coeff_radix_7;
    }

    // Use temporary storage. This is needed (as usually is) if x_in == x_out
    ComplexD x_temp_in[radix];
    for (size_t j = 0; j < radix; j++)
    {
        x_temp_in[j] = x_in[j];
    }

    for (size_t i = 0; i < radix; i++)
    {
        x_out[i] = load(0.0, 0.0);
        for (size_t j = 0; j < radix; j++)
        {
            ComplexD w = load128(coeff + 2*radix*i + 2*j);

            x_out[i] = x_out[i] + mul_w<forward>(x_temp_in[j], w);
        }
    }
}


template<size_t radix, bool forward> inline void multiply_twiddle(const ComplexD *x_in, ComplexD *x_out, const ComplexD *twiddle_factors)
{
    // It is assumed that first twiddle factors are always (1 + 0i)
    x_out[0] = x_in[0];

    // Read in the values used in this step and multiply them with twiddle factors
    for (size_t j = 1; j < radix; j++)
    {
        ComplexD x = x_in[j];
        ComplexD w = twiddle_factors[j];

        x_out[j] = mul_w<forward>(x, w);
    }
}

#endif
