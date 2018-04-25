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

// Loads same number four times: [x] -> [x x x x]
inline const ComplexD2 broadcast64_D2(const double x)
{
    return _mm256_broadcast_sd(&x);
}

// Combines complex number from two separate memory locations: [a1 a2], [b1 b2] -> [a1 a2 b1 b2]
inline const ComplexD2 load_two_128(const double *a, const double *b)
{
    // NOTE this should compile into two operations
    const ComplexD2 aa = _mm256_castpd128_pd256(load128(a));
    const ComplexD bb = load128(b);
    return _mm256_insertf128_pd (aa, bb, 1);
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

// Divides the complex numbers to two separate memory locations: [a1 a2 b1 b2] -> [a1 a2], [b1 b2]
inline void store_two_128(ComplexD2 val, double *a, double *b)
{
    // NOTE this should compile into three operations
    ComplexD aa = _mm256_castpd256_pd128(val);
    store(aa, a);
    ComplexD bb = _mm256_extractf128_pd(val, 1);
    store(bb, b);
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

template<size_t radix, bool forward> inline void multiply_coeff(const ComplexD2 *x_in, ComplexD2 *x_out)
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
        ComplexD2 t2 = mul_i(k1*(x_in[1] - x_in[2]));

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
            t3 = mul_i(x_in[3] - x_in[1]);
        else
            t3 = mul_i(x_in[1] - x_in[3]);

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
        ComplexD2 t6 = mul_i(k2*t2 + k4*t3);
        ComplexD2 t7 = mul_i(k4*t2 - k2*t3);

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
            ComplexD2 w = broadcast128(coeff + 2*radix*i + 2*j);

            x_out[i] = x_out[i] + mul_w<forward>(x_in[j], w);
        }
    }
}

template<size_t radix, bool forward> inline void multiply_twiddle(const ComplexD2 *x_in, ComplexD2 *x_out, const ComplexD2 *twiddle_factors)
{
    // It is assumed that first twiddle factors are always (1 + 0i)
    x_out[0] = x_in[0];

    // Read in the values used in this step and multiply them with twiddle factors
    for (size_t j = 1; j < radix; j++)
    {
        ComplexD2 x = x_in[j];
        ComplexD2 w = twiddle_factors[j];

        x_out[j] = mul_w<forward>(x, w);
    }
}
