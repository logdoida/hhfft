/*
*   Copyright Jouko Kalmari 2017
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

#include "hhfft_1d_avx_real_d.h"
#include "hhfft_1d_plain_real_impl.h"

#include <assert.h>
#include <immintrin.h>

using namespace hhfft;

// Functions to operate data using avx commands

// contains two complex numbers: [r1 i1 r2 i2]
typedef __m256d ComplexD2;

// contains a single complex number: [r i]
typedef __m128d ComplexD;

// Read complex numbers
inline ComplexD2 load(double r1, double i1, double r2, double i2)
{    
    return _mm256_setr_pd(r1,i1,r2,i2);
}
inline const ComplexD2 load(const double *v)
{
    return _mm256_loadu_pd(v);
}
inline ComplexD load(double r, double i)
{
    return _mm_setr_pd(r,i);
}
inline const ComplexD load128(const double *v)
{
    return _mm_loadu_pd(v);
}

// Loads two complex numbers [r1 i1], [r2 i2] and combine them to [r1 i1 r2 i2]
inline const ComplexD2 load_combine(const double *v1, const double *v2)
{
    // Only two instructions are actually needed: vmovupd + vinsertf128
    ComplexD a = _mm_loadu_pd(v1);
    ComplexD b = _mm_loadu_pd(v2);
    ComplexD2 out;
    out = _mm256_insertf128_pd(out, a, 0);
    out = _mm256_insertf128_pd(out, b, 1);
    return out;
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
// Stores two complex numbers [r1 i1], [r2 i2] from [r1 i1 r2 i2]
inline const void store_combined(ComplexD2 val, double *v1, double *v2)
{
    // Only two instructions are actually needed?
    ComplexD a = _mm256_extractf128_pd(val, 0);
    ComplexD b = _mm256_extractf128_pd(val, 1);
    _mm_storeu_pd(v1, a);
    _mm_storeu_pd(v2, b);
}

// Changes signs of [x1 x2 x3 x4] using [s1 s2 s3 s4]. s should contain only 0.0 and -0.0
// NOTE this seems to actually be bit slower than plain multiplication. Compare!
inline ComplexD2 change_sign(ComplexD2 x, ComplexD2 s)
{
    return _mm256_xor_pd(x,s);
}
inline ComplexD change_sign(ComplexD x, ComplexD s)
{
    return _mm_xor_pd(x,s);
}

// Defining the constants here or within the functions can have some difference. Compare!
const ComplexD2 const1 = load(0.0, -0.0, 0.0, -0.0);
const ComplexD const1_128 = load(0.0, -0.0);

// Multiplies complex numbers. If other of them changes more frequently, set it to b.
inline ComplexD2 mul(ComplexD2 a, ComplexD2 b)
{
    ComplexD2 a1 = change_sign(a, const1);
    ComplexD2 a2 = _mm256_permute_pd(a, 1 + 4);

    ComplexD2 t1 = _mm256_mul_pd(a1, b);
    ComplexD2 t2 = _mm256_mul_pd(a2, b);

    ComplexD2 y = _mm256_hadd_pd(t1, t2);
    return y;
}
inline ComplexD mul(ComplexD a, ComplexD b)
{
    ComplexD a1 = change_sign(a, const1_128);
    ComplexD a2 = _mm_permute_pd(a, 1);

    ComplexD t1 = _mm_mul_pd(a1, b);
    ComplexD t2 = _mm_mul_pd(a2, b);

    ComplexD y = _mm_hadd_pd(t1, t2);
    return y;
}

// Returns complex conjugate for complex numbers.
inline ComplexD2 conj(ComplexD2 x)
{
    return change_sign(x,const1);
}
inline ComplexD conj(ComplexD x)
{
    return change_sign(x,const1_128);
}

// Multiply constant 4x4 matrix with a vector
inline ComplexD2 mul_matrix(const double m[16], ComplexD2 x)
{
    const ComplexD2 const1 = load(m[0], m[1], m[10], m[11]);
    const ComplexD2 const2 = load(m[4], m[5], m[14], m[15]);
    const ComplexD2 const3 = load(m[2], m[3], m[8],  m[9]);
    const ComplexD2 const4 = load(m[6], m[7], m[12], m[13]);

    ComplexD2 x_perm = _mm256_permute2f128_pd(x,x,1);
    ComplexD2 t1 = _mm256_hadd_pd(const1*x, const2*x);
    ComplexD2 t2 = _mm256_hadd_pd(const3*x_perm, const4*x_perm);
    return t1 + t2;
}

// Multiplies packed complex numbers with i
inline ComplexD2 mul_i(ComplexD2 a)
{
    //const ComplexD2 const1 = load(0.0, -0.0, 0.0, -0.0);

    ComplexD2 a1 = change_sign(a, const1);
    ComplexD2 y = _mm256_permute_pd(a1, 1 + 4);
    return y;
}
inline ComplexD mul_i(ComplexD a)
{
    //const ComplexD const1 = load(0.0, -0.0);
    ComplexD a1 = change_sign(a, const1_128);
    ComplexD y = _mm_permute_pd(a1, 1);
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
inline std::ostream& operator<<(std::ostream& os, const ComplexD &x)
{
    double v[2];
    store(x, v);
    os << v[0] << ", " << v[1];
    return os;
}


void fft_1d_real_one_level_radix2_stride1(const double *data_in, double *data_out, hhfft::StepInfoRealD &step_info)
{
    assert (step_info.radix == 2);
    assert (step_info.stride == 1);
    assert (step_info.twiddle_factors == nullptr);

    const size_t repeats = step_info.repeats;

    const ComplexD2 const2 = load(0.0, -0.0, 0.0, -0.0);

    // First, process 4 inputs at a time
    size_t i = 0;    
    for (i = 0; i < repeats - 1; i+=2)
    {        
        // Load values. Values are actually real, not complex
        ComplexD2 x_in = load(data_in + 2*i);

        // Do the calculations
        ComplexD2 x_temp = change_sign(const2, x_in);
        ComplexD2 x_out = _mm256_hadd_pd(x_in, x_temp);

        // Save the output
        store(x_out, data_out + 2*i);
    }

    // If repeats is odd, there is one pair still to be processed
    if (i < repeats)
    {
        double x1_r = data_in[2*i + 0];
        double x2_r = data_in[2*i + 1];
        data_out[2*i + 0] = x1_r + x2_r;
        data_out[2*i + 1] = x1_r - x2_r;
    }

    /*
    for (size_t i = 0; i < repeats; i++)
    {
        double x1_r = data_in[2*i + 0];
        double x2_r = data_in[2*i + 1];
        data_out[2*i + 0] = x1_r + x2_r;
        data_out[2*i + 1] = x1_r - x2_r;
    }
    */
}

void fft_1d_real_one_level_radix4_stride1(const double *data_in, double *data_out, hhfft::StepInfoRealD &step_info)
{
    assert (step_info.radix == 4);
    assert (step_info.stride == 1);
    assert (step_info.twiddle_factors == nullptr);

    const size_t repeats = step_info.repeats;

    const ComplexD2 const1 = load(1.0, 1.0, -1.0, 0.0);
    const ComplexD2 const2 = load(1.0, -1.0, 0.0, 1.0);
    const ComplexD2 const3 = load(1.0, 1.0, 1.0, 0.0);
    const ComplexD2 const4 = load(1.0, -1.0, 0.0, -1.0);

    for (size_t i = 0; i < repeats; i++)
    {
        // Load values. Values are actually real, not complex
        ComplexD2 x_in = load(data_in + 4*i);

        // Do the calculations (4x4 matrix multiplied with a vector)
        ComplexD2 x_temp = _mm256_permute2f128_pd(x_in, x_in, 1);
        ComplexD2 x_temp2 = _mm256_hadd_pd(const1*x_in, const2*x_in);
        ComplexD2 x_temp3 = _mm256_hadd_pd(const3*x_temp, const4*x_temp);
        ComplexD2 x_out = x_temp2 + x_temp3;

        //std::cout << "x_in =   " << x_in << ", x_temp =   " << x_temp << std::endl;
        //std::cout << "x_temp2 = " << x_temp2 << ", x_temp3 = " << x_temp3 << std::endl;
        //std::cout << "x_out =  " << x_out << std::endl;

        // Save the output
        store(x_out, data_out + 4*i);
    }

    /*
    for (size_t i = 0; i < repeats; i++)
    {
        double x0_r = data_in[4*i + 0];
        double x1_r = data_in[4*i + 1];
        double x2_r = data_in[4*i + 2];
        double x3_r = data_in[4*i + 3];
        data_out[4*i + 0] = x0_r + x1_r + x2_r + x3_r;
        data_out[4*i + 1] = x0_r - x1_r + x2_r - x3_r;
        data_out[4*i + 2] = x0_r - x2_r;
        data_out[4*i + 3] = x3_r - x1_r;
    }
    */
}

void fft_1d_real_one_level_radix2_stride2(const double *data_in, double *data_out, hhfft::StepInfoRealD &step_info)
{
    assert (step_info.radix == 2);
    assert (step_info.stride == 2);
    assert (step_info.twiddle_factors == nullptr);

    const size_t repeats = step_info.repeats;

    const ComplexD2 const1 = load(1.0, 1.0, 0.0, 1.0);
    const ComplexD2 const2 = load(1.0, -1.0, -1.0, 0.0);

    for (size_t i = 0; i < repeats; i++)
    {
        // Load values. Values are actually real, not complex
        ComplexD2 x_in = load(data_in + 4*i);

        // Do the calculations
        ComplexD2 x_temp = _mm256_permute2f128_pd(x_in, x_in, 1);
        x_temp = _mm256_shuffle_pd (x_in,  x_temp, 4 + 8);
        ComplexD2 x_out = _mm256_hadd_pd(const1*x_temp, const2*x_temp);

        //std::cout << "x_in = " << x_in << ", x_temp = " << x_temp << ", x_out = " << x_out << std::endl;

        // Save the output
        store(x_out, data_out + 4*i);
    }    

    /*
    for (size_t i = 0; i < repeats; i++)
    {
        double x1_r = data_in[4*i + 0];
        double x2_r = data_in[4*i + 1];
        double x3_r = data_in[4*i + 2];
        double x4_r = data_in[4*i + 3];
        data_out[4*i + 0] = x1_r + x3_r;
        data_out[4*i + 1] = x1_r - x3_r;
        data_out[4*i + 2] = x2_r;
        data_out[4*i + 3] = -x4_r;
    }
    */
}

void fft_1d_real_one_level_twiddle_radix2(const double *data_in, double *data_out, hhfft::StepInfoRealD &step_info)
{
    assert (step_info.radix == 2);
    assert (step_info.stride%4 == 0); // This can be assumed as radix-2 steps are always called first
    assert (step_info.twiddle_factors != nullptr);

    const size_t repeats = step_info.repeats;
    const size_t stride = step_info.stride;

    const ComplexD const1 = load(0.0, -0.0);

    int dir_out = 1;
    for (size_t i = 0; i < repeats; i++)
    {
        // First the two real values in the beginning are handeld
        {
            ComplexD x1_in = load128(data_in + 2*stride*i);
            ComplexD x2_in = load128(data_in + 2*stride*i + stride);
            ComplexD x1_temp = _mm_shuffle_pd(x1_in, x2_in, 0);
            ComplexD x2_temp = _mm_shuffle_pd(x1_in, x2_in, 1 + 2);
            ComplexD x1_out = _mm_hadd_pd(x1_temp, change_sign(x1_temp, const1));
            ComplexD x2_out = change_sign(x2_temp, const1);
            store(x1_out, data_out + 2*stride*i);
            store(x2_out, data_out + 2*stride*i + stride);
        }

        // There are uneven number of complex numbers so the next ones are taken care of individually
        // NOTE these first two could be combined to use AVX?
        {
            ComplexD x1_in = load128(data_in + 2*stride*i + 2);
            ComplexD x2_in = load128(data_in + 2*stride*i + 2*stride - 2);
            ComplexD w2 = load128(step_info.twiddle_factors + 2*stride + 2);
            ComplexD x2_temp = mul(w2, x2_in);
            ComplexD x1_out = x1_in + x2_temp;
            ComplexD x2_out = conj(x1_in - x2_temp);

            // Direction affects how data is stored
            if (dir_out)
            {
                store(x1_out, data_out + 2*stride*i + 2);
                store(x2_out, data_out + 2*stride*i + 2*stride - 2);
            } else
            {
                store(x2_out, data_out + 2*stride*i + 2);
                store(x1_out, data_out + 2*stride*i + 2*stride - 2);
            }
        }

        // Then all the rest all complex number pairs
        for (size_t j = 4; j < stride; j+=4)
        {
            // Load data
            ComplexD2 x1_in = load(data_in + 2*stride*i + j);
            ComplexD2 x2_in = load(data_in + 2*stride*i + 2*stride - j - 2);
            x2_in = _mm256_permute2f128_pd(x2_in, x2_in, 1);

            // Multiply with twiddle factors
            ComplexD2 w2 = load(step_info.twiddle_factors + 2*stride + j);
            ComplexD2 x2_temp = mul(w2, x2_in);

            // Calculate output
            ComplexD2 x1_out = x1_in + x2_temp;
            ComplexD2 x2_out = conj(x1_in - x2_temp);

            // Direction affects how data is stored
            if (dir_out)
            {
                x2_out = _mm256_permute2f128_pd(x2_out, x2_out, 1);
                store(x1_out, data_out + 2*stride*i + j);
                store(x2_out, data_out + 2*stride*i + 2*stride - j - 2);
            } else
            {
                x1_out = _mm256_permute2f128_pd(x1_out, x1_out, 1);
                store(x2_out, data_out + 2*stride*i + j);
                store(x1_out, data_out + 2*stride*i + 2*stride - j - 2);
            }
        }

        dir_out = dir_out^1;
    }

    /*
    int dir_out = 1;
    for (size_t i = 0; i < repeats; i++)
    {
        double x1_r = data_in[2*stride*i + 0];
        double x2_r = data_in[2*stride*i + 1];
        double x3_r = data_in[2*stride*i + stride + 0];
        double x4_r = data_in[2*stride*i + stride + 1];
        data_out[2*stride*i + 0] = x1_r + x3_r;
        data_out[2*stride*i + 1] = x1_r - x3_r;
        data_out[2*stride*i + stride + 0] = x2_r;
        data_out[2*stride*i + stride + 1] = -x4_r;

        for (size_t j = 2; j < stride; j+=2)
        {
            double x1_r = data_in[2*stride*i + j + 0];
            double x1_i = data_in[2*stride*i + j + 1];
            double x2_r = data_in[2*stride*i + 2*stride - j + 0];
            double x2_i = data_in[2*stride*i + 2*stride - j + 1];

            // only the second one needs to be multiplied with twiddle factors
            double w2_r = step_info.twiddle_factors[2*stride + j + 0];
            double w2_i = step_info.twiddle_factors[2*stride + j + 1];
            double x3_r = w2_r*x2_r - w2_i*x2_i;
            double x3_i = w2_i*x2_r + w2_r*x2_i;

            if (dir_out)
            {
                data_out[2*stride*i + j + 0] = x1_r + x3_r;
                data_out[2*stride*i + j + 1] = x1_i + x3_i;
                data_out[2*stride*i + 2*stride - j + 0] =  x1_r - x3_r;
                data_out[2*stride*i + 2*stride - j + 1] = -x1_i + x3_i;
            } else
            {
                data_out[2*stride*i + j + 0] =  x1_r - x3_r;
                data_out[2*stride*i + j + 1] = -x1_i + x3_i;
                data_out[2*stride*i + 2*stride - j + 0] = x1_r + x3_r;
                data_out[2*stride*i + 2*stride - j + 1] = x1_i + x3_i;
            }

            //std::cout << "x1_r = " << x1_r << ", x1_i = " << x1_i << ", x2_r = " << x2_r << ", x2_i = " << x2_i << std::endl;
        }
        dir_out = dir_out^1;
    }
    */
}


void fft_1d_real_one_level_twiddle_radix4(const double *data_in, double *data_out, hhfft::StepInfoRealD &step_info)
{
    assert (step_info.radix == 4);
    assert (step_info.twiddle_factors != nullptr);

    const size_t repeats = step_info.repeats;
    const size_t stride = step_info.stride;

    const double m1[16] = {1,  1, 1,  1,
                           1, -1, 1, -1,
                           1,  0, -1, 0,
                           0, -1, 0, 1};

    const double w = sqrt(0.5);
    const double m2[16] = {1,  w,  0, -w,
                           0, -w, -1, -w,
                           1, -w,  0,  w,
                           0, -w,  1, -w};

    int dir_out = 1;
    for (size_t i = 0; i < repeats; i++)
    {
        const double *data_in2 = data_in + 4*stride*i;
        double *data_out2 = data_out + 4*stride*i;

        // First the two real values in the beginning are handeld
        {
            // Load and shuffle the data
            ComplexD2 x0_in = load_combine(data_in2 + 0*stride, data_in2 + 2*stride);
            ComplexD2 x1_in = load_combine(data_in2 + 1*stride, data_in2 + 3*stride);
            ComplexD2 x0_temp = _mm256_shuffle_pd (x0_in, x1_in, 0);
            ComplexD2 x1_temp = _mm256_shuffle_pd (x0_in, x1_in, 1+2+4+8);

            // Multiply with constant matrices
            ComplexD2 x0_out = mul_matrix(m1, x0_temp);
            ComplexD2 x1_out = mul_matrix(m2, x1_temp);

            // Store output
            store_combined(x0_out, data_out2 + 0*stride, data_out2 + 2*stride);
            if (dir_out)
            {
                store_combined(x1_out, data_out2 + 1*stride, data_out2 + 3*stride);
            } else
            {
                store_combined(x1_out, data_out2 + 3*stride, data_out2 + 1*stride);
            }

            /*
            std::cout << "x0_in = " << x0_in << std::endl;
            std::cout << "x1_in = " << x1_in << std::endl;
            std::cout << "x0_temp = " << x0_temp << std::endl;
            std::cout << "x1_temp = " << x1_temp << std::endl;
            std::cout << "x0_out = " << x0_out << std::endl;
            */
        }

        // There are uneven number of complex numbers so the next ones are taken care of individually        
        {
            ComplexD x0_in = load128(data_in2 + 0*stride + 2);
            ComplexD x1_in = load128(data_in2 + 2*stride - 2);
            ComplexD x2_in = load128(data_in2 + 2*stride + 2);
            ComplexD x3_in = load128(data_in2 + 4*stride - 2);

            // first one is not multiplied with the twiddle factors
            ComplexD w1 = load128(step_info.twiddle_factors + 2*stride + 2);
            ComplexD w2 = load128(step_info.twiddle_factors + 4*stride + 2);
            ComplexD w3 = load128(step_info.twiddle_factors + 6*stride + 2);
            ComplexD x1_temp = mul(w1, x1_in);
            ComplexD x2_temp = mul(w2, x2_in);
            ComplexD x3_temp = mul(w3, x3_in);

            // Radix 4 calculations
            ComplexD t0 = x0_in + x2_temp;
            ComplexD t1 = x0_in - x2_temp;
            ComplexD t2 = x1_temp + x3_temp;
            ComplexD t3 = mul_i(x1_temp - x3_temp);
            ComplexD x0_out = t0 + t2;
            ComplexD x1_out = conj(t1 + t3);
            ComplexD x2_out = t1 - t3;
            ComplexD x3_out = conj(t0 - t2);

            // Direction affects how data is stored
            if (dir_out)
            {
                store(x0_out, data_out2 + 2);
                store(x1_out, data_out2 + 2*stride - 2);
                store(x2_out, data_out2 + 2*stride + 2);
                store(x3_out, data_out2 + 4*stride - 2);
            } else
            {
                store(x3_out, data_out2 + 2);
                store(x2_out, data_out2 + 2*stride - 2);
                store(x1_out, data_out2 + 2*stride + 2);
                store(x0_out, data_out2 + 4*stride - 2);
            }            

            /*
            std::cout << "x0_in = " << x0_in << std::endl;
            std::cout << "x1_in = " << x1_in << ", w1 = " << w1 << ", x1_temp = " << x1_temp << std::endl;
            std::cout << "x2_in = " << x2_in << ", w2 = " << w1 << ", x2_temp = " << x2_temp << std::endl;
            std::cout << "x3_in = " << x3_in << ", w3 = " << w1 << ", x3_temp = " << x3_temp << std::endl;
            std::cout << "x0_out = " << x0_out << std::endl;
            std::cout << "x1_out = " << x1_out << std::endl;
            std::cout << "x2_out = " << x2_out << std::endl;
            std::cout << "x3_out = " << x3_out << std::endl;
            */
        }

        // Then all the rest are complex number pairs
        for (size_t j = 4; j < stride; j+=4)
        {
            // Load data
            ComplexD2 x0_in = load(data_in2 + 0*stride + j);
            ComplexD2 x1_in = load(data_in2 + 2*stride - j - 2);
            ComplexD2 x2_in = load(data_in2 + 2*stride + j);
            ComplexD2 x3_in = load(data_in2 + 4*stride - j - 2);
            x1_in = _mm256_permute2f128_pd(x1_in, x1_in, 1);
            x3_in = _mm256_permute2f128_pd(x3_in, x3_in, 1);

            // First one is not multiplied with the twiddle factors
            ComplexD2 w1 = load(step_info.twiddle_factors + 2*stride + j);
            ComplexD2 w2 = load(step_info.twiddle_factors + 4*stride + j);
            ComplexD2 w3 = load(step_info.twiddle_factors + 6*stride + j);
            ComplexD2 x1_temp = mul(w1, x1_in);
            ComplexD2 x2_temp = mul(w2, x2_in);
            ComplexD2 x3_temp = mul(w3, x3_in);

            // Radix 4 calculations
            ComplexD2 t0 = x0_in + x2_temp;
            ComplexD2 t1 = x0_in - x2_temp;
            ComplexD2 t2 = x1_temp + x3_temp;
            ComplexD2 t3 = mul_i(x1_temp - x3_temp);
            ComplexD2 x0_out = t0 + t2;
            ComplexD2 x1_out = conj(t1 + t3);
            ComplexD2 x2_out = t1 - t3;
            ComplexD2 x3_out = conj(t0 - t2);

            // Direction affects how data is stored
            if (dir_out)
            {
                x1_out = _mm256_permute2f128_pd(x1_out, x1_out, 1);
                x3_out = _mm256_permute2f128_pd(x3_out, x3_out, 1);
                store(x0_out, data_out2 + 0*stride + j);
                store(x1_out, data_out2 + 2*stride - j - 2);
                store(x2_out, data_out2 + 2*stride + j);
                store(x3_out, data_out2 + 4*stride - j - 2);
            } else
            {
                x0_out = _mm256_permute2f128_pd(x0_out, x0_out, 1);
                x2_out = _mm256_permute2f128_pd(x2_out, x2_out, 1);
                store(x3_out, data_out2 + 0*stride + j);
                store(x2_out, data_out2 + 2*stride - j - 2);
                store(x1_out, data_out2 + 2*stride + j);
                store(x0_out, data_out2 + 4*stride - j - 2);
            }

            /*
            std::cout << "x0_in = " << x0_in << std::endl;
            std::cout << "x1_in = " << x1_in << ", w1 = " << w1 << ", x1_temp = " << x1_temp << std::endl;
            std::cout << "x2_in = " << x2_in << ", w2 = " << w2 << ", x2_temp = " << x2_temp << std::endl;
            std::cout << "x3_in = " << x3_in << ", w3 = " << w3 << ", x3_temp = " << x3_temp << std::endl;
            std::cout << "x0_out = " << x0_out << std::endl;
            std::cout << "x1_out = " << x1_out << std::endl;
            std::cout << "x2_out = " << x2_out << std::endl;
            std::cout << "x3_out = " << x3_out << std::endl;
            */
        }        
        dir_out = dir_out^1;        
    }


    /*
    int dir_out = 1;
    const double w = sqrt(0.5);
    for (size_t i = 0; i < repeats; i++)
    {
        {
            double x0_r = data_in[4*stride*i + 0*stride + 0];
            double x1_r = data_in[4*stride*i + 0*stride + 1];
            double x2_r = data_in[4*stride*i + 1*stride + 0];
            double x3_r = data_in[4*stride*i + 1*stride + 1]*w;
            double x4_r = data_in[4*stride*i + 2*stride + 0];
            double x5_r = data_in[4*stride*i + 2*stride + 1];
            double x6_r = data_in[4*stride*i + 3*stride + 0];
            double x7_r = data_in[4*stride*i + 3*stride + 1]*w;

            data_out[4*stride*i + 0*stride + 0] = x0_r + x2_r + x4_r + x6_r;
            data_out[4*stride*i + 0*stride + 1] = x0_r - x2_r + x4_r - x6_r;
            data_out[4*stride*i + 2*stride + 0] = x0_r - x4_r;
            data_out[4*stride*i + 2*stride + 1] = -x2_r + x6_r;

            double x0_r_out =  x1_r + x3_r - x7_r;
            double x0_i_out = -x3_r - x5_r - x7_r;
            double x1_r_out =  x1_r - x3_r + x7_r;
            double x1_i_out = -x3_r + x5_r - x7_r;

            if (dir_out)
            {
                data_out[4*stride*i + 1*stride + 0] = x0_r_out;
                data_out[4*stride*i + 1*stride + 1] = x0_i_out;
                data_out[4*stride*i + 3*stride + 0] = x1_r_out;
                data_out[4*stride*i + 3*stride + 1] = x1_i_out;
            } else
            {
                data_out[4*stride*i + 1*stride + 0] = x1_r_out;
                data_out[4*stride*i + 1*stride + 1] = x1_i_out;
                data_out[4*stride*i + 3*stride + 0] = x0_r_out;
                data_out[4*stride*i + 3*stride + 1] = x0_i_out;
            }
        }

        for (size_t j = 2; j < stride; j+=2)
        {
            double x0_r = data_in[4*stride*i + 0*stride + j + 0];
            double x0_i = data_in[4*stride*i + 0*stride + j + 1];
            double x1_r = data_in[4*stride*i + 2*stride - j + 0];
            double x1_i = data_in[4*stride*i + 2*stride - j + 1];
            double x2_r = data_in[4*stride*i + 2*stride + j + 0];
            double x2_i = data_in[4*stride*i + 2*stride + j + 1];
            double x3_r = data_in[4*stride*i + 4*stride - j + 0];
            double x3_i = data_in[4*stride*i + 4*stride - j + 1];

            // first one is not multiplied with the twiddle factors
            double w1_r = step_info.twiddle_factors[2*stride + j + 0];
            double w1_i = step_info.twiddle_factors[2*stride + j + 1];
            double w2_r = step_info.twiddle_factors[4*stride + j + 0];
            double w2_i = step_info.twiddle_factors[4*stride + j + 1];
            double w3_r = step_info.twiddle_factors[6*stride + j + 0];
            double w3_i = step_info.twiddle_factors[6*stride + j + 1];
            double x1_w_r = w1_r*x1_r - w1_i*x1_i;
            double x1_w_i = w1_i*x1_r + w1_r*x1_i;
            double x2_w_r = w2_r*x2_r - w2_i*x2_i;
            double x2_w_i = w2_i*x2_r + w2_r*x2_i;
            double x3_w_r = w3_r*x3_r - w3_i*x3_i;
            double x3_w_i = w3_i*x3_r + w3_r*x3_i;

            double x0_r_out =  x0_r + x1_w_r + x2_w_r + x3_w_r;
            double x0_i_out =  x0_i + x1_w_i + x2_w_i + x3_w_i;
            double x1_r_out =  x0_r - x1_w_i - x2_w_r + x3_w_i;
            double x1_i_out = -x0_i - x1_w_r + x2_w_i + x3_w_r;
            double x2_r_out =  x0_r + x1_w_i - x2_w_r - x3_w_i;
            double x2_i_out =  x0_i - x1_w_r - x2_w_i + x3_w_r;
            double x3_r_out =  x0_r - x1_w_r + x2_w_r - x3_w_r;
            double x3_i_out = -x0_i + x1_w_i - x2_w_i + x3_w_i;

            if (dir_out)
            {
                data_out[4*stride*i + 0*stride + j + 0] = x0_r_out;
                data_out[4*stride*i + 0*stride + j + 1] = x0_i_out;
                data_out[4*stride*i + 2*stride - j + 0] = x1_r_out;
                data_out[4*stride*i + 2*stride - j + 1] = x1_i_out;
                data_out[4*stride*i + 2*stride + j + 0] = x2_r_out;
                data_out[4*stride*i + 2*stride + j + 1] = x2_i_out;
                data_out[4*stride*i + 4*stride - j + 0] = x3_r_out;
                data_out[4*stride*i + 4*stride - j + 1] = x3_i_out;
            } else
            {
                data_out[4*stride*i + 0*stride + j + 0] = x3_r_out;
                data_out[4*stride*i + 0*stride + j + 1] = x3_i_out;
                data_out[4*stride*i + 2*stride - j + 0] = x2_r_out;
                data_out[4*stride*i + 2*stride - j + 1] = x2_i_out;
                data_out[4*stride*i + 2*stride + j + 0] = x1_r_out;
                data_out[4*stride*i + 2*stride + j + 1] = x1_i_out;
                data_out[4*stride*i + 4*stride - j + 0] = x0_r_out;
                data_out[4*stride*i + 4*stride - j + 1] = x0_i_out;
            }


            //std::cout << "x0 = " << x0_r << ", " << x0_i << std::endl;
            //std::cout << "x1 = " << x1_r << ", " << x1_i << ", w1 = " << w1_r << ", " << w1_i << ", x1_w = " << x1_w_r << ", " << x1_w_i << std::endl;
            //std::cout << "x2 = " << x2_r << ", " << x2_i << ", w2 = " << w2_r << ", " << w2_i << ", x2_w = " << x2_w_r << ", " << x2_w_i << std::endl;
            //std::cout << "x2 = " << x3_r << ", " << x3_i << ", w3 = " << w3_r << ", " << w3_i << ", x3_w = " << x3_w_r << ", " << x3_w_i << std::endl;
            //std::cout << "x0_out = " << x0_r_out << ", " << x0_i_out << std::endl;
            //std::cout << "x1_out = " << x1_r_out << ", " << x1_i_out << std::endl;
            //std::cout << "x2_out = " << x2_r_out << ", " << x2_i_out << std::endl;
            //std::cout << "x3_out = " << x3_r_out << ", " << x3_i_out << std::endl;
        }        
        dir_out = dir_out^1;
    }    
    */
}



inline void set_fft_real_1d_one_level(StepInfoRealD &step_info)
{
    size_t radix = step_info.radix;

    if (radix == 2)
        step_info.step_function = fft_1d_real_one_level_radix2_stride1;
    if (radix == 3)
        step_info.step_function = fft_real_1d_one_level<double,3,1>;
    if (radix == 4)
        step_info.step_function = fft_1d_real_one_level_radix4_stride1;
    if (radix == 5)
        step_info.step_function = fft_real_1d_one_level<double,5,1>;
    if (radix == 7)
        step_info.step_function = fft_real_1d_one_level<double,7,1>;
}

inline void set_fft_real_1d_one_level_twiddle(StepInfoRealD &step_info)
{
    size_t radix = step_info.radix;
    size_t stride = step_info.stride;

    if (stride%2 == 0)
    {
        if (radix == 2)
        {
            if (stride == 2)
                step_info.step_function = fft_1d_real_one_level_radix2_stride2;
            else
                step_info.step_function = fft_1d_real_one_level_twiddle_radix2;
        }
        if (radix == 3)
            step_info.step_function = fft_real_1d_one_level_stride_even<double,3,1>;
        if (radix == 4)
            step_info.step_function = fft_1d_real_one_level_twiddle_radix4;
        if (radix == 5)
            step_info.step_function = fft_real_1d_one_level_stride_even<double,5,1>;
        if (radix == 7)
            step_info.step_function = fft_real_1d_one_level_stride_even<double,7,1>;
    } else
    {
        // Odd stride is only possible for odd radices
        if (radix == 3)
            step_info.step_function = fft_real_1d_one_level_stride_odd<double,3,0>;
        if (radix == 5)
            step_info.step_function = fft_real_1d_one_level_stride_odd<double,5,0>;
        if (radix == 7)
            step_info.step_function = fft_real_1d_one_level_stride_odd<double,7,0>;
    }
}


void hhfft::HHFFT_1D_AVX_real_set_function(StepInfoRealD &step_info)
{
    step_info.step_function = nullptr;

    if (step_info.reorder_table != nullptr)
    {
        // TODO how to use in-place if algorithm if input actually points to output?
        if (step_info.forward)
            step_info.step_function = fft_real_1d_reorder<double,1,true>;
        //else
        // TODO ifft should use in-place reordering!
        // step_info.step_function = fft_real_1d_reorder<double,1,false>;
        return;
    }

    if (step_info.twiddle_factors == nullptr)
    {
        if (step_info.forward)
            set_fft_real_1d_one_level(step_info);
        //else
        //   TODO ifft should actually be DIF
    } else
    {
        if (step_info.forward)
            set_fft_real_1d_one_level_twiddle(step_info);
        //else
        //   TODO ifft should actually be DIF
    }

    if (step_info.step_function == nullptr)
    {
        throw(std::runtime_error("HHFFT error: Unable to set a function!"));
    }
}
