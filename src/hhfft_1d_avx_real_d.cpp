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
    assert (step_info.twiddle_factors == nullptr);

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

            //std::cout << "x1_in =   " << x1_in << ", x2_in =   " << x2_in << std::endl;
            //std::cout << "w2 = " << w2 << ", x2_temp = " << x2_temp << std::endl;
            //std::cout << "x1_out =  " << x1_out << ", x2_out =  " << x2_out << std::endl;
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



inline void set_fft_real_1d_one_level(StepInfoRealD &step_info)
{
    size_t radix = step_info.radix;

    if (radix == 2)
        step_info.step_function = fft_1d_real_one_level_radix2_stride1;
    if (radix == 3)
        step_info.step_function = fft_real_1d_one_level<double,3,1>;
    if (radix == 4)
        step_info.step_function = fft_real_1d_one_level<double,4,1>;
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
            step_info.step_function = fft_real_1d_one_level_stride_even<double,4,1>;
        if (radix == 5)
            step_info.step_function = fft_real_1d_one_level_stride_even<double,5,1>;
        if (radix == 7)
            step_info.step_function = fft_real_1d_one_level_stride_even<double,7,1>;
    } else
    {
        // TODO stride odd (no need for radix 2 or 4)
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
