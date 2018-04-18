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

#include "hhfft_1d_avx_d.h"
#include "hhfft_1d_plain_impl.h"

#include <assert.h>
#include <immintrin.h>

using namespace hhfft;

// Functions to operate data using avx commands

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
const ComplexD2 const1 = load(0.0, -0.0, 0.0, -0.0);
//const ComplexD2 const2 = load(0.0, 0.0, -0.0, -0.0);

// Multiplies two packed complex numbers. If other of them changes more frequently, set it to b.
inline ComplexD2 mul(ComplexD2 a, ComplexD2 b)
{    
    //const ComplexD2 const1 = load(0.0, -0.0, 0.0, -0.0);

    ComplexD2 a1 = change_sign(a, const1);
    ComplexD2 a2 = _mm256_permute_pd(a, 1 + 4);

    ComplexD2 t1 = _mm256_mul_pd(a1, b);
    ComplexD2 t2 = _mm256_mul_pd(a2, b);

    ComplexD2 y = _mm256_hadd_pd(t1, t2);
    return y;
}

// Multiplies a packed complex numbers with twiddle factor. The forward / inverse fft is taken into account
template<bool forward> inline ComplexD2 mul_w(ComplexD2 w, ComplexD2 b)
{
    //const ComplexD2 const1 = load(0.0, -0.0, 0.0, -0.0);

    ComplexD2 a1, a2;
    if (forward)
    {
        a1 = change_sign(w, const1);
        a2 = _mm256_permute_pd(w, 1 + 4);
    } else
    {
        a1 = w;
        a2 = _mm256_permute_pd(change_sign(w, const1), 1 + 4);
    }

    ComplexD2 t1 = _mm256_mul_pd(a1, b);
    ComplexD2 t2 = _mm256_mul_pd(a2, b);

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

// TODO it might be possible to make this also compatible with stride%2 == 1, but is it ever needed (radix 2 and 4 are always done first)?
template<bool forward> void fft_1d_one_level_twiddle_radix4_stridemod2_0(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    assert (step_info.radix == 4);
    assert (step_info.stride%2 == 0);

    size_t stride = step_info.stride;
    size_t repeats = step_info.repeats;

    for (size_t i = 0; i < repeats; i++)
    {
        for (size_t k = 0; k < stride; k+=2)
        {
            // Load values
            ComplexD2 x_in_1 = load(data_in + 8*i*stride            + 2*k);
            ComplexD2 x_in_2 = load(data_in + 8*i*stride + 2*stride + 2*k);
            ComplexD2 x_in_3 = load(data_in + 8*i*stride + 4*stride + 2*k);
            ComplexD2 x_in_4 = load(data_in + 8*i*stride + 6*stride + 2*k);

            // Load twiddle factors
            // It is assumed that first twiddle factors are always (1 + 0i)
            ComplexD2 w_2 = load(step_info.twiddle_factors + 2*stride + 2*k);
            ComplexD2 w_3 = load(step_info.twiddle_factors + 4*stride + 2*k);
            ComplexD2 w_4 = load(step_info.twiddle_factors + 6*stride + 2*k);

            // Multiply with twiddle factors            
            ComplexD2 x_2 = mul_w<forward>(w_2, x_in_2);
            ComplexD2 x_3 = mul_w<forward>(w_3, x_in_3);
            ComplexD2 x_4 = mul_w<forward>(w_4, x_in_4);

            // Calculate some temporary variables
            ComplexD2 t_1 = x_in_1 + x_3;
            ComplexD2 t_2 = x_in_1 - x_3;
            ComplexD2 t_3 = x_2 + x_4;
            ComplexD2 t_4;
            if (forward)
                t_4 = mul_i(x_4 - x_2);
            else
                t_4 = mul_i(x_2 - x_4);

            // Calculate outputs
            ComplexD2 x_out_1 = t_1 + t_3;
            ComplexD2 x_out_2 = t_2 + t_4;
            ComplexD2 x_out_3 = t_1 - t_3;
            ComplexD2 x_out_4 = t_2 - t_4;

            // Save the output
            store(x_out_1, data_out + 8*i*stride            + 2*k);
            store(x_out_2, data_out + 8*i*stride + 2*stride + 2*k);
            store(x_out_3, data_out + 8*i*stride + 4*stride + 2*k);
            store(x_out_4, data_out + 8*i*stride + 6*stride + 2*k);
        }

        /*
        for (size_t k = 0; k < stride; k++)
        {
            double x_in1_r = data_in[8*i*stride            + 2*k + 0];
            double x_in1_i = data_in[8*i*stride            + 2*k + 1];
            double x_in2_r = data_in[8*i*stride + 2*stride + 2*k + 0];
            double x_in2_i = data_in[8*i*stride + 2*stride + 2*k + 1];
            double x_in3_r = data_in[8*i*stride + 4*stride + 2*k + 0];
            double x_in3_i = data_in[8*i*stride + 4*stride + 2*k + 1];
            double x_in4_r = data_in[8*i*stride + 6*stride + 2*k + 0];
            double x_in4_i = data_in[8*i*stride + 6*stride + 2*k + 1];

            // It is assumed that first twiddle factors are always (1 + 0i)
            double w2_r = step_info.twiddle_factors[2*stride + 2*k + 0];
            double w2_i = step_info.twiddle_factors[2*stride + 2*k + 1];
            double w3_r = step_info.twiddle_factors[4*stride + 2*k + 0];
            double w3_i = step_info.twiddle_factors[4*stride + 2*k + 1];
            double w4_r = step_info.twiddle_factors[6*stride + 2*k + 0];
            double w4_i = step_info.twiddle_factors[6*stride + 2*k + 1];

            // Multiply with twiddle factors
            double x2_r = w2_r*x_in2_r - w2_i*x_in2_i;
            double x2_i = w2_i*x_in2_r + w2_r*x_in2_i;
            double x3_r = w3_r*x_in3_r - w3_i*x_in3_i;
            double x3_i = w3_i*x_in3_r + w3_r*x_in3_i;
            double x4_r = w4_r*x_in4_r - w4_i*x_in4_i;
            double x4_i = w4_i*x_in4_r + w4_r*x_in4_i;


            data_out[8*i*stride            + 2*k + 0] = x_in1_r + x2_r + x3_r + x4_r;
            data_out[8*i*stride            + 2*k + 1] = x_in1_i + x2_i + x3_i + x4_i;
            data_out[8*i*stride + 2*stride + 2*k + 0] = x_in1_r + x2_i - x3_r - x4_i;
            data_out[8*i*stride + 2*stride + 2*k + 1] = x_in1_i - x2_r - x3_i + x4_r;
            data_out[8*i*stride + 4*stride + 2*k + 0] = x_in1_r - x2_r + x3_r - x4_r;
            data_out[8*i*stride + 4*stride + 2*k + 1] = x_in1_i - x2_i + x3_i - x4_i;
            data_out[8*i*stride + 6*stride + 2*k + 0] = x_in1_r - x2_i - x3_r + x4_i;
            data_out[8*i*stride + 6*stride + 2*k + 1] = x_in1_i + x2_r - x3_i - x4_r;
        }
        */
    }
}

template<bool forward> void fft_1d_one_level_twiddle_radix4_stridemod2_0_DIF(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    assert (step_info.radix == 4);
    assert (step_info.stride%2 == 0);

    size_t stride = step_info.stride;
    size_t repeats = step_info.repeats;

    for (size_t i = 0; i < repeats; i++)
    {
        // Load twiddle factors
        // It is assumed that first twiddle factors are always (1 + 0i)
        ComplexD2 w_2 = broadcast128(step_info.twiddle_factors + 8*i + 2);
        ComplexD2 w_3 = broadcast128(step_info.twiddle_factors + 8*i + 4);
        ComplexD2 w_4 = broadcast128(step_info.twiddle_factors + 8*i + 6);

        for (size_t k = 0; k < stride; k+=2)
        {
            // Load values
            ComplexD2 x_in_1 = load(data_in + 8*i*stride            + 2*k);
            ComplexD2 x_in_2 = load(data_in + 8*i*stride + 2*stride + 2*k);
            ComplexD2 x_in_3 = load(data_in + 8*i*stride + 4*stride + 2*k);
            ComplexD2 x_in_4 = load(data_in + 8*i*stride + 6*stride + 2*k);

            // Multiply with twiddle factors
            ComplexD2 x_2 = mul_w<forward>(w_2, x_in_2);
            ComplexD2 x_3 = mul_w<forward>(w_3, x_in_3);
            ComplexD2 x_4 = mul_w<forward>(w_4, x_in_4);

            // Calculate some temporary variables
            ComplexD2 t_1 = x_in_1 + x_3;
            ComplexD2 t_2 = x_in_1 - x_3;
            ComplexD2 t_3 = x_2 + x_4;
            ComplexD2 t_4;
            if (forward)
                t_4 = mul_i(x_4 - x_2);
            else
                t_4 = mul_i(x_2 - x_4);

            // Calculate outputs
            ComplexD2 x_out_1 = t_1 + t_3;
            ComplexD2 x_out_2 = t_2 + t_4;
            ComplexD2 x_out_3 = t_1 - t_3;
            ComplexD2 x_out_4 = t_2 - t_4;

            // Save the output
            store(x_out_1, data_out + 8*i*stride            + 2*k);
            store(x_out_2, data_out + 8*i*stride + 2*stride + 2*k);
            store(x_out_3, data_out + 8*i*stride + 4*stride + 2*k);
            store(x_out_4, data_out + 8*i*stride + 6*stride + 2*k);
        }
    }

    /*
    for (size_t i = 0; i < repeats; i++)
    {
        double w2_r = step_info.twiddle_factors[8*i + 2 + 0];
        double w2_i = step_info.twiddle_factors[8*i + 2 + 1];
        double w3_r = step_info.twiddle_factors[8*i + 4 + 0];
        double w3_i = step_info.twiddle_factors[8*i + 4 + 1];
        double w4_r = step_info.twiddle_factors[8*i + 6 + 0];
        double w4_i = step_info.twiddle_factors[8*i + 6 + 1];

        for (size_t k = 0; k < stride; k++)
        {
            double x_in1_r = data_in[8*i*stride            + 2*k + 0];
            double x_in1_i = data_in[8*i*stride            + 2*k + 1];
            double x_in2_r = data_in[8*i*stride + 2*stride + 2*k + 0];
            double x_in2_i = data_in[8*i*stride + 2*stride + 2*k + 1];
            double x_in3_r = data_in[8*i*stride + 4*stride + 2*k + 0];
            double x_in3_i = data_in[8*i*stride + 4*stride + 2*k + 1];
            double x_in4_r = data_in[8*i*stride + 6*stride + 2*k + 0];
            double x_in4_i = data_in[8*i*stride + 6*stride + 2*k + 1];

            // Multiply with twiddle factors
            double x2_r, x2_i, x3_r, x3_i, x4_r, x4_i;
            if (forward)
            {
                x2_r = w2_r*x_in2_r - w2_i*x_in2_i;
                x2_i = w2_i*x_in2_r + w2_r*x_in2_i;
                x3_r = w3_r*x_in3_r - w3_i*x_in3_i;
                x3_i = w3_i*x_in3_r + w3_r*x_in3_i;
                x4_r = w4_r*x_in4_r - w4_i*x_in4_i;
                x4_i = w4_i*x_in4_r + w4_r*x_in4_i;
            }
            else
            {
                x2_r =  w2_r*x_in2_r + w2_i*x_in2_i;
                x2_i = -w2_i*x_in2_r + w2_r*x_in2_i;
                x3_r =  w3_r*x_in3_r + w3_i*x_in3_i;
                x3_i = -w3_i*x_in3_r + w3_r*x_in3_i;
                x4_r =  w4_r*x_in4_r + w4_i*x_in4_i;
                x4_i = -w4_i*x_in4_r + w4_r*x_in4_i;
            }

            if (forward)
            {
                data_out[8*i*stride            + 2*k + 0] = x_in1_r + x2_r + x3_r + x4_r;
                data_out[8*i*stride            + 2*k + 1] = x_in1_i + x2_i + x3_i + x4_i;
                data_out[8*i*stride + 2*stride + 2*k + 0] = x_in1_r + x2_i - x3_r - x4_i;
                data_out[8*i*stride + 2*stride + 2*k + 1] = x_in1_i - x2_r - x3_i + x4_r;
                data_out[8*i*stride + 4*stride + 2*k + 0] = x_in1_r - x2_r + x3_r - x4_r;
                data_out[8*i*stride + 4*stride + 2*k + 1] = x_in1_i - x2_i + x3_i - x4_i;
                data_out[8*i*stride + 6*stride + 2*k + 0] = x_in1_r - x2_i - x3_r + x4_i;
                data_out[8*i*stride + 6*stride + 2*k + 1] = x_in1_i + x2_r - x3_i - x4_r;

            } else
            {
                data_out[8*i*stride            + 2*k + 0] = x_in1_r + x2_r + x3_r + x4_r;
                data_out[8*i*stride            + 2*k + 1] = x_in1_i + x2_i + x3_i + x4_i;
                data_out[8*i*stride + 2*stride + 2*k + 0] = x_in1_r - x2_i - x3_r + x4_i;
                data_out[8*i*stride + 2*stride + 2*k + 1] = x_in1_i + x2_r - x3_i - x4_r;
                data_out[8*i*stride + 4*stride + 2*k + 0] = x_in1_r - x2_r + x3_r - x4_r;
                data_out[8*i*stride + 4*stride + 2*k + 1] = x_in1_i - x2_i + x3_i - x4_i;
                data_out[8*i*stride + 6*stride + 2*k + 0] = x_in1_r + x2_i - x3_r - x4_i;
                data_out[8*i*stride + 6*stride + 2*k + 1] = x_in1_i - x2_r - x3_i + x4_r;
            }
        }
    }
    */
}

template<bool forward> void fft_1d_one_level_radix4_stride1(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{    
    assert (step_info.radix == 4);
    assert (step_info.stride == 1);

    const size_t repeats = step_info.repeats;

    const ComplexD2 const2 = load(0.0, 0.0, -0.0, -0.0);

    ComplexD2 const3, const4;
    if (forward)
    {
        const3 = load(0.0, 0.0, -0.0, 0.0);
        const4 = load(0.0, 0.0, 0.0, -0.0);
    } else
    {
        const3 = load(0.0, 0.0, 0.0, -0.0);
        const4 = load(0.0, 0.0, -0.0, 0.0);
    }

    // TODO this could be done a bit more efficiently (see fft_1d_one_level_twiddle_radix4_stride1_DIF)
    for (size_t i = 0; i < repeats; i++)
    {
        // Load values and change some signs        
        ComplexD2 x_in_1 = broadcast128(data_in + 8*i + 0);
        ComplexD2 x_in_2 = broadcast128(data_in + 8*i + 2);
        ComplexD2 x_in_3 = broadcast128(data_in + 8*i + 4);
        ComplexD2 x_in_4 = broadcast128(data_in + 8*i + 6);

        x_in_2 = change_sign(x_in_2, const3);
        x_in_3 = change_sign(x_in_3, const2);
        x_in_4 = change_sign(x_in_4, const4);

        // Calculate some temporary variables
        ComplexD2 temp_1 = x_in_1 + x_in_3;
        ComplexD2 temp_2 = x_in_2 + x_in_4;
        temp_2 = _mm256_permute_pd(temp_2, 2 + 4);

        // Calculate output
        ComplexD2 x_out_1 = temp_1 + temp_2;
        ComplexD2 x_out_2 = temp_1 - temp_2;

        // Save the output
        store(x_out_1, data_out + 8*i + 0);
        store(x_out_2, data_out + 8*i + 4);
    }

    /*
    for (size_t i = 0; i < repeats; i++)
    {
        double x1_r = data_in[8*i + 0];
        double x1_i = data_in[8*i + 1];
        double x2_r = data_in[8*i + 2];
        double x2_i = data_in[8*i + 3];
        double x3_r = data_in[8*i + 4];
        double x3_i = data_in[8*i + 5];
        double x4_r = data_in[8*i + 6];
        double x4_i = data_in[8*i + 7];

        data_out[8*i + 0] = x1_r + x2_r + x3_r + x4_r;
        data_out[8*i + 1] = x1_i + x2_i + x3_i + x4_i;
        data_out[8*i + 2] = x1_r + x2_i - x3_r - x4_i;
        data_out[8*i + 3] = x1_i - x2_r - x3_i + x4_r;
        data_out[8*i + 4] = x1_r - x2_r + x3_r - x4_r;
        data_out[8*i + 5] = x1_i - x2_i + x3_i - x4_i;
        data_out[8*i + 6] = x1_r - x2_i - x3_r + x4_i;
        data_out[8*i + 7] = x1_i + x2_r - x3_i - x4_r;
    }
    */
}

template<bool forward> void fft_1d_one_level_radix4_repeats1(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{    
    assert (step_info.radix == 4);
    assert (step_info.repeats == 1);
    assert (step_info.stride > 1);
    assert (step_info.twiddle_factors == nullptr);

    const size_t stride = step_info.stride;

    size_t i;
    for (i = 0; i < (stride-1); i+=2)
    {
        ComplexD2 x_in_1 = load(data_in + 2*i);
        ComplexD2 x_in_2 = load(data_in + 2*i + 2*stride);
        ComplexD2 x_in_3 = load(data_in + 2*i + 4*stride);
        ComplexD2 x_in_4 = load(data_in + 2*i + 6*stride);

        // Perform the calculations
        ComplexD2 t_1 = x_in_1 + x_in_3;
        ComplexD2 t_2 = x_in_1 - x_in_3;
        ComplexD2 t_3 = x_in_2 + x_in_4;
        ComplexD2 t_4;
        if (forward)
            t_4 = mul_i(x_in_4 - x_in_2);
        else
            t_4 = mul_i(x_in_2 - x_in_4);

        // Calculate outputs
        ComplexD2 x_out_1 = t_1 + t_3;
        ComplexD2 x_out_2 = t_2 + t_4;
        ComplexD2 x_out_3 = t_1 - t_3;
        ComplexD2 x_out_4 = t_2 - t_4;

        // Save the output
        store(x_out_1, data_out + 2*i);
        store(x_out_2, data_out + 2*i + 2*stride);
        store(x_out_3, data_out + 2*i + 4*stride);
        store(x_out_4, data_out + 2*i + 6*stride);
    }

    // If stride is uneven, there is one to go
    if (i < stride)
    {
        double x1_r = data_in[2*i            + 0];
        double x1_i = data_in[2*i            + 1];
        double x2_r = data_in[2*i + 2*stride + 0];
        double x2_i = data_in[2*i + 2*stride + 1];
        double x3_r = data_in[2*i + 4*stride + 0];
        double x3_i = data_in[2*i + 4*stride + 1];
        double x4_r = data_in[2*i + 6*stride + 0];
        double x4_i = data_in[2*i + 6*stride + 1];

        if (forward)
        {
            data_out[2*i            + 0] = x1_r + x2_r + x3_r + x4_r;
            data_out[2*i            + 1] = x1_i + x2_i + x3_i + x4_i;
            data_out[2*i + 2*stride + 0] = x1_r + x2_i - x3_r - x4_i;
            data_out[2*i + 2*stride + 1] = x1_i - x2_r - x3_i + x4_r;
            data_out[2*i + 4*stride + 0] = x1_r - x2_r + x3_r - x4_r;
            data_out[2*i + 4*stride + 1] = x1_i - x2_i + x3_i - x4_i;
            data_out[2*i + 6*stride + 0] = x1_r - x2_i - x3_r + x4_i;
            data_out[2*i + 6*stride + 1] = x1_i + x2_r - x3_i - x4_r;
        } else
        {

            data_out[2*i            + 0] = x1_r + x2_r + x3_r + x4_r;
            data_out[2*i            + 1] = x1_i + x2_i + x3_i + x4_i;
            data_out[2*i + 2*stride + 0] = x1_r - x2_i - x3_r + x4_i;
            data_out[2*i + 2*stride + 1] = x1_i + x2_r - x3_i - x4_r;
            data_out[2*i + 4*stride + 0] = x1_r - x2_r + x3_r - x4_r;
            data_out[2*i + 4*stride + 1] = x1_i - x2_i + x3_i - x4_i;
            data_out[2*i + 6*stride + 0] = x1_r + x2_i - x3_r - x4_i;
            data_out[2*i + 6*stride + 1] = x1_i - x2_r - x3_i + x4_r;
        }

    }

    /*
    for (size_t i = 0; i < stride; i++)
    {
        double x1_r = data_in[2*i            + 0];
        double x1_i = data_in[2*i            + 1];
        double x2_r = data_in[2*i + 2*stride + 0];
        double x2_i = data_in[2*i + 2*stride + 1];
        double x3_r = data_in[2*i + 4*stride + 0];
        double x3_i = data_in[2*i + 4*stride + 1];
        double x4_r = data_in[2*i + 6*stride + 0];
        double x4_i = data_in[2*i + 6*stride + 1];

        if (forward)
        {
            data_out[2*i            + 0] = x1_r + x2_r + x3_r + x4_r;
            data_out[2*i            + 1] = x1_i + x2_i + x3_i + x4_i;
            data_out[2*i + 2*stride + 0] = x1_r + x2_i - x3_r - x4_i;
            data_out[2*i + 2*stride + 1] = x1_i - x2_r - x3_i + x4_r;
            data_out[2*i + 4*stride + 0] = x1_r - x2_r + x3_r - x4_r;
            data_out[2*i + 4*stride + 1] = x1_i - x2_i + x3_i - x4_i;
            data_out[2*i + 6*stride + 0] = x1_r - x2_i - x3_r + x4_i;
            data_out[2*i + 6*stride + 1] = x1_i + x2_r - x3_i - x4_r;
        } else
        {

            data_out[2*i            + 0] = x1_r + x2_r + x3_r + x4_r;
            data_out[2*i            + 1] = x1_i + x2_i + x3_i + x4_i;
            data_out[2*i + 2*stride + 0] = x1_r - x2_i - x3_r + x4_i;
            data_out[2*i + 2*stride + 1] = x1_i + x2_r - x3_i - x4_r;
            data_out[2*i + 4*stride + 0] = x1_r - x2_r + x3_r - x4_r;
            data_out[2*i + 4*stride + 1] = x1_i - x2_i + x3_i - x4_i;
            data_out[2*i + 6*stride + 0] = x1_r + x2_i - x3_r - x4_i;
            data_out[2*i + 6*stride + 1] = x1_i - x2_r - x3_i + x4_r;
        }
    }
    */
}

//
template<bool forward> void fft_1d_one_level_radix2_repeats1(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    assert (step_info.radix == 2);
    assert (step_info.repeats == 1);
    assert (step_info.stride > 1);
    assert (step_info.twiddle_factors == nullptr);

    const size_t stride = step_info.stride;

    size_t i;
    for (i = 0; i < (stride-1); i+=2)
    {
        ComplexD2 x_in_1 = load(data_in + 2*i);
        ComplexD2 x_in_2 = load(data_in + 2*i + 2*stride);

        // Perform the calculations
        ComplexD2 x_out_1 = x_in_1 + x_in_2;
        ComplexD2 x_out_2 = x_in_1 - x_in_2;

        // Save the output
        store(x_out_1, data_out + 2*i);
        store(x_out_2, data_out + 2*i + 2*stride);
    }

    // If stride is uneven, there is one to go
    if (i < stride)
    {
        double x1_r = data_in[2*i            + 0];
        double x1_i = data_in[2*i            + 1];
        double x2_r = data_in[2*i + 2*stride + 0];
        double x2_i = data_in[2*i + 2*stride + 1];

        data_out[2*i + 0]            = x1_r + x2_r;
        data_out[2*i + 1]            = x1_i + x2_i;
        data_out[2*i + 2*stride + 0] = x1_r - x2_r;
        data_out[2*i + 2*stride + 1] = x1_i - x2_i;
    }

    /*
    for (size_t i = 0; i < stride; i++)
    {
        double x1_r = data_in[2*i            + 0];
        double x1_i = data_in[2*i            + 1];
        double x2_r = data_in[2*i + 2*stride + 0];
        double x2_i = data_in[2*i + 2*stride + 1];

        data_out[2*i + 0]            = x1_r + x2_r;
        data_out[2*i + 1]            = x1_i + x2_i;
        data_out[2*i + 2*stride + 0] = x1_r - x2_r;
        data_out[2*i + 2*stride + 1] = x1_i - x2_i;
    }
    */
}

template<bool forward> void fft_1d_one_level_radix2_stride1(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{    
    assert (step_info.radix == 2);
    assert (step_info.stride == 1);
    assert (step_info.twiddle_factors == nullptr);

    const size_t repeats = step_info.repeats;

    const ComplexD2 const2 = load(0.0, 0.0, -0.0, -0.0);

    for (size_t i = 0; i < repeats; i++)
    {
        // Load values
        ComplexD2 x_in_1 = broadcast128(data_in + 4*i + 0);
        ComplexD2 x_in_2 = broadcast128(data_in + 4*i + 2);

        // Do the calculations        
        ComplexD2 x_2 = change_sign(const2, x_in_2);
        ComplexD2 x_out = x_in_1 + x_2;

        // Save the output
        store(x_out, data_out + 4*i);
    }

    /*
    for (size_t i = 0; i < repeats; i++)
    {
        double x1_r = data_in[4*i + 0];
        double x1_i = data_in[4*i + 1];
        double x2_r = data_in[4*i + 2];
        double x2_i = data_in[4*i + 3];

        data_out[4*i + 0] = x1_r + x2_r;
        data_out[4*i + 1] = x1_i + x2_i;
        data_out[4*i + 2] = x1_r - x2_r;
        data_out[4*i + 3] = x1_i - x2_i;
    }
    */
}

template<bool forward> void fft_1d_one_level_twiddle_radix4_stride1_DIF(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    assert (step_info.radix == 4);
    assert (step_info.stride == 1);

    size_t repeats = step_info.repeats;

    const ComplexD2 const1 = load(0.0, 0.0, 0.0, -0.0);
    const ComplexD2 const2 = load(0.0, 0.0, -0.0, -0.0);
    for (size_t i = 0; i < repeats; i++)
    {
        // Load data and multiply with twiddle factors
        ComplexD2 x_in_1 = load(data_in + 8*i + 0);
        ComplexD2 x_in_2 = load(data_in + 8*i + 4);
        ComplexD2 w_1 = load(step_info.twiddle_factors + 8*i + 0);
        ComplexD2 w_2 = load(step_info.twiddle_factors + 8*i + 4);
        x_in_1 = mul_w<forward>(w_1, x_in_1);
        x_in_2 = mul_w<forward>(w_2, x_in_2);

        // Calculate some temporary variables
        ComplexD2 t_1 = x_in_1 + x_in_2;
        ComplexD2 t_2 = x_in_1 - x_in_2;
        if (forward)
        {
            t_2 = change_sign(t_2, const1);
        }
        {
            t_2 = change_sign(t_2, const2);
        }
        t_2 = _mm256_permute_pd(t_2, 2 + 4);

        // Reorder
        ComplexD2 t_3 = _mm256_permute2f128_pd(t_1, t_2, 0 + 2*16);
        ComplexD2 t_4 = _mm256_permute2f128_pd(t_1, t_2, 1 + 3*16);

        // Calculate and save the output
        ComplexD2 x_out_1 = t_3 + t_4;
        ComplexD2 x_out_2 = t_3 - t_4;
        store(x_out_1, data_out + 8*i + 0);
        store(x_out_2, data_out + 8*i + 4);
    }


    /*
    // TODO this works, but not the most efficient way!
    const ComplexD2 const2 = load(0.0, 0.0, -0.0, -0.0);
    ComplexD2 const3, const4;
    if (forward)
    {
        const3 = load(0.0, 0.0, -0.0, 0.0);
        const4 = load(0.0, 0.0, 0.0, -0.0);
    } else
    {
        const3 = load(0.0, 0.0, 0.0, -0.0);
        const4 = load(0.0, 0.0, -0.0, 0.0);
    }

    for (size_t i = 0; i < repeats; i++)
    {
        // Load values and change some signs
        ComplexD2 x_in_1 = broadcast128(data_in + 8*i + 0);
        ComplexD2 x_in_2 = broadcast128(data_in + 8*i + 2);
        ComplexD2 x_in_3 = broadcast128(data_in + 8*i + 4);
        ComplexD2 x_in_4 = broadcast128(data_in + 8*i + 6);

        ComplexD2 w_2 = broadcast128(step_info.twiddle_factors + 8*i + 2);
        ComplexD2 w_3 = broadcast128(step_info.twiddle_factors + 8*i + 4);
        ComplexD2 w_4 = broadcast128(step_info.twiddle_factors + 8*i + 6);

        // Multiply with twiddle factors
        x_in_2 = mul_w<forward>(w_2, x_in_2);
        x_in_3 = mul_w<forward>(w_3, x_in_3);
        x_in_4 = mul_w<forward>(w_4, x_in_4);

        x_in_2 = change_sign(x_in_2, const3);
        x_in_3 = change_sign(x_in_3, const2);
        x_in_4 = change_sign(x_in_4, const4);

        // Calculate some temporary variables
        ComplexD2 temp_1 = x_in_1 + x_in_3;
        ComplexD2 temp_2 = x_in_2 + x_in_4;
        temp_2 = _mm256_permute_pd(temp_2, 2 + 4);

        // Calculate output
        ComplexD2 x_out_1 = temp_1 + temp_2;
        ComplexD2 x_out_2 = temp_1 - temp_2;

        // Save the output
        store(x_out_1, data_out + 8*i + 0);
        store(x_out_2, data_out + 8*i + 4);
    }
    */


    /*
    for (size_t i = 0; i < repeats; i++)
    {
        double w2_r = step_info.twiddle_factors[8*i + 2 + 0];
        double w2_i = step_info.twiddle_factors[8*i + 2 + 1];
        double w3_r = step_info.twiddle_factors[8*i + 4 + 0];
        double w3_i = step_info.twiddle_factors[8*i + 4 + 1];
        double w4_r = step_info.twiddle_factors[8*i + 6 + 0];
        double w4_i = step_info.twiddle_factors[8*i + 6 + 1];

        double x_in1_r = data_in[8*i     + 0];
        double x_in1_i = data_in[8*i     + 1];
        double x_in2_r = data_in[8*i + 2 + 0];
        double x_in2_i = data_in[8*i + 2 + 1];
        double x_in3_r = data_in[8*i + 4 + 0];
        double x_in3_i = data_in[8*i + 4 + 1];
        double x_in4_r = data_in[8*i + 6 + 0];
        double x_in4_i = data_in[8*i + 6 + 1];

        // Multiply with twiddle factors
        double x2_r, x2_i, x3_r, x3_i, x4_r, x4_i;
        if (forward)
        {
            x2_r = w2_r*x_in2_r - w2_i*x_in2_i;
            x2_i = w2_i*x_in2_r + w2_r*x_in2_i;
            x3_r = w3_r*x_in3_r - w3_i*x_in3_i;
            x3_i = w3_i*x_in3_r + w3_r*x_in3_i;
            x4_r = w4_r*x_in4_r - w4_i*x_in4_i;
            x4_i = w4_i*x_in4_r + w4_r*x_in4_i;
        }
        else
        {
            x2_r =  w2_r*x_in2_r + w2_i*x_in2_i;
            x2_i = -w2_i*x_in2_r + w2_r*x_in2_i;
            x3_r =  w3_r*x_in3_r + w3_i*x_in3_i;
            x3_i = -w3_i*x_in3_r + w3_r*x_in3_i;
            x4_r =  w4_r*x_in4_r + w4_i*x_in4_i;
            x4_i = -w4_i*x_in4_r + w4_r*x_in4_i;
        }

        // Calculate some temporary variables
        double t1_r = x_in1_r + x3_r;
        double t1_i = x_in1_i + x3_i;
        double t2_r = x_in1_r - x3_r;
        double t2_i = x_in1_i - x3_i;
        double t3_r = x2_r + x4_r;
        double t3_i = x2_i + x4_i;
        double t4_r, t4_i;
        if (forward)
        {
            t4_r = -(x4_i - x2_i);
            t4_i = (x4_r - x2_r);
        }
        else
        {
            t4_r = -(x2_i - x4_i);
            t4_i = (x2_r - x4_r);
        }

        data_out[8*i     + 0] = t1_r + t3_r;
        data_out[8*i     + 1] = t1_i + t3_i;
        data_out[8*i + 2 + 0] = t2_r + t4_r;
        data_out[8*i + 2 + 1] = t2_i + t4_i;
        data_out[8*i + 4 + 0] = t1_r - t3_r;
        data_out[8*i + 4 + 1] = t1_i - t3_i;
        data_out[8*i + 6 + 0] = t2_r - t4_r;
        data_out[8*i + 6 + 1] = t2_i - t4_i;
    }
    */
}

template<bool forward> void fft_1d_one_level_twiddle_radix2_stride1_DIF(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{    
    assert (step_info.radix == 2);
    assert (step_info.stride == 1);

    size_t repeats = step_info.repeats;

    const ComplexD2 const2 = load(0.0, 0.0, -0.0, -0.0);

    // TODO there is probably a more efficient way of doing this. Now all inputs are broadcasted in!
    for (size_t i = 0; i < repeats; i++)
    {
        // Load values
        ComplexD2 x_in_1 = broadcast128(data_in + 4*i + 0);
        ComplexD2 x_in_2 = broadcast128(data_in + 4*i + 2);

        // Multiply with twiddle factors (It is assumed that first twiddle factors are always 1 + 0i)
        ComplexD2 w = broadcast128(step_info.twiddle_factors + 4*i + 2);
        ComplexD2 temp = mul_w<forward>(w, x_in_2);

        // Do the calculations
        ComplexD2 x_2 = change_sign(const2, temp);
        ComplexD2 x_out = x_in_1 + x_2;

        // Save the output
        store(x_out, data_out + 4*i);
    }

    /*
    for (size_t i = 0; i < repeats; i++)
    {
        // It is assumed that first twiddle factors are always (1 + 0i)
        double x_in1_r = data_in[4*i + 0];
        double x_in1_i = data_in[4*i + 1];
        double x_in2_r = data_in[4*i + 2];
        double x_in2_i = data_in[4*i + 3];

        double w_r = step_info.twiddle_factors[2*(i*2 + 1) + 0];
        double w_i = step_info.twiddle_factors[2*(i*2 + 1) + 1];

        double x2_r, x2_i;
        if (forward == 1)
        {
            x2_r = w_r*x_in2_r - w_i*x_in2_i;
            x2_i = w_i*x_in2_r + w_r*x_in2_i;
        } else
        {
            x2_r =  w_r*x_in2_r + w_i*x_in2_i;
            x2_i = -w_i*x_in2_r + w_r*x_in2_i;
        }

        data_out[4*i + 0] = x_in1_r + x2_r;
        data_out[4*i + 1] = x_in1_i + x2_i;
        data_out[4*i + 2] = x_in1_r - x2_r;
        data_out[4*i + 3] = x_in1_i - x2_i;
    }
    */
}

template<bool forward> void fft_1d_one_level_twiddle_radix2_stridemod2_0_DIF(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    assert (step_info.radix == 2);
    assert (step_info.stride%2 == 0);

    size_t stride = step_info.stride;
    size_t repeats = step_info.repeats;

    for (size_t i = 0; i < repeats; i++)
    {
        ComplexD2 w = broadcast128(step_info.twiddle_factors + 4*i + 2);

        for (size_t k = 0; k < stride; k+=2)
        {
            // It is assumed that first twiddle factors are always (1 + 0i)
            // Load values
            ComplexD2 x_in_1 = load(data_in + 4*i*stride + 2*k);
            ComplexD2 x_in_2 = load(data_in + 4*i*stride + 2*stride + 2*k);

            // Perform the calculations
            ComplexD2 x_2 = mul_w<forward>(w, x_in_2);
            ComplexD2 x_out_1 = x_in_1 + x_2;
            ComplexD2 x_out_2 = x_in_1 - x_2;

            // Save the output
            store(x_out_1, data_out + 4*i*stride + 2*k);
            store(x_out_2, data_out + 4*i*stride + 2*stride + 2*k);
        }


        /*
        double w_r = step_info.twiddle_factors[4*i + 2 + 0];
        double w_i = step_info.twiddle_factors[4*i + 2 + 1];

        for (size_t k = 0; k < stride; k++)
        {
            // It is assumed that first twiddle factors are always (1 + 0i)
            double x_in1_r = data_in[4*i*stride + 2*k + 0];
            double x_in1_i = data_in[4*i*stride + 2*k + 1];
            double x_in2_r = data_in[4*i*stride + 2*stride + 2*k + 0];
            double x_in2_i = data_in[4*i*stride + 2*stride + 2*k + 1];

            double x2_r, x2_i;
            if (forward == 1)
            {
                x2_r = w_r*x_in2_r - w_i*x_in2_i;
                x2_i = w_i*x_in2_r + w_r*x_in2_i;
            } else
            {
                x2_r =  w_r*x_in2_r + w_i*x_in2_i;
                x2_i = -w_i*x_in2_r + w_r*x_in2_i;
            }

            data_out[4*i*stride + 2*k + 0]              = x_in1_r + x2_r;
            data_out[4*i*stride + 2*k + 1]              = x_in1_i + x2_i;
            data_out[4*i*stride + 2*stride + 2*k + 0]   = x_in1_r - x2_r;
            data_out[4*i*stride + 2*stride + 2*k + 1]   = x_in1_i - x2_i;
        }
        */
    }
}

// TODO it might be possible to make this also compatible with stride%2 == 1, but is it ever needed (radix 2 and 4 are always done first)?
template<bool forward> void fft_1d_one_level_twiddle_radix2_stridemod2_0(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    assert (step_info.radix == 2);
    assert (step_info.stride%2 == 0);    

    size_t stride = step_info.stride;
    size_t repeats = step_info.repeats;

    for (size_t i = 0; i < repeats; i++)
    {
        for (size_t k = 0; k < stride; k+=2)
        {
            // It is assumed that first twiddle factors are always (1 + 0i)
            // Load values
            ComplexD2 x_in_1 = load(data_in + 4*i*stride + 2*k);
            ComplexD2 x_in_2 = load(data_in + 4*i*stride + 2*stride + 2*k);
            ComplexD2 w = load(step_info.twiddle_factors + 2*stride + 2*k);

            // Perform the calculations
            ComplexD2 x_2 = mul_w<forward>(w, x_in_2);
            ComplexD2 x_out_1 = x_in_1 + x_2;
            ComplexD2 x_out_2 = x_in_1 - x_2;

            // Save the output
            store(x_out_1, data_out + 4*i*stride + 2*k);
            store(x_out_2, data_out + 4*i*stride + 2*stride + 2*k);
        }

        /*
        for (size_t k = 0; k < stride; k++)
        {
            // It is assumed that first twiddle factors are always (1 + 0i)
            double x_in1_r = data_in[4*i*stride + 2*k + 0];
            double x_in1_i = data_in[4*i*stride + 2*k + 1];
            double x_in2_r = data_in[4*i*stride + 2*stride + 2*k + 0];
            double x_in2_i = data_in[4*i*stride + 2*stride + 2*k + 1];
            double w_r = step_info.twiddle_factors[2*stride + 2*k + 0];
            double w_i = step_info.twiddle_factors[2*stride + 2*k + 1];

            double x2_r = w_r*x_in2_r - w_i*x_in2_i;
            double x2_i = w_i*x_in2_r + w_r*x_in2_i;

            data_out[4*i*stride + 2*k + 0]              = x_in1_r + x2_r;
            data_out[4*i*stride + 2*k + 1]              = x_in1_i + x2_i;
            data_out[4*i*stride + 2*stride + 2*k + 0]   = x_in1_r - x2_r;
            data_out[4*i*stride + 2*stride + 2*k + 1]   = x_in1_i - x2_i;
        }
        */
    }
}

template<bool forward> void set_fft_1d_one_level(StepInfoD &step_info)
{
    size_t radix = step_info.radix;
    size_t stride = step_info.stride;
    size_t repeats = step_info.repeats;

    if (radix == 2)
    {
        if (stride == 1)
            step_info.step_function = fft_1d_one_level_radix2_stride1<forward>;  // Needed for first step in DIT
        else if (repeats == 1)
            step_info.step_function = fft_1d_one_level_radix2_repeats1<forward>;  //Needed for first step in DIF (TODO it should not be necassery to require repeats to be 1)
        else
            step_info.step_function = nullptr; //step_info.step_function = fft_1d_one_level<double,2,1,forward>; //Not ever needed!?
    }
    if (radix == 3)
        step_info.step_function = fft_1d_one_level<double,3,1,forward>;
    if (radix == 4)
    {
        if (stride == 1)
            step_info.step_function = fft_1d_one_level_radix4_stride1<forward>; // Needed for first step in DIT
        else if (repeats == 1)
            step_info.step_function = fft_1d_one_level_radix4_repeats1<forward>;  //Needed for first step in DIF (TODO it should not be necassery to require repeats to be 1)
        else
            step_info.step_function = nullptr; //step_info.step_function = fft_1d_one_level<double,4,1,forward>; //Not ever needed!?
    }
    if (radix == 5)
        step_info.step_function = fft_1d_one_level<double,5,1,forward>;
    if (radix == 7)
        step_info.step_function = fft_1d_one_level<double,7,1,forward>;
}

template<bool forward> void set_fft_1d_one_level_twiddle(StepInfoD &step_info)
{
    size_t radix = step_info.radix;
    size_t stride = step_info.stride;

    if (radix == 2)
    {
        if (stride%2 == 0)
            step_info.step_function = fft_1d_one_level_twiddle_radix2_stridemod2_0<forward>;
        else
            step_info.step_function = fft_1d_one_level_twiddle<double,2,1,forward>; // Not ever needed if 2 and 4 radices are always done first
    }
    if (radix == 3)
        step_info.step_function = fft_1d_one_level_twiddle<double,3,1,forward>;
    if (radix == 4)
    {
        if (stride%2 == 0)
            step_info.step_function = fft_1d_one_level_twiddle_radix4_stridemod2_0<forward>;
        else
            step_info.step_function = fft_1d_one_level_twiddle<double,4,1,forward>; // Not ever needed if 2 and 4 radices are always done first
    }
    if (radix == 5)
        step_info.step_function = fft_1d_one_level_twiddle<double,5,1,forward>;
    if (radix == 7)
        step_info.step_function = fft_1d_one_level_twiddle<double,7,1,forward>;
}

template<bool forward> void set_fft_1d_one_level_twiddle_DIF(StepInfoD &step_info)
{
    size_t radix = step_info.radix;
    size_t stride = step_info.stride;

    if (radix == 2)
    {
        if (stride%2 == 0)
            step_info.step_function = fft_1d_one_level_twiddle_radix2_stridemod2_0_DIF<forward>;
        else if (stride == 1)
            step_info.step_function = fft_1d_one_level_twiddle_radix2_stride1_DIF<forward>; // Needed for the last step in DIF
        else
            step_info.step_function = fft_1d_one_level_twiddle_DIF<double,2,1,forward>; // Not ever needed?
    }
    if (radix == 3)
        step_info.step_function = fft_1d_one_level_twiddle_DIF<double,3,1,forward>;
    if (radix == 4)
    {
        if (stride%2 == 0)
            step_info.step_function = fft_1d_one_level_twiddle_radix4_stridemod2_0_DIF<forward>;
        else if (stride == 1)
            step_info.step_function = fft_1d_one_level_twiddle_radix4_stride1_DIF<forward>; // Needed for the last step in DIF
        else
            step_info.step_function = fft_1d_one_level_twiddle_DIF<double,4,1,forward>; // Not ever needed?
    }
    if (radix == 5)
        step_info.step_function = fft_1d_one_level_twiddle_DIF<double,5,1,forward>;
    if (radix == 7)
        step_info.step_function = fft_1d_one_level_twiddle_DIF<double,7,1,forward>;
}

void hhfft::HHFFT_1D_AVX_set_function(StepInfoD &step_info)
{
    step_info.step_function = nullptr;

    if (step_info.reorder_table != nullptr || step_info.reorder_table_inplace != nullptr)
    {
        if (step_info.forward)
            step_info.step_function = fft_1d_reorder<double,1,true>;
        else
            step_info.step_function = fft_1d_reorder<double,1,false>;
        return;
    }

    if (step_info.twiddle_factors == nullptr)
    {
        if (step_info.forward)
            set_fft_1d_one_level<true>(step_info);
        else
            set_fft_1d_one_level<false>(step_info);
    } else
    {
        if (step_info.forward)
            set_fft_1d_one_level_twiddle<true>(step_info);
        else
            set_fft_1d_one_level_twiddle<false>(step_info);
    }

    if (step_info.step_function == nullptr)
    {
        throw(std::runtime_error("HHFFT error: Unable to set a function!"));
    }
}

void hhfft::HHFFT_1D_AVX_set_function_DIF(StepInfoD &step_info)
{
    step_info.step_function = nullptr;

    if (step_info.reorder_table != nullptr || step_info.reorder_table_inplace != nullptr)
    {
        if (step_info.data_type_in == step_info.data_type_out)
        {
            // In-place
            if (step_info.forward)
                step_info.step_function = fft_1d_reorder_in_place<double,1,true>;
            else
                step_info.step_function = fft_1d_reorder_in_place<double,1,false>;
        } else
        {
            // Out-of-place
            if (step_info.forward)
                step_info.step_function = fft_1d_reorder<double,1,true>;
            else
                step_info.step_function = fft_1d_reorder<double,1,false>;
        }
        return;
    }

    if (step_info.twiddle_factors == nullptr)
    {
        if (step_info.forward)
            set_fft_1d_one_level<true>(step_info);
        else
            set_fft_1d_one_level<false>(step_info);
    } else
    {
        if (step_info.forward)
            set_fft_1d_one_level_twiddle_DIF<true>(step_info);
        else
            set_fft_1d_one_level_twiddle_DIF<false>(step_info);
    }

    if (step_info.step_function == nullptr)
    {
        throw(std::runtime_error("HHFFT error: Unable to set a function!"));
    }
}




