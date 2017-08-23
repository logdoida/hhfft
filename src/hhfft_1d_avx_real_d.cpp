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
// contains four real double precision numbers: [r1 r2 r3 r4]
typedef __m256d RealD4;

// Read four doubles
inline RealD4 load(double r1, double r2, double r3, double r4)
{
    //return _mm256_set_pd(i2,r2,i1,r1); // Why this order?
    return _mm256_setr_pd(r1,r2,r3,r4); // Reversed. Why this order?
}
inline const RealD4 load(const double *v)
{
    return _mm256_loadu_pd(v);
}

// Loads same number four time: r -> [r r r r]
inline const RealD4 broadcast64(const double *v)
{    
    return _mm256_broadcast_sd(v);
}

// Loads same two numbers two time: [r1 r2] -> [r1 r2 r1 r2]
inline const RealD4 broadcast128(const double *v)
{
    return _mm256_broadcast_pd((const __m128d*) v);
}

// Store a complex number
inline void store(RealD4 val, double &r1, double &r2, double &r3, double &r4)
{
    double v[4];
    _mm256_storeu_pd(v, val);
    r1 = val[0]; r2 = val[1]; r3 = val[2]; r4 = val[3];
}
inline void store(RealD4 val, double *v)
{
    _mm256_storeu_pd(v, val);
}

// Changes signs of [x1 x2 x3 x4] using [s1 s2 s3 s4]. s should contain only 0.0 and -0.0
// NOTE this seems to actually be bit slower than plain multiplication. Compare!
inline RealD4 change_sign(RealD4 x, RealD4 s)
{
    return _mm256_xor_pd(x,s);
}

// Changes order from [x0 x1 x2 x3] [x0 x3 x2 x1]
inline RealD4 reorder(RealD4 x)
{
    RealD4 x_reord = _mm256_permute2f128_pd(x,x,1);
    return _mm256_blend_pd(x, x_reord, 2+8);
}

// Changes order from [X x1 x2 x3], [x0 X X X] to [x0 x3 x2 x1]
inline RealD4 reorder2(RealD4 x0, RealD4 x1)
{
    RealD4 x_reord = _mm256_permute2f128_pd(x0,x0,1);
    x_reord = _mm256_blend_pd(x_reord, x0, 4);
    return _mm256_blend_pd(x_reord, x1, 1);
}

// For testing
inline std::ostream& operator<<(std::ostream& os, const RealD4 &x)
{
    double v[4];
    store(x, v);
    os << v[0] << ", " << v[1] << ", " << v[2] << ", " << v[3];
    return os;
}


// TODO it might be a good idea to combine reordering (out-of-place) step with the first step
void dht_1d_one_level_radix2_stride1(const double *data_in, double *data_out, hhfft::StepInfoReal<double> &step_info)
{
    //std::cout << "dht_1d_one_level_radix2_stride1" << std::endl;

    assert (step_info.radix == 2);
    assert (step_info.stride == 1);

    size_t repeats = step_info.repeats;

    RealD4 const1 = load(0.0, -0.0, 0.0, -0.0);

    size_t i = 0;
    for (; i < repeats-1; i+=2)
    {
        RealD4 x_in = load(data_in + 2*i); // = [x0 x1 x2 x3]
        RealD4 temp = change_sign(x_in, const1); // = [x0 -x1 x2 -x3]

        RealD4 x_out = _mm256_hadd_pd(x_in, temp); // = [x0 + x1, x0 - x1, x2 + x3, x2 - x3]
        store(x_out, data_out + 2*i);
    }

    // There might be uneven number of repeats, so the last one is handeld separately
    if (i < repeats)
    {
        double x_in_1 = data_in[2*i + 0];
        double x_in_2 = data_in[2*i + 1];

        double x_out_1 = x_in_1 + x_in_2;
        double x_out_2 = x_in_1 - x_in_2;

        data_out[2*i + 0] = x_out_1;
        data_out[2*i + 1] = x_out_2;
    }

    /*
    for (size_t i = 0; i < repeats; i++)
    {
        double x_in_0 = data_in[2*i + 0];
        double x_in_1 = data_in[2*i + 1];

        data_out[2*i + 0] = x_in_0 + x_in_1;
        data_out[2*i + 1] = x_in_0 - x_in_1;
    }
    */
}

// TODO it might be a good idea to combine reordering (out-of-place) step with the first step
void dht_1d_one_level_radix4_stride1(const double *data_in, double *data_out, hhfft::StepInfoReal<double> &step_info)
{
    //std::cout << "dht_1d_one_level_radix4_stride1" << std::endl;

    assert (step_info.radix == 2);
    assert (step_info.stride == 1);

    size_t repeats = step_info.repeats;

    RealD4 const1 = load(0.0, 0.0, 0.0, -0.0);
    RealD4 const2 = load(-0.0, -0.0, -0.0, 0.0);

    for (size_t i = 0; i < repeats; i++)
    {
        // TODO is there a more efficient way of doing this?
        RealD4 x_in_0 = broadcast128(data_in + 4*i + 0); // = [x0 x1 x0 x1]
        RealD4 x_in_1 = broadcast128(data_in + 4*i + 2); // = [x2 x3 x2 x3]
        RealD4 temp_0 = change_sign(x_in_0, const1); // = [x0 x1 x0 -x1]
        RealD4 temp_1 = change_sign(x_in_1, const1); // = [x2 x3 x2 -x3]
        RealD4 temp_2 = change_sign(x_in_1, const2); // = [-x2 -x3 -x2 x3]
        RealD4 temp_3 = _mm256_hadd_pd(temp_0,temp_0); // [x0 + x1, x0 + x1, x0 - x1, x0 -x1]
        RealD4 temp_4 = _mm256_hadd_pd(temp_1,temp_2); // [x2 + x3, -x2 - x3, x2 - x3, -x2 +x3]
        RealD4 x_out = temp_3 + temp_4;
        store(x_out, data_out + 4*i);
    }

    /*
    for (size_t i = 0; i < repeats; i++)
    {
        double x_in_0 = data_in[4*i + 0];
        double x_in_1 = data_in[4*i + 1];
        double x_in_2 = data_in[4*i + 2];
        double x_in_3 = data_in[4*i + 3];

        data_out[4*i + 0] = x_in_0 + x_in_1 + x_in_2 + x_in_3;
        data_out[4*i + 1] = x_in_0 + x_in_1 - x_in_2 - x_in_3;
        data_out[4*i + 2] = x_in_0 - x_in_1 + x_in_2 - x_in_3;
        data_out[4*i + 3] = x_in_0 - x_in_1 - x_in_2 + x_in_3;
    }
    */
}



void dht_1d_one_level_radix2_stride2(const double *data_in, double *data_out, hhfft::StepInfoReal<double> &step_info)
{
    //std::cout << "dht_1d_one_level_radix2_stride2" << std::endl;

    assert (step_info.radix == 2);
    assert (step_info.stride == 2);

    size_t repeats = step_info.repeats;

    RealD4 const1 = load(0.0, 0.0, -0.0, -0.0);

    for (size_t i = 0; i < repeats; i++)
    {
        RealD4 x_in_0 = broadcast128(data_in + 4*i + 0); // = [x0 x1 x0 x1]
        RealD4 x_in_1 = broadcast128(data_in + 4*i + 2); // = [x2 x3 x2 x3]

        RealD4 temp = change_sign(x_in_1, const1); // = [x2 x3 -x2 -x3]

        RealD4 x_out = x_in_0 + temp; // = [x0 + x2, x1 + x3, x0 - x2, x1 - x3]
        store(x_out, data_out + 4*i);
    }

    /*
    for (size_t i = 0; i < repeats; i++)
    {
        double x_in_0 = data_in[4*i + 0];
        double x_in_1 = data_in[4*i + 1];
        double x_in_2 = data_in[4*i + 2];
        double x_in_3 = data_in[4*i + 3];

        double x_out_0 = x_in_0 + x_in_2;
        double x_out_1 = x_in_1 + x_in_3;
        double x_out_2 = x_in_0 - x_in_2;
        double x_out_3 = x_in_1 - x_in_3;

        data_out[4*i + 0] = x_out_0;
        data_out[4*i + 1] = x_out_1;
        data_out[4*i + 2] = x_out_2;
        data_out[4*i + 3] = x_out_3;
    }
    */
}

void dht_1d_one_level_twiddle_radix2_stride4(const double *data_in, double *data_out, hhfft::StepInfoReal<double> &step_info)
{
    //std::cout << "fft_1d_one_level_twiddle_radix2_stride4" << std::endl;

    assert (step_info.radix == 2);
    assert (step_info.stride == 4);

    size_t repeats = step_info.repeats;

    RealD4 c = load(step_info.cos_factors + 4); // = [c4 c5 c6 c7]
    RealD4 s = load(step_info.sin_factors + 4); // = [s4 s5 s6 s7]

    for (size_t i = 0; i < repeats; i++)
    {
        // Load an reorder data
        RealD4 x_in_0 = load(data_in + 8*i + 0); // = [x0 x1 x2 x3]
        RealD4 x_in_1 = load(data_in + 8*i + 4); // = [x4 x5 x6 x7]
        RealD4 x_in_1_reord = reorder(x_in_1); // = [x4 x7 x6 x5]

        // Perform the calculations
        RealD4 temp = c*x_in_1 + s*x_in_1_reord;
        RealD4 x_out_0 = x_in_0 + temp;
        RealD4 x_out_1 = x_in_0 - temp;

        // Save the results
        store(x_out_0, data_out + 8*i + 0);
        store(x_out_1, data_out + 8*i + 4);
    }

    /*
    const double *c = step_info.cos_factors;
    const double *s = step_info.sin_factors;
    for (size_t i = 0; i < repeats; i++)
    {
        double x_in_0 = data_in[8*i + 0];
        double x_in_1 = data_in[8*i + 1];
        double x_in_2 = data_in[8*i + 2];
        double x_in_3 = data_in[8*i + 3];
        double x_in_4 = data_in[8*i + 4];
        double x_in_5 = data_in[8*i + 5];
        double x_in_6 = data_in[8*i + 6];
        double x_in_7 = data_in[8*i + 7];

        double t_0 = x_in_4;
        double t_1 = c[5]*x_in_5 + s[5]*x_in_7;
        double t_2 = c[6]*x_in_6 + s[6]*x_in_6;
        double t_3 = c[7]*x_in_7 + s[7]*x_in_5;

        data_out[8*i + 0] = x_in_0 + t_0;
        data_out[8*i + 1] = x_in_1 + t_1;
        data_out[8*i + 2] = x_in_2 + t_2;
        data_out[8*i + 3] = x_in_3 + t_3;
        data_out[8*i + 4] = x_in_0 - t_0;
        data_out[8*i + 5] = x_in_1 - t_1;
        data_out[8*i + 6] = x_in_2 - t_2;
        data_out[8*i + 7] = x_in_3 - t_3;
    }*/
}

void dht_1d_one_level_twiddle_radix4_stride4(const double *data_in, double *data_out, hhfft::StepInfoReal<double> &step_info)
{
    //std::cout << "fft_1d_one_level_twiddle_radix4_stride4" << std::endl;

    assert (step_info.radix == 4);
    assert (step_info.stride == 4);

    size_t repeats = step_info.repeats;

    // NOTE: these could also be hard-coded
    RealD4 c_1 = load(step_info.cos_factors + 4);  // = [c4 c5 c6 c7]
    RealD4 s_1 = load(step_info.sin_factors + 4);  // = [s4 s5 s6 s7]
    RealD4 c_2 = load(step_info.cos_factors + 8);  // = [c8 c9 c10 c11]
    RealD4 s_2 = load(step_info.sin_factors + 8);  // = [s8 s9 s10 s11]
    RealD4 c_3 = load(step_info.cos_factors + 12); // = [c12 c13 c14 c15]
    RealD4 s_3 = load(step_info.sin_factors + 12); // = [s12 s13 s14 s15]

    for (size_t i = 0; i < repeats; i++)
    {
        // Load an reorder data
        RealD4 x_in_0 = load(data_in + 16*i + 0);  // = [x0  x1  x2  x3]
        RealD4 x_in_1 = load(data_in + 16*i + 4);  // = [x4  x5  x6  x7]
        RealD4 x_in_2 = load(data_in + 16*i + 8);  // = [x8  x9  x10 x11]
        RealD4 x_in_3 = load(data_in + 16*i + 12); // = [x12 x13 x14 x15]
        RealD4 x_in_1_reord = reorder(x_in_1); // = [x4 x7 x6 x5]
        RealD4 x_in_2_reord = reorder(x_in_2); // = [x8 x11 x10 x9]
        RealD4 x_in_3_reord = reorder(x_in_3); // = [x12 x15 x14 x13]

        // Perform the calculations
        RealD4 temp_b = c_1*x_in_1 + s_1*x_in_1_reord;
        RealD4 temp_c = c_2*x_in_2 + s_2*x_in_2_reord;
        RealD4 temp_d = c_3*x_in_3 + s_3*x_in_3_reord;
        RealD4 temp_0 = x_in_0 + temp_b;
        RealD4 temp_1 = x_in_0 - temp_b;
        RealD4 temp_2 = temp_c + temp_d;
        RealD4 temp_3 = temp_c - temp_d;

        RealD4 x_out_0 = temp_0 + temp_2;
        RealD4 x_out_1 = temp_0 - temp_2;
        RealD4 x_out_2 = temp_1 + temp_3;
        RealD4 x_out_3 = temp_1 - temp_3;

        // Save the results
        store(x_out_0, data_out + 16*i + 0);
        store(x_out_1, data_out + 16*i + 4);
        store(x_out_2, data_out + 16*i + 8);
        store(x_out_3, data_out + 16*i + 12);
    }

    /*
    const double *c = step_info.cos_factors;
    const double *s = step_info.sin_factors;
    for (size_t i = 0; i < repeats; i++)
    {
        double x_in_0  = data_in[16*i + 0],  x_in_1  = data_in[16*i + 1],  x_in_2  = data_in[16*i + 2],  x_in_3  = data_in[16*i + 3];
        double x_in_4  = data_in[16*i + 4],  x_in_5  = data_in[16*i + 5],  x_in_6  = data_in[16*i + 6],  x_in_7  = data_in[16*i + 7];
        double x_in_8  = data_in[16*i + 8],  x_in_9  = data_in[16*i + 9],  x_in_10 = data_in[16*i + 10], x_in_11 = data_in[16*i + 11];
        double x_in_12 = data_in[16*i + 12], x_in_13 = data_in[16*i + 13], x_in_14 = data_in[16*i + 14], x_in_15 = data_in[16*i + 15];

        double b_0 = c[4]*x_in_4 +   s[4] *x_in_4; // = x_in_4
        double b_1 = c[5]*x_in_5 +   s[5] *x_in_7;
        double b_2 = c[6]*x_in_6 +   s[6] *x_in_6;
        double b_3 = c[7]*x_in_7 +   s[7] *x_in_5;
        double c_0 = c[8]*x_in_8 +   s[8] *x_in_8;
        double c_1 = c[9]*x_in_9 +   s[9] *x_in_11;
        double c_2 = c[10]*x_in_10 + s[10]*x_in_10;
        double c_3 = c[11]*x_in_11 + s[11]*x_in_9;
        double d_0 = c[12]*x_in_12 + s[12]*x_in_12;
        double d_1 = c[13]*x_in_13 + s[13]*x_in_15;
        double d_2 = c[14]*x_in_14 + s[14]*x_in_14;
        double d_3 = c[15]*x_in_15 + s[15]*x_in_13;

        data_out[16*i + 0]  = x_in_0 + b_0 + c_0 + d_0;
        data_out[16*i + 1]  = x_in_1 + b_1 + c_1 + d_1;
        data_out[16*i + 2]  = x_in_2 + b_2 + c_2 + d_2;
        data_out[16*i + 3]  = x_in_3 + b_3 + c_3 + d_3;
        data_out[16*i + 4]  = x_in_0 + b_0 - c_0 - d_0;
        data_out[16*i + 5]  = x_in_1 + b_1 - c_1 - d_1;
        data_out[16*i + 6]  = x_in_2 + b_2 - c_2 - d_2;
        data_out[16*i + 7]  = x_in_3 + b_3 - c_3 - d_3;
        data_out[16*i + 8]  = x_in_0 - b_0 + c_0 - d_0;
        data_out[16*i + 9]  = x_in_1 - b_1 + c_1 - d_1;
        data_out[16*i + 10] = x_in_2 - b_2 + c_2 - d_2;
        data_out[16*i + 11] = x_in_3 - b_3 + c_3 - d_3;
        data_out[16*i + 12] = x_in_0 - b_0 - c_0 + d_0;
        data_out[16*i + 13] = x_in_1 - b_1 - c_1 + d_1;
        data_out[16*i + 14] = x_in_2 - b_2 - c_2 + d_2;
        data_out[16*i + 15] = x_in_3 - b_3 - c_3 + d_3;
    }*/
}

void dht_1d_one_level_twiddle_radix4_stridemod8_0(const double *data_in, double *data_out, hhfft::StepInfoReal<double> &step_info)
{
    //std::cout << "fft_1d_one_level_twiddle_radix4_stridemod8_0" << std::endl;

    assert (step_info.radix == 2);
    assert (step_info.stride%8 == 0);

    size_t stride = step_info.stride;
    size_t repeats = step_info.repeats;

    const double *c = step_info.cos_factors;
    const double *s = step_info.sin_factors;

    RealD4 x5_prev = load(data_in); // Exact value here is not important as it is not actually used
    RealD4 x6_prev = load(data_in); // Exact value here is not important as it is not actually used
    RealD4 x7_prev = load(data_in); // Exact value here is not important as it is not actually used

    for (size_t i = 0; i < repeats; i++)
    {
        RealD4 x2_next = load(data_in + 1*stride);
        RealD4 x3_next = load(data_in + 2*stride);
        RealD4 x4_next = load(data_in + 3*stride);

        for (size_t k = 0; k < stride/2; k+=4)
        {
            size_t k2 = stride - k;

            // Load inputs
            RealD4 x0 = load(data_in + 0*stride + k);
            RealD4 x1 = load(data_in + 0*stride + k2 - 4);
            RealD4 x2 = x2_next;
            RealD4 x3 = x3_next;
            RealD4 x4 = x4_next;
            x2_next   = load(data_in + 1*stride + k  + 4);
            x3_next   = load(data_in + 2*stride + k  + 4);
            x4_next   = load(data_in + 3*stride + k  + 4);
            RealD4 x5 = load(data_in + 1*stride + k2 - 4);
            RealD4 x6 = load(data_in + 2*stride + k2 - 4);
            RealD4 x7 = load(data_in + 3*stride + k2 - 4);

            // Do some reordering                        
            RealD4 x2_reord = reorder2(x2, x2_next);
            RealD4 x3_reord = reorder2(x3, x3_next);
            RealD4 x4_reord = reorder2(x4, x4_next);
            RealD4 x5_reord = reorder2(x5, x5_prev);
            RealD4 x6_reord = reorder2(x6, x6_prev);
            RealD4 x7_reord = reorder2(x7, x7_prev);

            // Load twiddle factors and do the calculations
            // TODO it should be possible to load sin factors from cos factors
            RealD4 c2 = load(c + 1*stride + k);
            RealD4 s2 = load(s + 1*stride + k);
            RealD4 c3 = load(c + 2*stride + k);
            RealD4 s3 = load(s + 2*stride + k);
            RealD4 c4 = load(c + 3*stride + k);
            RealD4 s4 = load(s + 3*stride + k);
            RealD4 c5 = load(c + 1*stride + k2 - 4);
            RealD4 s5 = load(s + 1*stride + k2 - 4);
            RealD4 c6 = load(c + 2*stride + k2 - 4);
            RealD4 s6 = load(s + 2*stride + k2 - 4);
            RealD4 c7 = load(c + 3*stride + k2 - 4);
            RealD4 s7 = load(s + 3*stride + k2 - 4);

            RealD4 temp_a = c2*x2 + s2*x5_reord;
            RealD4 temp_b = c3*x3 + s3*x6_reord;
            RealD4 temp_c = c4*x4 + s4*x7_reord;
            RealD4 temp_d = c5*x5 + s5*x2_reord;
            RealD4 temp_e = c6*x6 + s6*x3_reord;
            RealD4 temp_f = c7*x7 + s7*x4_reord;

            RealD4 temp_0 = x0 + temp_a;
            RealD4 temp_1 = x0 - temp_a;
            RealD4 temp_2 = temp_b + temp_c;
            RealD4 temp_3 = temp_b - temp_c;

            RealD4 temp_4 = x1 + temp_d;
            RealD4 temp_5 = x1 - temp_d;
            RealD4 temp_6 = temp_e + temp_f;
            RealD4 temp_7 = temp_e - temp_f;

            // Save the results
            store(temp_0 + temp_2, data_out + 0*stride + k);
            store(temp_4 + temp_6, data_out + 0*stride+ k2 - 4);
            store(temp_0 - temp_2, data_out + 1*stride + k);
            store(temp_4 - temp_6, data_out + 1*stride + k2 - 4);

            store(temp_1 + temp_3, data_out + 2*stride + k);
            store(temp_5 + temp_7, data_out + 2*stride + k2 - 4);
            store(temp_1 - temp_3, data_out + 3*stride + k);
            store(temp_5 - temp_7, data_out + 3*stride + k2 - 4);

            //std::cout << std::endl;
            //std::cout << "x0 = " << x0 << ", x1 = " << x1  << ", x2 = " << x2  << ", x3 = " << x3 << std::endl;
            //std::cout << "x3 = " << x3 << ", x3_reord = " << x3_reord << std::endl;
            //std::cout << "x2 = " << x2 << ", x2_next = " << x2_next << ", x2_reord = " << x2_reord << std::endl;
            //std::cout << "c2 = " << c2 << ", c3 = " << c3 << ", s2 = " << s2 << ", s3 = " << s3 << std::endl;

            x5_prev = x5;
            x6_prev = x6;
            x7_prev = x7;
        }
        data_in += 4*stride;
        data_out += 4*stride;
    }
}


void dht_1d_one_level_twiddle_radix2_stridemod8_0(const double *data_in, double *data_out, hhfft::StepInfoReal<double> &step_info)
{
    //std::cout << "dht_1d_one_level_twiddle_radix2_stridemod8_0" << std::endl;

    assert (step_info.radix == 2);
    assert (step_info.stride%8 == 0);

    size_t stride = step_info.stride;
    size_t repeats = step_info.repeats;

    const double *c = step_info.cos_factors;
    const double *s = step_info.sin_factors;

    RealD4 x3_prev = load(data_in); // Exact value here is not important as it is not actually used

    for (size_t i = 0; i < repeats; i++)
    {
        RealD4 x2_next = load(data_in + stride);

        for (size_t k = 0; k < stride/2; k+=4)
        {
            size_t k2 = stride - k;

            // Load inputs
            RealD4 x0 = load(data_in + k);
            RealD4 x1 = load(data_in + k2 - 4);
            RealD4 x2 = x2_next;
            x2_next   = load(data_in + stride + k  + 4);
            RealD4 x3 = load(data_in + stride + k2 - 4);

            // Do some reordering
            RealD4 x3_reord = reorder2(x3, x3_prev);
            RealD4 x2_reord = reorder2(x2, x2_next);

            // Load twiddle factors and do the calculations
            // TODO it should be possible to load sin factors from cos factors
            RealD4 c2 = load(c + stride + k);
            RealD4 c3 = load(c + stride + k2 - 4);
            RealD4 s2 = load(s + stride + k);
            RealD4 s3 = load(s + stride + k2 - 4);
            RealD4 t0 = c2*x2 + s2*x3_reord;
            RealD4 t1 = c3*x3 + s3*x2_reord;

            // Save the results
            store(x0 + t0, data_out + k);
            store(x1 + t1, data_out + k2 - 4);
            store(x0 - t0, data_out + stride + k);
            store(x1 - t1, data_out + stride + k2 - 4);

            //std::cout << std::endl;
            //std::cout << "x0 = " << x0 << ", x1 = " << x1  << ", x2 = " << x2  << ", x3 = " << x3 << std::endl;
            //std::cout << "x3 = " << x3 << ", x3_reord = " << x3_reord << std::endl;
            //std::cout << "x2 = " << x2 << ", x2_next = " << x2_next << ", x2_reord = " << x2_reord << std::endl;
            //std::cout << "c2 = " << c2 << ", c3 = " << c3 << ", s2 = " << s2 << ", s3 = " << s3 << std::endl;

            x3_prev = x3;
        }

        data_in += 2*stride;
        data_out += 2*stride;
    }

    /*
    for (size_t i = 0; i < repeats; i++)
    {
        // First one, no cos/sin factors
        {
            double x0 = data_in[2*i*stride + 0*stride];
            double x1 = data_in[2*i*stride + 1*stride];
            data_out[2*i*stride + 0*stride] = x0 + x1;
            data_out[2*i*stride + 1*stride] = x0 - x1;
        }

        // Loop over most, start from top and bottom
        for (size_t k = 1; k < (stride+1)/2; k++)
        {
            size_t k2 = stride - k;

            // Read the values used
            double x0 = data_in[2*i*stride + k];
            double x1 = data_in[2*i*stride + k2];
            double x2 = data_in[2*i*stride + 1*stride + k];
            double x3 = data_in[2*i*stride + 1*stride + k2];

            double c0 = step_info.cos_factors[1*stride + k];
            double s0 = step_info.sin_factors[1*stride + k];
            double c1 = step_info.cos_factors[1*stride + k2];
            double s1 = step_info.sin_factors[1*stride + k2];

            // To the calculations
            double t0 = c0*x2 + s0*x3;
            double t1 = c1*x3 + s1*x2;

            // Write out the result
            data_out[2*i*stride + k] = x0 + t0;
            data_out[2*i*stride + k2] = x1 + t1;
            data_out[2*i*stride + 1*stride + k] = x0 - t0;
            data_out[2*i*stride + 1*stride + k2] = x1 - t1;
        }

        // As stride is even there is still one more to go
        {
            size_t k = stride/2;
            double x0 = data_in[2*i*stride + k];
            double x1 = data_in[2*i*stride + 1*stride + k];
            double c = step_info.cos_factors[1*stride + k];
            double s = step_info.sin_factors[1*stride + k];
            double t = (c+s)*x1;
            data_out[2*i*stride + k] = x0 + t;
            data_out[2*i*stride + 1*stride + k] = x0 - t;
        }
    }
    */
}


template<bool forward> void set_dht_1d_one_level(StepInfoRealD &step_info)
{
    size_t radix = step_info.radix;
    size_t stride = step_info.stride;    

    if (radix == 2)
    {
        if (stride == 1)
            step_info.step_function = dht_1d_one_level_radix2_stride1;  // Needed for first step when radix = 2
        else
            step_info.step_function = nullptr; //step_info.step_function = dht_1d_one_level<double,2,1>; //Should not be ever needed
    }
    if (radix == 3)
        step_info.step_function = dht_1d_one_level<double,3,1>;
    if (radix == 4)
    {
        if (stride == 1)
            step_info.step_function = dht_1d_one_level_radix4_stride1;  // Needed for first step when radix = 2
        else
            step_info.step_function = nullptr; //step_info.step_function = dht_1d_one_level<double,4,1>; //Should not be ever needed

    }
    if (radix == 5)
        step_info.step_function = dht_1d_one_level<double,5,1>;
    if (radix == 7)
        step_info.step_function = dht_1d_one_level<double,7,1>;

    /*
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
        */
}

template<bool forward> void set_dht_1d_one_level_twiddle(StepInfoRealD &step_info)
{
    size_t radix = step_info.radix;
    size_t stride = step_info.stride;

    if (radix == 2)
    {
        if (stride == 2)
            step_info.step_function = dht_1d_one_level_radix2_stride2;  // Needed for second step when radix = 2. Twiddle factors are always same c = [1 1 1 0], s = [0 0 0 1].
        else if (stride == 4)
            step_info.step_function = dht_1d_one_level_twiddle_radix2_stride4; // Needed for third step when radix = 2
        else if (stride%8 == 0)
            step_info.step_function = dht_1d_one_level_twiddle_radix2_stridemod8_0;
        else
            step_info.step_function = nullptr; //dht_1d_one_level_twiddle<double,2,1>; // Should not ever be needed
    }
    if (radix == 3)
        step_info.step_function = dht_1d_one_level_twiddle<double,3,1>;
    if (radix == 4)
    {
        if (stride == 4)
            step_info.step_function = dht_1d_one_level_twiddle_radix4_stride4; // Needed for second step when radix = 4
        else if (stride%8 == 0)
            step_info.step_function = dht_1d_one_level_twiddle_radix4_stridemod8_0;
        else
            step_info.step_function = nullptr; //step_info.step_function = dht_1d_one_level_twiddle<double,4,1>; // Should not ever be needed
    }
    if (radix == 5)
        step_info.step_function = dht_1d_one_level_twiddle<double,5,1>;
    if (radix == 7)
        step_info.step_function = dht_1d_one_level_twiddle<double,7,1>;

    /*
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
        */
}

void hhfft::HHFFT_1D_AVX_real_set_function(StepInfoRealD &step_info)
{
    step_info.step_function = nullptr;

    if (step_info.reorder_table != nullptr)
    {
        // TODO how to use in-place if algorithm if input actually points to output?
        if (step_info.forward)
            step_info.step_function = dht_1d_reorder<double,0,true>;
        else
            step_info.step_function = dht_1d_reorder<double,0,false>;
        return;
    }

    if (step_info.cos_factors == nullptr)
    {
        if (step_info.forward)
            set_dht_1d_one_level<true>(step_info);
        else
            set_dht_1d_one_level<false>(step_info);
    } else
    {
        if (step_info.forward)
            set_dht_1d_one_level_twiddle<true>(step_info);
        else
            set_dht_1d_one_level_twiddle<false>(step_info);
    }

    if (step_info.step_function == nullptr)
    {
        throw(std::runtime_error("HHFFT error: Unable to set a function!"));
    }
}




