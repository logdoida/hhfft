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

#include "architecture.h"
#include "step_info.h"
#include <stdlib.h>
#include <assert.h>
#include <cmath>

#include "../common/hhfft_1d_complex_avx_common_d.h"

using namespace hhfft;

template<bool forward>
    void fft_1d_complex_to_complex_packed_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    const double *packing_table = step_info.twiddle_factors;
    size_t n = 2*step_info.repeats; // n = number of original real numbers

    const ComplexD2 const1 = load_D2(0.0, -0.0, 0.0, -0.0);
    const ComplexD2 const2 = load_D2(-0.0, 0.0, -0.0, 0.0);

    // Input/output way
    if (forward)
    {
        double x_r = data_in[0];
        double x_i = data_in[1];
        data_out[0] = x_r + x_i;
        data_out[1] = 0.0;
        data_out[n] = x_r - x_i;
        data_out[n+1] = 0.0;
    } else
    {
        double x_r = data_in[0];
        double x_i = data_in[n];
        data_out[0] = 0.5*(x_r + x_i);
        data_out[1] = 0.5*(x_r - x_i);
    }

    if (n%4 == 0)
    {
        ComplexD x_in = load_D(data_in + n/2);
        ComplexD x_out = change_sign_D(x_in, const1_128);
        store_D(x_out, data_out + n/2);
    }

    size_t i;
    for (i = 2; i + 2 < n/2; i+=4)
    {
        ComplexD2 sssc = load_D2(packing_table + i);
        ComplexD2 x0_in = load_D2(data_in + i);
        ComplexD2 x1_in = load_two_128_D2(data_in + n - i, data_in + n - i - 2);

        if(!forward)
        {
            sssc = change_sign_D2(sssc, const1);
        }

        ComplexD2 temp0 = x0_in + change_sign_D2(x1_in, const2);
        ComplexD2 temp1 = mul_D2(sssc, temp0);

        ComplexD2 x0_out = temp1 + x0_in;
        ComplexD2 x1_out = change_sign_D2(temp1, const2) + x1_in;

        store_D2(x0_out, data_out + i);
        store_two_128_D2(x1_out, data_out + n - i, data_out + n - i - 2);
    }

    if (i < n/2)
    {
        ComplexD sssc = load_D(packing_table + i);
        ComplexD x0_in = load_D(data_in + i);
        ComplexD x1_in = load_D(data_in + n - i);

        if(!forward)
        {
            sssc = change_sign_D(sssc, const1_128);
        }

        ComplexD temp0 = x0_in + change_sign_D(x1_in, const2_128);
        ComplexD temp1 = mul_D(sssc, temp0);

        ComplexD x0_out = temp1 + x0_in;
        ComplexD x1_out = change_sign_D(temp1, const2_128) + x1_in;

        store_D(x0_out, data_out + i);
        store_D(x1_out, data_out + n - i);
    }
}

// This is found in hhfft_1d_complex_avx_d.cpp
template<bool scale> void fft_1d_complex_reorder_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

void fft_1d_complex_to_complex_packed_ifft_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    // If data_in == data_out,
    if(data_in == data_out)
    {
        fft_1d_complex_to_complex_packed_avx_d<false>(data_in, data_out, step_info);
        fft_1d_complex_reorder_avx_d<true>(data_out, data_out, step_info);
        return;
    }

    const double *packing_table = step_info.twiddle_factors;
    size_t n = step_info.repeats; // 2*n = number of original real numbers
    uint32_t *reorder_table_inverse = step_info.reorder_table;

    double k = step_info.norm_factor;
    ComplexD k128 = broadcast64_D(k);
    ComplexD2 k256 = broadcast64_D2(k);

    const ComplexD2 const1 = load_D2(0.0, -0.0, 0.0, -0.0);
    const ComplexD2 const2 = load_D2(-0.0, 0.0, -0.0, 0.0);

    double x_r = data_in[0];
    double x_i = data_in[2*n];
    data_out[0] = 0.5*k*(x_r + x_i);
    data_out[1] = 0.5*k*(x_r - x_i);

    if (n%2 == 0)
    {
        size_t i = reorder_table_inverse[n/2];

        ComplexD x_in = load_D(data_in + n);
        ComplexD x_out = k128*change_sign_D(x_in, const1_128);
        store_D(x_out, data_out + 2*i);
    }

    size_t i;
    for (i = 1; i + 1 < (n+1)/2; i+=2)
    {
        ComplexD2 sssc = load_D2(packing_table + 2*i);
        sssc = change_sign_D2(sssc, const1);
        ComplexD2 x0_in = k256*load_D2(data_in + 2*i);
        ComplexD2 x1_in = k256*load_two_128_D2(data_in + 2*(n - i), data_in + 2*(n - i) - 2);

        ComplexD2 temp0 = x0_in + change_sign_D2(x1_in, const2);
        ComplexD2 temp1 = mul_D2(sssc, temp0);

        ComplexD2 x0_out = temp1 + x0_in;
        ComplexD2 x1_out = change_sign_D2(temp1, const2) + x1_in;

        size_t i2 = reorder_table_inverse[i];
        size_t i3 = reorder_table_inverse[i + 1];
        size_t i4 = reorder_table_inverse[n - i];
        size_t i5 = reorder_table_inverse[n - i - 1];

        store_two_128_D2(x0_out, data_out + 2*i2, data_out + 2*i3);
        store_two_128_D2(x1_out, data_out + 2*i4, data_out + 2*i5);
    }

    if (i < (n+1)/2)
    {
        ComplexD sssc = load_D(packing_table + 2*i);
        sssc = change_sign_D(sssc, const1_128);
        ComplexD x0_in = k128*load_D(data_in + 2*i);
        ComplexD x1_in = k128*load_D(data_in + 2*(n - i));

        ComplexD temp0 = x0_in + change_sign_D(x1_in, const2_128);
        ComplexD temp1 = mul_D(sssc, temp0);

        ComplexD x0_out = temp1 + x0_in;
        ComplexD x1_out = change_sign_D(temp1, const2_128) + x1_in;

        size_t i2 = reorder_table_inverse[i];
        size_t i3 = reorder_table_inverse[n - i];

        store_D(x0_out, data_out + 2*i2);
        store_D(x1_out, data_out + 2*i3);
    }
}

// for small sizes
template<bool forward, size_t n> void fft_1d_complex_to_complex_packed_1level_avx_d(ComplexD *x)
{
    const double *packing_table = get_packing_table<n>();

    // Input/output way
    if (forward)
    {
        ComplexD zeros = load_D(0,0);
        ComplexD t0 = _mm_unpacklo_pd(x[0], zeros);
        ComplexD t1 = _mm_unpackhi_pd(x[0], zeros);
        x[0] = t0 + t1;
        x[n/2] = t0 - t1;
    } else
    {
        ComplexD half = load_D(0.5,0.5);
        ComplexD t0 = x[0] + x[n/2];
        ComplexD t1 = x[0] - x[n/2];
        ComplexD t3 = _mm_unpacklo_pd(t0, t1);
        x[0] = half*t3;
    }

    if (n%4 == 0)
    {
        x[n/4] = change_sign_D(x[n/4], const1_128);
    }

    for (size_t i = 2; i < n/2; i+=2)
    {
        ComplexD sssc = load_D(packing_table + i);
        ComplexD x0_in = x[i/2];
        ComplexD x1_in = x[n/2 - i/2];

        if(!forward)
        {
            sssc = change_sign_D(sssc, const1_128);
        }

        ComplexD temp0 = x0_in + change_sign_D(x1_in, const2_128);
        ComplexD temp1 = mul_D(sssc, temp0);

        x[i/2] = temp1 + x0_in;
        x[n/2 - i/2] = change_sign_D(temp1, const2_128) + x1_in;
    }
}

////////////////////////////////////// Odd real FFT ////////////////////////////////////////////////////

// This function is used on the first level of odd real fft
// Note this does not use avx
template<size_t radix> void fft_1d_real_first_level_forward_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    size_t repeats = step_info.repeats;
    uint32_t *reorder_table = step_info.reorder_table;

    ComplexD x_temp_in[radix];
    ComplexD x_temp_out[radix];
    bool dir_out = true;
    for (size_t i = 0; i < repeats; i++)
    {
        // Copy input data taking reordering into account
        for (size_t j = 0; j < radix; j++)
        {
            size_t ind = reorder_table[i*radix + j];
            x_temp_in[j] = load_real_D(data_in + ind);
        }

        // Multiply with coefficients
        multiply_coeff_D<radix,true>(x_temp_in, x_temp_out);

        // Save only about half of the output
        // First/ last one is real
        if (dir_out) // direction normal
        {
            store_real_D(x_temp_out[0], data_out + i*radix); // only real part

            for (size_t j = 1; j < radix/2 + 1; j++)
            {
                store_D(x_temp_out[j], data_out + i*radix + 2*j - 1);
            }
        } else // direction inverted
        {
            store_real_D(x_temp_out[0], data_out + i*radix + radix - 1); // only real part

            for (size_t j = 1; j < radix/2 + 1; j++)
            {
                store_D(x_temp_out[j], data_out + i*radix + radix - 2*j - 1);
            }
        }
        dir_out = !dir_out;
    }
}

inline size_t index_dir_stride_odd(size_t dir_in, size_t stride, size_t k)
{
    return dir_in*(4*k - stride) + stride - 2*k - 1;
}

template<size_t radix> inline __attribute__((always_inline)) void fft_1d_real_one_level_forward_avx_d_internal(const double *data_in, double *data_out, const double *twiddle_factors, size_t stride, bool dir_out)
{
    // The first/last value in each stride is real
    {
        ComplexD x0_temp_in[radix];
        ComplexD x0_temp_out[radix];

        // Read the inputs
        bool dir_in = dir_out;
        for (size_t j = 0; j < radix; j++)
        {
            if (dir_in)
            {
                x0_temp_in[j] = load_real_D(data_in + j*stride);
            } else
            {
                x0_temp_in[j] = load_real_D(data_in + j*stride + stride - 1);
            }

            dir_in = !dir_in;
        }

        multiply_coeff_D<radix,true>(x0_temp_in, x0_temp_out);

        // first/last output is real
        if (dir_out)
        {
            store_real_D(x0_temp_out[0], data_out);
        } else
        {
            store_real_D(x0_temp_out[0], data_out + radix*stride - 1);
        }
        // only about half is written
        for (size_t j = 1; j < (radix+1)/2; j++)
        {
            if (dir_out)
            {
                store_D(x0_temp_out[j], data_out + 2*j*stride - 1);
            } else
            {
                store_D(x0_temp_out[(radix+1)/2 - j], data_out + 2*j*stride - stride - 1);
            }
        }
    }

    size_t k = 1;
    // First use 256-bit variables
    for (; k + 1 < (stride+1)/2; k+=2)
    {
        ComplexD2 x_temp_in[radix];
        ComplexD2 x_temp_out[radix];
        ComplexD2 twiddle_temp[radix];
        size_t dir_in = dir_out;

        // Copy the values and twiddle factors
        for (size_t j = 0; j < radix; j++)
        {
            if (dir_in)
            {
                x_temp_in[j] = load_D2(data_in + j*stride + 2*k - 1);
            } else
            {
                x_temp_in[j] = load_two_128_D2(data_in + j*stride + stride - 2*k - 1, data_in + j*stride + stride - 2*k - 3);
            }

            twiddle_temp[j] = load_D2(twiddle_factors + 2*k + 2*j*stride);
            dir_in = !dir_in;
        }

        multiply_twiddle_D2<radix,true>(x_temp_in, x_temp_in, twiddle_temp);

        multiply_coeff_real_odd_forward_D2<radix>(x_temp_in, x_temp_out);

        // save output taking the directions into account
        dir_in = dir_out;
        for (size_t j = 0; j < radix; j++)
        {
            // reverse the output order if required
            ComplexD2 temp_out;
            if (dir_out)
            {
                temp_out = x_temp_out[j];
            } else
            {
                temp_out = x_temp_out[radix - j - 1];
            }

            if (dir_in)
            {
                store_D2(temp_out, data_out + j*stride + 2*k - 1);
            } else
            {
                store_two_128_D2(temp_out, data_out + j*stride + stride - 2*k - 1, data_out + j*stride + stride - 2*k - 3);
            }

            dir_in = !dir_in;
        }
    }


    // Then, if necessary, use 128-bit variables
    if (k < (stride+1)/2)
    {
        ComplexD x_temp_in[radix];
        ComplexD x_temp_out[radix];
        ComplexD twiddle_temp[radix];
        size_t dir_in = dir_out;

        // Copy the values and twiddle factors
        for (size_t j = 0; j < radix; j++)
        {
            size_t index2 = index_dir_stride_odd(dir_in, stride, k);

            x_temp_in[j] = load_D(data_in + j*stride + index2);
            twiddle_temp[j] = load_D(twiddle_factors + 2*k + 2*j*stride);
            dir_in = !dir_in;
        }

        multiply_twiddle_D<radix,true>(x_temp_in, x_temp_in, twiddle_temp);

        multiply_coeff_real_odd_forward_D<radix>(x_temp_in, x_temp_out);

        // save output taking the directions into account
        dir_in = dir_out;
        for (size_t j = 0; j < radix; j++)
        {
            size_t index2 = index_dir_stride_odd(dir_in, stride, k);

            // reverse the output order if required
            if (dir_out)
            {
                store_D(x_temp_out[j], data_out + j*stride + index2);
            } else
            {
                store_D(x_temp_out[radix - j - 1], data_out + j*stride + index2);
            }

            dir_in = !dir_in;
        }
    }
}

// This function is used on rest of the odd real fft
template<size_t radix> void fft_1d_real_one_level_forward_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    size_t stride = step_info.stride;
    size_t repeats = step_info.repeats;
    double *twiddle_factors = step_info.twiddle_factors;

    bool dir_out = true;
    for (size_t i = 0; i < repeats; i++)
    {
        fft_1d_real_one_level_forward_avx_d_internal<radix>(data_in + i*radix*stride, data_out + i*radix*stride, twiddle_factors, stride, dir_out);

        dir_out = !dir_out;
    }
}

// This function is used on rest of the odd real 2d fft
template<size_t radix> void fft_2d_real_odd_rows_forward_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    size_t n = step_info.size;
    size_t stride = step_info.stride;
    size_t repeats = step_info.repeats;
    size_t m = stride*repeats*radix + 1;
    double *twiddle_factors = step_info.twiddle_factors;

    for (size_t j = 0; j < n; j++)
    {
        bool dir_out = true;
        for (size_t i = 0; i < repeats; i++)
        {
            fft_1d_real_one_level_forward_avx_d_internal<radix>(data_in + j*m + i*radix*stride + 1, data_out + j*m + i*radix*stride + 1,
                                                                  twiddle_factors, stride, dir_out);

            dir_out = !dir_out;
        }
    }
}

////////////////////////////////////// Odd real IFFT ////////////////////////////////////////////////////

// This function is used on the first level of odd real ifft
// Note this does not use avx
template<size_t radix> void fft_1d_real_first_level_inverse_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    size_t repeats = step_info.repeats;
    uint32_t *reorder_table = step_info.reorder_table;
    size_t n = radix * (2*step_info.repeats - 1);
    ComplexD norm_factor = broadcast64_D(step_info.norm_factor);

    ComplexD x_temp_in[radix];
    ComplexD x_temp_out[radix];

    // In the first repeat input is r, (r+i), (r+i) ... and output is r,r,r,r,r...
    {
        x_temp_in[0] = norm_factor*load_real_D(data_in + 0);

        // Read other inputs and conjugate them
        for (size_t j = 1; j <= radix/2; j++)
        {
            size_t ind = reorder_table[j];
            ComplexD x = norm_factor*load_D(data_in + 2*ind);
            x_temp_in[j] = x;
            x_temp_in[radix-j] = conj_D(x);
        }

        // NOTE this could be optimized as actually only the real part of output is used and input has symmetry
        // Multiply with coefficients
        multiply_coeff_D<radix,false>(x_temp_in, x_temp_out);

        // Write only real parts of the data
        for (size_t j = 0; j < radix; j++)
        {
            store_real_D(x_temp_out[j], data_out + j);
        }
    }

    // Other repeats are more usual, data ordering changes from r,i,r,i,r,i... to r,r,r...i,i,i...
    for (size_t i = 1; i < repeats; i++)
    {
        // Copy input data taking reordering into account
        for (size_t j = 0; j < radix; j++)
        {
            size_t ind = reorder_table[i*radix - radix/2 + j];
            if (ind <= n/2)
            {
                x_temp_in[j] = norm_factor*load_D(data_in + 2*ind);
            } else
            {
                // If input is from the lower part, it needs to be conjugated
                ind = n - ind;
                x_temp_in[j] = conj_D(norm_factor*load_D(data_in + 2*ind));
            }
        }

        // Multiply with coefficients
        multiply_coeff_D<radix,false>(x_temp_in, x_temp_out);

        // Store real and imag parts separately
        for (size_t j = 0; j < radix; j++)
        {
            store_D(x_temp_out[j], data_out[2*i*radix - radix + j], data_out[2*i*radix + j]);
        }
    }
}

// This function is used on first level of the odd real 2d ifft
// Note this does not use avx
template<size_t radix> void fft_2d_real_odd_rows_first_level_inverse_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    size_t repeats = step_info.repeats;
    size_t m = radix * (2*step_info.repeats - 1);
    size_t n = step_info.size;

    ComplexD x_temp_in[radix];
    ComplexD x_temp_out[radix];

    // Loop over all rows
    for (size_t k = 0; k < n; k++)
    {
        // In the first repeat input is r, (r+i), (r+i) ... and output is r,r,r,r,r...
        {
            x_temp_in[0] = load_real_D(data_in + k*m);

            // Read other inputs and conjugate them
            for (size_t j = 1; j <= radix/2; j++)
            {
                ComplexD x = load_D(data_in + k*m + 2*j - 1);
                x_temp_in[j] = x;
                x_temp_in[radix-j] = conj_D(x);
            }

            // Multiply with coefficients
            multiply_coeff_D<radix,false>(x_temp_in, x_temp_out);

            // Write only real parts of the data
            for (size_t j = 0; j < radix; j++)
            {
                store_real_D(x_temp_out[j], data_out + k*m + j);
            }
        }

        // Other repeats are more usual, data ordering changes from r,i,r,i,r,i... to r,r,r...i,i,i...
        for (size_t i = 1; i < repeats; i++)
        {
            // Copy input data
            for (size_t j = 0; j < radix; j++)
            {
                x_temp_in[j] = load_D(data_in + k*m + 2*i*radix - radix + 2*j);
            }

            // Multiply with coefficients
            multiply_coeff_D<radix,false>(x_temp_in, x_temp_out);

            // Store real and imag parts separately
            for (size_t j = 0; j < radix; j++)
            {
                store_D(x_temp_out[j], data_out[k*m + 2*i*radix - radix + j], data_out[k*m + 2*i*radix + j]);
            }
        }
    }
}

// This function is used on on rest of the odd real ifft
template<size_t radix> void fft_1d_real_one_level_inverse_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    size_t repeats = step_info.repeats;
    size_t stride = step_info.stride;
    double *twiddle_factors = step_info.twiddle_factors;

    // In the first repeat input is r,r,r,... r,r,r, ... i,i,i, ... and output is r,r,r,r,r...
    {
        size_t k;
        for (k = 0; k + 1 < stride; k+=2)
        {
            ComplexD2 x_temp_in[radix];
            ComplexD2 x_temp_out[radix];
            ComplexD2 twiddle_temp[radix];

            x_temp_in[0] = load_real_D2(data_in + k);

            // Read other inputs, only about half of them is needed
            for (size_t j = 1; j <= radix/2; j++)
            {
                x_temp_in[j] = load_D2(data_in[2*j*stride - stride + k], data_in[2*j*stride + k],
                                       data_in[2*j*stride - stride + k + 1], data_in[2*j*stride + k + 1]);
                twiddle_temp[j] = load_D2(twiddle_factors + 2*j*stride + 2*k + 0);
            }

            // Multiply with twiddle factors
            multiply_conj_twiddle_odd_D2<radix>(x_temp_in, x_temp_in, twiddle_temp);

            // NOTE this could be optimized as actually only the real part of output is used and input has symmetry
            // Multiply with coefficients
            multiply_coeff_D2<radix,false>(x_temp_in, x_temp_out);

            // Write only real parts of the data
            for (size_t j = 0; j < radix; j++)
            {
                store_real_D2(x_temp_out[j], data_out + j*stride + k);
            }
        }

        if (k < stride)
        {
            ComplexD x_temp_in[radix];
            ComplexD x_temp_out[radix];
            ComplexD twiddle_temp[radix];

            x_temp_in[0] = load_real_D(data_in + k);

            // Read other inputs, only about half of them is needed
            for (size_t j = 1; j <= radix/2; j++)
            {
                x_temp_in[j] = load_D(data_in[2*j*stride - stride + k], data_in[2*j*stride + k]);
                twiddle_temp[j] = load_D(twiddle_factors + 2*j*stride + 2*k + 0);
            }

            // Multiply with twiddle factors
            multiply_conj_twiddle_odd_D<radix>(x_temp_in, x_temp_in, twiddle_temp);

            // NOTE this could be optimized as actually only the real part of output is used and input has symmetry
            // Multiply with coefficients
            multiply_coeff_D<radix,false>(x_temp_in, x_temp_out);

            // Write only real parts of the data
            for (size_t j = 0; j < radix; j++)
            {
                store_real_D(x_temp_out[j], data_out + j*stride + k);
            }
        }
    }

    // Other repeats are more usual, however both inputs and outputs have real and imag parts separated
    for (size_t i = 1; i < repeats; i++)
    {
        // Load four complex numbers at a time (r,r,r,r) and (i,i,i,i)
        size_t k;
        for (k = 0; k + 3 < stride; k+=4)
        {
            ComplexD4S x_temp_in[radix];
            ComplexD4S x_temp_out[radix];
            ComplexD4S twiddle_temp[radix];

            // Read real and imag parts separately
            for (size_t j = 0; j < radix; j++)
            {
                size_t index = 2*i*stride*radix + 2*j*stride - stride*radix;
                x_temp_in[j] = load_D4S(data_in + index + k, data_in + index + stride + k);
                twiddle_temp[j] = load512_D4S(twiddle_factors + 2*j*stride + 2*k);
            }

            // Multiply with twiddle factors
            multiply_twiddle_D4S<radix,false>(x_temp_in, x_temp_in, twiddle_temp);

            // Multiply with coefficients
            multiply_coeff_D4S<radix,false>(x_temp_in, x_temp_out);

            // Store real and imag parts separately
            for (size_t j = 0; j < radix; j++)
            {
                size_t index = 2*i*stride*radix + j*stride;
                store_D4S(x_temp_out[j], data_out + index - stride*radix + k, data_out + index + k);
            }
        }

        // Load two complex numbers at a time (r,r) and (i,i)        
        if (k + 1 < stride)
        {
            ComplexD2S x_temp_in[radix];
            ComplexD2S x_temp_out[radix];
            ComplexD2S twiddle_temp[radix];

            // Read real and imag parts separately
            for (size_t j = 0; j < radix; j++)
            {
                size_t index = 2*i*stride*radix + 2*j*stride - stride*radix;
                x_temp_in[j] = load_D2S(data_in + index + k, data_in + index + stride + k);
                twiddle_temp[j] = load256s_D2S(twiddle_factors + 2*j*stride + 2*k);
            }

            // Multiply with twiddle factors
            multiply_twiddle_D2S<radix,false>(x_temp_in, x_temp_in, twiddle_temp);

            // Multiply with coefficients
            multiply_coeff_D2S<radix,false>(x_temp_in, x_temp_out);

            // Store real and imag parts separately
            for (size_t j = 0; j < radix; j++)
            {
                size_t index = 2*i*stride*radix + j*stride;
                store_D2S(x_temp_out[j], data_out + index - stride*radix + k, data_out + index + k);
            }
            k += 2;
        }        

        // Load one complex number at a time r and i separated
        if (k < stride)
        {
            ComplexD x_temp_in[radix];
            ComplexD x_temp_out[radix];
            ComplexD twiddle_temp[radix];

            // Read real and imag parts separately
            for (size_t j = 0; j < radix; j++)
            {
                size_t index = 2*i*stride*radix + 2*j*stride - stride*radix;
                x_temp_in[j] = load_D(data_in[index + k], data_in[index + stride + k]);

                twiddle_temp[j] = load_D(twiddle_factors + 2*j*stride + 2*k + 0);
            }

            // Multiply with twiddle factors
            multiply_twiddle_D<radix,false>(x_temp_in, x_temp_in, twiddle_temp);

            // Multiply with coefficients
            multiply_coeff_D<radix,false>(x_temp_in, x_temp_out);

            // Store real and imag parts separately
            for (size_t j = 0; j < radix; j++)
            {
                size_t index = 2*i*stride*radix + j*stride;
                store_D(x_temp_out[j], data_out[index - stride*radix + k], data_out[index + k]);
            }
        }
    }
}

// This is used in 2d real for odd row sizes
template<size_t radix> void fft_2d_real_odd_rows_inverse_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    size_t n = step_info.size;
    size_t stride = step_info.stride;
    size_t m = stride*radix*(2*step_info.repeats - 1);

    // Process all rows separately
    for (size_t j = 0; j < n; j++)
    {
        fft_1d_real_one_level_inverse_avx_d<radix>(data_in + j*m, data_out + j*m, step_info);
    }
}

// fft for small sizes (2,3,4,5,6,7,8,10,14,16) where only one level is needed
template<size_t n, bool forward> void fft_1d_real_1level_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &)
{
    ComplexD k = broadcast64_D(2.0/n);

    if (n == 1)
    {
        data_out[0] = data_in[0];
        data_out[1] = 0;
    } else if (n%2 == 0)
    {
        // n even

        ComplexD x_temp_in[n/2+1];
        ComplexD x_temp_out[n/2+1];

        if (forward)
        {
            // Copy input data
            for (size_t i = 0; i < n/2; i++)
            {
                x_temp_in[i] = load_D(data_in + 2*i);
            }

            // Multiply with coefficients
            multiply_coeff_D<n/2,forward>(x_temp_in, x_temp_out);

            // Make the conversion
            fft_1d_complex_to_complex_packed_1level_avx_d<forward,n>(x_temp_out);

            // Copy output data
            for (size_t i = 0; i < n/2 + 1; i++)
            {
                store_D(x_temp_out[i], data_out + 2*i);
            }
        } else
        {
            // Copy input data
            for (size_t i = 0; i < n/2 + 1; i++)
            {
                x_temp_in[i] = load_D(data_in + 2*i);
            }

            // Make the conversion
            fft_1d_complex_to_complex_packed_1level_avx_d<forward,n>(x_temp_in);

            // Multiply with coefficients
            multiply_coeff_D<n/2,forward>(x_temp_in, x_temp_out);

            // Copy output data
            for (size_t i = 0; i < n/2; i++)
            {
                store_D(k*x_temp_out[i], data_out + 2*i);
            }
        }
    } else
    {
        ComplexD x_temp_in[n];
        ComplexD x_temp_out[n];

        // odd n
        if (forward)
        {
            // Copy real input data
            for (size_t j = 0; j < n; j++)
            {
                x_temp_in[j] = load_real_D(data_in + j);
            }

            // Multiply with coefficients
            multiply_coeff_D<n,true>(x_temp_in, x_temp_out);

            // Save only about half of the output
            for (size_t j = 0; j < n/2 + 1; j++)
            {
                store_D(x_temp_out[j], data_out + 2*j);
            }
        } else
        {
            ComplexD norm_factor = load_D(1.0/n, 1.0/n);

            // First input is real
            x_temp_in[0] = norm_factor*load_real_D(data_in + 0);

            // Read other inputs and conjugate them
            for (size_t j = 1; j <= n/2; j++)
            {
                ComplexD x = norm_factor*load_D(data_in + 2*j);
                x_temp_in[j] = x;
                x_temp_in[n-j] = conj_D(x);
            }

            // Multiply with coefficients
            multiply_coeff_D<n,false>(x_temp_in, x_temp_out);

            // Write only real parts of the data
            for (size_t j = 0; j < n; j++)
            {
                store_real_D(x_temp_out[j], data_out + j);
            }
        }
    }
}


// Instantiations of the functions defined in this class
template void fft_1d_complex_to_complex_packed_avx_d<false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_to_complex_packed_avx_d<true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_1d_real_1level_avx_d<1, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_1level_avx_d<2, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_1level_avx_d<3, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_1level_avx_d<4, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_1level_avx_d<5, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_1level_avx_d<6, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_1level_avx_d<7, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_1level_avx_d<8, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_1level_avx_d<10, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_1level_avx_d<12, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_1level_avx_d<14, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_1level_avx_d<16, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_1level_avx_d<2, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_1level_avx_d<1, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_1level_avx_d<3, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_1level_avx_d<4, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_1level_avx_d<5, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_1level_avx_d<6, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_1level_avx_d<7, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_1level_avx_d<8, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_1level_avx_d<10, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_1level_avx_d<12, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_1level_avx_d<14, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_1level_avx_d<16, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_1d_real_first_level_forward_avx_d<3>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_first_level_forward_avx_d<5>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_first_level_forward_avx_d<7>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_1d_real_first_level_inverse_avx_d<3>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_first_level_inverse_avx_d<5>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_first_level_inverse_avx_d<7>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_1d_real_one_level_forward_avx_d<3>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_one_level_forward_avx_d<5>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_one_level_forward_avx_d<7>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_1d_real_one_level_inverse_avx_d<3>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_one_level_inverse_avx_d<5>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_one_level_inverse_avx_d<7>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_2d_real_odd_rows_forward_avx_d<3>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_forward_avx_d<5>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_forward_avx_d<7>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_2d_real_odd_rows_first_level_inverse_avx_d<3>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_first_level_inverse_avx_d<5>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_first_level_inverse_avx_d<7>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_2d_real_odd_rows_inverse_avx_d<3>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_inverse_avx_d<5>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_inverse_avx_d<7>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
