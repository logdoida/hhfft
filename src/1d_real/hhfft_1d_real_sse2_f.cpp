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

#include "architecture.h"
#include "step_info.h"
#include <stdlib.h>
#include <assert.h>
#include <cmath>

#include "../common/hhfft_common_sse2_f.h"
#include "../raders/raders_sse2_f.h"

using namespace hhfft;

const ComplexF const2_F = load_F(-0.0f, 0.0f);

template<bool forward>
    void fft_1d_complex_to_complex_packed_sse2_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info)
{
    const float *packing_table = step_info.twiddle_factors;
    size_t n = 2*step_info.repeats; // n = number of original real numbers

    // Input/output way
    if (forward)
    {
        float x_r = data_in[0];
        float x_i = data_in[1];
        data_out[0] = x_r + x_i;
        data_out[1] = 0.0;
        data_out[n] = x_r - x_i;
        data_out[n+1] = 0.0;
    } else
    {
        float x_r = data_in[0];
        float x_i = data_in[n];
        data_out[0] = 0.5f*(x_r + x_i);
        data_out[1] = 0.5f*(x_r - x_i);
    }

    if (n%4 == 0)
    {
        ComplexF x_in = load_F(data_in + n/2);
        ComplexF x_out = change_sign_F(x_in, const1_F);
        store_F(x_out, data_out + n/2);
    }

    size_t i;
    for (i = 2; i + 2 < n/2; i+=4)
    {
        ComplexF2 sssc = load_F2(packing_table + i);
        ComplexF2 x0_in = load_F2(data_in + i);
        ComplexF2 x1_in = load_two_64_F2(data_in + n - i, data_in + n - i - 2);

        if(!forward)
        {
            sssc = change_sign_F2(sssc, const1_F2);
        }

        ComplexF2 temp0 = x0_in + change_sign_F2(x1_in, const2_F2);
        ComplexF2 temp1 = mul_F2(sssc, temp0);

        ComplexF2 x0_out = temp1 + x0_in;
        ComplexF2 x1_out = change_sign_F2(temp1, const2_F2) + x1_in;

        store_F2(x0_out, data_out + i);
        store_two_64_F2(x1_out, data_out + n - i, data_out + n - i - 2);
    }

    if (i < n/2)
    {
        ComplexF sssc = load_F(packing_table + i);
        ComplexF x0_in = load_F(data_in + i);
        ComplexF x1_in = load_F(data_in + n - i);

        if(!forward)
        {
            sssc = change_sign_F(sssc, const1_F);
        }

        ComplexF temp0 = x0_in + change_sign_F(x1_in, const2_F);
        ComplexF temp1 = mul_F(sssc, temp0);

        ComplexF x0_out = temp1 + x0_in;
        ComplexF x1_out = change_sign_F(temp1, const2_F) + x1_in;

        store_F(x0_out, data_out + i);
        store_F(x1_out, data_out + n - i);
    }
}

void fft_1d_complex_to_complex_packed_ifft_sse2_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info)
{
    const float *packing_table = step_info.twiddle_factors;
    size_t n = step_info.repeats; // 2*n = number of original real numbers
    uint32_t *reorder_table_inverse = step_info.reorder_table;
    float k = step_info.norm_factor;
    ComplexF k64 = broadcast32_F(k);
    ComplexF2 k128 = broadcast32_F2(k);

    float x_r = data_in[0];
    float x_i = data_in[2*n];
    data_out[0] = 0.5f*k*(x_r + x_i);
    data_out[1] = 0.5f*k*(x_r - x_i);

    if (n%2 == 0)
    {
        size_t i = reorder_table_inverse[n/2];

        ComplexF x_in = load_F(data_in + n);
        ComplexF x_out = k64*change_sign_F(x_in, const1_F);
        store_F(x_out, data_out + 2*i);
    }

    size_t i;
    for (i = 1; i + 1 < (n+1)/2; i+=2)
    {
        ComplexF2 sssc = load_F2(packing_table + 2*i);
        sssc = change_sign_F2(sssc, const1_F2);
        ComplexF2 x0_in = k128*load_F2(data_in + 2*i);
        ComplexF2 x1_in = k128*load_two_64_F2(data_in + 2*(n - i), data_in + 2*(n - i) - 2);

        ComplexF2 temp0 = x0_in + change_sign_F2(x1_in, const2_F2);
        ComplexF2 temp1 = mul_F2(sssc, temp0);

        ComplexF2 x0_out = temp1 + x0_in;
        ComplexF2 x1_out = change_sign_F2(temp1, const2_F2) + x1_in;

        size_t i2 = reorder_table_inverse[n - i];
        size_t i3 = reorder_table_inverse[n - i - 1];
        size_t i4 = reorder_table_inverse[i];
        size_t i5 = reorder_table_inverse[i + 1];

        store_two_64_F2(x0_out, data_out + 2*i2, data_out + 2*i3);
        store_two_64_F2(x1_out, data_out + 2*i4, data_out + 2*i5);
    }

    if (i < (n+1)/2)
    {
        ComplexF sssc = load_F(packing_table + 2*i);
        sssc = change_sign_F(sssc, const1_F);
        ComplexF x0_in = k128*load_F(data_in + 2*i);
        ComplexF x1_in = k128*load_F(data_in + 2*(n - i));

        ComplexF temp0 = x0_in + change_sign_F(x1_in, const2_F);
        ComplexF temp1 = mul_F(sssc, temp0);

        ComplexF x0_out = temp1 + x0_in;
        ComplexF x1_out = change_sign_F(temp1, const2_F) + x1_in;

        size_t i2 = reorder_table_inverse[n - i];
        size_t i3 = reorder_table_inverse[i];

        store_F(x0_out, data_out + 2*i2);
        store_F(x1_out, data_out + 2*i3);
    }
}

// for small sizes
template<bool forward, size_t n> void fft_1d_complex_to_complex_packed_1level_sse2_f(ComplexF *x)
{
    const float *packing_table = get_packing_table_f<n>();

    // Input/output way
    if (forward)
    {        
        float t0, t1;
        store_F(x[0], t0, t1);
        x[0] = _mm_set_ss(t0 + t1);
        x[n/2] = _mm_set_ss(t0 - t1);
    } else
    {
        ComplexF half = load_F(0.5,0.5);
        ComplexF t0 = x[0] + x[n/2];
        ComplexF t1 = x[0] - x[n/2];
        ComplexF t3 = _mm_unpacklo_ps(t0, t1);
        x[0] = half*t3;
    }

    if (n%4 == 0)
    {
        x[n/4] = change_sign_F(x[n/4], const1_F);
    }

    for (size_t i = 2; i < n/2; i+=2)
    {
        ComplexF sssc = load_F(packing_table + i);
        ComplexF x0_in = x[i/2];
        ComplexF x1_in = x[n/2 - i/2];

        if(!forward)
        {
            sssc = change_sign_F(sssc, const1_F);
        }

        ComplexF temp0 = x0_in + change_sign_F(x1_in, const2_F);
        ComplexF temp1 = mul_F(sssc, temp0);

        x[i/2] = temp1 + x0_in;
        x[n/2 - i/2] = change_sign_F(temp1, const2_F) + x1_in;
    }
}


////////////////////////////////////// Odd real FFT ////////////////////////////////////////////////////

// This function is used on the first level of odd real fft
// Note 128-bit type is not used
template<RadixType radix_type> void fft_1d_real_first_level_forward_sse2_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info)
{
    size_t repeats = step_info.repeats;
    uint32_t *reorder_table = step_info.reorder_table;
    const hhfft::RadersF &raders = *step_info.raders;
    size_t radix = get_actual_radix<radix_type>(raders);

    // Allocate memory for Rader's algorithm if needed
    float *data_raders = allocate_raders_F<radix_type>(raders);

    ComplexF x_temp_in[radix_type];
    ComplexF x_temp_out[radix_type];
    bool dir_out = true;
    for (size_t i = 0; i < repeats; i++)
    {
        // Initialize raders data with zeros
        init_coeff_F<radix_type>(data_raders, raders);

        // Copy input data taking reordering into account
        for (size_t j = 0; j < radix; j++)
        {
            size_t ind = reorder_table[i*radix + j];
            ComplexF x = load_real_F(data_in + ind);
            set_value_F<radix_type>(x_temp_in, data_raders, j, raders, x);
        }

        // Multiply with coefficients
        multiply_coeff_forward_F<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

        // Save only about half of the output
        // First/ last one is real
        if (dir_out) // direction normal
        {
            ComplexF x0 = get_value_F<radix_type>(x_temp_out, data_raders, 0, raders);
            store_real_F(x0, data_out + i*radix); // only real part

            for (size_t j = 1; j < radix/2 + 1; j++)
            {
                ComplexF x = get_value_F<radix_type>(x_temp_out, data_raders, j, raders);
                store_F(x, data_out + i*radix + 2*j - 1);
            }
        } else // direction inverted
        {
            ComplexF x0 = get_value_F<radix_type>(x_temp_out, data_raders, 0, raders);
            store_real_F(x0, data_out + i*radix + radix - 1); // only real part

            for (size_t j = 1; j < radix/2 + 1; j++)
            {
                ComplexF x = get_value_F<radix_type>(x_temp_out, data_raders, j, raders);
                store_F(x, data_out + i*radix + radix - 2*j - 1);
            }
        }
        dir_out = !dir_out;
    }

    // Free temporary memory
    free_raders_F<radix_type>(raders, data_raders);
}

inline size_t index_dir_stride_odd(size_t dir_in, size_t stride, size_t k)
{
    return dir_in*(4*k - stride) + stride - 2*k - 1;
}

template<RadixType radix_type> void fft_1d_real_one_level_forward_sse2_d_internal(const float *data_in, float *data_out, const float *twiddle_factors, float *data_raders, const hhfft::RadersF &raders, size_t stride, bool dir_out)
{
    size_t radix = get_actual_radix<radix_type>(raders);

    // The first/last value in each stride is real
    {
        // Initialize raders data with zeros
        init_coeff_F<radix_type>(data_raders, raders);

        ComplexF x0_temp_in[radix_type];
        ComplexF x0_temp_out[radix_type];

        // Read the inputs
        bool dir_in = dir_out;
        for (size_t j = 0; j < radix; j++)
        {
            ComplexF x0;
            if (dir_in)
            {
                x0 = load_real_F(data_in + j*stride);
            } else
            {
                x0 = load_real_F(data_in + j*stride + stride - 1);
            }
            set_value_F<radix_type>(x0_temp_in, data_raders, j, raders, x0);

            dir_in = !dir_in;
        }

        multiply_coeff_forward_F<radix_type>(x0_temp_in, x0_temp_out, data_raders, raders);

        // first/last output is real
        if (dir_out)
        {
            ComplexF x = get_value_F<radix_type>(x0_temp_out, data_raders, 0, raders);
            store_real_F(x, data_out);
        } else
        {
            ComplexF x = get_value_F<radix_type>(x0_temp_out, data_raders, 0, raders);
            store_real_F(x, data_out + radix*stride - 1);
        }
        // only about half is written
        for (size_t j = 1; j < (radix+1)/2; j++)
        {
            if (dir_out)
            {
                ComplexF x = get_value_F<radix_type>(x0_temp_out, data_raders, j, raders);
                store_F(x, data_out + 2*j*stride - 1);
            } else
            {
                ComplexF x = get_value_F<radix_type>(x0_temp_out, data_raders, (radix+1)/2 - j, raders);
                store_F(x, data_out + 2*j*stride - stride - 1);
            }
        }
    }

    size_t k = 1;
    // First use 128-bit variables
    for (; k + 1 < (stride+1)/2; k+=2)
    {
        // Initialize raders data with zeros
        init_coeff_F2<radix_type>(data_raders, raders);

        ComplexF2 x_temp_in[radix_type];
        ComplexF2 x_temp_out[radix_type];
        ComplexF2 twiddle_temp[radix_type];
        size_t dir_in = dir_out;

        // Copy the values and twiddle factors
        for (size_t j = 0; j < radix; j++)
        {
            ComplexF2 x0;
            if (dir_in)
            {
                x0 = load_F2(data_in + j*stride + 2*k - 1);
            } else
            {
                x0 = load_two_64_F2(data_in + j*stride + stride - 2*k - 1, data_in + j*stride + stride - 2*k - 3);
            }
            ComplexF2 w = load_F2(twiddle_factors + 2*k + 2*j*stride);
            set_value_twiddle_F2<radix_type>(x_temp_in, data_raders, twiddle_temp, j, raders, x0, w);
            dir_in = !dir_in;
        }

        // Multiply with coefficients
        multiply_twiddle_F2<radix_type,true>(x_temp_in, x_temp_in, twiddle_temp);
        multiply_coeff_forward_F2<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

        // save output taking the directions into account
        dir_in = dir_out;
        for (size_t j = 0; j < radix; j++)
        {
            // reverse the output order if required
            ComplexF2 x;
            if (dir_out)
            {
                x = get_value_real_odd_forward_F2<radix_type>(x_temp_out, data_raders, j, raders);
            } else
            {
                x = get_value_real_odd_forward_F2<radix_type>(x_temp_out, data_raders, radix - j - 1, raders);
            }

            if (dir_in)
            {
                store_F2(x, data_out + j*stride + 2*k - 1);
            } else
            {
                store_two_64_F2(x, data_out + j*stride + stride - 2*k - 1, data_out + j*stride + stride - 2*k - 3);
            }

            dir_in = !dir_in;
        }
    }

    // Then, if necessary, use 64-bit variables
    if (k < (stride+1)/2)
    {
        // Initialize raders data with zeros
        init_coeff_F<radix_type>(data_raders, raders);

        ComplexF x_temp_in[radix_type];
        ComplexF x_temp_out[radix_type];
        ComplexF twiddle_temp[radix_type];
        size_t dir_in = dir_out;

        // Copy the values and twiddle factors
        for (size_t j = 0; j < radix; j++)
        {
            size_t index2 = index_dir_stride_odd(dir_in, stride, k);

            ComplexF x = load_F(data_in + j*stride + index2);
            ComplexF w = load_F(twiddle_factors + 2*k + 2*j*stride);
            set_value_twiddle_F<radix_type>(x_temp_in, data_raders, twiddle_temp, j, raders, x, w);
            dir_in = !dir_in;
        }

        // Multiply with coefficients
        multiply_twiddle_F<radix_type,true>(x_temp_in, x_temp_in, twiddle_temp);
        multiply_coeff_forward_F<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

        // save output taking the directions into account
        dir_in = dir_out;
        for (size_t j = 0; j < radix; j++)
        {
            size_t index2 = index_dir_stride_odd(dir_in, stride, k);

            // reverse the output order if required
            if (dir_out)
            {
                ComplexF x = get_value_real_odd_forward_F<radix_type>(x_temp_out, data_raders, j, raders);
                store_F(x, data_out + j*stride + index2);
            } else
            {
                ComplexF x = get_value_real_odd_forward_F<radix_type>(x_temp_out, data_raders, radix - j - 1, raders);
                store_F(x, data_out + j*stride + index2);
            }

            dir_in = !dir_in;
        }
    }
}

// This function is used on rest of the odd real fft
template<RadixType radix_type> void fft_1d_real_one_level_forward_sse2_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info)
{
    size_t stride = step_info.stride;
    size_t repeats = step_info.repeats;
    float *twiddle_factors = step_info.twiddle_factors;
    const hhfft::RadersF &raders = *step_info.raders;
    size_t radix = get_actual_radix<radix_type>(raders);

    // Allocate memory for Rader's algorithm if needed
    float *data_raders;
    if (stride < 3)
        data_raders = allocate_raders_F<radix_type>(raders);
    else
        data_raders = allocate_raders_F2<radix_type>(raders);

    bool dir_out = true;
    for (size_t i = 0; i < repeats; i++)
    {
        fft_1d_real_one_level_forward_sse2_d_internal<radix_type>(data_in + i*radix*stride, data_out + i*radix*stride, twiddle_factors, data_raders, raders, stride, dir_out);

        dir_out = !dir_out;
    }

    // Free temporary memory
    free_raders_F<radix_type>(raders, data_raders);
}

// This function is used on rest of the odd real 2d fft
template<RadixType radix_type> void fft_2d_real_odd_rows_forward_sse2_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info)
{
    const hhfft::RadersF &raders = *step_info.raders;
    size_t radix = get_actual_radix<radix_type>(raders);
    size_t n = step_info.size;
    size_t stride = step_info.stride;
    size_t repeats = step_info.repeats;
    size_t m = stride*repeats*radix + 1;
    float *twiddle_factors = step_info.twiddle_factors;

    // Allocate memory for Rader's algorithm if needed
    float *data_raders;
    if (stride < 3)
        data_raders = allocate_raders_F<radix_type>(raders);
    else
        data_raders = allocate_raders_F2<radix_type>(raders);

    for (size_t j = 0; j < n; j++)
    {
        bool dir_out = true;
        for (size_t i = 0; i < repeats; i++)
        {
            fft_1d_real_one_level_forward_sse2_d_internal<radix_type>(data_in + j*m + i*radix*stride + 1, data_out + j*m + i*radix*stride + 1,
                                                                  twiddle_factors, data_raders, raders, stride, dir_out);

            dir_out = !dir_out;
        }
    }

    // Free temporary memory
    free_raders_F<radix_type>(raders, data_raders);
}

////////////////////////////////////// Odd real IFFT ////////////////////////////////////////////////////
// Note this does not use 128-bit variables
// This function is used on the first level of odd real ifft
template<RadixType radix_type> void fft_1d_real_first_level_inverse_sse2_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info)
{
    size_t repeats = step_info.repeats;
    uint32_t *reorder_table = step_info.reorder_table;
    const hhfft::RadersF &raders = *step_info.raders;
    size_t radix = get_actual_radix<radix_type>(raders);
    size_t n = radix * (2*step_info.repeats - 1);
    ComplexF norm_factor = broadcast32_F(step_info.norm_factor);

    // Allocate memory for Rader's algorithm if needed
    float *data_raders = allocate_raders_F<radix_type>(raders);

    ComplexF x_temp_in[radix_type];
    ComplexF x_temp_out[radix_type];

    // In the first repeat input is r, (r+i), (r+i) ... and output is r,r,r,r,r...
    {
        // Initialize raders data with zeros
        init_coeff_F<radix_type>(data_raders, raders);

        ComplexF x0 = norm_factor*load_real_F(data_in + 0);
        set_value_F<radix_type>(x_temp_in, data_raders, 0, raders, x0);

        // Read other inputs and conjugate them
        for (size_t j = 1; j <= radix/2; j++)
        {
            size_t ind = reorder_table[j];
            ComplexF x = norm_factor*load_F(data_in + 2*ind);

            set_value_F<radix_type>(x_temp_in, data_raders, j, raders, conj_F(x));
            set_value_F<radix_type>(x_temp_in, data_raders, radix-j, raders, x);
        }

        // Multiply with coefficients
        multiply_coeff_forward_F<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

        // Write only real parts of the data
        for (size_t j = 0; j < radix; j++)
        {
            ComplexF x = get_value_F<radix_type>(x_temp_out, data_raders, j, raders);
            store_real_F(x, data_out + j);
        }
    }

    // Other repeats are more usual, data ordering changes from r,i,r,i,r,i... to r,r,r...i,i,i...
    for (size_t i = 1; i < repeats; i++)
    {
        // Initialize raders data with zeros
        init_coeff_F<radix_type>(data_raders, raders);

        // Copy input data taking reordering into account
        for (size_t j = 0; j < radix; j++)
        {
            size_t ind = n - reorder_table[i*radix - radix/2 + j];
            if (ind <= n/2)
            {
                ComplexF x = norm_factor*load_F(data_in + 2*ind);
                set_value_F<radix_type>(x_temp_in, data_raders, j, raders, x);
            } else
            {
                // If input is from the lower part, it needs to be conjugated
                ind = n - ind;
                ComplexF x = conj_F(norm_factor*load_F(data_in + 2*ind));
                set_value_F<radix_type>(x_temp_in, data_raders, j, raders, x);
            }
        }

        // Multiply with coefficients
        multiply_coeff_forward_F<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

        // Store real and imag parts separately
        for (size_t j = 0; j < radix; j++)
        {
            ComplexF x = get_value_F<radix_type>(x_temp_out, data_raders, j, raders);
            store_F(x, data_out[2*i*radix - radix + j], data_out[2*i*radix + j]);
        }
    }

    // Free temporary memory
    free_raders_F<radix_type>(raders, data_raders);
}

// This function is used on first level of the odd real 2d ifft
// Note this does not use 128-bit variables
template<RadixType radix_type> void fft_2d_real_odd_rows_first_level_inverse_sse2_f(const float *data_in, float *data_out, const hhfft::StepInfo<float> &step_info)
{
    const hhfft::RadersF &raders = *step_info.raders;
    size_t radix = get_actual_radix<radix_type>(raders);
    size_t repeats = step_info.repeats;
    size_t m = radix * (2*step_info.repeats - 1);
    size_t n = step_info.size;

    // Allocate memory for Rader's algorithm if needed
    float *data_raders = allocate_raders_F<radix_type>(raders);

    ComplexF x_temp_in[radix_type];
    ComplexF x_temp_out[radix_type];

    // Loop over all rows
    for (size_t k = 0; k < n; k++)
    {
        // In the first repeat input is r, (r+i), (r+i) ... and output is r,r,r,r,r...
        {
            // Initialize raders data with zeros
            init_coeff_F<radix_type>(data_raders, raders);

            ComplexF x0 = load_real_F(data_in + k*m);
            set_value_F<radix_type>(x_temp_in, data_raders, 0, raders, x0);

            // Read other inputs and conjugate them
            for (size_t j = 1; j <= radix/2; j++)
            {
                ComplexF x = load_F(data_in + k*m + 2*j - 1);
                set_value_F<radix_type>(x_temp_in, data_raders, j, raders, x);
                set_value_F<radix_type>(x_temp_in, data_raders, radix-j, raders, conj_F(x));
            }

            // Multiply with coefficients
            multiply_coeff_forward_F<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

            // Write only real parts of the data
            for (size_t j = 0; j < radix; j++)
            {
                ComplexF x = get_value_F<radix_type>(x_temp_out, data_raders, j, raders);
                store_real_F(x, data_out + k*m + j);
            }
        }

        // Other repeats are more usual, data ordering changes from r,i,r,i,r,i... to r,r,r...i,i,i...
        for (size_t i = 1; i < repeats; i++)
        {
            // Initialize raders data with zeros
            init_coeff_F<radix_type>(data_raders, raders);

            // Copy input data
            for (size_t j = 0; j < radix; j++)
            {
                ComplexF x = load_F(data_in + k*m + 2*i*radix - radix + 2*j);
                set_value_F<radix_type>(x_temp_in, data_raders, j, raders, x);
            }

            // Multiply with coefficients
            multiply_coeff_forward_F<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

            // Store real and imag parts separately
            for (size_t j = 0; j < radix; j++)
            {
                ComplexF x = get_value_F<radix_type>(x_temp_out, data_raders, j, raders);
                store_F(x, data_out[k*m + 2*i*radix - radix + j], data_out[k*m + 2*i*radix + j]);
            }
        }
    }

    // Free temporary memory
    free_raders_F<radix_type>(raders, data_raders);
}


// This function is used on on rest of the odd real ifft
template<RadixType radix_type> inline void fft_1d_real_one_level_inverse_sse2_internal_f(const float *data_in, float *data_out, float *data_raders, const hhfft::RadersF &raders, const hhfft::StepInfo<float> &step_info)
{
    size_t repeats = step_info.repeats;
    size_t stride = step_info.stride;
    float *twiddle_factors = step_info.twiddle_factors;
    size_t radix = get_actual_radix<radix_type>(raders);

    // In the first repeat input is r,r,r,... r,r,r, ... i,i,i, ... and output is r,r,r,r,r...
    {        
        size_t k;
        for (k = 0; k + 1 < stride; k+=2)
        {
            // Initialize raders data with zeros
            init_coeff_F2<radix_type>(data_raders, raders);
            ComplexF2 x_temp_in[radix_type];
            ComplexF2 x_temp_out[radix_type];
            ComplexF2 twiddle_temp[radix_type];

            // Set first real value
            ComplexF2 x0 = load_real_F2(data_in + k);
            ComplexF2 w0 = load_F2(1,0,1,0);
            set_value_twiddle_F2<radix_type>(x_temp_in, data_raders, twiddle_temp, 0, raders, x0, w0);

            // Read other inputs, only about half of them is needed, conjugate other half
            for (size_t j = 1; j <= radix/2; j++)
            {
                ComplexF2 x = load_F2(data_in[2*j*stride - stride + k], data_in[2*j*stride + k],
                                       data_in[2*j*stride - stride + k + 1], data_in[2*j*stride + k + 1]);
                ComplexF2 w = load_F2(twiddle_factors + 2*j*stride + 2*k + 0);

                set_value_twiddle_F2<radix_type>(x_temp_in, data_raders, twiddle_temp, j, raders, x, w);
                set_value_twiddle_F2<radix_type>(x_temp_in, data_raders, twiddle_temp, radix - j, raders, conj_F2(x), conj_F2(w));
            }

            // Multiply with twiddle factors and coefficients
            multiply_twiddle_F2<radix_type,true>(x_temp_in, x_temp_in, twiddle_temp);
            multiply_coeff_forward_F2<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

            // Write only real parts of the data
            for (size_t j = 0; j < radix; j++)
            {
                ComplexF2 x = get_value_F2<radix_type>(x_temp_out, data_raders, j, raders);
                store_real_F2(x, data_out + j*stride + k);
            }
        }

        if (k < stride)
        {
            // Initialize raders data with zeros
            init_coeff_F<radix_type>(data_raders, raders);

            ComplexF x_temp_in[radix_type];
            ComplexF x_temp_out[radix_type];
            ComplexF twiddle_temp[radix_type];

            // Set first real value
            ComplexF x0 = load_real_F(data_in + k);
            ComplexF w0 = load_F(1,0);
            set_value_twiddle_F<radix_type>(x_temp_in, data_raders, twiddle_temp, 0, raders, x0, w0);

            // Read other inputs, only about half of them is needed, conjugate other half
            for (size_t j = 1; j <= radix/2; j++)
            {
                ComplexF x = load_F(data_in[2*j*stride - stride + k], data_in[2*j*stride + k]);
                ComplexF w = load_F(twiddle_factors + 2*j*stride + 2*k + 0);

                set_value_twiddle_F<radix_type>(x_temp_in, data_raders, twiddle_temp, j, raders, x, w);
                set_value_twiddle_F<radix_type>(x_temp_in, data_raders, twiddle_temp, radix - j, raders, conj_F(x), conj_F(w));
            }

            // Multiply with twiddle factors and coefficients
            multiply_twiddle_F<radix_type,true>(x_temp_in, x_temp_in, twiddle_temp);
            multiply_coeff_forward_F<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

            // Write only real parts of the data
            for (size_t j = 0; j < radix; j++)
            {
                ComplexF x = get_value_F<radix_type>(x_temp_out, data_raders, j, raders);
                store_real_F(x, data_out + j*stride + k);
            }
        }
    }

    // Other repeats are more usual, however both inputs and outputs have real and imag parts separated
    for (size_t i = 1; i < repeats; i++)
    {
        size_t k;
        // First use 128 bit variables        
        for (k = 0; k + 1 < stride; k+=2)
        {
            // Initialize raders data with zeros
            init_coeff_F2<radix_type>(data_raders, raders);

            ComplexF2 x_temp_in[radix_type];
            ComplexF2 x_temp_out[radix_type];
            ComplexF2 twiddle_temp[radix_type];

            // Read real and imag parts separately
            for (size_t j = 0; j < radix; j++)
            {
                size_t index = 2*i*stride*radix + 2*j*stride - stride*radix;
                ComplexF2 x = load_real_imag_F2(data_in + index + k, data_in + index + stride + k);
                ComplexF2 w = load_F2(twiddle_factors + 2*j*stride + 2*k);
                set_value_twiddle_F2<radix_type>(x_temp_in, data_raders, twiddle_temp, j, raders, x, w);
            }

            // Multiply with twiddle factors and coefficients
            multiply_twiddle_F2<radix_type,true>(x_temp_in, x_temp_in, twiddle_temp);
            multiply_coeff_forward_F2<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

            // Store real and imag parts separately
            for (size_t j = 0; j < radix; j++)
            {
                size_t index = 2*i*stride*radix + j*stride;
                ComplexF2 x = get_value_F2<radix_type>(x_temp_out, data_raders, j, raders);
                store_real_imag_F2(x, data_out + index - stride*radix + k , data_out + index + k);
            }
        }

        // Then 64 bit variables
        if (k < stride)
        {
            // Initialize raders data with zeros
            init_coeff_F<radix_type>(data_raders, raders);

            ComplexF x_temp_in[radix_type];
            ComplexF x_temp_out[radix_type];
            ComplexF twiddle_temp[radix_type];

            // Read real and imag parts separately
            for (size_t j = 0; j < radix; j++)
            {
                size_t index = 2*i*stride*radix + 2*j*stride - stride*radix;
                ComplexF x = load_F(data_in[index + k], data_in[index + stride + k]);
                ComplexF w = load_F(twiddle_factors + 2*j*stride + 2*k);

                set_value_twiddle_F<radix_type>(x_temp_in, data_raders, twiddle_temp, j, raders, x, w);
            }

            // Multiply with twiddle factors and coefficients
            multiply_twiddle_F<radix_type,true>(x_temp_in, x_temp_in, twiddle_temp);
            multiply_coeff_forward_F<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

            // Store real and imag parts separately
            for (size_t j = 0; j < radix; j++)
            {
                size_t index = 2*i*stride*radix + j*stride;
                ComplexF x = get_value_F<radix_type>(x_temp_out, data_raders, j, raders);
                store_F(x, data_out[index - stride*radix + k], data_out[index + k]);
            }
        }
    }
}

template<RadixType radix_type> void fft_1d_real_one_level_inverse_sse2_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info)
{
    const hhfft::RadersF &raders = *step_info.raders;

    // Allocate memory for Rader's algorithm if needed
    float *data_raders = allocate_raders_F2<radix_type>(raders);

    // Call the actual function
    fft_1d_real_one_level_inverse_sse2_internal_f<radix_type>(data_in, data_out, data_raders, raders, step_info);

    // Free temporary memory
    free_raders_F<radix_type>(raders, data_raders);
}

// This is used in 2d real for odd row sizes
template<RadixType radix_type> void fft_2d_real_odd_rows_inverse_sse2_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info)
{
    const hhfft::RadersF &raders = *step_info.raders;
    size_t radix = get_actual_radix<radix_type>(raders);
    size_t n = step_info.size;
    size_t stride = step_info.stride;
    size_t m = stride*radix*(2*step_info.repeats - 1);

    // Allocate memory for Rader's algorithm if needed
    float *data_raders = allocate_raders_F2<radix_type>(raders);

    // Process all rows separately
    for (size_t j = 0; j < n; j++)
    {
        fft_1d_real_one_level_inverse_sse2_internal_f<radix_type>(data_in + j*m, data_out + j*m, data_raders, raders, step_info);
    }

    // Free temporary memory
    free_raders_F<radix_type>(raders, data_raders);
}

/////////////////////////////////// Small sizes ///////////////////////////////////////

// fft for small sizes (2,3,4,5,6,7,8,10,14,16) where only one level is needed
template<size_t n, bool forward> void fft_1d_real_1level_sse2_f(const float *data_in, float *data_out, const hhfft::StepInfo<float> &)
{
    ComplexF k = broadcast32_F(float(2.0/n));

    if (n == 1)
    {
        data_out[0] = data_in[0];
        data_out[1] = 0;
    } else if (n%2 == 0)
    {
        // even n

        ComplexF x_temp_in[n/2+1];
        ComplexF x_temp_out[n/2+1];

        if (forward)
        {
            // Copy input data
            for (size_t i = 0; i < n/2; i++)
            {
                x_temp_in[i] = load_F(data_in + 2*i);
            }

            // Multiply with coefficients
            multiply_coeff_F<n/2,forward>(x_temp_in, x_temp_out);

            // Make the conversion
            fft_1d_complex_to_complex_packed_1level_sse2_f<forward,n>(x_temp_out);

            // Copy output data
            for (size_t i = 0; i < n/2 + 1; i++)
            {
                store_F(x_temp_out[i], data_out + 2*i);
            }
        } else
        {
            // Copy input data
            for (size_t i = 0; i < n/2 + 1; i++)
            {
                x_temp_in[i] = load_F(data_in + 2*i);
            }

            // Make the conversion
            fft_1d_complex_to_complex_packed_1level_sse2_f<forward,n>(x_temp_in);

            // Multiply with coefficients
            multiply_coeff_F<n/2,forward>(x_temp_in, x_temp_out);

            // Copy output data
            for (size_t i = 0; i < n/2; i++)
            {
                store_F(k*x_temp_out[i], data_out + 2*i);
            }
        }
    } else
    {
        // odd n
        if (forward)
        {
            ComplexF x_temp_in[n];
            ComplexF x_temp_out[n];

            // Copy real input data
            for (size_t j = 0; j < n; j++)
            {
                x_temp_in[j] = load_real_F(data_in + j);
            }

            // Multiply with coefficients
            multiply_coeff_F<n,true>(x_temp_in, x_temp_out);

            // Save only about half of the output
            for (size_t j = 0; j < n/2 + 1; j++)
            {
                store_F(x_temp_out[j], data_out + 2*j);
            }
        } else
        {
            ComplexF x_temp_in[n];
            ComplexF x_temp_out[n];

            ComplexF norm_factor = load_F(float(1.0/n), float(1.0/n));

            // First input is real
            x_temp_in[0] = norm_factor*load_real_F(data_in + 0);

            // Read other inputs and conjugate them
            for (size_t j = 1; j <= n/2; j++)
            {
                ComplexF x = norm_factor*load_F(data_in + 2*j);
                x_temp_in[j] = x;
                x_temp_in[n-j] = conj_F(x);
            }

            // Multiply with coefficients
            multiply_coeff_F<n,false>(x_temp_in, x_temp_out);

            // Write only real parts of the data
            for (size_t j = 0; j < n; j++)
            {
                store_real_F(x_temp_out[j], data_out + j);
            }
        }
    }
}

// For problems that need only one level Rader's
template<bool forward> void fft_1d_real_1level_raders_sse2_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info)
{
    size_t n = step_info.radix;
    ComplexF k = broadcast32_F(float(1.0/n));

    // Allocate memory for Rader's algorithm
    const hhfft::RadersF &raders = *step_info.raders;
    float *data_raders = allocate_raders_F<Raders>(raders);

    // Initialize raders data with zeros
    init_coeff_F<Raders>(data_raders, raders);

    if (forward)
    {
        // FFT

        for (size_t j = 0; j < n; j++)
        {
            ComplexF x = load_real_F(data_in + j);
            set_value_F<Raders>(nullptr, data_raders, j, raders, x);
        }

        multiply_coeff_forward_F<Raders>(nullptr, nullptr, data_raders, raders);

        for (size_t j = 0; j < (n+1)/2; j++)
        {
            ComplexF x = get_value_F<Raders>(nullptr, data_raders, j, raders);
            store_F(x, data_out + 2*j);
        }
    } else
    {
        // IFFT
        ComplexF x0 = load_real_F(data_in + 0);
        set_value_inverse_F<Raders>(nullptr, data_raders, 0, raders, x0);

        for (size_t j = 1; j < (n+1)/2; j++)
        {
            ComplexF x = load_F(data_in + 2*j);
            set_value_inverse_F<Raders>(nullptr, data_raders, j, raders, x);
            set_value_inverse_F<Raders>(nullptr, data_raders, n-j, raders, conj_F(x));
        }

        multiply_coeff_forward_F<Raders>(nullptr, nullptr, data_raders, raders);

        for (size_t j = 0; j < n; j++)
        {
            ComplexF x = k*get_value_F<Raders>(nullptr, data_raders, j, raders);
            store_real_F(x, data_out + j);
        }
    }

    // Free temporary memory
    free_raders_F<Raders>(raders, data_raders);
}

// Instantiations of the functions defined in this class
template void fft_1d_complex_to_complex_packed_sse2_f<false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_to_complex_packed_sse2_f<true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template void fft_1d_real_1level_sse2_f<1, false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_real_1level_sse2_f<2, false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_real_1level_sse2_f<3, false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_real_1level_sse2_f<4, false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_real_1level_sse2_f<5, false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_real_1level_sse2_f<6, false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_real_1level_sse2_f<7, false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_real_1level_sse2_f<8, false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_real_1level_sse2_f<10, false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_real_1level_sse2_f<12, false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_real_1level_sse2_f<14, false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_real_1level_sse2_f<16, false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_real_1level_sse2_f<1, true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_real_1level_sse2_f<2, true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_real_1level_sse2_f<3, true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_real_1level_sse2_f<4, true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_real_1level_sse2_f<5, true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_real_1level_sse2_f<6, true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_real_1level_sse2_f<7, true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_real_1level_sse2_f<8, true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_real_1level_sse2_f<10, true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_real_1level_sse2_f<12, true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_real_1level_sse2_f<14, true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_real_1level_sse2_f<16, true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template void fft_1d_real_first_level_forward_sse2_f<Raders>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_real_first_level_forward_sse2_f<Radix3>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_real_first_level_forward_sse2_f<Radix5>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_real_first_level_forward_sse2_f<Radix7>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template void fft_1d_real_first_level_inverse_sse2_f<Raders>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_real_first_level_inverse_sse2_f<Radix3>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_real_first_level_inverse_sse2_f<Radix5>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_real_first_level_inverse_sse2_f<Radix7>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template void fft_1d_real_one_level_forward_sse2_f<Raders>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_real_one_level_forward_sse2_f<Radix3>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_real_one_level_forward_sse2_f<Radix5>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_real_one_level_forward_sse2_f<Radix7>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template void fft_1d_real_one_level_inverse_sse2_f<Raders>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_real_one_level_inverse_sse2_f<Radix3>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_real_one_level_inverse_sse2_f<Radix5>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_real_one_level_inverse_sse2_f<Radix7>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template void fft_2d_real_odd_rows_forward_sse2_f<Raders>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_2d_real_odd_rows_forward_sse2_f<Radix3>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_2d_real_odd_rows_forward_sse2_f<Radix5>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_2d_real_odd_rows_forward_sse2_f<Radix7>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template void fft_2d_real_odd_rows_first_level_inverse_sse2_f<Raders>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_2d_real_odd_rows_first_level_inverse_sse2_f<Radix3>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_2d_real_odd_rows_first_level_inverse_sse2_f<Radix5>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_2d_real_odd_rows_first_level_inverse_sse2_f<Radix7>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template void fft_2d_real_odd_rows_inverse_sse2_f<Raders>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_2d_real_odd_rows_inverse_sse2_f<Radix3>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_2d_real_odd_rows_inverse_sse2_f<Radix5>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_2d_real_odd_rows_inverse_sse2_f<Radix7>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template void fft_1d_real_1level_raders_sse2_f<false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_real_1level_raders_sse2_f<true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
