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

#include "../common/hhfft_common_avx_f.h"
#include "../raders/raders_avx_f.h"

const ComplexF const2_F = load_F(-0.0f, 0.0f);

using namespace hhfft;

void fft_2d_real_reorder_rows_in_place_avx_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info)
{
    size_t n = step_info.stride; // number of rows
    size_t m = step_info.size; // number of columns
    size_t reorder_table_size = step_info.reorder_table_inplace_size;
    uint32_t *reorder_table = step_info.reorder_table_inplace;

    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < reorder_table_size; j++)
        {
            size_t ind1 = j + 1; // First one has been omitted!
            size_t ind2 = reorder_table[j];

            ComplexF temp1 = load_F(data_in + 2*i*m + 2*ind1);
            ComplexF temp2 = load_F(data_in + 2*i*m + 2*ind2);
            store_F(temp1, data_out + 2*i*m + 2*ind2);
            store_F(temp2, data_out + 2*i*m + 2*ind1);
        }
    }
}

// Recalculate first column and add last column
void fft_2d_complex_to_complex_packed_first_column_avx_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info)
{
    //const float *packing_table = step_info.twiddle_factors;
    size_t n = step_info.repeats; // n = number of original real rows
    size_t m = step_info.size;    // m = number of original real columns

    size_t m2 = m + 2;

    // First copy data so that row size increases from m/2 to m/2 + 1
    for (size_t i = 0; i < n; i++)
    {
        size_t i2 = n - i - 1;
        size_t j = 0;
        for (j = 0; j + 2 < m; j+=4)
        {
            size_t j2 = m - j - 4;
            ComplexF2 x = load_F2(data_out + i2*m + j2);
            store_F2(x, data_out + i2*m2 + j2);
        }

        if (j < m)
        {
            size_t j2 = m - j - 2;
            ComplexF x = load_F(data_out + i2*m + j2);
            store_F(x, data_out + i2*m2 + j2);
        }
    }

    // Re-calculate first and last columns
    // First
    {
        float x_r = data_in[0];
        float x_i = data_in[1];
        data_out[0] = x_r + x_i;
        data_out[1] = 0.0;
        data_out[m + 0] = x_r - x_i;
        data_out[m + 1] = 0.0;
    }

    // Middle
    if (n%2 == 0)
    {
        float x_r = data_in[n/2*m2 + 0];
        float x_i = data_in[n/2*m2 + 1];
        data_out[n/2*m2 + 0] = x_r + x_i;
        data_out[n/2*m2 + 1] = 0.0;
        data_out[n/2*m2 + m + 0] = x_r - x_i;
        data_out[n/2*m2 + m + 1] = 0.0;
    }

    // Others in the first column
    for (size_t i = 1; i < (n+1)/2; i++)
    {
        size_t i2 = n - i;
        float x0_r = data_in[i*m2 + 0];
        float x0_i = data_in[i*m2 + 1];
        float x1_r = data_in[i2*m2 + 0];
        float x1_i = data_in[i2*m2 + 1];

        float t1 = 0.5*(x0_r + x1_r);
        float t2 = 0.5*(x0_r - x1_r);
        float t3 = 0.5*(x0_i + x1_i);
        float t4 = 0.5*(x0_i - x1_i);

        data_out[i*m2 + 0] = t1 + t3;
        data_out[i*m2 + 1] = t4 - t2;
        data_out[i*m2 + m + 0] = t1 - t3;
        data_out[i*m2 + m + 1] = t2 + t4;
        data_out[i2*m2 + 0] = t1 + t3;
        data_out[i2*m2 + 1] = -(t4 - t2);
        data_out[i2*m2 + m + 0] = t1 - t3;
        data_out[i2*m2 + m + 1] = -(t2 + t4);
    }
}


template<bool forward>
    void fft_2d_complex_to_complex_packed_avx_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info)
{        
    const float *packing_table = step_info.twiddle_factors;
    size_t n = step_info.repeats; // n = number of original real rows
    size_t m = step_info.size;    // m = number of original real columns

    const ComplexF2 const1 = load_F2(0.0, -0.0, 0.0, -0.0);
    const ComplexF2 const2 = load_F2(-0.0, 0.0, -0.0, 0.0);

    if (m%4 == 0)
    {
        for (size_t i = 0; i < n; i++)
        {
            ComplexF x_in = load_F(data_in + i*m + m/2);
            ComplexF x_out = change_sign_F(x_in, const1_F);
            store_F(x_out, data_out + i*m + m/2);
        }
    }

    for (size_t i = 0; i < n; i++)
    {
        size_t j = 0;

        // First use avx
        for (j = 2; j + 2 < m/2; j+=4)
        {
            ComplexF2 sssc = load_F2(packing_table + j);
            if (!forward)
            {
                sssc = change_sign_F2(sssc, const1);
            }
            ComplexF2 x0_in = load_F2(data_out + i*m + j);
            ComplexF2 x1_in = load_two_64_F2(data_out + i*m + (m-j), data_out + i*m + (m-j) - 2);

            ComplexF2 temp0 = x0_in + change_sign_F2(x1_in, const2);
            ComplexF2 temp1 = mul_F2(sssc, temp0);

            ComplexF2 x0_out = temp1 + x0_in;
            ComplexF2 x1_out = change_sign_F2(temp1, const2) + x1_in;

            store_F2(x0_out, data_out + i*m + j);
            store_two_64_F2(x1_out, data_out + i*m + (m-j), data_out + i*m + (m-j) - 2);
        }

        // Then, if needed use sse2
        if (j < m/2)
        {
            ComplexF sssc = load_F(packing_table + j);
            if (!forward)
            {
                sssc = change_sign_F(sssc, const1_F);
            }
            ComplexF x0_in = load_F(data_out + i*m + j);
            ComplexF x1_in = load_F(data_out + i*m + (m-j));

            ComplexF temp0 = x0_in + change_sign_F(x1_in, const2_F);
            ComplexF temp1 = mul_F(sssc, temp0);

            ComplexF x0_out = temp1 + x0_in;
            ComplexF x1_out = change_sign_F(temp1, const2_F) + x1_in;

            store_F(x0_out, data_out + i*m + j);
            store_F(x1_out, data_out + i*m + (m-j));
        }
    }
}

template<RadixType radix_type>
void fft_2d_real_reorder2_inverse_avx_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info)
{    
    size_t m = step_info.stride;  // number of columns
    size_t m2 = m + 1;            // number of columns in input
    size_t repeats = step_info.repeats;    
    uint32_t *reorder_table_columns = step_info.reorder_table;
    size_t reorder_table_size = step_info.reorder_table_size;
    ComplexF norm_factor = broadcast32_F(step_info.norm_factor);
    ComplexF2 norm_factor_256 = broadcast32_F2(step_info.norm_factor);
    const hhfft::RadersF &raders = *step_info.raders;
    size_t radix = get_actual_radix<radix_type>(raders);

    // Allocate memory for Rader's algorithm if needed (m should always be > 1)
    float *data_raders = allocate_raders_F2<radix_type>(raders);

    for (size_t i = 0; i < repeats; i++)
    {
        // The first column is calculated from first and last column
        // k = 0
        {
            // Initialize raders data with zeros
            init_coeff_F<radix_type>(data_raders, raders);

            ComplexF x_temp_in[radix_type];
            ComplexF x_temp_out[radix_type];

            // Copy input data (squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                size_t j1 = i*radix + j;
                size_t j2 = reorder_table_columns[reorder_table_size - j1 - 1];

                float x0_r = data_in[2*j2*m2 + 0];
                float x0_i = data_in[2*j2*m2 + 1];
                float x1_r = data_in[2*j2*m2 + 2*m + 0];
                float x1_i = data_in[2*j2*m2 + 2*m + 1];

                float t1 = 0.5*(x0_r + x1_r);
                float t2 = 0.5*(x1_i - x0_i);
                float t3 = 0.5*(x0_r - x1_r);
                float t4 = 0.5*(x0_i + x1_i);

                ComplexF x = norm_factor*load_F(t1 + t2, t3 + t4);
                set_value_F<radix_type>(x_temp_in, data_raders, j, raders, x);
            }

            // Multiply with coefficients
            multiply_coeff_forward_F<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

            // Copy output data (un-squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                ComplexF x = get_value_F<radix_type>(x_temp_out, data_raders, j, raders);
                store_F(x, data_out + 2*i*radix*m + 2*j*m);
            }
        }

        // Second column using SSE (Better alignment for AVX)
        size_t k = 1;
        if (k < m)
        {
            // Initialize raders data with zeros
            init_coeff_F<radix_type>(data_raders, raders);

            ComplexF x_temp_in[radix_type];
            ComplexF x_temp_out[radix_type];

            // Copy input data (squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                size_t j1 = i*radix + j;
                size_t j2 = reorder_table_columns[reorder_table_size - j1 - 1];
                ComplexF x = norm_factor*load_F(data_in + 2*j2*m2 + 2*k);
                set_value_F<radix_type>(x_temp_in, data_raders, j, raders, x);
            }

            // Multiply with coefficients
            multiply_coeff_forward_F<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

            // Copy output data (un-squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                ComplexF x = get_value_F<radix_type>(x_temp_out, data_raders, j, raders);
                store_F(x, data_out + 2*i*radix*m + 2*j*m + 2*k);
            }
        }

        // Then use 256-bit variables as many times as possible
        for (k = 2; k+1 < m; k+=2)
        {
            // Initialize raders data with zeros
            init_coeff_F2<radix_type>(data_raders, raders);

            ComplexF2 x_temp_in[radix_type];
            ComplexF2 x_temp_out[radix_type];

            // Copy input data (squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                size_t j1 = i*radix + j;
                size_t j2 = reorder_table_columns[reorder_table_size - j1 - 1];
                ComplexF2 x = norm_factor_256*load_F2(data_in + 2*j2*m2 + 2*k);
                set_value_F2<radix_type>(x_temp_in, data_raders, j, raders, x);
            }

            // Multiply with coefficients
            multiply_coeff_forward_F2<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

            // Copy output data (un-squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                ComplexF2 x = get_value_F2<radix_type>(x_temp_out, data_raders, j, raders);
                store_F2(x, data_out + 2*i*radix*m + 2*j*m + 2*k);
            }
        }

        if (k < m)
        {
            // Initialize raders data with zeros
            init_coeff_F<radix_type>(data_raders, raders);

            ComplexF x_temp_in[radix_type];
            ComplexF x_temp_out[radix_type];

            // Copy input data (squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                size_t j1 = i*radix + j;                
                size_t j2 = reorder_table_columns[reorder_table_size - j1 - 1];
                ComplexF x = norm_factor*load_F(data_in + 2*j2*m2 + 2*k);
                set_value_F<radix_type>(x_temp_in, data_raders, j, raders, x);
            }

            // Multiply with coefficients
            multiply_coeff_forward_F<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

            // Copy output data (un-squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                ComplexF x = get_value_F<radix_type>(x_temp_out, data_raders, j, raders);
                store_F(x, data_out + 2*i*radix*m + 2*j*m + 2*k);
            }
        }
    }

    // Free temporary memory
    free_raders_F2<radix_type>(raders, data_raders);
}

//////////////////////// Odd number of columns ////////////////////////////

// Combine reordering and first row wise FFT
// Note this does not use avx
template<RadixType radix_type> void fft_2d_real_reorder2_odd_rows_forward_avx_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info)
{    
    const hhfft::RadersF &raders = *step_info.raders;
    size_t radix = get_actual_radix<radix_type>(raders);
    size_t n = step_info.size;
    size_t m = step_info.repeats * radix;  // row size in input
    size_t m2 = m+1; // row size in output
    size_t repeats = step_info.repeats;
    uint32_t *reorder_table_columns = step_info.reorder_table;
    uint32_t *reorder_table_rows = step_info.reorder_table2;

    // Allocate memory for Rader's algorithm if needed
    float *data_raders = allocate_raders_F<radix_type>(raders);

    // FFT and reordering
    for (size_t i = 0; i < n; i++)
    {
        size_t i2 = reorder_table_columns[i];

        // As the row size increases by one, set first value to be zero
        data_out[i*m2] = 0;

        bool dir_out = true;
        for (size_t j = 0; j < repeats; j++)
        {
            // Initialize raders data with zeros
            init_coeff_F<radix_type>(data_raders, raders);

            ComplexF x_temp_in[radix_type];
            ComplexF x_temp_out[radix_type];

            // Copy input data taking reordering into account
            for (size_t k = 0; k < radix; k++)
            {
                size_t j2 = reorder_table_rows[j*radix + k];

                ComplexF x = load_real_F(data_in + i2*m + j2);
                set_value_F<radix_type>(x_temp_in, data_raders, k, raders, x);
            }

            // Multiply with coefficients
            multiply_coeff_forward_F<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

            // Save only about half of the output
            // First/ last one is real
            if (dir_out) // direction normal
            {
                ComplexF x = get_value_F<radix_type>(x_temp_out, data_raders, 0, raders);
                store_real_F(x, data_out + i*m2 + j*radix + 1);

                for (size_t k = 1; k < radix/2 + 1; k++)
                {
                    ComplexF x = get_value_F<radix_type>(x_temp_out, data_raders, k, raders);
                    store_F(x, data_out + i*m2 + j*radix + 2*k);
                }
            } else // direction inverted
            {
                ComplexF x = get_value_F<radix_type>(x_temp_out, data_raders, 0, raders);
                store_real_F(x, data_out + i*m2 + j*radix + radix);

                for (size_t k = 1; k < radix/2 + 1; k++)
                {
                    ComplexF x = get_value_F<radix_type>(x_temp_out, data_raders, k, raders);
                    store_F(x, data_out + i*m2 + j*radix + radix - 2*k);
                }
            }
            dir_out = !dir_out;
        }
    }

    // Free temporary memory
    free_raders_F<radix_type>(raders, data_raders);
}


// Calculates first ifft step for the first column and saves it to a temporary variable
template<RadixType radix_type> void fft_2d_real_odd_rows_reorder_first_column_avx_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info)
{
    size_t m2 = 2*step_info.stride; // row size in input
    size_t repeats = step_info.repeats;
    uint32_t *reorder_table_columns = step_info.reorder_table;
    size_t reorder_table_columns_size = step_info.reorder_table_size;
    float k = step_info.norm_factor;
    const hhfft::RadersF &raders = *step_info.raders;
    size_t radix = get_actual_radix<radix_type>(raders);

    // Allocate memory for Rader's algorithm if needed
    float *data_raders;
    if (repeats == 1)
        data_raders = allocate_raders_F<radix_type>(raders);
    else
        data_raders = allocate_raders_F2<radix_type>(raders);

    // First use AVX
    size_t i = 0;
    for (; i + 1 < repeats; i+=2)
    {
        // Initialize raders data with zeros
        init_coeff_F2<radix_type>(data_raders, raders);

        ComplexF2 x_temp_in[radix_type];
        ComplexF2 x_temp_out[radix_type];

        // Copy input data taking reordering and scaling into account
        for (size_t j = 0; j < radix; j++)
        {
            size_t i1 = i*radix + j;
            size_t i2 = reorder_table_columns[reorder_table_columns_size - i1 - 1];
            size_t i3 = reorder_table_columns[reorder_table_columns_size - i1 - radix - 1];

            ComplexF2 x = load_two_64_F2(data_in + i2*m2, data_in + i3*m2)*k;
            set_value_F2<radix_type>(x_temp_in, data_raders, j, raders, x);
        }

        // Multiply with coefficients
        multiply_coeff_forward_F2<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

        // save output
        for (size_t j = 0; j < radix; j++)
        {
            ComplexF2 x = get_value_F2<radix_type>(x_temp_out, data_raders, j, raders);
            store_two_64_F2(x, data_out + 2*(i*radix + j), data_out + 2*((i+1)*radix + j));
        }
    }

    // Then use sse2 if needed
    if (i < repeats)
    {
        // Initialize raders data with zeros
        init_coeff_F<radix_type>(data_raders, raders);

        ComplexF x_temp_in[radix_type];
        ComplexF x_temp_out[radix_type];

        // Copy input data taking reordering and scaling into account
        for (size_t j = 0; j < radix; j++)
        {
            size_t i1 = i*radix + j;
            size_t i2 = reorder_table_columns[reorder_table_columns_size - i1 - 1];
            ComplexF x = load_F(data_in + i2*m2)*k;
            set_value_F<radix_type>(x_temp_in, data_raders, j, raders, x);
        }

        // Multiply with coefficients
        multiply_coeff_forward_F<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

        // save output
        for (size_t j = 0; j < radix; j++)
        {
            ComplexF x = get_value_F<radix_type>(x_temp_out, data_raders, j, raders);
            store_F(x, data_out + 2*(i*radix + j));
        }
    }

    // Free temporary memory
    free_raders_F<radix_type>(raders, data_raders);
}

// Reordering row- and columnwise, and first IFFT-step combined
template<RadixType radix_type> void fft_2d_real_odd_rows_reorder_columns_avx_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info)
{
    size_t m2 = step_info.size; // row size in input
    size_t m = 2*m2 - 1;        // row size originally
    size_t repeats = step_info.repeats;
    uint32_t *reorder_table_columns = step_info.reorder_table;
    uint32_t *reorder_table_rows = step_info.reorder_table2;
    size_t reorder_table_columns_size = step_info.reorder_table_size;
    float norm_factor = step_info.norm_factor;
    const hhfft::RadersF &raders = *step_info.raders;
    size_t radix = get_actual_radix<radix_type>(raders);

    // Allocate memory for Rader's algorithm if needed
    float *data_raders = allocate_raders_F2<radix_type>(raders);

    for (size_t i = 0; i < repeats; i++)
    {
        // all columns, skip the first one as it is processed in temp variable

        // First use avx
        size_t j = 1;
        for (; j + 1 < m2; j+=2)
        {
            // Initialize raders data with zeros
            init_coeff_F2<radix_type>(data_raders, raders);

            size_t j2 = m - reorder_table_rows[j];
            size_t j3 = m - reorder_table_rows[j+1];

            // For some of the columns the output should be conjugated
            // This is achieved by conjugating input and changing its order: conj(ifft(x)) = ifft(conj(swap(x))
            bool conj1 = false, conj2 = false;
            if (j2 > m/2)
            {
                j2 = m - j2;
                conj1 = true;
            }
            if (j3 > m/2)
            {
                j3 = m - j3;
                conj2 = true;
            }

            ComplexF2 x_temp_in[radix_type];
            ComplexF2 x_temp_out[radix_type];

            // Copy input data taking reordering and scaling into account
            for (size_t k = 0; k < radix; k++)
            {
                size_t i2, i3;
                if (conj1)
                {
                    i2 = reorder_table_columns[i*radix + k];
                } else
                {
                    i2 = reorder_table_columns[reorder_table_columns_size - i*radix - k - 1];
                }
                if (conj2)
                {
                    i3 = reorder_table_columns[i*radix + k];
                } else
                {
                    i3 = reorder_table_columns[reorder_table_columns_size - i*radix - k - 1];
                }

                ComplexF x1 = load_F(data_in + i2*2*m2 + 2*j2);
                ComplexF x2 = load_F(data_in + i3*2*m2 + 2*j3);

                if (conj1)
                {
                    x1 = conj_F(x1);
                }

                if (conj2)
                {
                    x2 = conj_F(x2);
                }

                ComplexF2 x = combine_two_64_F2(x1,x2)*norm_factor;
                set_value_F2<radix_type>(x_temp_in, data_raders, k, raders, x);
            }

            // Multiply with coefficients
            multiply_coeff_forward_F2<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

            // save output. The row size is decreased to m2-1 so that all data fits
            for (size_t k = 0; k < radix; k++)
            {
                ComplexF2 x = get_value_F2<radix_type>(x_temp_out, data_raders, k, raders);
                store_F2(x, data_out + 2*(i*radix + k)*(m2-1) + 2*j - 2);
            }
        }

        // Then use sse2 if needed
        if (j < m2)
        {
            // Initialize raders data with zeros
            init_coeff_F<radix_type>(data_raders, raders);

            size_t j2 = m - reorder_table_rows[j];

            // For some of the columns the output should be conjugated
            // This is achieved by conjugating input and changing its order: conj(ifft(x)) = ifft(conj(swap(x))
            // Also ifft(x) = fft(swap(x))!
            bool conj = false;
            if (j2 > m/2)
            {
                j2 = m - j2;
                conj = true;
            }

            ComplexF x_temp_in[radix_type];
            ComplexF x_temp_out[radix_type];

            // Copy input data taking reordering and scaling into account
            for (size_t k = 0; k < radix; k++)
            {
                size_t i2;
                if (conj)
                {
                    i2 = reorder_table_columns[i*radix + k];
                } else
                {
                    i2 = reorder_table_columns[reorder_table_columns_size - i*radix - k - 1];
                }

                ComplexF x = load_F(data_in + i2*2*m2 + 2*j2)*norm_factor;

                if (conj)
                {
                    set_value_F<radix_type>(x_temp_in, data_raders, k, raders, conj_F(x));
                } else
                {
                    set_value_F<radix_type>(x_temp_in, data_raders, k, raders, x);
                }
            }

            // Multiply with coefficients
            multiply_coeff_forward_F<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

            // save output. The row size is decreased to m2-1 so that all data fits
            for (size_t k = 0; k < radix; k++)
            {
                ComplexF x = get_value_F<radix_type>(x_temp_out, data_raders, k, raders);
                store_F(x, data_out + 2*(i*radix + k)*(m2-1) + 2*j - 2);
            }
        }
    }

    // Free temporary memory
    free_raders_F<radix_type>(raders, data_raders);
}

// Instantiations of the functions defined in this class
template void fft_2d_complex_to_complex_packed_avx_f<false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_2d_complex_to_complex_packed_avx_f<true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template void fft_2d_real_reorder2_inverse_avx_f<Raders>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_2d_real_reorder2_inverse_avx_f<Radix2>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_2d_real_reorder2_inverse_avx_f<Radix3>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_2d_real_reorder2_inverse_avx_f<Radix4>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_2d_real_reorder2_inverse_avx_f<Radix5>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_2d_real_reorder2_inverse_avx_f<Radix6>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_2d_real_reorder2_inverse_avx_f<Radix7>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_2d_real_reorder2_inverse_avx_f<Radix8>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template void fft_2d_real_reorder2_odd_rows_forward_avx_f<Raders>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_2d_real_reorder2_odd_rows_forward_avx_f<Radix3>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_2d_real_reorder2_odd_rows_forward_avx_f<Radix5>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_2d_real_reorder2_odd_rows_forward_avx_f<Radix7>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template void fft_2d_real_odd_rows_reorder_first_column_avx_f<Raders>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_2d_real_odd_rows_reorder_first_column_avx_f<Radix2>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_2d_real_odd_rows_reorder_first_column_avx_f<Radix3>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_2d_real_odd_rows_reorder_first_column_avx_f<Radix4>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_2d_real_odd_rows_reorder_first_column_avx_f<Radix5>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_2d_real_odd_rows_reorder_first_column_avx_f<Radix6>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_2d_real_odd_rows_reorder_first_column_avx_f<Radix7>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_2d_real_odd_rows_reorder_first_column_avx_f<Radix8>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template void fft_2d_real_odd_rows_reorder_columns_avx_f<Raders>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_2d_real_odd_rows_reorder_columns_avx_f<Radix2>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_2d_real_odd_rows_reorder_columns_avx_f<Radix3>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_2d_real_odd_rows_reorder_columns_avx_f<Radix4>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_2d_real_odd_rows_reorder_columns_avx_f<Radix5>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_2d_real_odd_rows_reorder_columns_avx_f<Radix6>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_2d_real_odd_rows_reorder_columns_avx_f<Radix7>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_2d_real_odd_rows_reorder_columns_avx_f<Radix8>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
