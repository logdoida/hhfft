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

void fft_2d_real_reorder_rows_in_place_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
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

            ComplexD temp1 = load_D(data_in + 2*i*m + 2*ind1);
            ComplexD temp2 = load_D(data_in + 2*i*m + 2*ind2);
            store_D(temp1, data_out + 2*i*m + 2*ind2);
            store_D(temp2, data_out + 2*i*m + 2*ind1);
        }
    }
}

// Recalculate first column and add last column
void fft_2d_complex_to_complex_packed_first_column_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    //const double *packing_table = step_info.twiddle_factors;
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
            ComplexD2 x = load_D2(data_out + i2*m + j2);
            store_D2(x, data_out + i2*m2 + j2);
        }

        if (j < m)
        {
            size_t j2 = m - j - 2;
            ComplexD x = load_D(data_out + i2*m + j2);
            store_D(x, data_out + i2*m2 + j2);
        }
    }

    // Re-calculate first and last columns
    // First
    {
        double x_r = data_in[0];
        double x_i = data_in[1];
        data_out[0] = x_r + x_i;
        data_out[1] = 0.0;
        data_out[m + 0] = x_r - x_i;
        data_out[m + 1] = 0.0;
    }

    // Middle
    if (n%2 == 0)
    {
        double x_r = data_in[n/2*m2 + 0];
        double x_i = data_in[n/2*m2 + 1];
        data_out[n/2*m2 + 0] = x_r + x_i;
        data_out[n/2*m2 + 1] = 0.0;
        data_out[n/2*m2 + m + 0] = x_r - x_i;
        data_out[n/2*m2 + m + 1] = 0.0;
    }

    // Others in the first column
    for (size_t i = 1; i < (n+1)/2; i++)
    {
        size_t i2 = n - i;
        double x0_r = data_in[i*m2 + 0];
        double x0_i = data_in[i*m2 + 1];
        double x1_r = data_in[i2*m2 + 0];
        double x1_i = data_in[i2*m2 + 1];

        double t1 = 0.5*(x0_r + x1_r);
        double t2 = 0.5*(x0_r - x1_r);
        double t3 = 0.5*(x0_i + x1_i);
        double t4 = 0.5*(x0_i - x1_i);

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
    void fft_2d_complex_to_complex_packed_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{        
    const double *packing_table = step_info.twiddle_factors;
    size_t n = step_info.repeats; // n = number of original real rows
    size_t m = step_info.size;    // m = number of original real columns

    const ComplexD2 const1 = load_D2(0.0, -0.0, 0.0, -0.0);
    const ComplexD2 const2 = load_D2(-0.0, 0.0, -0.0, 0.0);

    if (m%4 == 0)
    {
        for (size_t i = 0; i < n; i++)
        {
            ComplexD x_in = load_D(data_in + i*m + m/2);
            ComplexD x_out = change_sign_D(x_in, const1_128);
            store_D(x_out, data_out + i*m + m/2);
        }
    }

    for (size_t i = 0; i < n; i++)
    {
        size_t j = 0;

        // First use avx
        for (j = 2; j + 2 < m/2; j+=4)
        {
            ComplexD2 sssc = load_D2(packing_table + j);
            if (!forward)
            {
                sssc = change_sign_D2(sssc, const1);
            }
            ComplexD2 x0_in = load_D2(data_out + i*m + j);
            ComplexD2 x1_in = load_two_128_D2(data_out + i*m + (m-j), data_out + i*m + (m-j) - 2);

            ComplexD2 temp0 = x0_in + change_sign_D2(x1_in, const2);
            ComplexD2 temp1 = mul_D2(sssc, temp0);

            ComplexD2 x0_out = temp1 + x0_in;
            ComplexD2 x1_out = change_sign_D2(temp1, const2) + x1_in;

            store_D2(x0_out, data_out + i*m + j);
            store_two_128_D2(x1_out, data_out + i*m + (m-j), data_out + i*m + (m-j) - 2);
        }

        // Then, if needed use sse2
        if (j < m/2)
        {
            ComplexD sssc = load_D(packing_table + j);
            if (!forward)
            {
                sssc = change_sign_D(sssc, const1_128);
            }
            ComplexD x0_in = load_D(data_out + i*m + j);
            ComplexD x1_in = load_D(data_out + i*m + (m-j));

            ComplexD temp0 = x0_in + change_sign_D(x1_in, const2_128);
            ComplexD temp1 = mul_D(sssc, temp0);

            ComplexD x0_out = temp1 + x0_in;
            ComplexD x1_out = change_sign_D(temp1, const2_128) + x1_in;

            store_D(x0_out, data_out + i*m + j);
            store_D(x1_out, data_out + i*m + (m-j));
        }
    }
}

template<size_t radix>
void fft_2d_real_reorder2_inverse_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    // In-place not supported
    assert (data_in != data_out);

    size_t m = step_info.stride;  // number of columns
    size_t m2 = m + 1;            // number of columns in input
    size_t repeats = step_info.repeats;
    size_t n = repeats*radix;
    uint32_t *reorder_table_columns = step_info.reorder_table;
    ComplexD norm_factor = broadcast64_D(step_info.norm_factor);
    ComplexD2 norm_factor_256 = broadcast64_D2(step_info.norm_factor);

    for (size_t i = 0; i < repeats; i++)
    {
        // The first column is calculated from first and last column
        // k = 0
        {
            ComplexD x_temp_in[radix];
            ComplexD x_temp_out[radix];

            // Copy input data (squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                size_t j1 = i*radix + j;
                size_t j2 = reorder_table_columns[j1];
                if (j1 > 0)
                {
                    j2 = n - j2;
                }

                double x0_r = data_in[2*j2*m2 + 0];
                double x0_i = data_in[2*j2*m2 + 1];
                double x1_r = data_in[2*j2*m2 + 2*m + 0];
                double x1_i = data_in[2*j2*m2 + 2*m + 1];

                double t1 = 0.5*(x0_r + x1_r);
                double t2 = 0.5*(x1_i - x0_i);
                double t3 = 0.5*(x0_r - x1_r);
                double t4 = 0.5*(x0_i + x1_i);

                x_temp_in[j] = norm_factor*load_D(t1 + t2, t3 + t4);
            }

            multiply_coeff_D<radix,true>(x_temp_in, x_temp_out);

            // Copy output data (un-squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                store_D(x_temp_out[j], data_out + 2*i*radix*m + 2*j*m);
            }
        }

        // Second column using SSE (Better alignment for AVX)
        size_t k = 1;
        if (k < m)
        {
            ComplexD x_temp_in[radix];
            ComplexD x_temp_out[radix];

            // Copy input data (squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                size_t j1 = i*radix + j;
                size_t j2 = reorder_table_columns[j1];

                if (j1 > 0)
                {
                    j2 = n - j2;
                }
                x_temp_in[j] = norm_factor*load_D(data_in + 2*j2*m2 + 2*k);
            }

            multiply_coeff_D<radix,true>(x_temp_in, x_temp_out);

            // Copy output data (un-squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                store_D(x_temp_out[j], data_out + 2*i*radix*m + 2*j*m + 2*k);
            }
        }

        // Then use 256-bit variables as many times as possible
        for (k = 2; k+1 < m; k+=2)
        {
            ComplexD2 x_temp_in[radix];
            ComplexD2 x_temp_out[radix];

            // Copy input data (squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                size_t j1 = i*radix + j;
                size_t j2 = reorder_table_columns[j1];
                if (j1 > 0)
                {
                    j2 = n - j2;
                }
                x_temp_in[j] = norm_factor_256*load_D2(data_in + 2*j2*m2 + 2*k);
            }

            multiply_coeff_D2<radix,true>(x_temp_in, x_temp_out);

            // Copy output data (un-squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                store_D2(x_temp_out[j], data_out + 2*i*radix*m + 2*j*m + 2*k);
            }
        }

        if (k < m)
        {
            ComplexD x_temp_in[radix];
            ComplexD x_temp_out[radix];

            // Copy input data (squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                size_t j1 = i*radix + j;
                size_t j2 = reorder_table_columns[j1];
                if (j1 > 0)
                {
                    j2 = n - j2;
                }
                x_temp_in[j] = norm_factor*load_D(data_in + 2*j2*m2 + 2*k);
            }

            multiply_coeff_D<radix,true>(x_temp_in, x_temp_out);

            // Copy output data (un-squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                store_D(x_temp_out[j], data_out + 2*i*radix*m + 2*j*m + 2*k);
            }
        }
    }
}

//////////////////////// Odd number of columns ////////////////////////////

// Combine reordering and first row wise FFT
// Note this does not use avx
template<size_t radix> void fft_2d_real_reorder2_odd_rows_forward_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    // Only out of place reordering supported
    assert(data_in != data_out);

    size_t n = step_info.size;
    size_t m = step_info.repeats * radix;  // row size in input
    size_t m2 = m+1; // row size in output
    size_t repeats = step_info.repeats;
    uint32_t *reorder_table_columns = step_info.reorder_table;
    uint32_t *reorder_table_rows = step_info.reorder_table2;

    // FFT and reordering
    for (size_t i = 0; i < n; i++)
    {
        size_t i2 = reorder_table_columns[i];

        // As the row size increases by one, set first value to be zero
        data_out[i*m2] = 0;

        bool dir_out = true;
        for (size_t j = 0; j < repeats; j++)
        {
            ComplexD x_temp_in[radix];
            ComplexD x_temp_out[radix];

            // Copy input data taking reordering into account
            for (size_t k = 0; k < radix; k++)
            {
                size_t j2 = reorder_table_rows[j*radix + k];

                x_temp_in[k] = load_real_D(data_in + i2*m + j2);
            }

            multiply_coeff_D<radix,true>(x_temp_in, x_temp_out);

            // Save only about half of the output
            // First/ last one is real
            if (dir_out) // direction normal
            {
                store_real_D(x_temp_out[0], data_out + i*m2 + j*radix + 1);

                for (size_t k = 1; k < radix/2 + 1; k++)
                {
                    store_D(x_temp_out[k], data_out + i*m2 + j*radix + 2*k);
                }
            } else // direction inverted
            {
                store_real_D(x_temp_out[0], data_out + i*m2 + j*radix + radix);

                for (size_t k = 1; k < radix/2 + 1; k++)
                {
                    store_D(x_temp_out[k], data_out + i*m2 + j*radix + radix - 2*k);
                }
            }
            dir_out = !dir_out;
        }
    }
}


// Calculates first ifft step for the first column and saves it to a temporary variable
template<size_t radix> void fft_2d_real_odd_rows_reorder_first_column_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    size_t m2 = 2*step_info.stride; // row size in input
    size_t repeats = step_info.repeats;
    uint32_t *reorder_table_columns = step_info.reorder_table;
    double k = step_info.norm_factor;

    // First use AVX
    size_t i = 0;
    for (; i + 1 < repeats; i+=2)
    {
        ComplexD2 x_temp_in[radix];
        ComplexD2 x_temp_out[radix];

        // Copy input data taking reordering and scaling into account
        for (size_t j = 0; j < radix; j++)
        {
            size_t i2 = reorder_table_columns[i*radix + j];
            size_t i3 = reorder_table_columns[(i+1)*radix + j];

            x_temp_in[j] = load_two_128_D2(data_in + i2*m2, data_in + i3*m2)*k;
        }

        multiply_coeff_D2<radix,false>(x_temp_in, x_temp_out);

        // save output
        for (size_t j = 0; j < radix; j++)
        {
            store_two_128_D2(x_temp_out[j], data_out + 2*(i*radix + j), data_out + 2*((i+1)*radix + j));
        }
    }

    // Then use sse2 if needed
    if (i < repeats)
    {
        ComplexD x_temp_in[radix];
        ComplexD x_temp_out[radix];

        // Copy input data taking reordering and scaling into account
        for (size_t j = 0; j < radix; j++)
        {
            size_t i2 = reorder_table_columns[i*radix + j];

            x_temp_in[j] = load_D(data_in + i2*m2)*k;
        }

        multiply_coeff_D<radix,false>(x_temp_in, x_temp_out);

        // save output
        for (size_t j = 0; j < radix; j++)
        {
            store_D(x_temp_out[j], data_out + 2*(i*radix + j));
        }
    }
}

// Reordering row- and columnwise, and first IFFT-step combined
template<size_t radix> void fft_2d_real_odd_rows_reorder_columns_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    size_t n = step_info.repeats * radix;  // number of rows
    size_t m2 = step_info.size; // row size in input
    size_t m = 2*m2 - 1;        // row size originally
    size_t repeats = step_info.repeats;
    uint32_t *reorder_table_columns = step_info.reorder_table;
    uint32_t *reorder_table_rows = step_info.reorder_table2;
    double norm_factor = step_info.norm_factor;

    for (size_t i = 0; i < repeats; i++)
    {
        // all columns, skip the first one as it is processed in temp variable

        // First use avx
        size_t j = 1;
        for (; j + 1 < m2; j+=2)
        {
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

            ComplexD2 x_temp_in[radix];
            ComplexD2 x_temp_out[radix];

            // Copy input data taking reordering and scaling into account
            for (size_t k = 0; k < radix; k++)
            {
                size_t i2 = reorder_table_columns[i*radix + k];
                size_t i3 = i2;

                // 0->0, 1->n-1, 2->n-2 ...
                if (i2 > 0 && conj1)
                {
                    i2 = n - i2;
                }
                if (i3 > 0 && conj2)
                {
                    i3 = n - i3;
                }

                ComplexD x1 = load_D(data_in + i2*2*m2 + 2*j2);
                ComplexD x2 = load_D(data_in + i3*2*m2 + 2*j3);

                if (conj1)
                {
                    x1 = conj_D(x1);
                }

                if (conj2)
                {
                    x2 = conj_D(x2);
                }

                x_temp_in[k] = combine_two_128_D2(x1,x2)*norm_factor;
            }

            multiply_coeff_D2<radix,false>(x_temp_in, x_temp_out);

            // save output. The row size is decreased to m2-1 so that all data fits
            for (size_t k = 0; k < radix; k++)
            {
                store_D2(x_temp_out[k], data_out + 2*(i*radix + k)*(m2-1) + 2*j - 2);
            }
        }

        // Then use sse2 if needed
        if (j < m2)
        {
            size_t j2 = m - reorder_table_rows[j];

            // For some of the columns the output should be conjugated
            // This is achieved by conjugating input and changing its order: conj(ifft(x)) = ifft(conj(swap(x))
            bool conj = false;
            if (j2 > m/2)
            {
                j2 = m - j2;
                conj = true;
            }

            ComplexD x_temp_in[radix];
            ComplexD x_temp_out[radix];

            // Copy input data taking reordering and scaling into account
            for (size_t k = 0; k < radix; k++)
            {
                size_t i2 = reorder_table_columns[i*radix + k];

                // 0->0, 1->n-1, 2->n-2 ...
                if (i2 > 0 && conj)
                {
                    i2 = n - i2;
                }

                ComplexD x = load_D(data_in + i2*2*m2 + 2*j2)*norm_factor;

                if (conj)
                {
                    x_temp_in[k] = conj_D(x);
                } else
                {
                    x_temp_in[k] = x;
                }
            }

            multiply_coeff_D<radix,false>(x_temp_in, x_temp_out);

            // save output. The row size is decreased to m2-1 so that all data fits
            for (size_t k = 0; k < radix; k++)
            {
                store_D(x_temp_out[k], data_out + 2*(i*radix + k)*(m2-1) + 2*j - 2);
            }
        }
    }
}

// Instantiations of the functions defined in this class
template void fft_2d_complex_to_complex_packed_avx_d<false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_to_complex_packed_avx_d<true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_2d_real_reorder2_inverse_avx_d<2>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_avx_d<3>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_avx_d<4>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_avx_d<5>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_avx_d<6>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_avx_d<7>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_avx_d<8>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_2d_real_reorder2_odd_rows_forward_avx_d<3>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_odd_rows_forward_avx_d<5>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_odd_rows_forward_avx_d<7>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_2d_real_odd_rows_reorder_first_column_avx_d<2>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_first_column_avx_d<3>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_first_column_avx_d<4>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_first_column_avx_d<5>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_first_column_avx_d<6>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_first_column_avx_d<7>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_first_column_avx_d<8>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_2d_real_odd_rows_reorder_columns_avx_d<2>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_columns_avx_d<3>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_columns_avx_d<4>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_columns_avx_d<5>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_columns_avx_d<6>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_columns_avx_d<7>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_columns_avx_d<8>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
