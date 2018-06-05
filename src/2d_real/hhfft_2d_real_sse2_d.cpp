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

#include "../common/hhfft_1d_complex_sse2_common_d.h"

using namespace hhfft;

void fft_2d_real_reorder_rows_in_place_sse2_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
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

            ComplexD temp1 = load128(data_in + 2*i*m + 2*ind1);
            ComplexD temp2 = load128(data_in + 2*i*m + 2*ind2);
            store(temp1, data_out + 2*i*m + 2*ind2);
            store(temp2, data_out + 2*i*m + 2*ind1);
        }
    }
}

// Recalculate first column and add last column
void fft_2d_complex_to_complex_packed_first_column_sse2_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    //const double *packing_table = step_info.twiddle_factors;
    size_t n = step_info.repeats; // n = number of original real rows
    size_t m = step_info.size;    // m = number of original real columns

    size_t m2 = m + 2;

    // First copy data so that row size increases from m/2 to m/2 + 1
    for (size_t i = 0; i < n; i++)
    {
        size_t i2 = n - i - 1;

        for (size_t j = 0; j < m; j+=2)
        {
            size_t j2 = m - j - 2;
            ComplexD x = load128(data_out + i2*m + j2);
            store(x, data_out + i2*m2 + j2);
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

    // Others
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
    void fft_2d_complex_to_complex_packed_sse2_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    const double *packing_table = step_info.twiddle_factors;
    size_t n = step_info.repeats; // n = number of original real rows
    size_t m = step_info.size;    // m = number of original real columns

    if (m%4 == 0)
    {
        for (size_t i = 0; i < n; i++)
        {
            ComplexD x_in = load128(data_in + i*m + m/2);
            ComplexD x_out = change_sign(x_in, const1_128);
            store(x_out, data_out + i*m + m/2);
        }
    }

    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 2; j < m/2; j+=2)
        {
            ComplexD sssc = load128(packing_table + j);
            if (!forward)
            {
                sssc = change_sign(sssc, const1_128);
            }

            ComplexD x0_in = load128(data_out + i*m + j);
            ComplexD x1_in = load128(data_out + i*m + (m-j));

            ComplexD temp0 = x0_in + change_sign(x1_in, const2_128);
            ComplexD temp1 = mul(sssc, temp0);

            ComplexD x0_out = temp1 + x0_in;
            ComplexD x1_out = change_sign(temp1, const2_128) + x1_in;

            store(x0_out, data_out + i*m + j);
            store(x1_out, data_out + i*m + (m-j));
        }
    }
}

template<size_t radix>
void fft_2d_real_reorder2_inverse_sse2_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    // In-place not supported
    assert (data_in != data_out);

    size_t m = step_info.stride;  // number of columns
    size_t m2 = m + 1;            // number of columns in input
    size_t repeats = step_info.repeats;
    uint32_t *reorder_table_columns = step_info.reorder_table;
    ComplexD norm_factor = broadcast64(step_info.norm_factor);

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
                size_t j2 = reorder_table_columns[i*radix + j];
                double x0_r = data_in[2*j2*m2 + 0];
                double x0_i = data_in[2*j2*m2 + 1];
                double x1_r = data_in[2*j2*m2 + 2*m + 0];
                double x1_i = data_in[2*j2*m2 + 2*m + 1];

                double t1 = 0.5*(x0_r + x1_r);
                double t2 = 0.5*(x1_i - x0_i);
                double t3 = 0.5*(x0_r - x1_r);
                double t4 = 0.5*(x0_i + x1_i);

                x_temp_in[j] = norm_factor*load(t1 + t2, t3 + t4);
            }

            multiply_coeff<radix,false>(x_temp_in, x_temp_out);

            // Copy output data (un-squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                store(x_temp_out[j], data_out + 2*i*radix*m + 2*j*m);
            }
        }

        for (size_t k = 1; k < m; k++)
        {
            ComplexD x_temp_in[radix];
            ComplexD x_temp_out[radix];

            // Copy input data (squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                size_t j2 = reorder_table_columns[i*radix + j];
                x_temp_in[j] = norm_factor*load128(data_in + 2*j2*m2 + 2*k);
            }

            multiply_coeff<radix,false>(x_temp_in, x_temp_out);

            // Copy output data (un-squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                store(x_temp_out[j], data_out + 2*i*radix*m + 2*j*m + 2*k);
            }
        }
    }
}

// Instantiations of the functions defined in this class
template void fft_2d_complex_to_complex_packed_sse2_d<false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_to_complex_packed_sse2_d<true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_2d_real_reorder2_inverse_sse2_d<2>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_sse2_d<3>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_sse2_d<4>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_sse2_d<5>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_sse2_d<6>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_sse2_d<7>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_sse2_d<8>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
