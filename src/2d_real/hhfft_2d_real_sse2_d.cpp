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

template<bool forward>
    void fft_2d_complex_to_complex_packed_sse2_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    const double *packing_table = step_info.twiddle_factors;
    size_t n = step_info.repeats; // n = number of original real rows
    size_t m = step_info.size;    // m = number of original real columns

    // In forward the input is n x (m/2), output is n x (m/2 + 1) complex numbers
    if (forward)
    {
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

        // First / Last column
        for (size_t i = 0; i < n; i++)
        {
            double x_r = data_in[i*m2 + 0];
            double x_i = data_in[i*m2 + 1];
            data_out[i*m2 + 0] = x_r + x_i;
            data_out[i*m2 + 1] = 0.0;
            data_out[i*m2 + m + 0] = x_r - x_i;
            data_out[i*m2 + m + 1] = 0.0;
        }

        if (m%4 == 0)
        {
            for (size_t i = 0; i < n; i++)
            {
                ComplexD x_in = load128(data_in + i*m2 + m/2);
                ComplexD x_out = change_sign(x_in, const1_128);
                store(x_out, data_out + i*m2 + m/2);
            }
        }

        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 2; j < m/2; j+=2)
            {
                ComplexD k = load128(packing_table + j);

                ComplexD x0_in = load128(data_in + i*m2 + j);
                ComplexD x1_in = load128(data_in + i*m2 + (m-j));

                ComplexD temp0 = x0_in + change_sign(x1_in, const2_128);
                ComplexD temp1 = mul(k, temp0);

                ComplexD x0_out = temp1 + x0_in;
                ComplexD x1_out = change_sign(temp1, const2_128) + x1_in;

                store(x0_out, data_out + i*m2 + j);
                store(x1_out, data_out + i*m2 + (m-j));
            }
        }

    } else
    {
        // In inverse the input is n x (m/2) and n x 1, output is n x (m/2)

        // First / Last column
        for (size_t i = 0; i < n; i++)
        {
            double x_r = data_out[i*m + 0];
            double x_i = data_in[2*i + 0];  // temp column
            data_out[i*m + 0] = 0.5*(x_r + x_i);
            data_out[i*m + 1] = 0.5*(x_r - x_i);
        }

        if (m%4 == 0)
        {
            for (size_t i = 0; i < n; i++)
            {
                ComplexD x_in = load128(data_out + i*m + m/2);
                ComplexD x_out = change_sign(x_in, const1_128);
                store(x_out, data_out + i*m + m/2);
            }
        }

        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 2; j < m/2; j+=2)
            {
                ComplexD k = load128(packing_table + j);
                k = change_sign(k, const1_128);

                ComplexD x0_in = load128(data_out + i*m + j);
                ComplexD x1_in = load128(data_out + i*m + (m-j));

                ComplexD temp0 = x0_in + change_sign(x1_in, const2_128);
                ComplexD temp1 = mul(k, temp0);

                ComplexD x0_out = temp1 + x0_in;
                ComplexD x1_out = change_sign(temp1, const2_128) + x1_in;

                store(x0_out, data_out + i*m + j);
                store(x1_out, data_out + i*m + (m-j));
            }
        }
    }
}

template<size_t radix>
void fft_2d_real_reorder2_inverse_sse2_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    // In-place not supported
    assert (data_in != data_out);

    size_t m = step_info.stride; // number of columns
    size_t m2 = m + 1;           // number of columns in input
    size_t repeats = step_info.repeats;
    uint32_t *reorder_table_columns = step_info.reorder_table;
    double norm_factor = step_info.norm_factor;

    ComplexD x_temp_in[radix];
    ComplexD x_temp_out[radix];

    for (size_t i = 0; i < repeats; i++)
    {
        for (size_t k = 0; k < m; k++)
        {
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

template<size_t radix>
void fft_2d_real_reorder_last_column_sse2_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    // In-place not supported
    assert (data_in != data_out);

    size_t m = step_info.stride; // number of columns
    size_t m2 = m + 1;           // number of columns in input
    size_t repeats = step_info.repeats;
    uint32_t *reorder_table_columns = step_info.reorder_table;
    double norm_factor = step_info.norm_factor;

    ComplexD x_temp_in[radix];
    ComplexD x_temp_out[radix];

    for (size_t i = 0; i < repeats; i++)
    {
        // Copy input data (squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            size_t j2 = reorder_table_columns[i*radix + j];
            x_temp_in[j] = norm_factor*load128(data_in + 2*j2*m2 + 2*m);
        }

        multiply_coeff<radix,false>(x_temp_in, x_temp_out);

        // Copy output data (un-squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            store(x_temp_out[j], data_out + 2*i*radix + 2*j);
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
template void fft_2d_real_reorder2_inverse_sse2_d<7>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_sse2_d<8>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_2d_real_reorder_last_column_sse2_d<2>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder_last_column_sse2_d<3>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder_last_column_sse2_d<4>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder_last_column_sse2_d<5>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder_last_column_sse2_d<7>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder_last_column_sse2_d<8>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
