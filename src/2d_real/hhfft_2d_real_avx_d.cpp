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
            ComplexD2 x = load(data_out + i2*m + j2);
            store(x, data_out + i2*m2 + j2);
        }

        if (j < m)
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

    const ComplexD2 const1 = load(0.0, -0.0, 0.0, -0.0);
    const ComplexD2 const2 = load(-0.0, 0.0, -0.0, 0.0);

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
        size_t j = 0;

        // First use avx
        for (j = 2; j + 2 < m/2; j+=4)
        {
            ComplexD2 sssc = load(packing_table + j);
            if (!forward)
            {
                sssc = change_sign(sssc, const1);
            }
            ComplexD2 x0_in = load(data_out + i*m + j);
            ComplexD2 x1_in = load_two_128(data_out + i*m + (m-j), data_out + i*m + (m-j) - 2);

            ComplexD2 temp0 = x0_in + change_sign(x1_in, const2);
            ComplexD2 temp1 = mul(sssc, temp0);

            ComplexD2 x0_out = temp1 + x0_in;
            ComplexD2 x1_out = change_sign(temp1, const2) + x1_in;

            store(x0_out, data_out + i*m + j);
            store_two_128(x1_out, data_out + i*m + (m-j), data_out + i*m + (m-j) - 2);
        }

        // Then, if needed use sse2
        if (j < m/2)
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
void fft_2d_real_reorder2_inverse_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    // In-place not supported
    assert (data_in != data_out);

    size_t m = step_info.stride;  // number of columns
    size_t m2 = m + 1;            // number of columns in input
    size_t repeats = step_info.repeats;
    uint32_t *reorder_table_columns = step_info.reorder_table;
    ComplexD norm_factor = broadcast64(step_info.norm_factor);
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

        // Second column using SSE (Better alignment for AVX)
        size_t k = 1;
        if (k < m)
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

        // Then use 256-bit variables as many times as possible
        for (k = 2; k+1 < m; k+=2)
        {
            ComplexD2 x_temp_in[radix];
            ComplexD2 x_temp_out[radix];

            // Copy input data (squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                size_t j2 = reorder_table_columns[i*radix + j];
                x_temp_in[j] = norm_factor_256*load(data_in + 2*j2*m2 + 2*k);
            }

            multiply_coeff<radix,false>(x_temp_in, x_temp_out);

            // Copy output data (un-squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                store(x_temp_out[j], data_out + 2*i*radix*m + 2*j*m + 2*k);
            }
        }

        if (k < m)
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
template void fft_2d_complex_to_complex_packed_avx_d<false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_to_complex_packed_avx_d<true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_2d_real_reorder2_inverse_avx_d<2>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_avx_d<3>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_avx_d<4>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_avx_d<5>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_avx_d<7>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_avx_d<8>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
