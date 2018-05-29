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

/*
template<bool forward>
    void fft_2d_complex_to_complex_packed_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    const double *packing_table = step_info.twiddle_factors;
    size_t n = step_info.repeats; // n = number of original real rows
    size_t m = step_info.size;    // m = number of original real columns

    const ComplexD2 const1 = load(0.0, -0.0, 0.0, -0.0);
    const ComplexD2 const2 = load(-0.0, 0.0, -0.0, 0.0);

    if (forward)
    {
        for (size_t j = 0; j < 2*m; j+=2)
        {
            double x_r = data_in[j + 0];
            double x_i = data_in[j + 1];
            data_out[j + 0] = x_r + x_i;
            data_out[j + 1] = 0.0;
            data_out[j + n*m + 0] = x_r - x_i;
            data_out[j + n*m + 1] = 0.0;
        }
    } else
    {
        // For IFFT: data_in = last row (temporary variable), data_out is actually both input and output!
        for (size_t j = 0; j < 2*m; j+=2)
        {
            double x_r = data_out[j];
            double x_i = data_in[j];
            data_out[j + 0] = 0.5*(x_r + x_i);
            data_out[j + 1] = 0.5*(x_r - x_i);
        }

        // !
        data_in = data_out;
    }

    if (n%4 == 0)
    {
        for (size_t j = 0; j < 2*m; j+=2)
        {
            ComplexD x_in = load128(data_in + j + n*m/2);
            ComplexD x_out = change_sign(x_in, const1_128);
            store(x_out, data_out + j + n*m/2);
        }
    }

    for (size_t i = 2; i < n/2; i+=2)
    {
        ComplexD2 k = broadcast128(packing_table + i);

        if(!forward)
        {
            k = change_sign(k, const1);
        }

        size_t j = 0;
        // First use avx
        for (j = 0; j + 2 < 2*m; j+=4)
        {
            ComplexD2 x0_in = load(data_in + i*m + j);
            ComplexD2 x1_in = load(data_in + (n - i)*m + j);

            ComplexD2 temp0 = x0_in + change_sign(x1_in, const2);
            ComplexD2 temp1 = mul(k, temp0);

            ComplexD2 x0_out = temp1 + x0_in;
            ComplexD2 x1_out = change_sign(temp1, const2) + x1_in;

            store(x0_out, data_out + i*m + j);
            store(x1_out, data_out + (n - i)*m + j);
        }

        // Then if needed, use sse2
        if (j < 2*m)
        {
            ComplexD k = load128(packing_table + i);
            if(!forward)
            {
                k = change_sign(k, const1_128);
            }

            ComplexD x0_in = load128(data_in + i*m + j);
            ComplexD x1_in = load128(data_in + (n - i)*m + j);

            ComplexD temp0 = x0_in + change_sign(x1_in, const2_128);
            ComplexD temp1 = mul(k, temp0);

            ComplexD x0_out = temp1 + x0_in;
            ComplexD x1_out = change_sign(temp1, const2_128) + x1_in;

            store(x0_out, data_out + i*m + j);
            store(x1_out, data_out + (n - i)*m + j);
        }
    }
}

template<size_t radix>
void fft_2d_real_reorder2_inverse_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    // In-place not supported
    assert (data_in != data_out);

    size_t n = step_info.stride; // number of rows
    size_t m = step_info.size; // number of columns
    size_t repeats = step_info.repeats;
    uint32_t *reorder_table_rows = step_info.reorder_table2;
    ComplexD norm_factor = broadcast64(step_info.norm_factor);
    ComplexD2 norm_factor256 = broadcast64_D2(step_info.norm_factor);

    for (size_t i = 0; i < n; i++)
    {
        size_t j = 0;

        // First use 256-bit variables
        {
            ComplexD2 x_temp_in[radix];
            ComplexD2 x_temp_out[radix];

            for (j = 0; j+1 < repeats; j+=2)
            {
                // Copy input data (squeeze)
                for (size_t k = 0; k < radix; k++)
                {
                    size_t j2 = reorder_table_rows[j*radix + k];
                    size_t j3 = reorder_table_rows[(j+1)*radix + k];

                    x_temp_in[k] = norm_factor256*load_two_128(data_in + 2*i*m + 2*j2, data_in + 2*i*m + 2*j3);
                }

                // Multiply with coefficients
                multiply_coeff<radix,false>(x_temp_in, x_temp_out);

                // Copy output data (un-squeeze)
                for (size_t k = 0; k < radix; k++)
                {
                    store_two_128(x_temp_out[k], data_out + 2*i*m + 2*(j*radix + k), data_out + 2*i*m + 2*((j+1)*radix + k));
                }
            }
        }

        // Then, if necassery, use 128-bit variables
        if (j < repeats)
        {
            ComplexD x_temp_in[radix];
            ComplexD x_temp_out[radix];

            // Copy input data
            for (size_t k = 0; k < radix; k++)
            {
                size_t j2 = reorder_table_rows[j*radix + k];

                x_temp_in[k] = norm_factor*load128(data_in + 2*i*m + 2*j2);
            }

            multiply_coeff<radix,false>(x_temp_in, x_temp_out);

            // Copy input data (un-squeeze)
            for (size_t k = 0; k < radix; k++)
            {
                store(x_temp_out[k], data_out + 2*i*m + 2*(j*radix + k));
            }
        }
    }
}
*/

template<bool forward>
    void fft_2d_complex_to_complex_packed_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    const double *packing_table = step_info.twiddle_factors;
    size_t n = step_info.repeats; // n = number of original real rows
    size_t m = step_info.size;    // m = number of original real columns

    const ComplexD2 const1 = load(0.0, -0.0, 0.0, -0.0);
    const ComplexD2 const2 = load(-0.0, 0.0, -0.0, 0.0);

    // In forward the input is n x (m/2), output is n x (m/2 + 1) complex numbers
    if (forward)
    {
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
            size_t j = 0;

            // First use avx
            for (j = 2; j + 2 < m/2; j+=4)
            {
                ComplexD2 sssc = load(packing_table + j);
                ComplexD2 x0_in = load(data_in + i*m2 + j);
                ComplexD2 x1_in = load_two_128(data_in + i*m2 + (m-j), data_in + i*m2 + (m-j) - 2);

                ComplexD2 temp0 = x0_in + change_sign(x1_in, const2);
                ComplexD2 temp1 = mul(sssc, temp0);

                ComplexD2 x0_out = temp1 + x0_in;
                ComplexD2 x1_out = change_sign(temp1, const2) + x1_in;

                store(x0_out, data_out + i*m2 + j);
                store_two_128(x1_out, data_out + i*m2 + (m-j), data_out + i*m2 + (m-j) - 2);
            }

            // Then, if needed use sse2
            if (j < m/2)
            {
                ComplexD sssc = load128(packing_table + j);
                ComplexD x0_in = load128(data_in + i*m2 + j);
                ComplexD x1_in = load128(data_in + i*m2 + (m-j));

                ComplexD temp0 = x0_in + change_sign(x1_in, const2_128);
                ComplexD temp1 = mul(sssc, temp0);

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
            size_t j = 0;

            // First use avx
            for (j = 2; j + 2 < m/2; j+=4)
            {
                ComplexD2 sssc = load(packing_table + j);
                sssc = change_sign(sssc, const1);
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
                sssc = change_sign(sssc, const1_128);
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
}

template<size_t radix>
void fft_2d_real_reorder2_inverse_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    // In-place not supported
    assert (data_in != data_out);

    size_t m = step_info.stride; // number of columns
    size_t m2 = m + 1;           // number of columns in input
    size_t repeats = step_info.repeats;
    uint32_t *reorder_table_columns = step_info.reorder_table;
    double norm_factor = step_info.norm_factor;

    for (size_t i = 0; i < repeats; i++)
    {
        size_t k = 0;

        // First use 256-bit variables as many times as possible
        for (k = 0; k+1 < m; k+=2)
        {
            ComplexD2 x_temp_in[radix];
            ComplexD2 x_temp_out[radix];

            // Copy input data (squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                size_t j2 = reorder_table_columns[i*radix + j];
                x_temp_in[j] = norm_factor*load(data_in + 2*j2*m2 + 2*k);
            }

            multiply_coeff<radix,false>(x_temp_in, x_temp_out);

            // Copy output data (un-squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                store(x_temp_out[j], data_out + 2*i*radix*m + 2*j*m + 2*k);
            }
        }

        // Then, if necassery, use 128-bit variables
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

template<size_t radix>
void fft_2d_real_reorder_last_column_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
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
template void fft_2d_complex_to_complex_packed_avx_d<false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_to_complex_packed_avx_d<true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_2d_real_reorder2_inverse_avx_d<2>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_avx_d<3>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_avx_d<4>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_avx_d<5>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_avx_d<7>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_avx_d<8>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_2d_real_reorder_last_column_avx_d<2>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder_last_column_avx_d<3>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder_last_column_avx_d<4>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder_last_column_avx_d<5>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder_last_column_avx_d<7>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder_last_column_avx_d<8>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

