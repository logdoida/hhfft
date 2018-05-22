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

#include "../common/hhfft_1d_complex_plain_common_d.h"

#include <iostream> // TESTING

using namespace hhfft;

template<bool forward>
    void fft_2d_complex_to_complex_packed_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{        
    const double *packing_table = step_info.twiddle_factors;
    size_t n = step_info.repeats; // n = number of original real rows
    size_t m = step_info.size;    // m = number of original real columns

    // Input/output way
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
        // For inverse data_in = last row (temporary variable), data_out is actually both input and output!
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
            double x_r = data_in[j + n*m/2 + 0];
            double x_i = data_in[j + n*m/2 + 1];

            data_out[j + n*m/2 + 0] =  x_r;
            data_out[j + n*m/2 + 1] = -x_i;
        }
    }

    for (size_t i = 2; i < n/2; i+=2)
    {        
        double ss = -packing_table[i + 0];
        double sc = -packing_table[i + 1];

        if (!forward)
        {
            sc = -sc;
        }

        for (size_t j = 0; j < 2*m; j+=2)
        {
            double x0_r = data_in[i*m + j + 0];
            double x0_i = data_in[i*m + j + 1];
            double x1_r = data_in[(n - i)*m + j + 0];
            double x1_i = data_in[(n - i)*m + j + 1];

            //std::cout << "x0_r = " << x0_r << ", x0_i = " << x0_i << std::endl;
            //std::cout << "x1_r = " << x1_r << ", x1_i = " << x1_i << std::endl;

            double temp0 = -ss*(x0_r - x1_r) + sc*(x0_i + x1_i);
            double temp1 = -sc*(x0_r - x1_r) - ss*(x0_i + x1_i);

            data_out[i*m + j + 0]     = temp0 + x0_r;
            data_out[i*m + j + 1]     = temp1 + x0_i;
            data_out[(n - i)*m + j + 0] = -temp0 + x1_r;
            data_out[(n - i)*m + j + 1] = temp1 + x1_i;
        }
    }
}

// Column reordering, shuffling and first FFT step. Used as a first step in FFT 2D real
template<size_t radix>
    void fft_2d_real_reorder2_forward_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    // In-place not supported
    assert (data_in != data_out);

    size_t m = step_info.size;
    uint32_t *reorder_table_columns = step_info.reorder_table;
    uint32_t *reorder_table_rows = step_info.reorder_table2;
    size_t repeats = step_info.repeats;

    // FFT and reordering
    for (size_t i = 0; i < repeats; i++)
    {
        double x_temp_in[2*radix];
        double x_temp_out[2*radix];

        for (size_t k = 0; k < m; k++)
        {
            size_t k2 = reorder_table_rows[k];

            // Copy input data (squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                size_t j2 = reorder_table_columns[i*radix + j];

                x_temp_in[2*j + 0] = data_in[2*j2*m + k2];
                x_temp_in[2*j + 1] = data_in[(2*j2+1)*m + k2];
            }

            multiply_coeff<radix,true>(x_temp_in, x_temp_out);

            // Copy input data (un-squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                data_out[2*i*radix*m + 2*j*m + 2*k + 0] = x_temp_out[2*j + 0];
                data_out[2*i*radix*m + 2*j*m + 2*k + 1] = x_temp_out[2*j + 1];
            }
        }
    }
}

template<size_t radix>
void fft_2d_real_reorder2_inverse_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    // In-place not supported
    assert (data_in != data_out);

    size_t n = step_info.stride; // number of rows
    size_t m = step_info.size; // number of columns
    size_t repeats = step_info.repeats;
    uint32_t *reorder_table_rows = step_info.reorder_table2;
    double norm_factor = step_info.norm_factor;

    double x_temp_in[2*radix];
    double x_temp_out[2*radix];

    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < repeats; j++)
        {
            // Copy input data
            for (size_t k = 0; k < radix; k++)
            {
                size_t j2 = reorder_table_rows[j*radix + k];
                x_temp_in[2*k + 0] = norm_factor*data_in[2*i*m + 2*j2 + 0];
                x_temp_in[2*k + 1] = norm_factor*data_in[2*i*m + 2*j2 + 1];
            }

            multiply_coeff<radix,false>(x_temp_in, x_temp_out);

            // Copy input data (un-squeeze)
            for (size_t k = 0; k < radix; k++)
            {
                data_out[2*i*m + 2*(j*radix + k) + 0] = x_temp_out[2*k + 0];
                data_out[2*i*m + 2*(j*radix + k) + 1] = x_temp_out[2*k + 1];
            }
        }
    }
}

// Instantiations of the functions defined in this class
template void fft_2d_complex_to_complex_packed_plain_d<false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_to_complex_packed_plain_d<true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_2d_real_reorder2_forward_plain_d<2>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_forward_plain_d<3>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_forward_plain_d<4>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_forward_plain_d<5>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_forward_plain_d<7>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_forward_plain_d<8>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_2d_real_reorder2_inverse_plain_d<2>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_plain_d<3>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_plain_d<4>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_plain_d<5>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_plain_d<7>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_plain_d<8>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
