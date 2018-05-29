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

    // In forward the input is n x (m/2), output is n x (m/2 + 1) complex numbers
    if (forward)
    {
        size_t m2 = m + 2;

        // First copy data so that row size increases from m/2 to m/2 + 1
        for (size_t i = 0; i < n; i++)
        {
            size_t i2 = n - i - 1;
            for (size_t j = 0; j < m; j++)
            {
                size_t j2 = m - j - 1;
                data_out[i2*m2 + j2] = data_in[i2*m + j2];
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
                double x_r = data_in[i*m2 + m/2 + 0];
                double x_i = data_in[i*m2 + m/2 + 1];

                data_out[i*m2 + m/2 + 0] =  x_r;
                data_out[i*m2 + m/2 + 1] = -x_i;
            }
        }

        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 2; j < m/2; j+=2)
            {
                double ss = -packing_table[j + 0];
                double sc = -packing_table[j + 1];

                double x0_r = data_in[i*m2 + j + 0];
                double x0_i = data_in[i*m2 + j + 1];
                double x1_r = data_in[i*m2 + (m-j) + 0];
                double x1_i = data_in[i*m2 + (m-j) + 1];

                //std::cout << "x0_r = " << x0_r << ", x0_i = " << x0_i << std::endl;
                //std::cout << "x1_r = " << x1_r << ", x1_i = " << x1_i << std::endl;

                double temp0 = -ss*(x0_r - x1_r) + sc*(x0_i + x1_i);
                double temp1 = -sc*(x0_r - x1_r) - ss*(x0_i + x1_i);

                data_out[i*m2 + j + 0]     = temp0 + x0_r;
                data_out[i*m2 + j + 1]     = temp1 + x0_i;
                data_out[i*m2 + (m-j) + 0] = -temp0 + x1_r;
                data_out[i*m2 + (m-j) + 1] = temp1 + x1_i;
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
                double x_r = data_out[i*m + m/2 + 0];
                double x_i = data_out[i*m + m/2 + 1];

                data_out[i*m + m/2 + 0] =  x_r;
                data_out[i*m + m/2 + 1] = -x_i;
            }
        }

        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 2; j < m/2; j+=2)
            {
                double ss = -packing_table[j + 0];
                double sc = packing_table[j + 1];

                double x0_r = data_out[i*m + j + 0];
                double x0_i = data_out[i*m + j + 1];
                double x1_r = data_out[i*m + (m-j) + 0];
                double x1_i = data_out[i*m + (m-j) + 1];

                //std::cout << "x0_r = " << x0_r << ", x0_i = " << x0_i << std::endl;
                //std::cout << "x1_r = " << x1_r << ", x1_i = " << x1_i << std::endl;

                double temp0 = -ss*(x0_r - x1_r) + sc*(x0_i + x1_i);
                double temp1 = -sc*(x0_r - x1_r) - ss*(x0_i + x1_i);

                data_out[i*m + j + 0]     = temp0 + x0_r;
                data_out[i*m + j + 1]     = temp1 + x0_i;
                data_out[i*m + (m-j) + 0] = -temp0 + x1_r;
                data_out[i*m + (m-j) + 1] = temp1 + x1_i;
            }
        }
    }
}

template<size_t radix>
void fft_2d_real_reorder2_inverse_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    // In-place not supported
    assert (data_in != data_out);

    size_t m = step_info.stride; // number of columns
    size_t m2 = m + 1;           // number of columns in input
    size_t repeats = step_info.repeats;
    uint32_t *reorder_table_columns = step_info.reorder_table;
    double norm_factor = step_info.norm_factor;

    double x_temp_in[2*radix];
    double x_temp_out[2*radix];

    for (size_t i = 0; i < repeats; i++)
    {
        for (size_t k = 0; k < m; k++)
        {
            // Copy input data (squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                size_t j2 = reorder_table_columns[i*radix + j];
                x_temp_in[2*j + 0] = data_in[2*j2*m2 + 2*k + 0];
                x_temp_in[2*j + 1] = data_in[2*j2*m2 + 2*k + 1];
            }

            multiply_coeff<radix,false>(x_temp_in, x_temp_out);

            // Copy output data (un-squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                data_out[2*i*radix*m + 2*j*m + 2*k + 0] = norm_factor*x_temp_out[2*j + 0];
                data_out[2*i*radix*m + 2*j*m + 2*k + 1] = norm_factor*x_temp_out[2*j + 1];
            }
        }
    }
}


template<size_t radix>
void fft_2d_real_reorder_last_column_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    // In-place not supported
    assert (data_in != data_out);

    size_t m = step_info.stride; // number of columns
    size_t m2 = m + 1;           // number of columns in input
    size_t repeats = step_info.repeats;
    uint32_t *reorder_table_columns = step_info.reorder_table;
    double norm_factor = step_info.norm_factor;

    double x_temp_in[2*radix];
    double x_temp_out[2*radix];

    for (size_t i = 0; i < repeats; i++)
    {
        // Copy input data (squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            size_t j2 = reorder_table_columns[i*radix + j];
            x_temp_in[2*j + 0] = data_in[2*j2*m2 + 2*m + 0];
            x_temp_in[2*j + 1] = data_in[2*j2*m2 + 2*m + 1];
        }

        multiply_coeff<radix,false>(x_temp_in, x_temp_out);

        // Copy output data (un-squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            data_out[2*i*radix + 2*j + 0] = norm_factor*x_temp_out[2*j + 0];
            data_out[2*i*radix + 2*j + 1] = norm_factor*x_temp_out[2*j + 1];
        }
    }
}


// Instantiations of the functions defined in this class
template void fft_2d_complex_to_complex_packed_plain_d<false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_to_complex_packed_plain_d<true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_2d_real_reorder2_inverse_plain_d<2>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_plain_d<3>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_plain_d<4>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_plain_d<5>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_plain_d<7>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_plain_d<8>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_2d_real_reorder_last_column_plain_d<2>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder_last_column_plain_d<3>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder_last_column_plain_d<4>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder_last_column_plain_d<5>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder_last_column_plain_d<7>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder_last_column_plain_d<8>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

