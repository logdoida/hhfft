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

using namespace hhfft;


////////////////////////////////////// Column-wise ////////////////////////////////////////////////

template<size_t radix>
    inline __attribute__((always_inline)) void fft_2d_complex_column_twiddle_dit_plain_d_internal(
            const double *data_in, double *data_out, const double *twiddle_factors, size_t stride, size_t length)
{        
    double x_temp_in[2*radix];
    double x_temp_out[2*radix];
    double twiddle_temp[2*radix];

    for (size_t i = 0; i < stride; i++)
    {
        // Copy twiddle factors
        for (size_t j = 0; j < radix; j++)
        {
            twiddle_temp[2*j + 0] = twiddle_factors[2*i + 2*j*stride + 0];
            twiddle_temp[2*j + 1] = twiddle_factors[2*i + 2*j*stride + 1];
        }

        for (size_t k = 0; k < length; k++)
        {
            // Copy input data (squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                x_temp_in[2*j + 0] = data_in[2*j*stride*length + 2*i*length + 2*k + 0];
                x_temp_in[2*j + 1] = data_in[2*j*stride*length + 2*i*length + 2*k + 1];                
            }

            multiply_twiddle<radix,true>(x_temp_in, x_temp_in, twiddle_temp);

            multiply_coeff<radix,true>(x_temp_in, x_temp_out);

            // Copy output data (un-squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                data_out[2*j*stride*length + 2*i*length + 2*k + 0] = x_temp_out[2*j + 0];
                data_out[2*j*stride*length + 2*i*length + 2*k + 1] = x_temp_out[2*j + 1];
            }
        }
    }
}

template<size_t radix>
    void fft_2d_complex_column_twiddle_dit_plain_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info)
{
    size_t stride = step_info.stride;
    size_t repeats = step_info.repeats;
    size_t length = step_info.size;

    for (size_t i = 0; i < repeats; i++)
    {
        fft_2d_complex_column_twiddle_dit_plain_d_internal<radix>
                (data_in  + 2*i*radix*stride*length,
                 data_out + 2*i*radix*stride*length,
                 step_info.twiddle_factors,
                 stride, length);
    }
}

////////////////////////////////////// Column and row-wise + one FFT step ////////////////////////////////////////////////

// Combine reordering and first column wise FFT
template<size_t radix, bool forward>
    void fft_2d_complex_reorder2_plain_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info)
{
    size_t m = step_info.size;    
    uint32_t *reorder_table_columns = step_info.reorder_table;
    uint32_t *reorder_table_rows = step_info.reorder_table2;
    size_t repeats = step_info.repeats;
    size_t n = repeats*radix;

    // Needed only in ifft. Equal to 1/N
    double norm_factor = step_info.norm_factor;

    // FFT and reordering
    for (size_t i = 0; i < repeats; i++)
    {
        double x_temp_in[2*radix];
        double x_temp_out[2*radix];

        for (size_t k = 0; k < m; k++)
        {
            size_t k2 = reorder_table_rows[k];

            if (!forward && k > 0)
            {
                k2 = m - k2;
            }

            // Copy input data (squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                size_t j1 = i*radix + j;
                size_t j2 = reorder_table_columns[j1];

                if (forward)
                {
                    x_temp_in[2*j + 0] = data_in[2*j2*m + 2*k2 + 0];
                    x_temp_in[2*j + 1] = data_in[2*j2*m + 2*k2 + 1];
                } else
                {
                    if (j1 > 0)
                    {
                        j2 = n - j2;
                    }
                    x_temp_in[2*j + 0] = norm_factor*data_in[2*j2*m + 2*k2 + 0];
                    x_temp_in[2*j + 1] = norm_factor*data_in[2*j2*m + 2*k2 + 1];
                }
            }

            multiply_coeff<radix,true>(x_temp_in, x_temp_out);

            // Copy output data (un-squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                size_t j1 = i*radix + j;
                data_out[2*j1*m + 2*k + 0] = x_temp_out[2*j + 0];
                data_out[2*j1*m + 2*k + 1] = x_temp_out[2*j + 1];
            }
        }
    }
}

// Combine reordering and first row wise FFT
template<size_t radix> void fft_2d_complex_reorder2_rows_forward_plain_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info)
{    
    // Only out of place reordering supported
    assert(data_in != data_out);

    size_t n = step_info.size;
    size_t m = step_info.repeats * radix;
    size_t repeats = step_info.repeats;
    uint32_t *reorder_table_columns = step_info.reorder_table;
    uint32_t *reorder_table_rows = step_info.reorder_table2;

    // FFT and reordering
    for (size_t i = 0; i < n; i++)
    {
        size_t i2 = reorder_table_columns[i];

        for (size_t j = 0; j < repeats; j++)
        {
            double x_temp_in[2*radix];
            double x_temp_out[2*radix];

            // Copy input data (squeeze)
            for (size_t k = 0; k < radix; k++)
            {
                size_t j2 = reorder_table_rows[j*radix + k];

                x_temp_in[2*k + 0] = data_in[2*i2*m + 2*j2 + 0];
                x_temp_in[2*k + 1] = data_in[2*i2*m + 2*j2 + 1];
            }

            multiply_coeff<radix,true>(x_temp_in, x_temp_out);

            // Copy output data (un-squeeze)
            for (size_t k = 0; k < radix; k++)
            {
                data_out[2*i*m + 2*j*radix + 2*k + 0] = x_temp_out[2*k + 0];
                data_out[2*i*m + 2*j*radix + 2*k + 1] = x_temp_out[2*k + 1];
            }
        }
    }
}


// Instantiations of the functions defined in this class
template void fft_2d_complex_column_twiddle_dit_plain_d<2>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_plain_d<3>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_plain_d<4>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_plain_d<5>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_plain_d<6>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_plain_d<7>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_plain_d<8>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template void fft_2d_complex_reorder2_plain_d<2, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<2, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<3, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<3, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<4, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<4, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<5, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<5, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<6, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<6, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<7, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<7, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<8, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<8, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template void fft_2d_complex_reorder2_rows_forward_plain_d<2>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_rows_forward_plain_d<3>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_rows_forward_plain_d<4>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_rows_forward_plain_d<5>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_rows_forward_plain_d<6>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_rows_forward_plain_d<7>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_rows_forward_plain_d<8>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
