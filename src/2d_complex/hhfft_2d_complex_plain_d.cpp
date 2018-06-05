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

#include <iostream> // TESTING

#include "../common/hhfft_1d_complex_plain_common_d.h"

using namespace hhfft;


////////////////////////////////////// Column-wise ////////////////////////////////////////////////

template<size_t radix, bool forward>
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

            multiply_twiddle<radix,forward>(x_temp_in, x_temp_in, twiddle_temp);

            multiply_coeff<radix,forward>(x_temp_in, x_temp_out);

            // Copy output data (un-squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                data_out[2*j*stride*length + 2*i*length + 2*k + 0] = x_temp_out[2*j + 0];
                data_out[2*j*stride*length + 2*i*length + 2*k + 1] = x_temp_out[2*j + 1];
            }
        }
    }
}

template<size_t radix, bool forward>
    void fft_2d_complex_column_twiddle_dit_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    assert(step_info.forward == forward);

    size_t stride = step_info.stride;
    size_t repeats = step_info.repeats;
    size_t length = step_info.size;

    for (size_t i = 0; i < repeats; i++)
    {
        fft_2d_complex_column_twiddle_dit_plain_d_internal<radix,forward>
                (data_in  + 2*i*radix*stride*length,
                 data_out + 2*i*radix*stride*length,
                 step_info.twiddle_factors,
                 stride, length);
    }
}

////////////////////////////////////// Column and row-wise + one FFT step ////////////////////////////////////////////////

template<size_t radix, bool forward, bool scale>
    void fft_2d_complex_reorder2_inplace_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    size_t n = step_info.stride;
    size_t m = step_info.size;
    uint32_t *reorder_table_columns = step_info.reorder_table_inplace;
    uint32_t *reorder_table_rows = step_info.reorder_table2_inplace;
    size_t reorder_table_size_columns = step_info.reorder_table_inplace_size;
    size_t reorder_table_size_rows = step_info.reorder_table2_inplace_size;
    size_t repeats = step_info.repeats;

    // Reorder rows
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < reorder_table_size_rows; j++)
        {
            size_t ind1 = j + 1; // First one has been omitted!
            size_t ind2 = reorder_table_rows[j];

            double r_temp = data_out[2*i*m + 2*ind1+0];
            double c_temp = data_out[2*i*m + 2*ind1+1];
            data_out[2*i*m + 2*ind1+0] = data_out[2*i*m + 2*ind2+0];
            data_out[2*i*m + 2*ind1+1] = data_out[2*i*m + 2*ind2+1];
            data_out[2*i*m + 2*ind2+0] = r_temp;
            data_out[2*i*m + 2*ind2+1] = c_temp;
        }
    }

    // Reorder columns
    for (size_t i = 0; i < reorder_table_size_columns; i++)
    {
        size_t ind1 = i + 1; // First one has been omitted!
        size_t ind2 = reorder_table_columns[i];

        for (size_t j = 0; j < 2*m; j++)
        {
            double temp = data_out[2*ind1*m + j];
            data_out[2*ind1*m + j] = data_out[2*ind2*m + j];
            data_out[2*ind2*m + j] = temp;
        }
    }

    // Normal fft
    double norm_factor = step_info.norm_factor; // Needed only in ifft. Equal to 1/N
    for (size_t i = 0; i < repeats; i++)
    {
        double x_temp_in[2*radix];
        double x_temp_out[2*radix];

        for (size_t k = 0; k < m; k++)
        {
            // Copy input data (squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                x_temp_in[2*j + 0] = data_out[2*i*radix*m + 2*j*m + 2*k + 0];
                x_temp_in[2*j + 1] = data_out[2*i*radix*m + 2*j*m + 2*k + 1];
            }

            multiply_coeff<radix,forward>(x_temp_in, x_temp_out);

            // Copy output data (un-squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                if (scale)
                {
                    x_temp_out[2*j + 0] *= norm_factor;
                    x_temp_out[2*j + 1] *= norm_factor;
                }

                data_out[2*i*radix*m + 2*j*m + 2*k + 0] = x_temp_out[2*j + 0];
                data_out[2*i*radix*m + 2*j*m + 2*k + 1] = x_temp_out[2*j + 1];
            }
        }
    }
}

// Combine reordering and first column wise FFT
template<size_t radix, bool forward, bool scale>
    void fft_2d_complex_reorder2_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{    
    if (data_in == data_out)
    {
        fft_2d_complex_reorder2_inplace_plain_d<radix, forward, scale>(data_in, data_out, step_info);
        return;
    }

    size_t m = step_info.size;
    uint32_t *reorder_table_columns = step_info.reorder_table;
    uint32_t *reorder_table_rows = step_info.reorder_table2;
    size_t repeats = step_info.repeats;

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

            // Copy input data (squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                size_t j2 = reorder_table_columns[i*radix + j];

                x_temp_in[2*j + 0] = data_in[2*j2*m + 2*k2 + 0];
                x_temp_in[2*j + 1] = data_in[2*j2*m + 2*k2 + 1];                
            }

            multiply_coeff<radix,forward>(x_temp_in, x_temp_out);

            if (scale)
            {
                for (size_t j = 0; j < radix; j++)
                {
                    x_temp_out[2*j + 0] *= norm_factor;
                    x_temp_out[2*j + 1] *= norm_factor;
                }
            }

            // Copy output data (un-squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                data_out[2*i*radix*m + 2*j*m + 2*k + 0] = x_temp_out[2*j + 0];
                data_out[2*i*radix*m + 2*j*m + 2*k + 1] = x_temp_out[2*j + 1];
            }
        }
    }
}

// Combine reordering and first row wise FFT
template<size_t radix> void fft_2d_complex_reorder2_rows_forward_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
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
template void fft_2d_complex_column_twiddle_dit_plain_d<2, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_plain_d<2, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_plain_d<3, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_plain_d<3, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_plain_d<4, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_plain_d<4, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_plain_d<5, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_plain_d<5, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_plain_d<6, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_plain_d<6, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_plain_d<7, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_plain_d<7, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_plain_d<8, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_plain_d<8, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_2d_complex_reorder2_plain_d<2, true, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<2, true, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<2, false, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<2, false, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<3, true, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<3, true, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<3, false, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<3, false, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<4, true, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<4, true, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<4, false, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<4, false, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<5, true, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<5, true, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<5, false, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<5, false, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<6, true, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<6, true, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<6, false, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<6, false, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<7, true, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<7, true, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<7, false, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<7, false, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<8, true, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<8, true, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<8, false, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<8, false, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_2d_complex_reorder2_rows_forward_plain_d<2>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_rows_forward_plain_d<3>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_rows_forward_plain_d<4>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_rows_forward_plain_d<5>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_rows_forward_plain_d<6>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_rows_forward_plain_d<7>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_rows_forward_plain_d<8>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
