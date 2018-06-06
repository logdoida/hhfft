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


////////////////////////////////////// Column-wise ////////////////////////////////////////////////

template<size_t radix, bool forward>
    inline __attribute__((always_inline)) void fft_2d_complex_column_twiddle_dit_sse2_d_internal(
            const double *data_in, double *data_out, const double *twiddle_factors, size_t stride, size_t length)
{    
    ComplexD x_temp_in[radix];
    ComplexD x_temp_out[radix];
    ComplexD twiddle_temp[radix];

    for (size_t i = 0; i < stride; i++)
    {
        // Copy twiddle factors
        for (size_t j = 0; j < radix; j++)
        {
            twiddle_temp[j] = load_D(twiddle_factors + 2*i + 2*j*stride);
        }

        for (size_t k = 0; k < length; k++)
        {
            // Copy input data (squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                x_temp_in[j] = load_D(data_in + 2*j*stride*length + 2*i*length + 2*k);
            }

            multiply_twiddle_D<radix,forward>(x_temp_in, x_temp_in, twiddle_temp);
            multiply_coeff_D<radix,forward>(x_temp_in, x_temp_out);

            // Copy output data (un-squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                store_D(x_temp_out[j], data_out + 2*j*stride*length + 2*i*length + 2*k);
            }
        }
    }
}

template<size_t radix, SizeType size_type, bool forward>
    void fft_2d_complex_column_twiddle_dit_sse2_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{    
    size_t stride = step_info.stride;
    size_t repeats = step_info.repeats;    
    size_t length = get_size<size_type>(step_info.size);

    for (size_t i = 0; i < repeats; i++)
    {
        fft_2d_complex_column_twiddle_dit_sse2_d_internal<radix,forward>
                (data_in  + 2*i*radix*stride*length,
                 data_out + 2*i*radix*stride*length,
                 step_info.twiddle_factors,
                 stride, length);

    }
}

template<size_t radix, bool forward, bool scale>
    void fft_2d_complex_reorder2_inplace_sse2_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
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

            // Swap two doubles at a time
            ComplexD x_in0 = load_D(data_in + 2*i*m + 2*ind1);
            ComplexD x_in1 = load_D(data_in + 2*i*m + 2*ind2);
            store_D(x_in0, data_out + 2*i*m + 2*ind2);
            store_D(x_in1, data_out + 2*i*m + 2*ind1);
        }
    }

    // Reorder columns
    for (size_t i = 0; i < reorder_table_size_columns; i++)
    {
        size_t ind1 = i + 1; // First one has been omitted!
        size_t ind2 = reorder_table_columns[i];

        for (size_t j = 0; j < m; j++)
        {
            ComplexD x_in0 = load_D(data_out + 2*ind1*m + 2*j);
            ComplexD x_in1 = load_D(data_out + 2*ind2*m + 2*j);
            store_D(x_in0, data_out + 2*ind2*m + 2*j);
            store_D(x_in1, data_out + 2*ind1*m + 2*j);
        }
    }

    // Normal fft
    // Needed only in ifft. Equal to 1/N
    ComplexD norm_factor = broadcast64_D(step_info.norm_factor);
    for (size_t i = 0; i < repeats; i++)
    {
        ComplexD x_temp_in[radix];
        ComplexD x_temp_out[radix];

        for (size_t k = 0; k < m; k++)
        {
            // Copy input data (squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                x_temp_in[j] = load_D(data_out + 2*i*radix*m + 2*j*m + 2*k);
            }

            multiply_coeff_D<radix,forward>(x_temp_in, x_temp_out);
            if (scale)
            {
                for (size_t j = 0; j < radix; j++)
                {
                    x_temp_out[j] *= norm_factor;
                }
            }

            // Copy input data (un-squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                store_D(x_temp_out[j], data_out + 2*i*radix*m + 2*j*m + 2*k);
            }
        }
    }
}

// Combine reordering and first column wise reordering
template<size_t radix, bool forward, bool scale>
    void fft_2d_complex_reorder2_sse2_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    if (data_in == data_out)
    {
        fft_2d_complex_reorder2_inplace_sse2_d<radix, forward, scale>(data_in, data_out, step_info);
        return;
    }

    size_t m = step_info.size;
    uint32_t *reorder_table_columns = step_info.reorder_table;
    uint32_t *reorder_table_rows = step_info.reorder_table2;
    size_t repeats = step_info.repeats;

    // Needed only in ifft. Equal to 1/N
    ComplexD norm_factor = broadcast64_D(step_info.norm_factor);

    // FFT and reordering
    for (size_t i = 0; i < repeats; i++)
    {
        ComplexD x_temp_in[radix];
        ComplexD x_temp_out[radix];

        for (size_t k = 0; k < m; k++)
        {
            size_t k2 = reorder_table_rows[k];

            // Copy input data (squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                size_t j2 = reorder_table_columns[i*radix + j];

                x_temp_in[j] = load_D(data_in + 2*j2*m + 2*k2);
            }

            multiply_coeff_D<radix,forward>(x_temp_in, x_temp_out);
            if (scale)
            {
                for (size_t j = 0; j < radix; j++)
                {
                    x_temp_out[j] *= norm_factor;
                }
            }

            // Copy input data (un-squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                store_D(x_temp_out[j], data_out + 2*i*radix*m + 2*j*m + 2*k);
            }
        }
    }
}

// Combine reordering and first row wise FFT
template<size_t radix> void fft_2d_complex_reorder2_rows_forward_sse2_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
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
            ComplexD x_temp_in[radix];
            ComplexD x_temp_out[radix];

            // Copy input data (squeeze)
            for (size_t k = 0; k < radix; k++)
            {
                size_t j2 = reorder_table_rows[j*radix + k];

                x_temp_in[k] = load_D(data_in + 2*i2*m + 2*j2);
            }

            multiply_coeff_D<radix,true>(x_temp_in, x_temp_out);

            // Copy output data (un-squeeze)
            for (size_t k = 0; k < radix; k++)
            {
                store_D(x_temp_out[k], data_out + 2*i*m + 2*j*radix + 2*k);
            }
        }
    }
}


// Instantiations of the functions defined in this class
template void fft_2d_complex_column_twiddle_dit_sse2_d<2, SizeType::Size1, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_sse2_d<2, SizeType::Size1, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_sse2_d<3, SizeType::Size1, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_sse2_d<3, SizeType::Size1, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_sse2_d<4, SizeType::Size1, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_sse2_d<4, SizeType::Size1, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_sse2_d<5, SizeType::Size1, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_sse2_d<5, SizeType::Size1, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_sse2_d<6, SizeType::Size1, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_sse2_d<6, SizeType::Size1, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_sse2_d<7, SizeType::Size1, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_sse2_d<7, SizeType::Size1, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_sse2_d<8, SizeType::Size1, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_sse2_d<8, SizeType::Size1, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_2d_complex_column_twiddle_dit_sse2_d<2, SizeType::SizeN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_sse2_d<2, SizeType::SizeN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_sse2_d<3, SizeType::SizeN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_sse2_d<3, SizeType::SizeN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_sse2_d<4, SizeType::SizeN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_sse2_d<4, SizeType::SizeN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_sse2_d<5, SizeType::SizeN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_sse2_d<5, SizeType::SizeN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_sse2_d<6, SizeType::SizeN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_sse2_d<6, SizeType::SizeN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_sse2_d<7, SizeType::SizeN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_sse2_d<7, SizeType::SizeN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_sse2_d<8, SizeType::SizeN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_sse2_d<8, SizeType::SizeN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_2d_complex_reorder2_sse2_d<2, true, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_sse2_d<2, true, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_sse2_d<2, false, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_sse2_d<2, false, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_sse2_d<3, true, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_sse2_d<3, true, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_sse2_d<3, false, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_sse2_d<3, false, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_sse2_d<4, true, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_sse2_d<4, true, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_sse2_d<4, false, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_sse2_d<4, false, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_sse2_d<5, true, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_sse2_d<5, true, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_sse2_d<5, false, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_sse2_d<5, false, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_sse2_d<6, true, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_sse2_d<6, true, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_sse2_d<6, false, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_sse2_d<6, false, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_sse2_d<7, true, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_sse2_d<7, true, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_sse2_d<7, false, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_sse2_d<7, false, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_sse2_d<8, true, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_sse2_d<8, true, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_sse2_d<8, false, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_sse2_d<8, false, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_2d_complex_reorder2_rows_forward_sse2_d<2>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_rows_forward_sse2_d<3>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_rows_forward_sse2_d<4>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_rows_forward_sse2_d<5>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_rows_forward_sse2_d<6>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_rows_forward_sse2_d<7>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_rows_forward_sse2_d<8>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
