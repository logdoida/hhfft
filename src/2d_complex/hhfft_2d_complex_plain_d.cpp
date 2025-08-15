/*
*   Copyright Jouko Kalmari 2017-2019
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

#include "../common/hhfft_common_plain_d.h"
#include "../raders/raders_plain_d.h"

using namespace hhfft;


////////////////////////////////////// Column-wise ////////////////////////////////////////////////

template<RadixType radix_type>
    inline __attribute__((always_inline)) void fft_2d_complex_column_twiddle_dit_plain_d_internal(
            const double *data_in, double *data_out, const double *twiddle_factors, double *data_raders, const hhfft::RadersD &raders, size_t stride, size_t length)
{
    ComplexD x_temp_in[radix_type];
    ComplexD x_temp_out[radix_type];
    ComplexD twiddle_temp[radix_type];
    size_t radix = get_actual_radix<radix_type>(raders);

    for (size_t i = 0; i < stride; i++)
    {
        // NOTE it would be possible to copy twiddle factors only once here
        /*
        for (size_t j = 0; j < radix; j++)
        {
            twiddle_temp[j] = load_D(twiddle_factors + 2*i + 2*j*stride);
        }
        */

        for (size_t k = 0; k < length; k++)
        {
            // Initialize raders data with zeros
            init_coeff_D<radix_type>(data_raders, raders);

            // Copy input data (squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                ComplexD x = load_D(data_in + 2*j*stride*length + 2*i*length + 2*k);
                ComplexD w = load_D(twiddle_factors + 2*i + 2*j*stride);
                set_value_twiddle_D<radix_type>(x_temp_in, data_raders, twiddle_temp, j, raders, x, w);
            }

            // Multiply with coefficients
            multiply_twiddle_D<radix_type,true>(x_temp_in, x_temp_in, twiddle_temp);
            multiply_coeff_forward_D<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

            // Copy output data (un-squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                ComplexD x = get_value_D<radix_type>(x_temp_out, data_raders, j, raders);
                store_D(x, data_out + 2*j*stride*length + 2*i*length + 2*k);
            }
        }
    }
}

template<RadixType radix_type, SizeType size_type>
    void fft_2d_complex_column_twiddle_dit_plain_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info)
{
    size_t stride = step_info.stride;
    size_t repeats = step_info.repeats;
    size_t length = get_size<size_type>(step_info.size);
    const hhfft::RadersD &raders = *step_info.raders;
    size_t radix = get_actual_radix<radix_type>(raders);

    // Allocate memory for Rader's algorithm if needed
    double *data_raders = allocate_raders_D<radix_type>(raders);

    for (size_t i = 0; i < repeats; i++)
    {
        fft_2d_complex_column_twiddle_dit_plain_d_internal<radix_type>
                (data_in  + 2*i*radix*stride*length,
                 data_out + 2*i*radix*stride*length,
                 step_info.twiddle_factors, data_raders, raders,
                 stride, length);

    }

    // Free temporary memory
    free_raders_D<radix_type>(raders, data_raders);
}


// Combine reordering and first column wise reordering
template<RadixType radix_type, bool forward>
    void fft_2d_complex_reorder2_plain_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info)
{
    size_t m = step_info.size;
    uint32_t *reorder_table_columns = step_info.reorder_table;
    uint32_t *reorder_table_rows = step_info.reorder_table2;
    size_t repeats = step_info.repeats;
    size_t reorder_table_columns_size = step_info.reorder_table_size;
    size_t reorder_table_rows_size = step_info.reorder_table2_size;
    const hhfft::RadersD &raders = *step_info.raders;
    size_t radix = get_actual_radix<radix_type>(raders);

    // Needed only in ifft. Equal to 1/N
    ComplexD norm_factor = broadcast64_D(step_info.norm_factor);

    // Allocate memory for Rader's algorithm if needed
    double *data_raders = allocate_raders_D<radix_type>(raders);

    // FFT and reordering
    for (size_t i = 0; i < repeats; i++)
    {
        ComplexD x_temp_in[radix_type];
        ComplexD x_temp_out[radix_type];

        for (size_t k = 0; k < m; k++)
        {
            // Initialize raders data with zeros
            init_coeff_D<radix_type>(data_raders, raders);

            size_t k2;
            if (forward)
            {
                k2 = reorder_table_rows[k];
            } else
            {
                k2 = reorder_table_rows[reorder_table_rows_size - k - 1];
            }

            // Copy input data (squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                size_t j1 = i*radix + j;
                size_t j2;
                if (forward)
                {
                    j2 = reorder_table_columns[j1];
                } else
                {
                    j2 = reorder_table_columns[reorder_table_columns_size - j1 - 1];
                }
                ComplexD x = load_D(data_in + 2*j2*m + 2*k2);
                set_value_D<radix_type>(x_temp_in, data_raders, j, raders, x);
            }

            // Multiply with coefficients
            multiply_coeff_forward_D<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

            // Copy output data (un-squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                size_t j1 = i*radix + j;
                ComplexD x = get_value_D<radix_type>(x_temp_out, data_raders, j, raders);
                if (!forward)
                {
                    x = x*norm_factor;
                }
                store_D(x, data_out + 2*j1*m + 2*k);
            }
        }
    }

    // Free temporary memory
    free_raders_D<radix_type>(raders, data_raders);
}

// Combine reordering and first row wise FFT
template<RadixType radix_type> void fft_2d_complex_reorder2_rows_forward_plain_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info)
{
    const hhfft::RadersD &raders = *step_info.raders;
    size_t radix = get_actual_radix<radix_type>(raders);
    size_t n = step_info.size;
    size_t m = step_info.repeats * radix;
    size_t repeats = step_info.repeats;
    uint32_t *reorder_table_columns = step_info.reorder_table;
    uint32_t *reorder_table_rows = step_info.reorder_table2;

    // Allocate memory for Rader's algorithm if needed
    double *data_raders = allocate_raders_D<radix_type>(raders);

    // FFT and reordering
    for (size_t i = 0; i < n; i++)
    {
        size_t i2 = reorder_table_columns[i];

        for (size_t j = 0; j < repeats; j++)
        {
            // Initialize raders data with zeros
            init_coeff_D<radix_type>(data_raders, raders);

            ComplexD x_temp_in[radix_type];
            ComplexD x_temp_out[radix_type];

            // Copy input data (squeeze)
            for (size_t k = 0; k < radix; k++)
            {
                size_t j2 = reorder_table_rows[j*radix + k];

                ComplexD x = load_D(data_in + 2*i2*m + 2*j2);
                set_value_D<radix_type>(x_temp_in, data_raders, k, raders, x);
            }

            // Multiply with coefficients
            multiply_coeff_forward_D<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

            // Copy output data (un-squeeze)
            for (size_t k = 0; k < radix; k++)
            {
                ComplexD x = get_value_D<radix_type>(x_temp_out, data_raders, k, raders);
                store_D(x, data_out + 2*i*m + 2*j*radix + 2*k);
            }
        }
    }

    // Free temporary memory
    free_raders_D<radix_type>(raders, data_raders);
}


// Instantiations of the functions defined in this class
template void fft_2d_complex_column_twiddle_dit_plain_d<Raders, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_plain_d<Radix2, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_plain_d<Radix3, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_plain_d<Radix4, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_plain_d<Radix5, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_plain_d<Radix6, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_plain_d<Radix7, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_plain_d<Radix8, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_plain_d<Radix11, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_plain_d<Radix13, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template void fft_2d_complex_column_twiddle_dit_plain_d<Raders, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_plain_d<Radix2, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_plain_d<Radix3, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_plain_d<Radix4, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_plain_d<Radix5, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_plain_d<Radix6, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_plain_d<Radix7, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_plain_d<Radix8, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_plain_d<Radix11, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_plain_d<Radix13, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template void fft_2d_complex_reorder2_plain_d<Raders, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<Raders, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<Radix2, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<Radix2, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<Radix3, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<Radix3, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<Radix4, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<Radix4, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<Radix5, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<Radix5, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<Radix6, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<Radix6, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<Radix7, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<Radix7, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<Radix8, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<Radix8, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<Radix11, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<Radix11, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<Radix13, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_plain_d<Radix13, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template void fft_2d_complex_reorder2_rows_forward_plain_d<Raders>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_rows_forward_plain_d<Radix2>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_rows_forward_plain_d<Radix3>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_rows_forward_plain_d<Radix4>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_rows_forward_plain_d<Radix5>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_rows_forward_plain_d<Radix6>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_rows_forward_plain_d<Radix7>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_rows_forward_plain_d<Radix8>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_rows_forward_plain_d<Radix11>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_reorder2_rows_forward_plain_d<Radix13>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
