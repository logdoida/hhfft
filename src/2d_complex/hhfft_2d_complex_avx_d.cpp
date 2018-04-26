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

#include "../common/hhfft_1d_complex_avx_common_d.h"

using namespace hhfft;


////////////////////////////////////// Column-wise ////////////////////////////////////////////////

template<size_t radix, bool forward>
    inline __attribute__((always_inline)) void fft_2d_complex_column_twiddle_dit_avx_d_internal(
            const double *data_in, double *data_out, const double *twiddle_factors, size_t stride, size_t length)
{
    for (size_t i = 0; i < stride; i++)
    {
        size_t k = 0;

        // First use 256-bit variables as many times as possible
        {
            ComplexD2 x_temp_in[radix];
            ComplexD2 x_temp_out[radix];
            ComplexD2 twiddle_temp[radix];

            // Copy twiddle factors (squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                twiddle_temp[j] = broadcast128(twiddle_factors + 2*i + 2*j*stride);
            }

            for (k = 0; k+1 < length; k+=2)
            {
                // Copy input data (squeeze)
                for (size_t j = 0; j < radix; j++)
                {
                    x_temp_in[j] = load(data_in +  2*j*stride*length + 2*i*length + 2*k);
                }

                multiply_twiddle<radix,forward>(x_temp_in, x_temp_in, twiddle_temp);
                multiply_coeff<radix,forward>(x_temp_in, x_temp_out);

                // Copy ouput data (un-squeeze)
                for (size_t j = 0; j < radix; j++)
                {
                    store(x_temp_out[j], data_out +  2*j*stride*length + 2*i*length + 2*k);
                }
            }
        }

        //TODO
        //if (length_type is divisible by 2)
        //    return;

        // Then, if necassery, use 128-bit variables
        if (k < length)
        {
            ComplexD x_temp_in[radix];
            ComplexD x_temp_out[radix];
            ComplexD twiddle_temp[radix];

            // Copy twiddle factors (squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                twiddle_temp[j] = load128(twiddle_factors + 2*i + 2*j*stride);
            }

            // Copy input data (squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                x_temp_in[j] = load128(data_in +  2*j*stride*length + 2*i*length + 2*k);
            }

            multiply_twiddle<radix,forward>(x_temp_in, x_temp_in, twiddle_temp);
            multiply_coeff<radix,forward>(x_temp_in, x_temp_out);

            // Copy ouput data (un-squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                store(x_temp_out[j], data_out +  2*j*stride*length + 2*i*length + 2*k);
            }
        }
    }
}

template<size_t radix, SizeType size_type, bool forward>
    void fft_2d_complex_column_twiddle_dit_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{   
    size_t stride = step_info.stride;
    size_t repeats = step_info.repeats;    
    size_t length = get_size<size_type>(step_info.size);

    for (size_t i = 0; i < repeats; i++)
    {
        fft_2d_complex_column_twiddle_dit_avx_d_internal<radix,forward>
                (data_in  + 2*i*radix*stride*length,
                 data_out + 2*i*radix*stride*length,
                 step_info.twiddle_factors,
                 stride, length);

    }
}

// Instantiations of the functions defined in this class
template void fft_2d_complex_column_twiddle_dit_avx_d<2, SizeType::Size1, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_avx_d<2, SizeType::Size1, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_avx_d<3, SizeType::Size1, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_avx_d<3, SizeType::Size1, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_avx_d<4, SizeType::Size1, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_avx_d<4, SizeType::Size1, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_avx_d<5, SizeType::Size1, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_avx_d<5, SizeType::Size1, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_avx_d<7, SizeType::Size1, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_avx_d<7, SizeType::Size1, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_2d_complex_column_twiddle_dit_avx_d<2, SizeType::SizeN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_avx_d<2, SizeType::SizeN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_avx_d<3, SizeType::SizeN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_avx_d<3, SizeType::SizeN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_avx_d<4, SizeType::SizeN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_avx_d<4, SizeType::SizeN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_avx_d<5, SizeType::SizeN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_avx_d<5, SizeType::SizeN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_avx_d<7, SizeType::SizeN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_avx_d<7, SizeType::SizeN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
