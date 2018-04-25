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

            // Copy input data (un-squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                data_out[2*j*stride*length + 2*i*length + 2*k + 0] = x_temp_out[2*j + 0];
                data_out[2*j*stride*length + 2*i*length + 2*k + 1] = x_temp_out[2*j + 1];
            }

            /*if (radix == 2)
            {
                std::cout << "x_temp_in  = (" << x_temp_in[0] << ", " << x_temp_in[1] << "), (" << x_temp_in[2] << ", " << x_temp_in[3] << ")" << std::endl;
                std::cout << "x_temp_out = (" << x_temp_out[0] << ", " << x_temp_out[1] << "), (" << x_temp_out[2] << ", " << x_temp_out[3] << ")" << std::endl;
            }*/
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

    std::cout << "fft_2d_complex_column_twiddle_dit_plain_d_internal. repeats = " << repeats << ", radix = " << radix << ", stride = " << stride << ", length = " << length << std::endl;
    for (size_t i = 0; i < repeats; i++)
    {
        fft_2d_complex_column_twiddle_dit_plain_d_internal<radix,forward>
                (data_in  + 2*i*radix*stride*length,
                 data_out + 2*i*radix*stride*length,
                 step_info.twiddle_factors,
                 stride, length);

    }
}


////////////////////////////////////// Row-wise ////////////////////////////////////////////////

// NOTE this is exactly like fft_1d_complex_twiddle_dit_plain_d_internal
template<size_t radix, bool forward>
    inline __attribute__((always_inline)) void fft_2d_complex_row_twiddle_dit_plain_d_internal(const double *data_in, double *data_out, const double *twiddle_factors, size_t stride)
{
    double x_temp_in[2*radix];
    double x_temp_out[2*radix];
    double twiddle_temp[2*radix];

    for (size_t k = 0; k < stride; k++)
    {
        // Copy input data (squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            x_temp_in[2*j + 0] = data_in[2*k + 2*j*stride + 0];
            x_temp_in[2*j + 1] = data_in[2*k + 2*j*stride + 1];
        }

        // Copy twiddle factors (squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            twiddle_temp[2*j + 0] = twiddle_factors[2*k + 2*j*stride + 0];
            twiddle_temp[2*j + 1] = twiddle_factors[2*k + 2*j*stride + 1];
        }

        multiply_twiddle<radix,forward>(x_temp_in, x_temp_in, twiddle_temp);

        multiply_coeff<radix,forward>(x_temp_in, x_temp_out);

        // Copy input data (un-squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            data_out[2*k + 2*j*stride + 0] = x_temp_out[2*j + 0];
            data_out[2*k + 2*j*stride + 1] = x_temp_out[2*j + 1];
        }

        /*if (radix == 2)
        {
            std::cout << "x_temp_in  = (" << x_temp_in[0] << ", " << x_temp_in[1] << "), (" << x_temp_in[2] << ", " << x_temp_in[3] << ")" << std::endl;
            std::cout << "x_temp_out = (" << x_temp_out[0] << ", " << x_temp_out[1] << "), (" << x_temp_out[2] << ", " << x_temp_out[3] << ")" << std::endl;
        }*/
    }
}

template<size_t radix, bool forward>
    void fft_2d_complex_row_twiddle_dit_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    assert(step_info.forward == forward);

    size_t stride = step_info.stride;
    size_t repeats = step_info.repeats;
    size_t rows = step_info.size;

    std::cout << "fft_2d_complex_row_twiddle_dit_plain_d. repeats = " << repeats << ", radix = " << radix << ", stride = " << stride << ", rows = " << rows << std::endl;
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < repeats; j++)
        {
            fft_2d_complex_row_twiddle_dit_plain_d_internal<radix,forward>
                    (data_in  + 2*i*repeats*radix*stride + 2*j*radix*stride,
                     data_out + 2*i*repeats*radix*stride + 2*j*radix*stride,
                     step_info.twiddle_factors,
                     stride);
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
template void fft_2d_complex_column_twiddle_dit_plain_d<7, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_column_twiddle_dit_plain_d<7, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_2d_complex_row_twiddle_dit_plain_d<2, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_row_twiddle_dit_plain_d<2, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_row_twiddle_dit_plain_d<3, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_row_twiddle_dit_plain_d<3, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_row_twiddle_dit_plain_d<4, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_row_twiddle_dit_plain_d<4, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_row_twiddle_dit_plain_d<5, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_row_twiddle_dit_plain_d<5, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_row_twiddle_dit_plain_d<7, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_row_twiddle_dit_plain_d<7, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

/*
template void fft_1d_complex_column_twiddle_dif_plain_d<2, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_column_twiddle_dif_plain_d<2, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_column_twiddle_dif_plain_d<3, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_column_twiddle_dif_plain_d<3, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_column_twiddle_dif_plain_d<4, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_column_twiddle_dif_plain_d<4, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_column_twiddle_dif_plain_d<5, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_column_twiddle_dif_plain_d<5, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_column_twiddle_dif_plain_d<7, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_column_twiddle_dif_plain_d<7, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
*/
