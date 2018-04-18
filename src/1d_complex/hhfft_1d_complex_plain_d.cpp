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

#include "hhfft_1d_complex_common_d.h"

using namespace hhfft;

template<size_t radix, bool forward> inline void multiply_coeff(const double *x_in, double *x_out)
{
    const double *coeff = nullptr;

    if (radix == 2)
    {
        coeff = coeff_radix_2;
    } else if (radix == 3)
    {
        coeff = coeff_radix_3;
    } else if (radix == 4)
    {
        coeff = coeff_radix_4;
    } else if (radix == 5)
    {
        coeff = coeff_radix_5;
    } else if (radix == 7)
    {
        coeff = coeff_radix_7;
    }


    // Use temporary storage. This is needed in (as usually is) if x_in == x_out
    double x_temp_in[2*radix];
    for (size_t j = 0; j < radix; j++)
    {
        x_temp_in[2*j + 0] = x_in[2*j + 0];
        x_temp_in[2*j + 1] = x_in[2*j + 1];
    }

    for (size_t i = 0; i < radix; i++)
    {
        x_out[2*i + 0] = 0;
        x_out[2*i + 1] = 0;
        for (size_t j = 0; j < radix; j++)
        {
            double a = coeff[2*radix*i + 2*j + 0];
            double b = forward ? coeff[2*radix*i + 2*j + 1]: -coeff[2*radix*i + 2*j + 1];
            x_out[2*i + 0] += a*x_temp_in[2*j + 0] - b*x_temp_in[2*j + 1];
            x_out[2*i + 1] += b*x_temp_in[2*j + 0] + a*x_temp_in[2*j + 1];
        }
    }
}

template<size_t radix, bool forward> inline void multiply_twiddle(const double *x_in, double *x_out, const double *twiddle_factors)
{
    // It is assumed that first twiddle factors are always (1 + 0i)
    x_out[0] = x_in[0];
    x_out[1] = x_in[1];

    // Read in the values used in this step and multiply them with twiddle factors
    for (size_t j = 1; j < radix; j++)
    {
        double x_r = x_in[2*j + 0];
        double x_i = x_in[2*j + 1];
        double w_r = twiddle_factors[2*j + 0];
        double w_i = twiddle_factors[2*j + 1];

        if (forward == 1)
        {
            x_out[2*j + 0] = w_r*x_r - w_i*x_i;
            x_out[2*j + 1] = w_i*x_r + w_r*x_i;
        } else
        {
            x_out[2*j + 0] =  w_r*x_r + w_i*x_i;
            x_out[2*j + 1] = -w_i*x_r + w_r*x_i;
        }
    }
}

template<size_t radix, StrideType stride_type, bool forward>
    inline void fft_1d_complex_plain_d_internal(const double *data_in, double *data_out, size_t stride)
{
    double x_temp_in[2*radix];
    double x_temp_out[2*radix];

    for (size_t k = 0; k < stride; k++)
    {
        // Copy input data (squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            x_temp_in[2*j + 0] = data_in[2*k + 2*j*stride + 0];
            x_temp_in[2*j + 1] = data_in[2*k + 2*j*stride + 1];
        }

        // Multiply with coefficients
        multiply_coeff<radix,forward>(x_temp_in, x_temp_out);

        // Copy input data (un-squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            data_out[2*k + 2*j*stride + 0] = x_temp_out[2*j + 0];
            data_out[2*k + 2*j*stride + 1] = x_temp_out[2*j + 1];
        }
    }
}


template<size_t radix, StrideType stride_type, bool forward>
    inline void fft_1d_complex_twiddle_dit_plain_d_internal(const double *data_in, double *data_out, const double *twiddle_factors, size_t stride)
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
    }    
}

template<size_t radix, StrideType stride_type, bool forward>
    inline void fft_1d_complex_twiddle_dif_plain_d_internal(const double *data_in, double *data_out, const double *twiddle_factors, size_t stride)
{
    double x_temp_in[2*radix];
    double x_temp_out[2*radix];
    double twiddle_temp[2*radix];

    // Copy twiddle factors
    for (size_t j = 0; j < radix; j++)
    {
        twiddle_temp[2*j + 0] = twiddle_factors[2*j + 0];
        twiddle_temp[2*j + 1] = twiddle_factors[2*j + 1];
    }

    for (size_t k = 0; k < stride; k++)
    {
        // Copy input data (squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            x_temp_in[2*j + 0] = data_in[2*k + 2*j*stride + 0];
            x_temp_in[2*j + 1] = data_in[2*k + 2*j*stride + 1];
        }

        multiply_twiddle<radix,forward>(x_temp_in, x_temp_in, twiddle_temp);

        multiply_coeff<radix,forward>(x_temp_in, x_temp_out);

        // Copy input data (un-squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            data_out[2*k + 2*j*stride + 0] = x_temp_out[2*j + 0];
            data_out[2*k + 2*j*stride + 1] = x_temp_out[2*j + 1];
        }
    }    
}

// This function can help compiler to optimze the code
template<StrideType stride_type> size_t get_stride(size_t stride)
{
    if (stride_type == StrideType::Stride1)
    {
        return 1;
    } else if (stride_type == StrideType::Stride2)
    {
        return 2;
    } else if (stride_type == StrideType::Stride4)
    {
        return 4;
    } else
    {
        return stride;
    }
}

template<size_t radix, StrideType stride_type, bool forward>
    void fft_1d_complex_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    size_t stride = get_stride<stride_type>(step_info.stride);
    size_t repeats = step_info.repeats;

    for (size_t i = 0; i < repeats; i++)
    {
        fft_1d_complex_plain_d_internal<radix,stride_type, forward>
                (data_in + 2*i*radix*stride, data_out + 2*i*radix*stride, stride);
    }
}

template<size_t radix, StrideType stride_type, bool forward>
    void fft_1d_complex_twiddle_dit_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    assert(step_info.forward == forward);

    size_t stride = get_stride<stride_type>(step_info.stride);
    size_t repeats = step_info.repeats;

    for (size_t i = 0; i < repeats; i++)
    {
        fft_1d_complex_twiddle_dit_plain_d_internal<radix,stride_type,forward>
                (data_in + 2*i*radix*stride, data_out + 2*i*radix*stride, step_info.twiddle_factors, stride);
    }
}

template<size_t radix, StrideType stride_type, bool forward>
    void fft_1d_complex_twiddle_dif_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{    
    assert(step_info.forward == forward);

    size_t stride = get_stride<stride_type>(step_info.stride);
    size_t repeats = step_info.repeats;

    for (size_t i = 0; i < repeats; i++)
    {
        fft_1d_complex_twiddle_dif_plain_d_internal<radix,stride_type,forward>
                (data_in + 2*i*radix*stride, data_out + 2*i*radix*stride, step_info.twiddle_factors + 2*i*radix, stride);
    }
}


// Instantiations of the functions defined in this class
template void fft_1d_complex_plain_d<2, StrideType::StrideN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_plain_d<2, StrideType::StrideN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_plain_d<3, StrideType::StrideN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_plain_d<3, StrideType::StrideN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_plain_d<4, StrideType::StrideN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_plain_d<4, StrideType::StrideN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_plain_d<5, StrideType::StrideN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_plain_d<5, StrideType::StrideN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_plain_d<7, StrideType::StrideN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_plain_d<7, StrideType::StrideN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_1d_complex_twiddle_dit_plain_d<2, StrideType::StrideN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_plain_d<2, StrideType::StrideN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_plain_d<3, StrideType::StrideN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_plain_d<3, StrideType::StrideN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_plain_d<4, StrideType::StrideN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_plain_d<4, StrideType::StrideN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_plain_d<5, StrideType::StrideN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_plain_d<5, StrideType::StrideN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_plain_d<7, StrideType::StrideN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_plain_d<7, StrideType::StrideN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_1d_complex_twiddle_dif_plain_d<2, StrideType::StrideN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dif_plain_d<2, StrideType::StrideN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dif_plain_d<3, StrideType::StrideN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dif_plain_d<3, StrideType::StrideN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dif_plain_d<4, StrideType::StrideN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dif_plain_d<4, StrideType::StrideN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dif_plain_d<5, StrideType::StrideN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dif_plain_d<5, StrideType::StrideN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dif_plain_d<7, StrideType::StrideN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dif_plain_d<7, StrideType::StrideN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

