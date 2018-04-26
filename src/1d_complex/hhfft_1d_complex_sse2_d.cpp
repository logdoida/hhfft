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

template<size_t radix, SizeType stride_type, bool forward>
    inline __attribute__((always_inline)) void fft_1d_complex_sse2_d_internal(const double *data_in, double *data_out, size_t stride)
{
    ComplexD x_temp_in[radix];
    ComplexD x_temp_out[radix];

    for (size_t k = 0; k < stride; k++)
    {
        // Copy input data (squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            x_temp_in[j] = load128(data_in + 2*k + 2*j*stride);
        }

        // Multiply with coefficients        
        multiply_coeff<radix,forward>(x_temp_in, x_temp_out);

        // Copy ouput data (un-squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            store(x_temp_out[j], data_out + 2*k + 2*j*stride);
        }
    }
}


template<size_t radix, SizeType stride_type, bool forward>
    inline __attribute__((always_inline)) void fft_1d_complex_twiddle_dit_sse2_d_internal(const double *data_in, double *data_out, const double *twiddle_factors, size_t stride)
{
    ComplexD x_temp_in[radix];
    ComplexD x_temp_out[radix];
    ComplexD twiddle_temp[radix];

    for (size_t k = 0; k < stride; k++)
    {
        // Copy input data (squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            x_temp_in[j] = load128(data_in + 2*k + 2*j*stride);
        }

        // Copy twiddle factors (squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            twiddle_temp[j] = load128(twiddle_factors + 2*k + 2*j*stride);
        }

        multiply_twiddle<radix,forward>(x_temp_in, x_temp_in, twiddle_temp);
        multiply_coeff<radix,forward>(x_temp_in, x_temp_out);

        // Copy ouput data (un-squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            store(x_temp_out[j], data_out + 2*k + 2*j*stride);
        }
    }
}

template<size_t radix, SizeType stride_type, bool forward>
    inline __attribute__((always_inline)) void fft_1d_complex_twiddle_dif_sse2_d_internal(const double *data_in, double *data_out, const double *twiddle_factors, size_t stride)
{
    ComplexD x_temp_in[radix];
    ComplexD x_temp_out[radix];
    ComplexD twiddle_temp[radix];

    // Copy twiddle factors (squeeze)
    for (size_t j = 0; j < radix; j++)
    {
        twiddle_temp[j] = load128(twiddle_factors + 2*j);
    }

    for (size_t k = 0; k < stride; k++)
    {
        // Copy input data (squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            x_temp_in[j] = load128(data_in + 2*k + 2*j*stride);
        }

        multiply_twiddle<radix,forward>(x_temp_in, x_temp_in, twiddle_temp);
        multiply_coeff<radix,forward>(x_temp_in, x_temp_out);

        // Copy ouput data (un-squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            store(x_temp_out[j], data_out + 2*k + 2*j*stride);
        }
    }
}

template<size_t radix, SizeType stride_type, bool forward>
    void fft_1d_complex_sse2_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    size_t stride = get_size<stride_type>(step_info.stride);
    size_t repeats = step_info.repeats;

    for (size_t i = 0; i < repeats; i++)
    {
        fft_1d_complex_sse2_d_internal<radix,stride_type, forward>
                (data_in + 2*i*radix*stride, data_out + 2*i*radix*stride, stride);
    }
}

template<size_t radix, SizeType stride_type, bool forward>
    void fft_1d_complex_twiddle_dit_sse2_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    assert(step_info.forward == forward);

    size_t stride = get_size<stride_type>(step_info.stride);
    size_t repeats = step_info.repeats;

    for (size_t i = 0; i < repeats; i++)
    {
        fft_1d_complex_twiddle_dit_sse2_d_internal<radix,stride_type,forward>
                (data_in + 2*i*radix*stride, data_out + 2*i*radix*stride, step_info.twiddle_factors, stride);
    }
}

template<size_t radix, SizeType stride_type, bool forward>
    void fft_1d_complex_twiddle_dif_sse2_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    assert(step_info.forward == forward);

    size_t stride = get_size<stride_type>(step_info.stride);
    size_t repeats = step_info.repeats;

    for (size_t i = 0; i < repeats; i++)
    {
        fft_1d_complex_twiddle_dif_sse2_d_internal<radix,stride_type,forward>
                (data_in + 2*i*radix*stride, data_out + 2*i*radix*stride, step_info.twiddle_factors + 2*i*radix, stride);
    }
}


// Instantiations of the functions defined in this class
template void fft_1d_complex_sse2_d<2, SizeType::SizeN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_sse2_d<2, SizeType::SizeN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_sse2_d<3, SizeType::SizeN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_sse2_d<3, SizeType::SizeN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_sse2_d<4, SizeType::SizeN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_sse2_d<4, SizeType::SizeN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_sse2_d<5, SizeType::SizeN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_sse2_d<5, SizeType::SizeN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_sse2_d<7, SizeType::SizeN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_sse2_d<7, SizeType::SizeN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_1d_complex_twiddle_dit_sse2_d<2, SizeType::SizeN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_sse2_d<2, SizeType::SizeN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_sse2_d<3, SizeType::SizeN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_sse2_d<3, SizeType::SizeN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_sse2_d<4, SizeType::SizeN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_sse2_d<4, SizeType::SizeN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_sse2_d<5, SizeType::SizeN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_sse2_d<5, SizeType::SizeN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_sse2_d<7, SizeType::SizeN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_sse2_d<7, SizeType::SizeN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_1d_complex_twiddle_dif_sse2_d<2, SizeType::SizeN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dif_sse2_d<2, SizeType::SizeN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dif_sse2_d<3, SizeType::SizeN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dif_sse2_d<3, SizeType::SizeN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dif_sse2_d<4, SizeType::SizeN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dif_sse2_d<4, SizeType::SizeN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dif_sse2_d<5, SizeType::SizeN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dif_sse2_d<5, SizeType::SizeN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dif_sse2_d<7, SizeType::SizeN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dif_sse2_d<7, SizeType::SizeN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
