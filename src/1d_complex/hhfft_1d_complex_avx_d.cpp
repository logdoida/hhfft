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

#define ENABLE_SSE2
#define ENABLE_AVX
#include "hhfft_1d_complex_common_d.h"

using namespace hhfft;

template<size_t radix, bool forward> inline void multiply_coeff(const ComplexD *x_in, ComplexD *x_out)
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

    // Use temporary storage. This is needed (as usually is) if x_in == x_out
    ComplexD x_temp_in[radix];
    for (size_t j = 0; j < radix; j++)
    {
        x_temp_in[j] = x_in[j];
    }

    for (size_t i = 0; i < radix; i++)
    {
        x_out[i] = load(0.0, 0.0);
        for (size_t j = 0; j < radix; j++)
        {
            ComplexD w = load128(coeff + 2*radix*i + 2*j);

            x_out[i] = x_out[i] + mul_w<forward>(x_temp_in[j], w);
        }
    }
}

template<size_t radix, bool forward> inline void multiply_twiddle(const ComplexD *x_in, ComplexD *x_out, const ComplexD *twiddle_factors)
{
    // It is assumed that first twiddle factors are always (1 + 0i)
    x_out[0] = x_in[0];

    // Read in the values used in this step and multiply them with twiddle factors
    for (size_t j = 1; j < radix; j++)
    {
        ComplexD x = x_in[j];
        ComplexD w = twiddle_factors[j];

        x_out[j] = mul_w<forward>(x, w);
    }
}

template<size_t radix, StrideType stride_type, bool forward>
    inline void fft_1d_complex_avx_d_internal(const double *data_in, double *data_out, size_t stride)
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


template<size_t radix, StrideType stride_type, bool forward>
    inline void fft_1d_complex_twiddle_dit_avx_d_internal(const double *data_in, double *data_out, const double *twiddle_factors, size_t stride)
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

template<size_t radix, StrideType stride_type, bool forward>
    inline void fft_1d_complex_twiddle_dif_avx_d_internal(const double *data_in, double *data_out, const double *twiddle_factors, size_t stride)
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
    void fft_1d_complex_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    size_t stride = get_stride<stride_type>(step_info.stride);
    size_t repeats = step_info.repeats;

    for (size_t i = 0; i < repeats; i++)
    {
        fft_1d_complex_avx_d_internal<radix,stride_type, forward>
                (data_in + 2*i*radix*stride, data_out + 2*i*radix*stride, stride);
    }
}

template<size_t radix, StrideType stride_type, bool forward>
    void fft_1d_complex_twiddle_dit_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    assert(step_info.forward == forward);

    size_t stride = get_stride<stride_type>(step_info.stride);
    size_t repeats = step_info.repeats;

    for (size_t i = 0; i < repeats; i++)
    {
        fft_1d_complex_twiddle_dit_avx_d_internal<radix,stride_type,forward>
                (data_in + 2*i*radix*stride, data_out + 2*i*radix*stride, step_info.twiddle_factors, stride);
    }
}

template<size_t radix, StrideType stride_type, bool forward>
    void fft_1d_complex_twiddle_dif_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    assert(step_info.forward == forward);

    size_t stride = get_stride<stride_type>(step_info.stride);
    size_t repeats = step_info.repeats;

    for (size_t i = 0; i < repeats; i++)
    {
        fft_1d_complex_twiddle_dif_avx_d_internal<radix,stride_type,forward>
                (data_in + 2*i*radix*stride, data_out + 2*i*radix*stride, step_info.twiddle_factors + 2*i*radix, stride);
    }
}


// Instantiations of the functions defined in this class
template void fft_1d_complex_avx_d<2, StrideType::StrideN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_avx_d<2, StrideType::StrideN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_avx_d<3, StrideType::StrideN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_avx_d<3, StrideType::StrideN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_avx_d<4, StrideType::StrideN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_avx_d<4, StrideType::StrideN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_avx_d<5, StrideType::StrideN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_avx_d<5, StrideType::StrideN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_avx_d<7, StrideType::StrideN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_avx_d<7, StrideType::StrideN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_1d_complex_twiddle_dit_avx_d<2, StrideType::StrideN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_avx_d<2, StrideType::StrideN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_avx_d<3, StrideType::StrideN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_avx_d<3, StrideType::StrideN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_avx_d<4, StrideType::StrideN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_avx_d<4, StrideType::StrideN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_avx_d<5, StrideType::StrideN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_avx_d<5, StrideType::StrideN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_avx_d<7, StrideType::StrideN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_avx_d<7, StrideType::StrideN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_1d_complex_twiddle_dif_avx_d<2, StrideType::StrideN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dif_avx_d<2, StrideType::StrideN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dif_avx_d<3, StrideType::StrideN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dif_avx_d<3, StrideType::StrideN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dif_avx_d<4, StrideType::StrideN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dif_avx_d<4, StrideType::StrideN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dif_avx_d<5, StrideType::StrideN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dif_avx_d<5, StrideType::StrideN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dif_avx_d<7, StrideType::StrideN, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dif_avx_d<7, StrideType::StrideN, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
