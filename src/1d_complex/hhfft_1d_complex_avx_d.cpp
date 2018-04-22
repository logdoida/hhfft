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

#define ENABLE_COMPLEX_D
#define ENABLE_COMPLEX_D2
#include "hhfft_1d_complex_common_d.h"

using namespace hhfft;

template<size_t radix, bool forward> inline void multiply_coeff(const ComplexD2 *x_in, ComplexD2 *x_out)
{
    // Implementation for radix = 2
    if (radix == 2)
    {
        x_out[0] = x_in[0] + x_in[1];
        x_out[1] = x_in[0] - x_in[1];
        return;
    }

    // Implementation for radix = 3
    if (radix == 3)
    {
        const ComplexD2 k0 = broadcast64_D2(0.5);
        const ComplexD2 k1 = broadcast64_D2(0.5*sqrt(3.0));
        ComplexD2 t0 = x_in[1] + x_in[2];
        ComplexD2 t1 = x_in[0] - k0*t0;
        ComplexD2 t2 = mul_i(k1*(x_in[1] - x_in[2]));

        x_out[0] = x_in[0] + t0;
        if (forward)
        {
            x_out[1] = t1 - t2;
            x_out[2] = t1 + t2;
        } else
        {
            x_out[1] = t1 + t2;
            x_out[2] = t1 - t2;
        }
        return;
    }

    // Implementation for radix = 4
    if (radix == 4)
    {
        ComplexD2 t0 = x_in[0] + x_in[2];
        ComplexD2 t1 = x_in[1] + x_in[3];
        ComplexD2 t2 = x_in[0] - x_in[2];
        ComplexD2 t3;
        if (forward)
            t3 = mul_i(x_in[3] - x_in[1]);
        else
            t3 = mul_i(x_in[1] - x_in[3]);

        x_out[0] = t0 + t1;
        x_out[1] = t2 + t3;
        x_out[2] = t0 - t1;
        x_out[3] = t2 - t3;
        return;
    }

    // Implementation for radix = 5
    if (radix == 5)
    {
        const ComplexD2 k1 = broadcast64_D2(cos(2.0*M_PI*1.0/5.0));
        const ComplexD2 k2 = broadcast64_D2(sin(2.0*M_PI*1.0/5.0));
        const ComplexD2 k3 = broadcast64_D2(-cos(2.0*M_PI*2.0/5.0));
        const ComplexD2 k4 = broadcast64_D2(sin(2.0*M_PI*2.0/5.0));

        ComplexD2 t0 = x_in[1] + x_in[4];
        ComplexD2 t1 = x_in[2] + x_in[3];
        ComplexD2 t2 = x_in[1] - x_in[4];
        ComplexD2 t3 = x_in[2] - x_in[3];
        ComplexD2 t4 = x_in[0] + k1*t0 - k3*t1;
        ComplexD2 t5 = x_in[0] + k1*t1 - k3*t0;
        ComplexD2 t6 = mul_i(k2*t2 + k4*t3);
        ComplexD2 t7 = mul_i(k4*t2 - k2*t3);

        x_out[0] = x_in[0] + t0 + t1;
        if (forward)
        {
            x_out[1] = t4 - t6;
            x_out[2] = t5 - t7;
            x_out[3] = t5 + t7;
            x_out[4] = t4 + t6;
        }
        else
        {
            x_out[1] = t4 + t6;
            x_out[2] = t5 + t7;
            x_out[3] = t5 - t7;
            x_out[4] = t4 - t6;
        }
        return;
    }

    // Other radices
    const double *coeff = nullptr;

    if (radix == 7)
    {
        coeff = coeff_radix_7;
    }

    // First row is (1,0)
    x_out[0] = x_in[0];
    for (size_t i = 1; i < radix; i++)
    {
        x_out[0] = x_out[0] + x_in[i];
    }

    for (size_t i = 1; i < radix; i++)
    {
        x_out[i] = x_in[0]; // First column is always (1,0)
        for (size_t j = 1; j < radix; j++)
        {
            ComplexD2 w = broadcast128(coeff + 2*radix*i + 2*j);

            x_out[i] = x_out[i] + mul_w<forward>(x_in[j], w);
        }
    }
}

template<size_t radix, bool forward> inline void multiply_twiddle(const ComplexD2 *x_in, ComplexD2 *x_out, const ComplexD2 *twiddle_factors)
{
    // It is assumed that first twiddle factors are always (1 + 0i)
    x_out[0] = x_in[0];

    // Read in the values used in this step and multiply them with twiddle factors
    for (size_t j = 1; j < radix; j++)
    {
        ComplexD2 x = x_in[j];
        ComplexD2 w = twiddle_factors[j];

        x_out[j] = mul_w<forward>(x, w);
    }
}

template<size_t radix, bool forward>
    inline __attribute__((always_inline)) void fft_1d_complex_avx_d_internal_stride1(const double *data_in, double *data_out, size_t repeats)
{
    size_t i = 0;

    // First use 256-bit variables
    {
        ComplexD2 x_temp_in[radix];
        ComplexD2 x_temp_out[radix];
        for (i = 0; i+1 < repeats; i+=2)
        {
            // Copy input data from two memory locations.
            for (size_t j = 0; j < radix; j++)
            {
                x_temp_in[j] = load_two_128(data_in + 2*i*radix + 2*j, data_in + 2*(i+1)*radix + 2*j);
            }

            // Multiply with coefficients
            multiply_coeff<radix,forward>(x_temp_in, x_temp_out);

            // Save output to two memory locations.
            for (size_t j = 0; j < radix; j++)
            {
                store_two_128(x_temp_out[j], data_out + 2*i*radix + 2*j, data_out + 2*(i+1)*radix + 2*j);
            }
        }
    }

    // Then, if necessary, use 128-bit variables
    if (i < repeats)
    {
        ComplexD x_temp_in[radix];
        ComplexD x_temp_out[radix];

        // Copy input data (squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            x_temp_in[j] = load128(data_in + 2*i*radix + 2*j);
        }

        // Multiply with coefficients
        multiply_coeff<radix,forward>(x_temp_in, x_temp_out);

        // Copy ouput data (un-squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            store(x_temp_out[j], data_out + 2*i*radix + 2*j);
        }
    }
}


template<size_t radix, StrideType stride_type, bool forward>
    inline __attribute__((always_inline)) void fft_1d_complex_avx_d_internal(const double *data_in, double *data_out, size_t stride)
{
    size_t k = 0;

    // First use 256-bit variables
    {
        ComplexD2 x_temp_in[radix];
        ComplexD2 x_temp_out[radix];

        for (k = 0; k+1 < stride; k+=2)
        {
            // Copy input data (squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                x_temp_in[j] = load(data_in + 2*k + 2*j*stride);
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

    //TODO
    //if (strid_type is divisible by 2)
    //    return;

    // Then, if necassery, use 128-bit variables
    if (k < stride)
    {
        ComplexD x_temp_in[radix];
        ComplexD x_temp_out[radix];

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
    inline __attribute__((always_inline)) void fft_1d_complex_twiddle_dit_avx_d_internal(const double *data_in, double *data_out, const double *twiddle_factors, size_t stride)
{
    size_t k = 0;

    // First use 256-bit variables
    {
        ComplexD2 x_temp_in[radix];
        ComplexD2 x_temp_out[radix];
        ComplexD2 twiddle_temp[radix];

        for (k = 0; k+1 < stride; k+=2)
        {
            // Copy input data (squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                x_temp_in[j] = load(data_in + 2*k + 2*j*stride);
            }

            // Copy twiddle factors (squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                twiddle_temp[j] = load(twiddle_factors + 2*k + 2*j*stride);
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

    //TODO
    //if (strid_type is divisible by 2)
    //    return;

    // Then, if necassery, use 128-bit variables
    if (k < stride)
    {
        ComplexD x_temp_in[radix];
        ComplexD x_temp_out[radix];
        ComplexD twiddle_temp[radix];

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
    inline __attribute__((always_inline)) void fft_1d_complex_twiddle_dif_avx_d_internal(const double *data_in, double *data_out, const double *twiddle_factors, size_t stride)
{
    size_t k = 0;

    // First use 256-bit variables
    {
        ComplexD2 x_temp_in[radix];
        ComplexD2 x_temp_out[radix];
        ComplexD2 twiddle_temp[radix];

        // Copy twiddle factors (squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            twiddle_temp[j] = broadcast128(twiddle_factors + 2*j);
        }

        for (k = 0; k+1 < stride; k+=2)
        {
            // Copy input data (squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                x_temp_in[j] = load(data_in + 2*k + 2*j*stride);
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

    //TODO
    //if (strid_type is divisible by 2)
    //    return;

    // Then, if necassery, use 128-bit variables
    if (k < stride)
    {
        ComplexD x_temp_in[radix];
        ComplexD x_temp_out[radix];
        ComplexD twiddle_temp[radix];

        // Copy twiddle factors (squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            twiddle_temp[j] = load128(twiddle_factors + 2*j);
        }

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

    // Implementation for stride == 1
    if (stride_type == StrideType::Stride1)
    {
        fft_1d_complex_avx_d_internal_stride1<radix, forward>(data_in, data_out, repeats);

        return;
    }

    // Other stride types
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
template void fft_1d_complex_avx_d<2, StrideType::Stride1, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_avx_d<2, StrideType::Stride1, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_avx_d<3, StrideType::Stride1, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_avx_d<3, StrideType::Stride1, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_avx_d<4, StrideType::Stride1, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_avx_d<4, StrideType::Stride1, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_avx_d<5, StrideType::Stride1, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_avx_d<5, StrideType::Stride1, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_avx_d<7, StrideType::Stride1, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_avx_d<7, StrideType::Stride1, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_1d_complex_twiddle_dit_avx_d<2, StrideType::Stride1, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_avx_d<2, StrideType::Stride1, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_avx_d<3, StrideType::Stride1, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_avx_d<3, StrideType::Stride1, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_avx_d<4, StrideType::Stride1, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_avx_d<4, StrideType::Stride1, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_avx_d<5, StrideType::Stride1, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_avx_d<5, StrideType::Stride1, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_avx_d<7, StrideType::Stride1, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_avx_d<7, StrideType::Stride1, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_1d_complex_twiddle_dif_avx_d<2, StrideType::Stride1, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dif_avx_d<2, StrideType::Stride1, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dif_avx_d<3, StrideType::Stride1, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dif_avx_d<3, StrideType::Stride1, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dif_avx_d<4, StrideType::Stride1, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dif_avx_d<4, StrideType::Stride1, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dif_avx_d<5, StrideType::Stride1, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dif_avx_d<5, StrideType::Stride1, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dif_avx_d<7, StrideType::Stride1, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dif_avx_d<7, StrideType::Stride1, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);


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


// TESTING, how well does it optimize?
/*

*/
