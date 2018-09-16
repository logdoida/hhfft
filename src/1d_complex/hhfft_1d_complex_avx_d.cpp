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

#include "../common/hhfft_1d_complex_avx_common_d.h"

using namespace hhfft;

// In-place reordering "swap"
void fft_1d_complex_reorder_in_place_avx_d(const double *, double *data_out,const hhfft::StepInfo<double> &step_info)
{
    size_t n = step_info.reorder_table_inplace_size;
    uint32_t *reorder_table = step_info.reorder_table_inplace;

    for (size_t i = 0; i < n; i++)
    {
        size_t ind1 = i + 1; // First one has been omitted!
        size_t ind2 = reorder_table[i];

        // Swap two doubles at a time
        ComplexD x_in0 = load_D(data_out + 2*ind1);
        ComplexD x_in1 = load_D(data_out + 2*ind2);
        store_D(x_in0, data_out + 2*ind2);
        store_D(x_in1, data_out + 2*ind1);
    }
}

template<bool scale> void fft_1d_complex_reorder_avx_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info)
{
    size_t n = step_info.radix*step_info.repeats;
    uint32_t *reorder_table = step_info.reorder_table;

    // Needed only in ifft. Equal to 1/N
    double k = step_info.norm_factor;
    ComplexD k128 = broadcast64_D(k);

    for (size_t i = 0; i < n; i++)
    {
        size_t i2 = reorder_table[i];
        if (scale)
        {
            ComplexD x_in = k128*load_D(data_in + 2*i2);
            store_D(x_in, data_out + 2*i);
        } else
        {
            ComplexD x_in = load_D(data_in + 2*i2);
            store_D(x_in, data_out + 2*i);
        }
    }
}

// To be used when stride is 1.
template<size_t radix>
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
                size_t ind0 = i*radix + j;
                size_t ind1 = i*radix + radix + j;
                x_temp_in[j] = load_two_128_D2(data_in + 2*ind0, data_in + 2*ind1);
            }

            // Multiply with coefficients
            multiply_coeff_D2<radix,true>(x_temp_in, x_temp_out);

            // Save output to two memory locations.
            for (size_t j = 0; j < radix; j++)
            {
                size_t ind0 = i*radix + j;
                size_t ind1 = i*radix + radix + j;
                store_two_128_D2(x_temp_out[j], data_out + 2*ind0, data_out + 2*ind1);
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
            size_t ind = i*radix + j;
            x_temp_in[j] = load_D(data_in + 2*ind);
        }

        // Multiply with coefficients
        multiply_coeff_D<radix,true>(x_temp_in, x_temp_out);

        // Copy output data (un-squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            store_D(x_temp_out[j], data_out + 2*i*radix + 2*j);
        }
    }
}

// To be used when stride is 1 and reordering is needed
template<size_t radix, bool forward>
    inline __attribute__((always_inline)) void fft_1d_complex_avx_d_internal_stride1_reorder(const double *data_in, double *data_out, size_t repeats, const hhfft::StepInfo<double> &step_info)
{
    size_t i = 0;
    uint32_t *reorder_table = step_info.reorder_table;
    double k = step_info.norm_factor;
    size_t n = repeats*radix;

    // First use 256-bit variables
    {
        ComplexD2 x_temp_in[radix];
        ComplexD2 x_temp_out[radix];
        for (i = 0; i+1 < repeats; i+=2)
        {
            // Copy input data from two memory locations.
            for (size_t j = 0; j < radix; j++)
            {
                size_t i2 = i*radix + j;
                size_t ind0 = reorder_table[i2];
                size_t ind1 = reorder_table[i2 + radix];

                if (forward)
                {
                    x_temp_in[j] = load_two_128_D2(data_in + 2*ind0, data_in + 2*ind1);
                } else
                {
                    if (i2 > 0)
                    {
                        ind0 = n - ind0;
                    }
                    ind1 = n - ind1;
                    x_temp_in[j] = k*load_two_128_D2(data_in + 2*ind0, data_in + 2*ind1);
                }
            }

            // Multiply with coefficients
            multiply_coeff_D2<radix,true>(x_temp_in, x_temp_out);

            // Save output to two memory locations.
            for (size_t j = 0; j < radix; j++)
            {
                size_t ind0 = i*radix + j;
                size_t ind1 = i*radix + radix + j;

                store_two_128_D2(x_temp_out[j], data_out + 2*ind0, data_out + 2*ind1);
            }
        }
    }

    // Then, if necessary, use 128-bit variables
    if (i < repeats)
    {
        ComplexD x_temp_in[radix];
        ComplexD x_temp_out[radix];

        // Copy input data taking reordering into account
        for (size_t j = 0; j < radix; j++)
        {
            size_t i2 = i*radix + j;
            size_t ind = reorder_table[i2];

            if (forward)
            {
                x_temp_in[j] = load_D(data_in + 2*ind);
            } else
            {
                if (i2 > 0)
                {
                    ind = n - ind;
                }
                x_temp_in[j] = k*load_D(data_in + 2*ind);
            }
        }

        // Multiply with coefficients
        multiply_coeff_D<radix,true>(x_temp_in, x_temp_out);

        // Save output to two memory locations.
        for (size_t j = 0; j < radix; j++)
        {
            store_D(x_temp_out[j], data_out + 2*i*radix + 2*j);
        }
    }
}

template<size_t radix, SizeType stride_type>
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
                x_temp_in[j] = load_D2(data_in + 2*k + 2*j*stride);
            }

            // Multiply with coefficients
            multiply_coeff_D2<radix,true>(x_temp_in, x_temp_out);

            // Copy output data (un-squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                store_D2(x_temp_out[j], data_out + 2*k + 2*j*stride);
            }
        }
    }

    // NOTE slightly more performance could be gain by this
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
            x_temp_in[j] = load_D(data_in + 2*k + 2*j*stride);
        }

        // Multiply with coefficients
        multiply_coeff_D<radix,true>(x_temp_in, x_temp_out);

        // Copy output data (un-squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            store_D(x_temp_out[j], data_out + 2*k + 2*j*stride);
        }    
    }
}


template<size_t radix, SizeType stride_type>
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
                x_temp_in[j] = load_D2(data_in + 2*k + 2*j*stride);
            }

            // Copy twiddle factors (squeeze)
            for (size_t j = 0; j < radix; j++)
            {                
                // NOTE aligned load can be used here if stride%2 == 0
                twiddle_temp[j] = load_D2(twiddle_factors + 2*k + 2*j*stride);
            }

            multiply_twiddle_D2<radix,true>(x_temp_in, x_temp_in, twiddle_temp);
            multiply_coeff_D2<radix,true>(x_temp_in, x_temp_out);

            // Copy output data (un-squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                store_D2(x_temp_out[j], data_out + 2*k + 2*j*stride);
            }
        }
    }

    //NOTE slightly more performance could be gain by this
    //if (stride_type is divisible by 2)
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
            x_temp_in[j] = load_D(data_in + 2*k + 2*j*stride);
        }

        // Copy twiddle factors (squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            twiddle_temp[j] = load_D(twiddle_factors + 2*k + 2*j*stride);
        }

        multiply_twiddle_D<radix,true>(x_temp_in, x_temp_in, twiddle_temp);
        multiply_coeff_D<radix,true>(x_temp_in, x_temp_out);

        // Copy output data (un-squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            store_D(x_temp_out[j], data_out + 2*k + 2*j*stride);
        }
    }
}

template<size_t radix, SizeType stride_type>
    void fft_1d_complex_avx_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info)
{
    size_t stride = get_size<stride_type>(step_info.stride);
    size_t repeats = step_info.repeats;

    // Implementation for stride == 1
    if (stride_type == SizeType::Size1)
    {
        fft_1d_complex_avx_d_internal_stride1<radix>(data_in, data_out, repeats);
        return;
    }

    // Other stride types
    for (size_t i = 0; i < repeats; i++)
    {
        fft_1d_complex_avx_d_internal<radix,stride_type>
                (data_in + 2*i*radix*stride, data_out + 2*i*radix*stride, stride);
    }    
}

// Reordering and fft-step combined
template<size_t radix, SizeType stride_type, bool forward>
    void fft_1d_complex_reorder2_avx_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info)
{
    size_t repeats = step_info.repeats;

    // Only for stride 1!
    assert(stride_type == SizeType::Size1);

    fft_1d_complex_avx_d_internal_stride1_reorder<radix, forward>(data_in, data_out, repeats, step_info);
}

template<size_t radix, SizeType stride_type>
    void fft_1d_complex_twiddle_dit_avx_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info)
{
    size_t stride = get_size<stride_type>(step_info.stride);
    size_t repeats = step_info.repeats;

    for (size_t i = 0; i < repeats; i++)
    {
        fft_1d_complex_twiddle_dit_avx_d_internal<radix,stride_type>
                (data_in + 2*i*radix*stride, data_out + 2*i*radix*stride, step_info.twiddle_factors, stride);
    }
}

void fft_1d_complex_convolution_avx_d(const double *data_in0, const double *data_in1, double *data_out, size_t n)
{
    size_t i;

    // First use 2*256-bit variables (real and complex parts are calculated separately)
    for (i = 0; i+6 < 2*n; i+=8)
    {
        ComplexD4S x_in0 = load512s_D4S(data_in0 + i);
        ComplexD4S x_in1 = load512s_D4S(data_in1 + i);
        ComplexD4S x_out = mul_D4S(x_in0, x_in1);
        store512s_D4S(x_out, data_out + i);
    }

    // Then, if necassery, use 128-bit variables
    for (; i < 2*n; i+=2)
    {
        ComplexD x_in0 = load_D(data_in0 + i);
        ComplexD x_in1 = load_D(data_in1 + i);
        ComplexD x_out = mul_D(x_in0, x_in1);
        store_D(x_out, data_out + i);
    }
}

// fft for small sizes (1,2,3,4,5,7,8) where only one level is needed
template<size_t n, bool forward> void fft_1d_complex_1level_avx_d(const double *data_in, double *data_out, const hhfft::StepInfo<double> &)
{
    ComplexD k = broadcast64_D(1.0/n);

    if (n == 1)
    {
        ComplexD x = load_D(data_in);
        store_D(x, data_out);
    } else
    {
        ComplexD x_temp_in[n];
        ComplexD x_temp_out[n];

        // Copy input data
        for (size_t i = 0; i < n; i++)
        {
            x_temp_in[i] = load_D(data_in + 2*i);
        }

        // Multiply with coefficients
        multiply_coeff_D<n,forward>(x_temp_in, x_temp_out);

        // Copy output data
        for (size_t i = 0; i < n; i++)
        {
            if(forward)
                store_D(x_temp_out[i], data_out + 2*i);
            else
                store_D(k*x_temp_out[i], data_out + 2*i);
        }
    }
}


// Instantiations of the functions defined in this class
template void fft_1d_complex_reorder_avx_d<false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder_avx_d<true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template void fft_1d_complex_reorder2_avx_d<2, SizeType::Size1, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_avx_d<2, SizeType::Size1, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_avx_d<3, SizeType::Size1, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_avx_d<3, SizeType::Size1, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_avx_d<4, SizeType::Size1, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_avx_d<4, SizeType::Size1, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_avx_d<5, SizeType::Size1, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_avx_d<5, SizeType::Size1, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_avx_d<6, SizeType::Size1, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_avx_d<6, SizeType::Size1, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_avx_d<7, SizeType::Size1, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_avx_d<7, SizeType::Size1, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_avx_d<8, SizeType::Size1, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_avx_d<8, SizeType::Size1, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template void fft_1d_complex_avx_d<2, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_avx_d<3, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_avx_d<4, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_avx_d<5, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_avx_d<6, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_avx_d<7, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_avx_d<8, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template void fft_1d_complex_avx_d<2, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_avx_d<3, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_avx_d<4, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_avx_d<5, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_avx_d<6, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_avx_d<7, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_avx_d<8, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template void fft_1d_complex_twiddle_dit_avx_d<2, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_avx_d<3, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_avx_d<4, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_avx_d<5, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_avx_d<6, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_avx_d<7, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_avx_d<8, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template void fft_1d_complex_twiddle_dit_avx_d<2, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_avx_d<3, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_avx_d<4, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_avx_d<5, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_avx_d<6, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_avx_d<7, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_avx_d<8, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template void fft_1d_complex_1level_avx_d<1, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_avx_d<2, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_avx_d<3, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_avx_d<4, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_avx_d<5, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_avx_d<6, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_avx_d<7, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_avx_d<8, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_avx_d<1, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_avx_d<2, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_avx_d<3, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_avx_d<4, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_avx_d<5, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_avx_d<6, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_avx_d<7, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_avx_d<8, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

