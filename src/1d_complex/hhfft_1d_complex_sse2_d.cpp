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
#include "../raders/raders_sse2_d.h"

using namespace hhfft;

// In-place reordering "swap"
void fft_1d_complex_reorder_in_place_sse2_d(const double *, double *data_out,const hhfft::StepInfo<double> &step_info)
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

template<bool scale> void fft_1d_complex_reorder_sse2_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info)
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

template<RadixType radix_type>
    inline __attribute__((always_inline)) void fft_1d_complex_sse2_d_internal(const double *data_in, double *data_out, double *data_raders, const hhfft::RadersD &raders, size_t stride)
{
    ComplexD x_temp_in[radix_type];
    ComplexD x_temp_out[radix_type];
    size_t radix = get_actual_radix<radix_type>(raders);

    for (size_t k = 0; k < stride; k++)
    {
        // Copy input data (squeeze)
        for (size_t j = 0; j < radix; j++)
        {            
            ComplexD x = load_D(data_in + 2*k + 2*j*stride);
            set_value_D<radix_type>(x_temp_in, data_raders, j, raders, x);
        }

        // Multiply with coefficients        
        multiply_coeff_forward_D<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

        // Copy output data (un-squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            ComplexD x = get_value_D<radix_type>(x_temp_out, data_raders, j, raders);
            store_D(x, data_out + 2*k + 2*j*stride);
        }
    }
}

// To be used when stride is 1 and reordering is needed
template<RadixType radix_type, bool forward>
    inline __attribute__((always_inline)) void fft_1d_complex_sse2_d_internal_stride1_reorder(const double *data_in, double *data_out, size_t repeats, const hhfft::StepInfo<double> &step_info)
{
    const hhfft::RadersD &raders = *step_info.raders;
    size_t radix = get_actual_radix<radix_type>(raders);
    uint32_t *reorder_table = step_info.reorder_table;
    double k = step_info.norm_factor;
    size_t n = repeats*radix;

    ComplexD x_temp_in[radix_type];
    ComplexD x_temp_out[radix_type];

    // Allocate memory for Rader's algorithm if needed
    double *data_raders = allocate_raders_D<radix_type>(raders);

    for (size_t i = 0; i < repeats; i++)
    {
        // Initialize raders data with zeros
        init_coeff_D<radix_type>(data_raders, raders);

        // Copy input data taking reordering into account
        // First index is copied seprately to improve performance
        {
            size_t i2 = i*radix;
            size_t ind = reorder_table[i2];
            if (!forward && i2 > 0)
            {
                ind = n - ind;
            }
            ComplexD x = load_D(data_in + 2*ind);
            set_value_D<radix_type>(x_temp_in, data_raders, 0, raders, x);
        }

        for (size_t j = 1; j < radix; j++)
        {
            size_t i2 = i*radix + j;
            size_t ind = reorder_table[i2];
            if (!forward)
            {
                ind = n - ind;
            }
            ComplexD x = load_D(data_in + 2*ind);
            set_value_D<radix_type>(x_temp_in, data_raders, j, raders, x);
        }

        // Multiply with coefficients
        multiply_coeff_forward_D<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

        // Save output to two memory locations.
        for (size_t j = 0; j < radix; j++)
        {
            ComplexD x = get_value_D<radix_type>(x_temp_out, data_raders, j, raders);
            if (!forward)
            {
                x = k*x;
            }
            store_D(x, data_out + 2*i*radix + 2*j);
        }
    }
}


template<RadixType radix_type>
    inline __attribute__((always_inline)) void fft_1d_complex_twiddle_dit_sse2_d_internal(const double *data_in, double *data_out, const double *twiddle_factors, double *data_raders, const hhfft::RadersD &raders, size_t stride)
{    
    ComplexD x_temp_in[radix_type];
    ComplexD x_temp_out[radix_type];
    ComplexD twiddle_temp[radix_type];
    size_t radix = get_actual_radix<radix_type>(raders);

    for (size_t k = 0; k < stride; k++)
    {
        // Initialize raders data with zeros
        init_coeff_D<radix_type>(data_raders, raders);

        // Copy input data and multiply with twiddle factors
        for (size_t j = 0; j < radix; j++)
        {
            size_t j2 = 2*k + 2*j*stride;
            ComplexD x = load_D(data_in + j2);
            ComplexD w = load_D(twiddle_factors + j2);
            set_value_twiddle_D<radix_type>(x_temp_in, data_raders, twiddle_temp, j, raders, x, w);
        }

        // Multiply with coefficients
        multiply_twiddle_D<radix_type,true>(x_temp_in, x_temp_in, twiddle_temp);
        multiply_coeff_forward_D<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

        // Copy output data (un-squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            size_t j2 = 2*k + 2*j*stride;
            ComplexD x = get_value_D<radix_type>(x_temp_out, data_raders, j, raders);
            store_D(x, data_out + j2);
        }
    }
}

template<RadixType radix_type, SizeType stride_type>
    void fft_1d_complex_sse2_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info)
{    
    size_t stride = get_size<stride_type>(step_info.stride);
    size_t repeats = step_info.repeats;
    const hhfft::RadersD &raders = *step_info.raders;
    size_t radix = get_actual_radix<radix_type>(raders);

    // Allocate memory for Rader's algorithm if needed
    double *data_raders = allocate_raders_D<radix_type>(raders);

    for (size_t i = 0; i < repeats; i++)
    {
        fft_1d_complex_sse2_d_internal<radix_type>
                (data_in + 2*i*radix*stride, data_out + 2*i*radix*stride, data_raders, *step_info.raders, stride);
    }

    // Free temporary memory
    free_raders_D<radix_type>(raders, data_raders);
}

// Reordering and fft-step combined
template<RadixType radix_type, SizeType stride_type, bool forward>
    void fft_1d_complex_reorder2_sse2_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info)
{
    size_t repeats = step_info.repeats;

    // Only for stride 1!
    assert(stride_type == SizeType::Size1);

    fft_1d_complex_sse2_d_internal_stride1_reorder<radix_type, forward>(data_in, data_out, repeats, step_info);
}


template<RadixType radix_type, SizeType stride_type>
    void fft_1d_complex_twiddle_dit_sse2_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info)
{        
    size_t stride = get_size<stride_type>(step_info.stride);
    size_t repeats = step_info.repeats;
    const hhfft::RadersD &raders = *step_info.raders;
    size_t radix = get_actual_radix<radix_type>(raders);

    // Allocate memory for Rader's algorithm if needed
    double *data_raders = allocate_raders_D<radix_type>(raders);

    for (size_t i = 0; i < repeats; i++)
    {
        fft_1d_complex_twiddle_dit_sse2_d_internal<radix_type>
                (data_in + 2*i*radix*stride, data_out + 2*i*radix*stride, step_info.twiddle_factors, data_raders, raders, stride);
    }

    // Free temporary memory
    free_raders_D<radix_type>(raders, data_raders);
}

void fft_1d_complex_convolution_sse2_d(const double *data_in1, const double *data_in2, double *data_out, size_t n)
{
    for(size_t i = 0; i < n; i++)
    {
        ComplexD x_in0 = load_D(data_in1 + 2*i);
        ComplexD x_in1 = load_D(data_in2 + 2*i);
        ComplexD x_out = mul_D(x_in0, x_in1);
        store_D(x_out, data_out + 2*i);
    }
}

// fft for small sizes (1,2,3,4,5,7,8) where only one level is needed
template<size_t n, bool forward> void fft_1d_complex_1level_sse2_d(const double *data_in, double *data_out, const hhfft::StepInfo<double> &)
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

// For problems that need only one level Rader's
template<bool forward> void fft_1d_complex_1level_raders_sse2_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info)
{
    size_t n = step_info.radix;
    ComplexD k = broadcast64_D(1.0/n);

    // Allocate memory for Rader's algorithm
    const hhfft::RadersD &raders = *step_info.raders;
    double *data_raders = allocate_raders_D<Raders>(raders);

    // Initialize raders data with zeros
    init_coeff_D<Raders>(data_raders, raders);

    // Copy input data (squeeze)
    for (size_t i = 0; i < n; i++)
    {
        ComplexD x = load_D(data_in + 2*i);
        if (forward)
        {
            set_value_D<Raders>(nullptr, data_raders, i, raders, x);
        } else
        {
            set_value_inverse_D<Raders>(nullptr, data_raders, i, raders, x);
        }
    }

    // Multiply with coefficients
    multiply_coeff_forward_D<Raders>(nullptr, nullptr, data_raders, raders);

    // Copy input data (un-squeeze)
    for (size_t i = 0; i < n; i++)
    {
        ComplexD x = get_value_D<Raders>(nullptr, data_raders, i, raders);

        if(forward)
            store_D(x, data_out + 2*i);
        else
            store_D(k*x, data_out + 2*i);
    }

    // Free temporary memory
    free_raders_D<Raders>(raders, data_raders);
}

// Instantiations of the functions defined in this class
template void fft_1d_complex_reorder_sse2_d<false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder_sse2_d<true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template void fft_1d_complex_reorder2_sse2_d<Raders, SizeType::Size1, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_sse2_d<Raders, SizeType::Size1, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_sse2_d<Radix2, SizeType::Size1, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_sse2_d<Radix2, SizeType::Size1, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_sse2_d<Radix3, SizeType::Size1, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_sse2_d<Radix3, SizeType::Size1, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_sse2_d<Radix4, SizeType::Size1, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_sse2_d<Radix4, SizeType::Size1, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_sse2_d<Radix5, SizeType::Size1, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_sse2_d<Radix5, SizeType::Size1, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_sse2_d<Radix6, SizeType::Size1, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_sse2_d<Radix6, SizeType::Size1, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_sse2_d<Radix7, SizeType::Size1, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_sse2_d<Radix7, SizeType::Size1, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_sse2_d<Radix8, SizeType::Size1, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_sse2_d<Radix8, SizeType::Size1, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template void fft_1d_complex_sse2_d<Raders, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_sse2_d<Radix2, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_sse2_d<Radix3, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_sse2_d<Radix4, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_sse2_d<Radix5, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_sse2_d<Radix6, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_sse2_d<Radix7, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_sse2_d<Radix8, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_sse2_d<Raders, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_sse2_d<Radix2, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_sse2_d<Radix3, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_sse2_d<Radix4, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_sse2_d<Radix5, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_sse2_d<Radix6, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_sse2_d<Radix7, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_sse2_d<Radix8, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template void fft_1d_complex_twiddle_dit_sse2_d<Raders, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_sse2_d<Radix2, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_sse2_d<Radix3, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_sse2_d<Radix4, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_sse2_d<Radix5, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_sse2_d<Radix6, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_sse2_d<Radix7, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_sse2_d<Radix8, SizeType::SizeN>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_sse2_d<Raders, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_sse2_d<Radix2, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_sse2_d<Radix3, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_sse2_d<Radix4, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_sse2_d<Radix5, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_sse2_d<Radix6, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_sse2_d<Radix7, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_sse2_d<Radix8, SizeType::Size1>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template void fft_1d_complex_1level_sse2_d<1, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_sse2_d<2, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_sse2_d<3, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_sse2_d<4, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_sse2_d<5, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_sse2_d<6, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_sse2_d<7, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_sse2_d<8, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_sse2_d<1, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_sse2_d<2, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_sse2_d<3, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_sse2_d<4, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_sse2_d<5, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_sse2_d<6, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_sse2_d<7, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_sse2_d<8, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template void fft_1d_complex_1level_raders_sse2_d<true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_raders_sse2_d<false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
