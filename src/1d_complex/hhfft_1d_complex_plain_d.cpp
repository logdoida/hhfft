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

#include "../common/hhfft_1d_complex_plain_common_d.h"
#include "../raders/raders_plain_d.h"

using namespace hhfft;

// In-place reordering "swap"
void fft_1d_complex_reorder_in_place_plain_d(const double *, double *data_out,const hhfft::StepInfo<double> &step_info)
{
    size_t n = step_info.reorder_table_inplace_size;
    uint32_t *reorder_table = step_info.reorder_table_inplace;

    for (size_t i = 0; i < n; i++)
    {
        size_t ind1 = i + 1; // First one has been omitted!
        size_t ind2 = reorder_table[i];

        double r_temp = data_out[2*ind1+0];
        double c_temp = data_out[2*ind1+1];
        data_out[2*ind1+0] = data_out[2*ind2+0];
        data_out[2*ind1+1] = data_out[2*ind2+1];
        data_out[2*ind2+0] = r_temp;
        data_out[2*ind2+1] = c_temp;
    }
}

template<bool forward> void fft_1d_complex_reorder_plain_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info)
{    
    size_t n = step_info.radix*step_info.repeats;
    uint32_t *reorder_table = step_info.reorder_table;
    size_t reorder_table_size = step_info.reorder_table_size;

    // Needed only in ifft. Equal to 1/N
    double k = step_info.norm_factor;

    for (size_t i = 0; i < n; i++)
    {        
        if (forward)
        {
            size_t i2 = reorder_table[i];
            data_out[2*i+0] = data_in[2*i2+0];
            data_out[2*i+1] = data_in[2*i2+1];
        } else
        {            
            size_t i2 = reorder_table[reorder_table_size - i - 1];
            data_out[2*i+0] = k*data_in[2*i2+0];
            data_out[2*i+1] = k*data_in[2*i2+1];
        }
    }
}


template<RadixType radix_type>
    inline void fft_1d_complex_plain_d_internal(const double *data_in, double *data_out, double *data_raders, const hhfft::RadersD &raders, size_t stride)
{
    double x_temp_in[2*radix_type];
    double x_temp_out[2*radix_type];
    size_t radix = get_actual_radix<radix_type>(raders);

    for (size_t k = 0; k < stride; k++)
    {
        // Initialize raders data with zeros
        init_coeff<radix_type>(data_raders, raders);

        // Copy input data (squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            set_value<radix_type>(x_temp_in, data_raders, j, raders, data_in[2*k + 2*j*stride + 0], data_in[2*k + 2*j*stride + 1]);
        }

        // Multiply with coefficients
        multiply_coeff_forward<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

        // Copy input data (un-squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            get_value<radix_type>(x_temp_out, data_raders, j, raders, data_out[2*k + 2*j*stride + 0], data_out[2*k + 2*j*stride + 1]);
        }
    }
}

template<RadixType radix_type>
    inline void fft_1d_complex_twiddle_dit_plain_d_internal(const double *data_in, double *data_out, const double *twiddle_factors, double *data_raders, const hhfft::RadersD &raders, size_t stride)
{    
    double x_temp_in[2*radix_type];
    double x_temp_out[2*radix_type];
    double twiddle_temp[2*radix_type];
    size_t radix = get_actual_radix<radix_type>(raders);

    for (size_t k = 0; k < stride; k++)
    {
        // Initialize raders data with zeros
        init_coeff<radix_type>(data_raders, raders);

        // Copy input data and multiply with twiddle factors
        for (size_t j = 0; j < radix; j++)
        {
            size_t j2 = 2*k + 2*j*stride;
            set_value_twiddle<radix_type>(x_temp_in, data_raders, twiddle_temp, j, raders, data_in[j2 + 0], data_in[j2 + 1], twiddle_factors[j2 + 0], twiddle_factors[j2 + 1]);
        }

        // Multiply with coefficients
        multiply_twiddle<radix_type,true>(x_temp_in, x_temp_in, twiddle_temp);
        multiply_coeff_forward<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

        // Copy input data (un-squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            size_t j2 = 2*k + 2*j*stride;
            get_value<radix_type>(x_temp_out, data_raders, j, raders, data_out[j2 + 0], data_out[j2 + 1]);
        }
    }    
}

template<RadixType radix_type>
    void fft_1d_complex_plain_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info)
{
    size_t stride = step_info.stride;
    size_t repeats = step_info.repeats;    
    const hhfft::RadersD &raders = *step_info.raders;
    size_t radix = get_actual_radix<radix_type>(raders);

    // Allocate memory for Rader's algorithm if needed
    double *data_raders = allocate_raders<radix_type>(raders);

    for (size_t i = 0; i < repeats; i++)
    {
        fft_1d_complex_plain_d_internal<radix_type>
                (data_in + 2*i*radix*stride, data_out + 2*i*radix*stride, data_raders, raders, stride);
    }

    // Free temporary memory
    free_raders<radix_type>(raders, data_raders);
}

// Reordering and fft-step combined. In other version they really are!
template<RadixType radix_type, bool forward>
    void fft_1d_complex_reorder2_plain_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info)
{   
    size_t stride = step_info.stride;
    size_t repeats = step_info.repeats;
    const hhfft::RadersD &raders = *step_info.raders;
    size_t radix = get_actual_radix<radix_type>(raders);

    fft_1d_complex_reorder_plain_d<forward>(data_in, data_out, step_info);

    // Allocate memory for Rader's algorithm if needed
    double *data_raders = allocate_raders<radix_type>(raders);

    for (size_t i = 0; i < repeats; i++)
    {
        fft_1d_complex_plain_d_internal<radix_type>
                (data_out + 2*i*radix*stride, data_out + 2*i*radix*stride, data_raders, raders, stride);
    }

    // Free temporary memory
    free_raders<radix_type>(raders, data_raders);
}

template<RadixType radix_type>
    void fft_1d_complex_twiddle_dit_plain_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info)
{
    size_t stride = step_info.stride;
    size_t repeats = step_info.repeats;
    const hhfft::RadersD &raders = *step_info.raders;
    size_t radix = get_actual_radix<radix_type>(raders);

    // Allocate memory for Rader's algorithm if needed
    double *data_raders = allocate_raders<radix_type>(raders);

    for (size_t i = 0; i < repeats; i++)
    {
        fft_1d_complex_twiddle_dit_plain_d_internal<radix_type>
                (data_in + 2*i*radix*stride, data_out + 2*i*radix*stride, step_info.twiddle_factors, data_raders, raders, stride);
    }

    // Free temporary memory
    free_raders<radix_type>(raders, data_raders);
}

void fft_1d_complex_convolution_plain_d(const double *in1, const double *in2, double *out, size_t n)
{
    for(size_t i = 0; i < n; i++)
    {
        double re1 = in1[2*i + 0];
        double im1 = in1[2*i + 1];
        double re2 = in2[2*i + 0];
        double im2 = in2[2*i + 1];

        out[2*i + 0] = re1*re2 - im1*im2;
        out[2*i + 1] = re1*im2 + im1*re2;
    }
}

// fft for small sizes (1,2,3,4,5,7,8) where only one level is needed
template<size_t n, bool forward> void fft_1d_complex_1level_plain_d(const double *data_in, double *data_out, const hhfft::StepInfo<double> &)
{
    double k = 1.0/n;

    if (n == 1)
    {
        data_out[0] = data_in[0];
        data_out[1] = data_in[1];
    } else
    {
        double x_temp_in[2*n];
        double x_temp_out[2*n];

        // Copy input data
        for (size_t i = 0; i < 2*n; i++)
        {
            x_temp_in[i] = data_in[i];
        }

        // Multiply with coefficients
        multiply_coeff<n,forward>(x_temp_in, x_temp_out);

        // Copy output data
        for (size_t i = 0; i < 2*n; i++)
        {
            if(forward)
                data_out[i] = x_temp_out[i];
            else
                data_out[i] = k*x_temp_out[i];
        }
    }
}

// For problems that need only one level Rader's
template<bool forward> void fft_1d_complex_1level_raders_plain_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info)
{
    size_t n = step_info.radix;
    double k = 1.0/n;

    // Allocate memory for Rader's algorithm
    const hhfft::RadersD &raders = *step_info.raders;
    double *data_raders = allocate_raders<Raders>(raders);

    // Initialize raders data with zeros
    init_coeff<Raders>(data_raders, raders);

    // Copy input data (squeeze)
    for (size_t j = 0; j < n; j++)
    {
        if (forward)
        {
            set_value<Raders>(nullptr, data_raders, j, raders, data_in[2*j + 0], data_in[2*j + 1]);
        } else
        {
            set_value_inverse<Raders>(nullptr, data_raders, j, raders, data_in[2*j + 0], data_in[2*j + 1]);
        }
    }

    // Multiply with coefficients
    multiply_coeff_forward<Raders>(nullptr, nullptr, data_raders, raders);

    // Copy input data (un-squeeze)
    for (size_t j = 0; j < n; j++)
    {
        double re, im;
        get_value<Raders>(nullptr, data_raders, j, raders, re, im);
        if (forward)
        {
            data_out[2*j + 0] = re;
            data_out[2*j + 1] = im;
        } else
        {
            data_out[2*j + 0] = k*re;
            data_out[2*j + 1] = k*im;
        }
    }

    // Free temporary memory
    free_raders<Raders>(raders, data_raders);
}

// Instantiations of the functions defined in this class
template void fft_1d_complex_reorder_plain_d<false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder_plain_d<true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template void fft_1d_complex_reorder2_plain_d<Raders, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_plain_d<Raders, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_plain_d<Radix2, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_plain_d<Radix2, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_plain_d<Radix3, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_plain_d<Radix3, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_plain_d<Radix4, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_plain_d<Radix4, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_plain_d<Radix5, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_plain_d<Radix5, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_plain_d<Radix6, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_plain_d<Radix6, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_plain_d<Radix7, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_plain_d<Radix7, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_plain_d<Radix8, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_plain_d<Radix8, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template void fft_1d_complex_plain_d<Raders>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_plain_d<Radix2>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_plain_d<Radix3>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_plain_d<Radix4>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_plain_d<Radix5>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_plain_d<Radix6>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_plain_d<Radix7>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_plain_d<Radix8>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template void fft_1d_complex_twiddle_dit_plain_d<Raders>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_plain_d<Radix2>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_plain_d<Radix3>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_plain_d<Radix4>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_plain_d<Radix5>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_plain_d<Radix6>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_plain_d<Radix7>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_plain_d<Radix8>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template void fft_1d_complex_1level_plain_d<1, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_plain_d<2, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_plain_d<3, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_plain_d<4, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_plain_d<5, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_plain_d<6, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_plain_d<7, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_plain_d<8, false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_plain_d<1, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_plain_d<2, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_plain_d<3, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_plain_d<4, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_plain_d<5, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_plain_d<6, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_plain_d<7, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_plain_d<8, true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template void fft_1d_complex_1level_raders_plain_d<true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_raders_plain_d<false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
