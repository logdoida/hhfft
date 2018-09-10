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

using namespace hhfft;

template<bool forward> void fft_1d_complex_reorder_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{    
    size_t n = step_info.radix*step_info.repeats;
    uint32_t *reorder_table = step_info.reorder_table;

    // Needed only in ifft. Equal to 1/N
    double k = step_info.norm_factor;

    for (size_t i = 0; i < n; i++)
    {
        size_t i2 = reorder_table[i];
        if (forward)
        {            
            data_out[2*i+0] = data_in[2*i2+0];
            data_out[2*i+1] = data_in[2*i2+1];
        } else
        {
            if (i > 0)
            {
                i2 = n - i2;
            }
            data_out[2*i+0] = k*data_in[2*i2+0];
            data_out[2*i+1] = k*data_in[2*i2+1];
        }
    }
}


template<size_t radix>
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
        multiply_coeff<radix,true>(x_temp_in, x_temp_out);

        // Copy input data (un-squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            data_out[2*k + 2*j*stride + 0] = x_temp_out[2*j + 0];
            data_out[2*k + 2*j*stride + 1] = x_temp_out[2*j + 1];
        }
    }
}


template<size_t radix>
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

        multiply_twiddle<radix,true>(x_temp_in, x_temp_in, twiddle_temp);

        multiply_coeff<radix,true>(x_temp_in, x_temp_out);

        // Copy input data (un-squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            data_out[2*k + 2*j*stride + 0] = x_temp_out[2*j + 0];
            data_out[2*k + 2*j*stride + 1] = x_temp_out[2*j + 1];
        }
    }    
}

template<size_t radix>
    void fft_1d_complex_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    size_t stride = step_info.stride;
    size_t repeats = step_info.repeats;    

    for (size_t i = 0; i < repeats; i++)
    {
        fft_1d_complex_plain_d_internal<radix>
                (data_in + 2*i*radix*stride, data_out + 2*i*radix*stride, stride);
    }
}

// Reordering and fft-step combined. In other version they really are!
template<size_t radix, bool forward>
    void fft_1d_complex_reorder2_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{   
    size_t stride = step_info.stride;
    size_t repeats = step_info.repeats;

    fft_1d_complex_reorder_plain_d<forward>(data_in, data_out, step_info);

    for (size_t i = 0; i < repeats; i++)
    {
        fft_1d_complex_plain_d_internal<radix>
                (data_out + 2*i*radix*stride, data_out + 2*i*radix*stride, stride);
    }
}

template<size_t radix>
    void fft_1d_complex_twiddle_dit_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    size_t stride = step_info.stride;
    size_t repeats = step_info.repeats;

    for (size_t i = 0; i < repeats; i++)
    {
        fft_1d_complex_twiddle_dit_plain_d_internal<radix>
                (data_in + 2*i*radix*stride, data_out + 2*i*radix*stride, step_info.twiddle_factors, stride);
    }
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
template<size_t n, bool forward> void fft_1d_complex_1level_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &)
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


// Instantiations of the functions defined in this class
template void fft_1d_complex_reorder_plain_d<false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder_plain_d<true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_1d_complex_reorder2_plain_d<2, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_plain_d<2, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_plain_d<3, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_plain_d<3, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_plain_d<4, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_plain_d<4, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_plain_d<5, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_plain_d<5, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_plain_d<6, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_plain_d<6, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_plain_d<7, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_plain_d<7, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_plain_d<8, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_reorder2_plain_d<8, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_1d_complex_plain_d<2>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_plain_d<3>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_plain_d<4>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_plain_d<5>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_plain_d<6>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_plain_d<7>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_plain_d<8>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_1d_complex_twiddle_dit_plain_d<2>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_plain_d<3>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_plain_d<4>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_plain_d<5>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_plain_d<6>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_plain_d<7>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_twiddle_dit_plain_d<8>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_1d_complex_1level_plain_d<1, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_plain_d<2, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_plain_d<3, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_plain_d<4, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_plain_d<5, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_plain_d<6, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_plain_d<7, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_plain_d<8, false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_plain_d<1, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_plain_d<2, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_plain_d<3, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_plain_d<4, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_plain_d<5, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_plain_d<6, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_plain_d<7, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_1level_plain_d<8, true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
