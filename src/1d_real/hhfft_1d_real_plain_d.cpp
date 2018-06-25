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

template<bool forward>
    void fft_1d_complex_to_complex_packed_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    const double *packing_table = step_info.twiddle_factors;
    size_t n = 2*step_info.repeats; // n = number of original real numbers

    // Input/output way
    if (forward)
    {
        double x_r = data_in[0];
        double x_i = data_in[1];
        data_out[0] = x_r + x_i;
        data_out[1] = 0.0;
        data_out[n] = x_r - x_i;
        data_out[n+1] = 0.0;
    } else
    {
        double x_r = data_in[0];
        double x_i = data_in[n];
        data_out[0] = 0.5*(x_r + x_i);
        data_out[1] = 0.5*(x_r - x_i);
    }

    if (n%4 == 0)
    {
        double x_r = data_in[n/2 + 0];
        double x_i = data_in[n/2 + 1];

        if (forward)
        {
            data_out[n/2 + 0] =  x_r;
            data_out[n/2 + 1] = -x_i;
        } else
        {
            data_out[n/2 + 0] =  x_r;
            data_out[n/2 + 1] = -x_i;
        }
    }

    for (size_t i = 2; i < n/2; i+=2)
    {        
        double ss = -packing_table[i + 0];
        double sc = -packing_table[i + 1];

        double x0_r = data_in[i + 0];
        double x0_i = data_in[i + 1];
        double x1_r = data_in[n - i + 0];
        double x1_i = data_in[n - i + 1];

        if (!forward)
        {
            ss = ss;
            sc = -sc;
        }

        double temp0 = -ss*(x0_r - x1_r) + sc*(x0_i + x1_i);
        double temp1 = -sc*(x0_r - x1_r) - ss*(x0_i + x1_i);

        data_out[i + 0]     = temp0 + x0_r;
        data_out[i + 1]     = temp1 + x0_i;
        data_out[n - i + 0] = -temp0 + x1_r;
        data_out[n - i + 1] = temp1 + x1_i;
    }
}

// This is found in hhfft_1d_complex_plain_d.cpp
template<bool scale> void fft_1d_complex_reorder_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

void fft_1d_complex_to_complex_packed_ifft_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    // If data_in == data_out,
    if(data_in == data_out)
    {
        fft_1d_complex_to_complex_packed_plain_d<false>(data_in, data_out, step_info);
        fft_1d_complex_reorder_plain_d<true>(data_out, data_out, step_info);
        return;
    }

    const double *packing_table = step_info.twiddle_factors;
    size_t n = step_info.repeats; // 2*n = number of original real numbers
    uint32_t *reorder_table_inverse = step_info.reorder_table;
    double k = step_info.norm_factor;

    double x_r = data_in[0];
    double x_i = data_in[2*n];
    data_out[0] = 0.5*k*(x_r + x_i);
    data_out[1] = 0.5*k*(x_r - x_i);

    if (n%2 == 0)
    {
        double x_r = data_in[n + 0];
        double x_i = data_in[n + 1];

        size_t i = reorder_table_inverse[n/2];

        data_out[2*i + 0] =  k*x_r;
        data_out[2*i + 1] = -k*x_i;
    }

    for (size_t i = 1; i < (n+1)/2; i++)
    {
        double ss = -packing_table[2*i + 0];
        double sc = packing_table[2*i + 1];

        double x0_r = k*data_in[2*i + 0];
        double x0_i = k*data_in[2*i + 1];
        double x1_r = k*data_in[2*(n - i) + 0];
        double x1_i = k*data_in[2*(n - i) + 1];

        double temp0 = -ss*(x0_r - x1_r) + sc*(x0_i + x1_i);
        double temp1 = -sc*(x0_r - x1_r) - ss*(x0_i + x1_i);

        size_t i2 = reorder_table_inverse[i];
        size_t i3 = reorder_table_inverse[n - i];

        data_out[2*i2 + 0] = temp0 + x0_r;
        data_out[2*i2 + 1] = temp1 + x0_i;
        data_out[2*i3 + 0] = -temp0 + x1_r;
        data_out[2*i3 + 1] = temp1 + x1_i;
    }
}

// This function is used on the first level of odd real fft
template<size_t radix> void fft_1d_real_first_level_forward_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    size_t repeats = step_info.repeats;
    uint32_t *reorder_table = step_info.reorder_table;

    double x_temp_in[2*radix];
    double x_temp_out[2*radix];
    bool dir_out = true;
    for (size_t i = 0; i < repeats; i++)
    {
        // Copy input data taking reordering into account
        for (size_t j = 0; j < radix; j++)
        {
            size_t ind = reorder_table[i*radix + j];
            x_temp_in[2*j + 0] = data_in[ind];
            x_temp_in[2*j + 1] = 0;
        }

        //std::cout << "x_temp_in = " << x_temp_in[0] << " " << x_temp_in[1] << " " << x_temp_in[2] << " " << x_temp_in[3] << " " << x_temp_in[4] << " " << x_temp_in[5] << std::endl;

        // Multiply with coefficients
        multiply_coeff<radix,true>(x_temp_in, x_temp_out);

        //std::cout << "x_temp_out = " << x_temp_out[0] << " " << x_temp_out[1] << " " << x_temp_out[2] << " " << x_temp_out[3] << " " << x_temp_out[4] << " " << x_temp_out[5] << std::endl;

        // Save only about half of the output
        // First/ last one is real
        if (dir_out) // direction normal
        {
            data_out[i*radix + 0] = x_temp_out[0];

            for (size_t j = 1; j < radix/2 + 1; j++)
            {
                data_out[i*radix + 2*j - 1] = x_temp_out[2*j + 0];
                data_out[i*radix + 2*j + 0] = x_temp_out[2*j + 1];
            }
        } else // direction inverted
        {
            data_out[i*radix + radix - 1] = x_temp_out[0];

            for (size_t j = 1; j < radix/2 + 1; j++)
            {
                data_out[i*radix + radix - 2*j - 1] = x_temp_out[2*j + 0];
                data_out[i*radix + radix - 2*j + 0] = x_temp_out[2*j + 1];
            }
        }
        dir_out = !dir_out;
    }
}

// This function is used on the first level of odd real ifft
template<size_t radix> void fft_1d_real_first_level_inverse_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    size_t repeats = step_info.repeats;
    uint32_t *reorder_table = step_info.reorder_table;
    size_t n = radix * (2*step_info.repeats - 1);
    double norm_factor = step_info.norm_factor;

    double x_temp_in[2*radix];
    double x_temp_out[2*radix];

    // In the first repeat input is r, (r+i), (r+i) ... and output is r,r,r,r,r...
    {
        x_temp_in[0] = norm_factor*data_in[0];
        x_temp_in[1] = 0;

        // Read other inputs and conjugate them
        for (size_t j = 1; j <= radix/2; j++)
        {
            size_t ind = reorder_table[j];
            double real =  norm_factor*data_in[2*ind + 0];
            double imag =  norm_factor*data_in[2*ind + 1];
            x_temp_in[2*j + 0] = real;
            x_temp_in[2*j + 1] = imag;
            x_temp_in[2*(radix-j) + 0] = real;
            x_temp_in[2*(radix-j) + 1] = -imag;
        }

        // Multiply with coefficients
        multiply_coeff<radix,false>(x_temp_in, x_temp_out);

        // Write only real parts of the data
        for (size_t j = 0; j < radix; j++)
        {
            data_out[j] = x_temp_out[2*j + 0];
        }
    }

    // Other repeats are more usual, data ordering changes from r,i,r,i,r,i... to r,r,r...i,i,i...
    for (size_t i = 1; i < repeats; i++)
    {
        // Copy input data taking reordering into account
        for (size_t j = 0; j < radix; j++)
        {
            size_t ind = reorder_table[i*radix - radix/2 + j];
            //std::cout << "i = " << i << ", j = " << j << ", ind = " << ind << std::endl;
            if (ind <= n/2)
            {
                x_temp_in[2*j + 0] = norm_factor*data_in[2*ind + 0];
                x_temp_in[2*j + 1] = norm_factor*data_in[2*ind + 1];
            } else
            {
                // If input is from the lower part, it needs to be conjugated
                ind = n - ind;
                x_temp_in[2*j + 0] = norm_factor*data_in[2*ind + 0];
                x_temp_in[2*j + 1] = -norm_factor*data_in[2*ind + 1];
            }
        }

        //std::cout << "x_temp_in = " << x_temp_in[0] << " " << x_temp_in[1] << " " << x_temp_in[2] << " " << x_temp_in[3] << " " << x_temp_in[4] << " " << x_temp_in[5] << std::endl;

        // Multiply with coefficients
        multiply_coeff<radix,false>(x_temp_in, x_temp_out);

        // Store real and imag parts separately
        for (size_t j = 0; j < radix; j++)
        {
            data_out[2*i*radix - radix + j] = x_temp_out[2*j + 0];
            data_out[2*i*radix + j] = x_temp_out[2*j + 1];
        }
    }
}

inline size_t index_dir_stride_odd(size_t dir_in, size_t stride, size_t k)
{
    return dir_in*(4*k - stride) + stride - 2*k - 1;
}

template<size_t radix> void fft_1d_real_one_level_forward_plain_d_internal(const double *data_in, double *data_out, const double *twiddle_factors, size_t stride, bool dir_out)
{
    // The first/last value in each stride is real
    {
        double x0_temp_in[2*radix];
        double x0_temp_out[2*radix];

        bool dir_in = dir_out;
        for (size_t j = 0; j < radix; j++)
        {
            if (dir_in)
            {
                x0_temp_in[2*j + 0] = data_in[j*stride + 0];
            } else
            {
                x0_temp_in[2*j + 0] = data_in[j*stride + stride - 1];
            }
            x0_temp_in[2*j + 1] = 0;

            dir_in = !dir_in;
        }

        multiply_coeff<radix,true>(x0_temp_in, x0_temp_out);

        // only about half is written
        for (size_t j = 1; j < (radix+1)/2; j++)
        {
            if (dir_out)
            {
                data_out[0] = x0_temp_out[0];
                data_out[2*j*stride - 1] = x0_temp_out[2*j + 0];
                data_out[2*j*stride + 0] = x0_temp_out[2*j + 1];
            } else
            {
                data_out[radix*stride - 1] = x0_temp_out[0];
                data_out[2*j*stride - stride - 1] = x0_temp_out[2*((radix+1)/2 - j) + 0];
                data_out[2*j*stride - stride + 0] = x0_temp_out[2*((radix+1)/2 - j) + 1];
            }
        }      
    }

    // Rest of the values represent complex numbers
    for (size_t k = 1; k < (stride+1)/2; k++)
    {
        double x_temp_in[2*radix];
        double x_temp_out[2*radix];
        double twiddle_temp[2*radix];
        size_t dir_in = dir_out;

        // Copy the values and twiddle factors
        for (size_t j = 0; j < radix; j++)
        {
            size_t index2 = index_dir_stride_odd(dir_in, stride, k);
            x_temp_in[2*j + 0]  = data_in[j*stride + index2 + 0];
            x_temp_in[2*j + 1]  = data_in[j*stride + index2 + 1];

            twiddle_temp[2*j + 0] = twiddle_factors[2*k + 2*j*stride + 0];
            twiddle_temp[2*j + 1] = twiddle_factors[2*k + 2*j*stride + 1];

            dir_in = !dir_in;
        }

        multiply_twiddle<radix,true>(x_temp_in, x_temp_in, twiddle_temp);

        multiply_coeff_real_odd_forward<radix>(x_temp_in, x_temp_out);

        // save output taking the directions into account
        dir_in = dir_out;
        for (size_t j = 0; j < radix; j++)
        {
            size_t index2 = index_dir_stride_odd(dir_in, stride, k);

            // reverse the output order if required
            if (dir_out)
            {
                data_out[j*stride + index2 + 0] = x_temp_out[2*j + 0];
                data_out[j*stride + index2 + 1] = x_temp_out[2*j + 1];
            } else
            {
                data_out[j*stride + index2 + 0] = x_temp_out[2*(radix - j - 1) + 0];
                data_out[j*stride + index2 + 1] = x_temp_out[2*(radix - j - 1) + 1];
            }

            dir_in = !dir_in;
        }
    }
}

// This function is used on rest of the odd real fft
template<size_t radix> void fft_1d_real_one_level_forward_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    size_t stride = step_info.stride;
    size_t repeats = step_info.repeats;
    double *twiddle_factors = step_info.twiddle_factors;

    bool dir_out = true;
    for (size_t i = 0; i < repeats; i++)
    {
        fft_1d_real_one_level_forward_plain_d_internal<radix>(data_in + i*radix*stride, data_out + i*radix*stride, twiddle_factors, stride, dir_out);

        dir_out = !dir_out;
    }
}

// This function is used on on rest of the odd real ifft
template<size_t radix> void fft_1d_real_one_level_inverse_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{    
    size_t repeats = step_info.repeats;
    size_t stride = step_info.stride;
    double *twiddle_factors = step_info.twiddle_factors;

    double x_temp_in[2*radix];
    double x_temp_out[2*radix];
    double twiddle_temp[2*radix];

    // In the first repeat input is r,r,r,... (r+i), (r+i) ... and output is r,r,r,r,r...
    for (size_t k = 0; k < stride; k++)
    {
        x_temp_in[0] = data_in[k];
        x_temp_in[1] = 0;

        // Read other inputs, multiply them with twiddle factors and conjugate them
        for (size_t j = 1; j <= radix/2; j++)
        {
            double re = data_in[2*j*stride - stride + k];
            double im = data_in[2*j*stride + k];

            // multiplication with conjugated twiddle factors is done first
            double w_r = twiddle_factors[2*j*stride + 2*k + 0];
            double w_i = twiddle_factors[2*j*stride + 2*k + 1];

            double re2 = w_r*re + w_i*im;
            double im2 = w_r*im - w_i*re;

            x_temp_in[2*j + 0] = re2;
            x_temp_in[2*j + 1] = im2;
            x_temp_in[2*(radix-j) + 0] = re2;
            x_temp_in[2*(radix-j) + 1] = -im2;
        }

        // Multiply with coefficients
        multiply_coeff<radix,false>(x_temp_in, x_temp_out);

        // Write only real parts of the data
        for (size_t j = 0; j < radix; j++)
        {
            data_out[j*stride + k] = x_temp_out[2*j + 0];
        }        
    }

    // Other repeats are more usual, however both inputs and outputs have real and imag parts separated
    for (size_t i = 1; i < repeats; i++)
    {
        for (size_t k = 0; k < stride; k++)
        {
            // Read real and imag parts separately
            for (size_t j = 0; j < radix; j++)
            {
                x_temp_in[2*j + 0] = data_in[2*i*stride*radix + 2*j*stride - stride*radix + k];
                x_temp_in[2*j + 1] = data_in[2*i*stride*radix + 2*j*stride - stride*radix + stride + k];

                twiddle_temp[2*j + 0] = twiddle_factors[2*j*stride + 2*k + 0];
                twiddle_temp[2*j + 1] = twiddle_factors[2*j*stride + 2*k + 1];
            }

            // Multiply with twiddle factors
            multiply_twiddle<radix,false>(x_temp_in, x_temp_in, twiddle_temp);

            // Multiply with coefficients
            multiply_coeff<radix,false>(x_temp_in, x_temp_out);

            // Store real and imag parts separately
            for (size_t j = 0; j < radix; j++)
            {
                data_out[2*i*stride*radix + j*stride - stride*radix + k] = x_temp_out[2*j + 0];
                data_out[2*i*stride*radix + j*stride + k] = x_temp_out[2*j + 1];
            }
        }
    }    
}

// Instantiations of the functions defined in this class
template void fft_1d_complex_to_complex_packed_plain_d<false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_to_complex_packed_plain_d<true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_1d_real_first_level_forward_plain_d<3>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_first_level_forward_plain_d<5>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_first_level_forward_plain_d<7>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_1d_real_first_level_inverse_plain_d<3>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_first_level_inverse_plain_d<5>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_first_level_inverse_plain_d<7>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_1d_real_one_level_forward_plain_d<3>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_one_level_forward_plain_d<5>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_one_level_forward_plain_d<7>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_1d_real_one_level_inverse_plain_d<3>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_one_level_inverse_plain_d<5>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_one_level_inverse_plain_d<7>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
