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

#include <iostream> // TESTING

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

// TODO move this to header
/*
template<size_t radix> void multiply_coeff_real_forward(const double *x_in, double *x_out)
{
    if (radix == 3)
    {
        double k1 = 0.5;
        double k2 = 0.5*sqrt(3);
        x_out[0] = x_in[0] + x_in[2] + x_in[4];
        x_out[1] = x_in[1] + x_in[3] + x_in[5];
        x_out[2] = x_in[0] - k1*x_in[2] - k2*x_in[3] - k1*x_in[4] + k2*x_in[5];
        x_out[3] = -x_in[1] - k2*x_in[2] + k1*x_in[3] + k2*x_in[4] + k1*x_in[5];
        x_out[4] = x_in[0] - k1*x_in[2] + k2*x_in[3] - k1*x_in[4] - k2*x_in[5];
        x_out[5] = x_in[1] - k2*x_in[2] - k1*x_in[3] + k2*x_in[4] - k1*x_in[5];
    }

    if (radix == 5)
    {
        double k1 = cos(2*M_PI*1.0/5.0);
        double k2 = sin(2*M_PI*1.0/5.0);
        double k3 =-cos(2*M_PI*2.0/5.0);
        double k4 = sin(2*M_PI*2.0/5.0);

        x_out[0] = x_in[0] + x_in[2] + x_in[4] + x_in[6] + x_in[8];
        x_out[1] = x_in[1] + x_in[3] + x_in[5] + x_in[7] + x_in[9];
        x_out[2] = x_in[0] + k1*x_in[2] - k2*x_in[3] - k3*x_in[4] - k4*x_in[5] - k3*x_in[6] + k4*x_in[7] + k1*x_in[8] + k2*x_in[9];
        x_out[3] = -(x_in[1] + k2*x_in[2] + k1*x_in[3] + k4*x_in[4] - k3*x_in[5] - k4*x_in[6] - k3*x_in[7] - k2*x_in[8] + k1*x_in[9]);
        x_out[4] = x_in[0] + k1*x_in[2] + k2*x_in[3] - k3*x_in[4] + k4*x_in[5] - k3*x_in[6] - k4*x_in[7] + k1*x_in[8] - k2*x_in[9];
        x_out[5] = x_in[1] - k2*x_in[2] + k1*x_in[3] - k4*x_in[4] - k3*x_in[5] + k4*x_in[6] - k3*x_in[7] + k2*x_in[8] + k1*x_in[9];
        x_out[6] = x_in[0] - k3*x_in[2] - k4*x_in[3] + k1*x_in[4] + k2*x_in[5] + k1*x_in[6] - k2*x_in[7] - k3*x_in[8] + k4*x_in[9];
        x_out[7] = -(x_in[1] + k4*x_in[2] - k3*x_in[3] - k2*x_in[4] + k1*x_in[5] + k2*x_in[6] + k1*x_in[7] - k4*x_in[8] - k3*x_in[9]);
        x_out[8] = x_in[0] - k3*x_in[2] + k4*x_in[3] + k1*x_in[4] - k2*x_in[5] + k1*x_in[6] + k2*x_in[7] - k3*x_in[8] - k4*x_in[9];
        x_out[9] = x_in[1] - k4*x_in[2] - k3*x_in[3] + k2*x_in[4] + k1*x_in[5] - k2*x_in[6] + k1*x_in[7] + k4*x_in[8] - k3*x_in[9];
    }

    if (radix == 7)
    {
        double k1 = cos(2*M_PI*1.0/7.0);
        double k2 = sin(2*M_PI*1.0/7.0);
        double k3 =-cos(2*M_PI*2.0/7.0);
        double k4 = sin(2*M_PI*2.0/7.0);
        double k5 =-cos(2*M_PI*3.0/7.0);
        double k6 = sin(2*M_PI*3.0/7.0);
        x_out[0] = x_in[0] + x_in[2] + x_in[4] + x_in[6] + x_in[8] + x_in[10] + x_in[12];
        x_out[1] = x_in[1] + x_in[3] + x_in[5] + x_in[7] + x_in[9] + x_in[11] + x_in[13];
        x_out[2] = x_in[0] + k1*x_in[2] - k2*x_in[3] - k3*x_in[4] - k4*x_in[5] - k5*x_in[6] - k6*x_in[7] - k5*x_in[8] + k6*x_in[9] - k3*x_in[10] + k4*x_in[11] + k1*x_in[12] + k2*x_in[13];
        x_out[3] = -(x_in[1] + k2*x_in[2] + k1*x_in[3] + k4*x_in[4] - k3*x_in[5] + k6*x_in[6] - k5*x_in[7] - k6*x_in[8] - k5*x_in[9] - k4*x_in[10] - k3*x_in[11] - k2*x_in[12] + k1*x_in[13]);
        x_out[4] = x_in[0] + k1*x_in[2] + k2*x_in[3] - k3*x_in[4] + k4*x_in[5] - k5*x_in[6] + k6*x_in[7] - k5*x_in[8] - k6*x_in[9] - k3*x_in[10] - k4*x_in[11] + k1*x_in[12] - k2*x_in[13];
        x_out[5] = x_in[1] - k2*x_in[2] + k1*x_in[3] - k4*x_in[4] - k3*x_in[5] - k6*x_in[6] - k5*x_in[7] + k6*x_in[8] - k5*x_in[9] + k4*x_in[10] - k3*x_in[11] + k2*x_in[12] + k1*x_in[13];
        x_out[6] = x_in[0] - k3*x_in[2] - k4*x_in[3] - k5*x_in[4] + k6*x_in[5] + k1*x_in[6] + k2*x_in[7] + k1*x_in[8] - k2*x_in[9] - k5*x_in[10] - k6*x_in[11] - k3*x_in[12] + k4*x_in[13];
        x_out[7] = -(x_in[1] + k4*x_in[2] - k3*x_in[3] - k6*x_in[4] - k5*x_in[5] - k2*x_in[6] + k1*x_in[7] + k2*x_in[8] + k1*x_in[9] + k6*x_in[10] - k5*x_in[11] - k4*x_in[12] - k3*x_in[13]);
        x_out[8] = x_in[0] - k3*x_in[2] + k4*x_in[3] - k5*x_in[4] - k6*x_in[5] + k1*x_in[6] - k2*x_in[7] + k1*x_in[8] + k2*x_in[9] - k5*x_in[10] + k6*x_in[11] - k3*x_in[12] - k4*x_in[13];
        x_out[9] = x_in[1] - k4*x_in[2] - k3*x_in[3] + k6*x_in[4] - k5*x_in[5] + k2*x_in[6] + k1*x_in[7] - k2*x_in[8] + k1*x_in[9] - k6*x_in[10] - k5*x_in[11] + k4*x_in[12] - k3*x_in[13];
        x_out[10] = x_in[0] - k5*x_in[2] - k6*x_in[3] + k1*x_in[4] + k2*x_in[5] - k3*x_in[6] - k4*x_in[7] - k3*x_in[8] + k4*x_in[9] + k1*x_in[10] - k2*x_in[11] - k5*x_in[12] + k6*x_in[13];
        x_out[11] = -(x_in[1] + k6*x_in[2] - k5*x_in[3] - k2*x_in[4] + k1*x_in[5] + k4*x_in[6] - k3*x_in[7] - k4*x_in[8] - k3*x_in[9] + k2*x_in[10] + k1*x_in[11] - k6*x_in[12] - k5*x_in[13]);
        x_out[12] = x_in[0] - k5*x_in[2] + k6*x_in[3] + k1*x_in[4] - k2*x_in[5] - k3*x_in[6] + k4*x_in[7] - k3*x_in[8] - k4*x_in[9] + k1*x_in[10] + k2*x_in[11] - k5*x_in[12] - k6*x_in[13];
        x_out[13] = x_in[1] - k6*x_in[2] - k5*x_in[3] + k2*x_in[4] + k1*x_in[5] - k4*x_in[6] - k3*x_in[7] + k4*x_in[8] - k3*x_in[9] - k2*x_in[10] + k1*x_in[11] + k6*x_in[12] - k5*x_in[13];
    }
}
*/

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

        //std::cout << "stride = " << stride << std::endl;
        //std::cout << "x0_temp_in = " << x0_temp_in[0] << ", " << x0_temp_in[1] << ", " << x0_temp_in[2] << ", " << x0_temp_in[3] << ", " << x0_temp_in[4] << ", " << x0_temp_in[5] << ", " << x0_temp_in[6] << ", " << x0_temp_in[7] << ", " << x0_temp_in[8] << ", " << x0_temp_in[9] << std::endl;
        //std::cout << "x0_temp_out = " << x0_temp_out[0] << ", " << x0_temp_out[1] << ", " << x0_temp_out[2] << ", " << x0_temp_out[3] << ", " << x0_temp_out[4] << ", " << x0_temp_out[5] << ", " << x0_temp_out[6] << ", " << x0_temp_out[7] << ", " << x0_temp_out[8] << ", " << x0_temp_out[9] << std::endl;
    }

    // Rest of the values represent complex numbers
    for (size_t k = 1; k < (stride+1)/2; k++)
    {
        double x_temp_in[2*radix];
        double x_temp_out[2*radix];
        double twiddle_temp[2*radix];
        size_t dir_in = dir_out;

        /*
        // Read in the values used in this step and multiply them with twiddle factors
        for (size_t j = 0; j < radix; j++)
        {
            // direction affects in which direction the inputs are read
            size_t index2 = index_dir_stride_odd(dir_in, stride, k);
            double x_r = data_in[j*stride + index2 + 0];
            double x_i = data_in[j*stride + index2 + 1];

            double w_r = twiddle_factors[2*(j*stride + k) + 0];
            double w_i = twiddle_factors[2*(j*stride + k) + 1];

            x_temp_in[2*j + 0] = w_r*x_r - w_i*x_i;
            x_temp_in[2*j + 1] = w_i*x_r + w_r*x_i;

            // TESTING
            //std::cout << "k = " << k << ", j = " << j << ", index2 = " << index2 << ", x_in = " << x_r << ", " << x_i << ", w = " << w_r << ", " << w_i << ", x_temp = " << x_temp_in[2*j + 0] << ", " << x_temp_in[2*j + 1] << std::endl;

            dir_in = !dir_in;
        }
        */

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


            //std::cout << "j = " << j << ", dir_in = " << dir_in << ", index2 = " << index2 << ", x_temp_out = " << x_temp_out[2*j + 0] << ", " << x_temp_out[2*j + 1] << std::endl;
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

// Instantiations of the functions defined in this class
template void fft_1d_complex_to_complex_packed_plain_d<false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_to_complex_packed_plain_d<true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_1d_real_first_level_forward_plain_d<3>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_first_level_forward_plain_d<5>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_first_level_forward_plain_d<7>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template void fft_1d_real_one_level_forward_plain_d<3>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_one_level_forward_plain_d<5>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_real_one_level_forward_plain_d<7>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
