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

#include <iostream> // For testing

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

// Instantiations of the functions defined in this class
template void fft_1d_complex_to_complex_packed_plain_d<false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_to_complex_packed_plain_d<true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

