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
    size_t n = step_info.repeats; // n = number of original real numbers

    // TODO this way is probably need in 2D FFT
    // Packed way
    /*
    if (forward)
    {
        double x_r = data_in[0];
        double x_i = data_in[1];
        data_out[0] = x_r + x_i;
        data_out[1] = x_r - x_i;
    } else
    {
        double x_r = data_in[0];
        double x_i = data_in[1];
        data_out[0] = 0.5*(x_r + x_i);
        data_out[1] = 0.5*(x_r - x_i);
    }
    */

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

        data_out[n/2 + 0] =  x_r;
        data_out[n/2 + 1] = -x_i;
    }

    for (size_t i = 2; i < n/2; i+=2)
    {        
        double ss = -packing_table[i + 0];
        double sc = -packing_table[i + 1];

        if (!forward)
        {
            sc = -sc;
        }

        double x0_r = data_in[i + 0];
        double x0_i = data_in[i + 1];
        double x1_r = data_in[n - i + 0];
        double x1_i = data_in[n - i + 1];

        double temp0 = -ss*(x0_r - x1_r) + sc*(x0_i + x1_i);
        double temp1 = -sc*(x0_r - x1_r) - ss*(x0_i + x1_i);

        data_out[i + 0]     = temp0 + x0_r;
        data_out[i + 1]     = temp1 + x0_i;
        data_out[n - i + 0] = -temp0 + x1_r;
        data_out[n - i + 1] = temp1 + x1_i;
    }
}

// Instantiations of the functions defined in this class
template void fft_1d_complex_to_complex_packed_plain_d<false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_to_complex_packed_plain_d<true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

