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

#include <iostream> // TESTING

using namespace hhfft;

template<bool forward>
    void fft_2d_complex_to_complex_packed_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{        
    const double *packing_table = step_info.twiddle_factors;
    size_t n = step_info.repeats; // n = number of original real rows
    size_t m = step_info.size;    // m = number of original real columns

    // Input/output way
    if (forward)
    {
        for (size_t j = 0; j < 2*m; j+=2)
        {
            double x_r = data_in[j + 0];
            double x_i = data_in[j + 1];
            data_out[j + 0] = x_r + x_i;
            data_out[j + 1] = 0.0;
            data_out[j + n*m + 0] = x_r - x_i;
            data_out[j + n*m + 1] = 0.0;
        }
    } else
    {
        // For inverse data_in = last row (temporary variable), data_out is actually both input and output!
        for (size_t j = 0; j < 2*m; j+=2)
        {
            double x_r = data_out[j];
            double x_i = data_in[j];
            data_out[j + 0] = 0.5*(x_r + x_i);
            data_out[j + 1] = 0.5*(x_r - x_i);
        }

        // !
        data_in = data_out;
    }

    if (n%4 == 0)
    {
        for (size_t j = 0; j < 2*m; j+=2)
        {
            double x_r = data_in[j + n*m/2 + 0];
            double x_i = data_in[j + n*m/2 + 1];

            data_out[j + n*m/2 + 0] =  x_r;
            data_out[j + n*m/2 + 1] = -x_i;
        }
    }

    for (size_t i = 2; i < n/2; i+=2)
    {        
        double ss = -packing_table[i + 0];
        double sc = -packing_table[i + 1];

        if (!forward)
        {
            sc = -sc;
        }

        for (size_t j = 0; j < 2*m; j+=2)
        {
            double x0_r = data_in[i*m + j + 0];
            double x0_i = data_in[i*m + j + 1];
            double x1_r = data_in[(n - i)*m + j + 0];
            double x1_i = data_in[(n - i)*m + j + 1];

            //std::cout << "x0_r = " << x0_r << ", x0_i = " << x0_i << std::endl;
            //std::cout << "x1_r = " << x1_r << ", x1_i = " << x1_i << std::endl;

            double temp0 = -ss*(x0_r - x1_r) + sc*(x0_i + x1_i);
            double temp1 = -sc*(x0_r - x1_r) - ss*(x0_i + x1_i);

            data_out[i*m + j + 0]     = temp0 + x0_r;
            data_out[i*m + j + 1]     = temp1 + x0_i;
            data_out[(n - i)*m + j + 0] = -temp0 + x1_r;
            data_out[(n - i)*m + j + 1] = temp1 + x1_i;
        }
    }
}

// Instantiations of the functions defined in this class
template void fft_2d_complex_to_complex_packed_plain_d<false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_to_complex_packed_plain_d<true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

