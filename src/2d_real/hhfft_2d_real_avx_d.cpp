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

template<bool forward>
    void fft_2d_complex_to_complex_packed_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    const double *packing_table = step_info.twiddle_factors;
    size_t n = step_info.repeats; // n = number of original real rows
    size_t m = step_info.size;    // m = number of original real columns

    const ComplexD2 const1 = load(0.0, -0.0, 0.0, -0.0);
    const ComplexD2 const2 = load(-0.0, 0.0, -0.0, 0.0);

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
        // For IFFT: data_in = last row (temporary variable), data_out is actually both input and output!
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
            ComplexD x_in = load128(data_in + j + n*m/2);
            ComplexD x_out = change_sign(x_in, const1_128);
            store(x_out, data_out + j + n*m/2);
        }
    }

    for (size_t i = 2; i < n/2; i+=2)
    {
        ComplexD2 k = broadcast128(packing_table + i);

        if(!forward)
        {
            k = change_sign(k, const1);
        }

        size_t j = 0;
        // First use avx
        for (j = 0; j + 2 < 2*m; j+=4)
        {
            ComplexD2 x0_in = load(data_in + i*m + j);
            ComplexD2 x1_in = load(data_in + (n - i)*m + j);

            ComplexD2 temp0 = x0_in + change_sign(x1_in, const2);
            ComplexD2 temp1 = mul(k, temp0);

            ComplexD2 x0_out = temp1 + x0_in;
            ComplexD2 x1_out = change_sign(temp1, const2) + x1_in;

            store(x0_out, data_out + i*m + j);
            store(x1_out, data_out + (n - i)*m + j);
        }

        // Then if needed, use sse2
        if (j < 2*m)
        {
            ComplexD k = load128(packing_table + i);
            if(!forward)
            {
                k = change_sign(k, const1_128);
            }

            ComplexD x0_in = load128(data_in + i*m + j);
            ComplexD x1_in = load128(data_in + (n - i)*m + j);

            ComplexD temp0 = x0_in + change_sign(x1_in, const2_128);
            ComplexD temp1 = mul(k, temp0);

            ComplexD x0_out = temp1 + x0_in;
            ComplexD x1_out = change_sign(temp1, const2_128) + x1_in;

            store(x0_out, data_out + i*m + j);
            store(x1_out, data_out + (n - i)*m + j);
        }
    }
}

// Instantiations of the functions defined in this class
template void fft_2d_complex_to_complex_packed_avx_d<false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_to_complex_packed_avx_d<true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

