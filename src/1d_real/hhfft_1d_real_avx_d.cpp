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
    void fft_1d_complex_to_complex_packed_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    const double *packing_table = step_info.twiddle_factors;
    size_t n = 2*step_info.repeats; // n = number of original real numbers

    const ComplexD2 const1 = load(0.0, -0.0, 0.0, -0.0);
    const ComplexD2 const2 = load(-0.0, 0.0, -0.0, 0.0);

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
        ComplexD x_in = load128(data_in + n/2);        
        ComplexD x_out = change_sign(x_in, const1_128);
        store(x_out, data_out + n/2);
    }

    size_t i;
    for (i = 2; i + 2 < n/2; i+=4)
    {
        ComplexD2 sssc = load(packing_table + i);
        ComplexD2 x0_in = load(data_in + i);
        ComplexD2 x1_in = load_two_128(data_in + n - i, data_in + n - i - 2);

        if(!forward)
        {
            sssc = change_sign(sssc, const1);           
        }

        ComplexD2 temp0 = x0_in + change_sign(x1_in, const2);
        ComplexD2 temp1 = mul(sssc, temp0);

        ComplexD2 x0_out = temp1 + x0_in;
        ComplexD2 x1_out = change_sign(temp1, const2) + x1_in;

        store(x0_out, data_out + i);
        store_two_128(x1_out, data_out + n - i, data_out + n - i - 2);
    }

    if (i < n/2)
    {
        ComplexD sssc = load128(packing_table + i);
        ComplexD x0_in = load128(data_in + i);
        ComplexD x1_in = load128(data_in + n - i);

        if(!forward)
        {
            sssc = change_sign(sssc, const1_128);            
        }

        ComplexD temp0 = x0_in + change_sign(x1_in, const2_128);
        ComplexD temp1 = mul(sssc, temp0);

        ComplexD x0_out = temp1 + x0_in;
        ComplexD x1_out = change_sign(temp1, const2_128) + x1_in;

        store(x0_out, data_out + i);
        store(x1_out, data_out + n - i);
    }
}

// This is found in hhfft_1d_complex_avx_d.cpp
template<bool scale> void fft_1d_complex_reorder_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

void fft_1d_complex_to_complex_packed_ifft_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    // If data_in == data_out,
    if(data_in == data_out)
    {
        fft_1d_complex_to_complex_packed_avx_d<false>(data_in, data_out, step_info);
        fft_1d_complex_reorder_avx_d<true>(data_out, data_out, step_info);
        return;
    }

    const double *packing_table = step_info.twiddle_factors;
    size_t n = step_info.repeats; // 2*n = number of original real numbers
    uint32_t *reorder_table_inverse = step_info.reorder_table;

    double k = step_info.norm_factor;
    ComplexD k128 = broadcast64(k);
    ComplexD2 k256 = broadcast64_D2(k);

    const ComplexD2 const1 = load(0.0, -0.0, 0.0, -0.0);
    const ComplexD2 const2 = load(-0.0, 0.0, -0.0, 0.0);

    double x_r = data_in[0];
    double x_i = data_in[2*n];
    data_out[0] = 0.5*k*(x_r + x_i);
    data_out[1] = 0.5*k*(x_r - x_i);

    if (n%2 == 0)
    {
        size_t i = reorder_table_inverse[n/2];

        ComplexD x_in = load128(data_in + n);
        ComplexD x_out = k128*change_sign(x_in, const1_128);
        store(x_out, data_out + 2*i);
    }

    size_t i;
    for (i = 1; i + 1 < (n+1)/2; i+=2)
    {
        ComplexD2 sssc = load(packing_table + 2*i);
        sssc = change_sign(sssc, const1);
        ComplexD2 x0_in = k256*load(data_in + 2*i);
        ComplexD2 x1_in = k256*load_two_128(data_in + 2*(n - i), data_in + 2*(n - i) - 2);

        ComplexD2 temp0 = x0_in + change_sign(x1_in, const2);
        ComplexD2 temp1 = mul(sssc, temp0);

        ComplexD2 x0_out = temp1 + x0_in;
        ComplexD2 x1_out = change_sign(temp1, const2) + x1_in;

        size_t i2 = reorder_table_inverse[i];
        size_t i3 = reorder_table_inverse[i + 1];
        size_t i4 = reorder_table_inverse[n - i];
        size_t i5 = reorder_table_inverse[n - i - 1];

        store_two_128(x0_out, data_out + 2*i2, data_out + 2*i3);
        store_two_128(x1_out, data_out + 2*i4, data_out + 2*i5);
    }

    if (i < (n+1)/2)
    {
        ComplexD sssc = load128(packing_table + 2*i);
        sssc = change_sign(sssc, const1_128);
        ComplexD x0_in = k128*load128(data_in + 2*i);
        ComplexD x1_in = k128*load128(data_in + 2*(n - i));

        ComplexD temp0 = x0_in + change_sign(x1_in, const2_128);
        ComplexD temp1 = mul(sssc, temp0);

        ComplexD x0_out = temp1 + x0_in;
        ComplexD x1_out = change_sign(temp1, const2_128) + x1_in;

        size_t i2 = reorder_table_inverse[i];
        size_t i3 = reorder_table_inverse[n - i];

        store(x0_out, data_out + 2*i2);
        store(x1_out, data_out + 2*i3);
    }
}

// Instantiations of the functions defined in this class
template void fft_1d_complex_to_complex_packed_avx_d<false>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template void fft_1d_complex_to_complex_packed_avx_d<true>(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
