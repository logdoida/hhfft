/*
*   Copyright Jouko Kalmari 2017
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

#ifndef HHFFT_1D_PLAIN_IMPL_H
#define HHFFT_1D_PLAIN_IMPL_H

#include "step_info.h"

#include <vector>
#include <array>
#include <cmath>
#include <assert.h>
#include <iostream> // TESTING

namespace hhfft
{

// TODO it might be a good idea to have versions that also do the first fft step (at least for radix 2) (NOT needed if using DIF)
// NOTE this is out of place reordering!
template<typename T, size_t arch, bool forward> void fft_1d_reorder(const T *data_in, T *data_out, hhfft::StepInfo<T> &step_info)
{
    size_t n = step_info.repeats;    
    uint32_t *reorder_table = step_info.reorder_table;

    // Needed only in ifft. Equal to 1/N
    T k = step_info.norm_factor;

    for (size_t i = 0; i < n; i++)
    {
        if (forward)
        {
            data_out[2*i+0] = data_in[2*reorder_table[i]+0];
            data_out[2*i+1] = data_in[2*reorder_table[i]+1];
        } else
        {
            data_out[2*i+0] = k*data_in[2*reorder_table[i]+0];
            data_out[2*i+1] = k*data_in[2*reorder_table[i]+1];
        }
    }
}

// In-place reordering "swap"
template<typename T, size_t arch, bool forward> void fft_1d_reorder_in_place(const T *data_in, T *data_out, hhfft::StepInfo<T> &step_info)
{
    size_t n = step_info.repeats;
    uint32_t *reorder_table = step_info.reorder_table;

    // In-place algorithm
    assert (data_in == data_out);

    for (size_t i = 0; i < n; i++)
    {
        size_t ind1 = i + 1; // First one has been omitted!
        size_t ind2 = reorder_table[i];

        T r_temp = data_in[2*ind1+0];
        T c_temp = data_in[2*ind1+1];
        data_out[2*ind1+0] = data_out[2*ind2+0];
        data_out[2*ind1+1] = data_out[2*ind2+1];
        data_out[2*ind2+0] = r_temp;
        data_out[2*ind2+1] = c_temp;
    }

    // Scaling needs to be done as a separate step as some data might be copied twice or zero times
    // TODO this is note very efficient. Is there some other way?
    size_t n2 = step_info.stride;
    if (!forward)
    {
        // Needed only in ifft. Equal to 1/N
        T k = step_info.norm_factor;

        for (size_t i = 0; i < 2*n2; i++)
        {
            data_out[i] *= k;
        }
    }
}

// In-place reordering "cycle"
/*
template<typename T, size_t arch, bool forward> void fft_1d_reorder_in_place(const T *data_in, T *data_out, hhfft::StepInfo<T> &step_info)
{
    size_t n = step_info.repeats;
    uint32_t *reorder_table = step_info.reorder_table;

    // In-place algorithm
    assert (data_in == data_out);

    // First index is uses as temporary storage, so it needs to be saved
    T r_temp = data_in[0];
    T c_temp = data_in[1];

    // A slightly faster implementation
    // TODO is there any way to make this even faster?
    size_t ind1 = 2*reorder_table[0];
    for (size_t i = 1; i < n; i++)
    {
        size_t ind2 = 2*reorder_table[i];

        for (size_t j = 0; j < 2; j++)
        {
            data_out[ind1+j] = data_out[ind2+j];
        }
        ind1 = ind2;
    }

    data_out[0] = r_temp;
    data_out[1] = c_temp;

    // Scaling needs to be done as a separate step as some data might be copied twice or zero times
    // TODO this is note very efficient. Is there some other way?
    size_t n2 = step_info.stride;
    if (!forward)
    {
        // Needed only in ifft. Equal to 1/N
        T k = step_info.norm_factor;

        for (size_t i = 0; i < 2*n2; i++)
        {
            data_out[i] *= k;
        }
    }
}
*/

template<typename T, size_t radix, bool forward> void multiply_coeff(const T *x_in, T *x_out)
{
    if (radix == 2)
    {
        x_out[0] = x_in[0] + x_in[2];
        x_out[1] = x_in[1] + x_in[3];
        x_out[2] = x_in[0] - x_in[2];
        x_out[3] = x_in[1] - x_in[3];
    }

    if (radix == 3)
    {
        if (forward)
        {
            T k1 = (T) 0.5;
            T k2 = (T) 0.5*sqrt(3);
            x_out[0] = x_in[0] + x_in[2] + x_in[4];
            x_out[1] = x_in[1] + x_in[3] + x_in[5];
            x_out[2] = x_in[0] - k1*x_in[2] + k2*x_in[3] - k1*x_in[4] - k2*x_in[5];
            x_out[3] = x_in[1] - k2*x_in[2] - k1*x_in[3] + k2*x_in[4] - k1*x_in[5];
            x_out[4] = x_in[0] - k1*x_in[2] - k2*x_in[3] - k1*x_in[4] + k2*x_in[5];
            x_out[5] = x_in[1] + k2*x_in[2] - k1*x_in[3] - k2*x_in[4] - k1*x_in[5];
        } else
        {
            T k1 = (T) 0.5;
            T k2 = (T) 0.5*sqrt(3);
            x_out[0] = x_in[0] + x_in[2] + x_in[4];
            x_out[1] = x_in[1] + x_in[3] + x_in[5];
            x_out[2] = x_in[0] - k1*x_in[2] - k2*x_in[3] - k1*x_in[4] + k2*x_in[5];
            x_out[3] = x_in[1] + k2*x_in[2] - k1*x_in[3] - k2*x_in[4] - k1*x_in[5];
            x_out[4] = x_in[0] - k1*x_in[2] + k2*x_in[3] - k1*x_in[4] - k2*x_in[5];
            x_out[5] = x_in[1] - k2*x_in[2] - k1*x_in[3] + k2*x_in[4] - k1*x_in[5];
        }
    }

    if (radix == 4)
    {                        
        if (forward)
        {
            x_out[0] = x_in[0] + x_in[2] + x_in[4] + x_in[6];
            x_out[1] = x_in[1] + x_in[3] + x_in[5] + x_in[7];
            x_out[2] = x_in[0] + x_in[3] - x_in[4] - x_in[7];
            x_out[3] = x_in[1] - x_in[2] - x_in[5] + x_in[6];
            x_out[4] = x_in[0] - x_in[2] + x_in[4] - x_in[6];
            x_out[5] = x_in[1] - x_in[3] + x_in[5] - x_in[7];
            x_out[6] = x_in[0] - x_in[3] - x_in[4] + x_in[7];
            x_out[7] = x_in[1] + x_in[2] - x_in[5] - x_in[6];
        } else
        {
            x_out[0] = x_in[0] + x_in[2] + x_in[4] + x_in[6];
            x_out[1] = x_in[1] + x_in[3] + x_in[5] + x_in[7];
            x_out[2] = x_in[0] - x_in[3] - x_in[4] + x_in[7];
            x_out[3] = x_in[1] + x_in[2] - x_in[5] - x_in[6];
            x_out[4] = x_in[0] - x_in[2] + x_in[4] - x_in[6];
            x_out[5] = x_in[1] - x_in[3] + x_in[5] - x_in[7];
            x_out[6] = x_in[0] + x_in[3] - x_in[4] - x_in[7];
            x_out[7] = x_in[1] - x_in[2] - x_in[5] + x_in[6];
        }
    }

    if (radix == 5)
    {
        T k1 = cos(2*M_PI*1.0/5.0);
        T k2 = sin(2*M_PI*1.0/5.0);
        T k3 =-cos(2*M_PI*2.0/5.0);
        T k4 = sin(2*M_PI*2.0/5.0);
        if (forward)
        {
            x_out[0] = x_in[0] + x_in[2] + x_in[4] + x_in[6] + x_in[8];
            x_out[1] = x_in[1] + x_in[3] + x_in[5] + x_in[7] + x_in[9];
            x_out[2] = x_in[0] + k1*x_in[2] + k2*x_in[3] - k3*x_in[4] + k4*x_in[5] - k3*x_in[6] - k4*x_in[7] + k1*x_in[8] - k2*x_in[9];
            x_out[3] = x_in[1] - k2*x_in[2] + k1*x_in[3] - k4*x_in[4] - k3*x_in[5] + k4*x_in[6] - k3*x_in[7] + k2*x_in[8] + k1*x_in[9];
            x_out[4] = x_in[0] - k3*x_in[2] + k4*x_in[3] + k1*x_in[4] - k2*x_in[5] + k1*x_in[6] + k2*x_in[7] - k3*x_in[8] - k4*x_in[9];
            x_out[5] = x_in[1] - k4*x_in[2] - k3*x_in[3] + k2*x_in[4] + k1*x_in[5] - k2*x_in[6] + k1*x_in[7] + k4*x_in[8] - k3*x_in[9];
            x_out[6] = x_in[0] - k3*x_in[2] - k4*x_in[3] + k1*x_in[4] + k2*x_in[5] + k1*x_in[6] - k2*x_in[7] - k3*x_in[8] + k4*x_in[9];
            x_out[7] = x_in[1] + k4*x_in[2] - k3*x_in[3] - k2*x_in[4] + k1*x_in[5] + k2*x_in[6] + k1*x_in[7] - k4*x_in[8] - k3*x_in[9];
            x_out[8] = x_in[0] + k1*x_in[2] - k2*x_in[3] - k3*x_in[4] - k4*x_in[5] - k3*x_in[6] + k4*x_in[7] + k1*x_in[8] + k2*x_in[9];
            x_out[9] = x_in[1] + k2*x_in[2] + k1*x_in[3] + k4*x_in[4] - k3*x_in[5] - k4*x_in[6] - k3*x_in[7] - k2*x_in[8] + k1*x_in[9];
        } else
        {
            x_out[0] = x_in[0] + x_in[2] + x_in[4] + x_in[6] + x_in[8];
            x_out[1] = x_in[1] + x_in[3] + x_in[5] + x_in[7] + x_in[9];
            x_out[2] = x_in[0] + k1*x_in[2] - k2*x_in[3] - k3*x_in[4] - k4*x_in[5] - k3*x_in[6] + k4*x_in[7] + k1*x_in[8] + k2*x_in[9];
            x_out[3] = x_in[1] + k2*x_in[2] + k1*x_in[3] + k4*x_in[4] - k3*x_in[5] - k4*x_in[6] - k3*x_in[7] - k2*x_in[8] + k1*x_in[9];
            x_out[4] = x_in[0] - k3*x_in[2] - k4*x_in[3] + k1*x_in[4] + k2*x_in[5] + k1*x_in[6] - k2*x_in[7] - k3*x_in[8] + k4*x_in[9];
            x_out[5] = x_in[1] + k4*x_in[2] - k3*x_in[3] - k2*x_in[4] + k1*x_in[5] + k2*x_in[6] + k1*x_in[7] - k4*x_in[8] - k3*x_in[9];
            x_out[6] = x_in[0] - k3*x_in[2] + k4*x_in[3] + k1*x_in[4] - k2*x_in[5] + k1*x_in[6] + k2*x_in[7] - k3*x_in[8] - k4*x_in[9];
            x_out[7] = x_in[1] - k4*x_in[2] - k3*x_in[3] + k2*x_in[4] + k1*x_in[5] - k2*x_in[6] + k1*x_in[7] + k4*x_in[8] - k3*x_in[9];
            x_out[8] = x_in[0] + k1*x_in[2] + k2*x_in[3] - k3*x_in[4] + k4*x_in[5] - k3*x_in[6] - k4*x_in[7] + k1*x_in[8] - k2*x_in[9];
            x_out[9] = x_in[1] - k2*x_in[2] + k1*x_in[3] - k4*x_in[4] - k3*x_in[5] + k4*x_in[6] - k3*x_in[7] + k2*x_in[8] + k1*x_in[9];
        }
    }

    if (radix == 7)
    {
        T k1 = cos(2*M_PI*1.0/7.0);
        T k2 = sin(2*M_PI*1.0/7.0);
        T k3 =-cos(2*M_PI*2.0/7.0);
        T k4 = sin(2*M_PI*2.0/7.0);
        T k5 =-cos(2*M_PI*3.0/7.0);
        T k6 = sin(2*M_PI*3.0/7.0);
        if (forward)
        {
            x_out[0] = x_in[0] + x_in[2] + x_in[4] + x_in[6] + x_in[8] + x_in[10] + x_in[12];
            x_out[1] = x_in[1] + x_in[3] + x_in[5] + x_in[7] + x_in[9] + x_in[11] + x_in[13];
            x_out[2] = x_in[0] + k1*x_in[2] + k2*x_in[3] - k3*x_in[4] + k4*x_in[5] - k5*x_in[6] + k6*x_in[7] - k5*x_in[8] - k6*x_in[9] - k3*x_in[10] - k4*x_in[11] + k1*x_in[12] - k2*x_in[13];
            x_out[3] = x_in[1] - k2*x_in[2] + k1*x_in[3] - k4*x_in[4] - k3*x_in[5] - k6*x_in[6] - k5*x_in[7] + k6*x_in[8] - k5*x_in[9] + k4*x_in[10] - k3*x_in[11] + k2*x_in[12] + k1*x_in[13];
            x_out[4] = x_in[0] - k3*x_in[2] + k4*x_in[3] - k5*x_in[4] - k6*x_in[5] + k1*x_in[6] - k2*x_in[7] + k1*x_in[8] + k2*x_in[9] - k5*x_in[10] + k6*x_in[11] - k3*x_in[12] - k4*x_in[13];
            x_out[5] = x_in[1] - k4*x_in[2] - k3*x_in[3] + k6*x_in[4] - k5*x_in[5] + k2*x_in[6] + k1*x_in[7] - k2*x_in[8] + k1*x_in[9] - k6*x_in[10] - k5*x_in[11] + k4*x_in[12] - k3*x_in[13];
            x_out[6] = x_in[0] - k5*x_in[2] + k6*x_in[3] + k1*x_in[4] - k2*x_in[5] - k3*x_in[6] + k4*x_in[7] - k3*x_in[8] - k4*x_in[9] + k1*x_in[10] + k2*x_in[11] - k5*x_in[12] - k6*x_in[13];
            x_out[7] = x_in[1] - k6*x_in[2] - k5*x_in[3] + k2*x_in[4] + k1*x_in[5] - k4*x_in[6] - k3*x_in[7] + k4*x_in[8] - k3*x_in[9] - k2*x_in[10] + k1*x_in[11] + k6*x_in[12] - k5*x_in[13];
            x_out[8] = x_in[0] - k5*x_in[2] - k6*x_in[3] + k1*x_in[4] + k2*x_in[5] - k3*x_in[6] - k4*x_in[7] - k3*x_in[8] + k4*x_in[9] + k1*x_in[10] - k2*x_in[11] - k5*x_in[12] + k6*x_in[13];
            x_out[9] = x_in[1] + k6*x_in[2] - k5*x_in[3] - k2*x_in[4] + k1*x_in[5] + k4*x_in[6] - k3*x_in[7] - k4*x_in[8] - k3*x_in[9] + k2*x_in[10] + k1*x_in[11] - k6*x_in[12] - k5*x_in[13];
            x_out[10] = x_in[0] - k3*x_in[2] - k4*x_in[3] - k5*x_in[4] + k6*x_in[5] + k1*x_in[6] + k2*x_in[7] + k1*x_in[8] - k2*x_in[9] - k5*x_in[10] - k6*x_in[11] - k3*x_in[12] + k4*x_in[13];
            x_out[11] = x_in[1] + k4*x_in[2] - k3*x_in[3] - k6*x_in[4] - k5*x_in[5] - k2*x_in[6] + k1*x_in[7] + k2*x_in[8] + k1*x_in[9] + k6*x_in[10] - k5*x_in[11] - k4*x_in[12] - k3*x_in[13];
            x_out[12] = x_in[0] + k1*x_in[2] - k2*x_in[3] - k3*x_in[4] - k4*x_in[5] - k5*x_in[6] - k6*x_in[7] - k5*x_in[8] + k6*x_in[9] - k3*x_in[10] + k4*x_in[11] + k1*x_in[12] + k2*x_in[13];
            x_out[13] = x_in[1] + k2*x_in[2] + k1*x_in[3] + k4*x_in[4] - k3*x_in[5] + k6*x_in[6] - k5*x_in[7] - k6*x_in[8] - k5*x_in[9] - k4*x_in[10] - k3*x_in[11] - k2*x_in[12] + k1*x_in[13];
        } else
        {
            x_out[0] = x_in[0] + x_in[2] + x_in[4] + x_in[6] + x_in[8] + x_in[10] + x_in[12];
            x_out[1] = x_in[1] + x_in[3] + x_in[5] + x_in[7] + x_in[9] + x_in[11] + x_in[13];
            x_out[2] = x_in[0] + k1*x_in[2] - k2*x_in[3] - k3*x_in[4] - k4*x_in[5] - k5*x_in[6] - k6*x_in[7] - k5*x_in[8] + k6*x_in[9] - k3*x_in[10] + k4*x_in[11] + k1*x_in[12] + k2*x_in[13];
            x_out[3] = x_in[1] + k2*x_in[2] + k1*x_in[3] + k4*x_in[4] - k3*x_in[5] + k6*x_in[6] - k5*x_in[7] - k6*x_in[8] - k5*x_in[9] - k4*x_in[10] - k3*x_in[11] - k2*x_in[12] + k1*x_in[13];
            x_out[4] = x_in[0] - k3*x_in[2] - k4*x_in[3] - k5*x_in[4] + k6*x_in[5] + k1*x_in[6] + k2*x_in[7] + k1*x_in[8] - k2*x_in[9] - k5*x_in[10] - k6*x_in[11] - k3*x_in[12] + k4*x_in[13];
            x_out[5] = x_in[1] + k4*x_in[2] - k3*x_in[3] - k6*x_in[4] - k5*x_in[5] - k2*x_in[6] + k1*x_in[7] + k2*x_in[8] + k1*x_in[9] + k6*x_in[10] - k5*x_in[11] - k4*x_in[12] - k3*x_in[13];
            x_out[6] = x_in[0] - k5*x_in[2] - k6*x_in[3] + k1*x_in[4] + k2*x_in[5] - k3*x_in[6] - k4*x_in[7] - k3*x_in[8] + k4*x_in[9] + k1*x_in[10] - k2*x_in[11] - k5*x_in[12] + k6*x_in[13];
            x_out[7] = x_in[1] + k6*x_in[2] - k5*x_in[3] - k2*x_in[4] + k1*x_in[5] + k4*x_in[6] - k3*x_in[7] - k4*x_in[8] - k3*x_in[9] + k2*x_in[10] + k1*x_in[11] - k6*x_in[12] - k5*x_in[13];
            x_out[8] = x_in[0] - k5*x_in[2] + k6*x_in[3] + k1*x_in[4] - k2*x_in[5] - k3*x_in[6] + k4*x_in[7] - k3*x_in[8] - k4*x_in[9] + k1*x_in[10] + k2*x_in[11] - k5*x_in[12] - k6*x_in[13];
            x_out[9] = x_in[1] - k6*x_in[2] - k5*x_in[3] + k2*x_in[4] + k1*x_in[5] - k4*x_in[6] - k3*x_in[7] + k4*x_in[8] - k3*x_in[9] - k2*x_in[10] + k1*x_in[11] + k6*x_in[12] - k5*x_in[13];
            x_out[10] = x_in[0] - k3*x_in[2] + k4*x_in[3] - k5*x_in[4] - k6*x_in[5] + k1*x_in[6] - k2*x_in[7] + k1*x_in[8] + k2*x_in[9] - k5*x_in[10] + k6*x_in[11] - k3*x_in[12] - k4*x_in[13];
            x_out[11] = x_in[1] - k4*x_in[2] - k3*x_in[3] + k6*x_in[4] - k5*x_in[5] + k2*x_in[6] + k1*x_in[7] - k2*x_in[8] + k1*x_in[9] - k6*x_in[10] - k5*x_in[11] + k4*x_in[12] - k3*x_in[13];
            x_out[12] = x_in[0] + k1*x_in[2] + k2*x_in[3] - k3*x_in[4] + k4*x_in[5] - k5*x_in[6] + k6*x_in[7] - k5*x_in[8] - k6*x_in[9] - k3*x_in[10] - k4*x_in[11] + k1*x_in[12] - k2*x_in[13];
            x_out[13] = x_in[1] - k2*x_in[2] + k1*x_in[3] - k4*x_in[4] - k3*x_in[5] - k6*x_in[6] - k5*x_in[7] + k6*x_in[8] - k5*x_in[9] + k4*x_in[10] - k3*x_in[11] + k2*x_in[12] + k1*x_in[13];
        }
    }
}

// This function can be used on the first level, when there are no twiddle factors
template<typename T, size_t radix, size_t arch, bool forward> void fft_1d_one_level(const T *data_in, T *data_out, hhfft::StepInfo<T> &step_info)
{
    size_t stride = step_info.stride;
    size_t repeats = step_info.repeats;

    T x_temp_in[2*radix];
    T x_temp_out[2*radix];

    for (size_t i = 0; i < repeats; i++)
    {
        for (size_t k = 0; k < stride; k++)
        {
            // Read in the values used in this step
            for (size_t j = 0; j < radix; j++)
            {
                x_temp_in[2*j + 0] = data_in[2*(i*radix*stride + j*stride + k) + 0];
                x_temp_in[2*j + 1] = data_in[2*(i*radix*stride + j*stride + k) + 1];
            }

            // Multiply with coefficients
            multiply_coeff<T,radix,forward>(x_temp_in, x_temp_out);

            // Write in the values used in this step
            for (size_t j = 0; j < radix; j++)
            {
                data_out[2*(i*radix*stride + j*stride + k) + 0] = x_temp_out[2*j + 0];
                data_out[2*(i*radix*stride + j*stride + k) + 1] = x_temp_out[2*j + 1];
            }
        }
    }    
}

// This function is to be used when there are twiddle factors for DIT
// forward = 1 (forward) or -1 (inverse)
template<typename T, size_t radix, size_t arch, bool forward> void fft_1d_one_level_twiddle(const T *data_in, T *data_out, hhfft::StepInfo<T> &step_info)
{    
    assert(step_info.forward == forward);

    size_t stride = step_info.stride;
    size_t repeats = step_info.repeats;

    T x_temp_in[2*radix];
    T x_temp_out[2*radix];

    for (size_t i = 0; i < repeats; i++)
    {
        for (size_t k = 0; k < stride; k++)
        {
            // It is assumed that first twiddle factors are always (1 + 0i)
            x_temp_in[0] = data_in[2*(i*radix*stride + k) + 0];
            x_temp_in[1] = data_in[2*(i*radix*stride + k) + 1];

            // Read in the values used in this step and multiply them with twiddle factors
            for (size_t j = 1; j < radix; j++)
            {
                T x_r = data_in[2*(i*radix*stride + j*stride + k) + 0];
                T x_i = data_in[2*(i*radix*stride + j*stride + k) + 1];
                T w_r = step_info.twiddle_factors[2*(j*stride + k) + 0];
                T w_i = step_info.twiddle_factors[2*(j*stride + k) + 1];

                if (forward == 1)
                {
                    x_temp_in[2*j + 0] = w_r*x_r - w_i*x_i;
                    x_temp_in[2*j + 1] = w_i*x_r + w_r*x_i;
                } else
                {
                    x_temp_in[2*j + 0] =  w_r*x_r + w_i*x_i;
                    x_temp_in[2*j + 1] = -w_i*x_r + w_r*x_i;
                }

                // TESTING save the data multiplied with twiddle factors
                //data_out[2*(i*radix*stride + j*stride + k) + 0] = x_temp_real[j];
                //data_out[2*(i*radix*stride + j*stride + k) + 1] = x_temp_imag[j];
            }

            multiply_coeff<T,radix,forward>(x_temp_in, x_temp_out);

            // Write in the values used in this step
            for (size_t j = 0; j < radix; j++)
            {
                data_out[2*(i*radix*stride + j*stride + k) + 0] = x_temp_out[2*j + 0];
                data_out[2*(i*radix*stride + j*stride + k) + 1] = x_temp_out[2*j + 1];
            }
        }
    }
}


// This function is to be used when there are twiddle factors for DIF
// forward = 1 (forward) or -1 (inverse)
template<typename T, size_t radix, size_t arch, bool forward> void fft_1d_one_level_twiddle_DIF(const T *data_in, T *data_out, hhfft::StepInfo<T> &step_info)
{
    assert(step_info.forward == forward);

    size_t stride = step_info.stride;
    size_t repeats = step_info.repeats;

    T x_temp_in[2*radix];
    T x_temp_out[2*radix];

    for (size_t i = 0; i < repeats; i++)
    {
        for (size_t k = 0; k < stride; k++)
        {
            // It is assumed that first twiddle factors are always (1 + 0i)
            x_temp_in[0] = data_in[2*(i*radix*stride + k) + 0];
            x_temp_in[1] = data_in[2*(i*radix*stride + k) + 1];

            // Read in the values used in this step and multiply them with twiddle factors
            for (size_t j = 1; j < radix; j++)
            {
                T x_r = data_in[2*(i*radix*stride + j*stride + k) + 0];
                T x_i = data_in[2*(i*radix*stride + j*stride + k) + 1];
                T w_r = step_info.twiddle_factors[2*(i*radix + j) + 0];
                T w_i = step_info.twiddle_factors[2*(i*radix + j) + 1];

                if (forward == 1)
                {
                    x_temp_in[2*j + 0] = w_r*x_r - w_i*x_i;
                    x_temp_in[2*j + 1] = w_i*x_r + w_r*x_i;
                } else
                {
                    x_temp_in[2*j + 0] =  w_r*x_r + w_i*x_i;
                    x_temp_in[2*j + 1] = -w_i*x_r + w_r*x_i;
                }

                // TESTING save the data multiplied with twiddle factors
                //data_out[2*(i*radix*stride + j*stride + k) + 0] = x_temp_real[j];
                //data_out[2*(i*radix*stride + j*stride + k) + 1] = x_temp_imag[j];
            }

            multiply_coeff<T,radix,forward>(x_temp_in, x_temp_out);

            // Write in the values used in this step
            for (size_t j = 0; j < radix; j++)
            {
                data_out[2*(i*radix*stride + j*stride + k) + 0] = x_temp_out[2*j + 0];
                data_out[2*(i*radix*stride + j*stride + k) + 1] = x_temp_out[2*j + 1];
            }
        }
    }
}

}

#endif // HHFFT_1D_PLAIN_IMPL_H
