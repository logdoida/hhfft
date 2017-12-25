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

#ifndef HHFFT_1D_PLAIN_REAL_IMPL_H
#define HHFFT_1D_PLAIN_REAL_IMPL_H

#include "step_info.h"
#include "aligned_arrays.h"

#include <vector>
#include <array>
#include <cmath>
#include <assert.h>
#include <immintrin.h>
#include <iostream> // TESTING

namespace hhfft
{

// TODO it might be a good idea to have versions that also do the first fft step (at least for radix 2)
// NOTE this is out of place reordering!
template<typename T, size_t arch, bool forward> void fft_real_1d_reorder(const T *data_in, T *data_out, hhfft::StepInfoReal<T> &step_info)
{
    size_t n = step_info.repeats;
    uint32_t *reorder_table = step_info.reorder_table;

    // Needed only in ifft. Equal to 1/N
    T k = step_info.norm_factor;

    for (size_t i = 0; i < n; i++)
    {
        size_t i2 = reorder_table[i];
        if (forward)
        {
            data_out[i] = data_in[i2];
        } else
        {
            data_out[i] = k*data_in[i2];
        }
    }
}

template<typename T, size_t radix> void multiply_coeff_real(const T *x_in, T *x_out)
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
        T k1 = (T) 0.5;
        T k2 = (T) 0.5*sqrt(3);
        x_out[0] = x_in[0] + x_in[2] + x_in[4];
        x_out[1] = x_in[1] + x_in[3] + x_in[5];
        x_out[2] = x_in[0] - k1*x_in[2] + k2*x_in[3] - k1*x_in[4] - k2*x_in[5];
        x_out[3] = x_in[1] - k2*x_in[2] - k1*x_in[3] + k2*x_in[4] - k1*x_in[5];
        x_out[4] = x_in[0] - k1*x_in[2] - k2*x_in[3] - k1*x_in[4] + k2*x_in[5];
        x_out[5] = x_in[1] + k2*x_in[2] - k1*x_in[3] - k2*x_in[4] - k1*x_in[5];
    }

    if (radix == 4)
    {
            x_out[0] = x_in[0] + x_in[2] + x_in[4] + x_in[6];
            x_out[1] = x_in[1] + x_in[3] + x_in[5] + x_in[7];
            x_out[2] = x_in[0] + x_in[3] - x_in[4] - x_in[7];
            x_out[3] = x_in[1] - x_in[2] - x_in[5] + x_in[6];
            x_out[4] = x_in[0] - x_in[2] + x_in[4] - x_in[6];
            x_out[5] = x_in[1] - x_in[3] + x_in[5] - x_in[7];
            x_out[6] = x_in[0] - x_in[3] - x_in[4] + x_in[7];
            x_out[7] = x_in[1] + x_in[2] - x_in[5] - x_in[6];
    }

    if (radix == 5)
    {
        T k1 = cos(2*M_PI*1.0/5.0);
        T k2 = sin(2*M_PI*1.0/5.0);
        T k3 =-cos(2*M_PI*2.0/5.0);
        T k4 = sin(2*M_PI*2.0/5.0);

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
    }

    if (radix == 7)
    {
        T k1 = cos(2*M_PI*1.0/7.0);
        T k2 = sin(2*M_PI*1.0/7.0);
        T k3 =-cos(2*M_PI*2.0/7.0);
        T k4 = sin(2*M_PI*2.0/7.0);
        T k5 =-cos(2*M_PI*3.0/7.0);
        T k6 = sin(2*M_PI*3.0/7.0);
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
    }
}

template<typename T, size_t radix> void multiply_coeff_real2(const T *x_in, T *x_out)
{
    if (radix == 2)
    {
        x_out[0] = x_in[0] + x_in[2];
        x_out[1] = x_in[1] + x_in[3];
        x_out[2] = x_in[0] - x_in[2];
        x_out[3] = -x_in[1] + x_in[3];
    }

    if (radix == 3)
    {
        // TODO
    }

    if (radix == 4)
    {
            x_out[0] = x_in[0] + x_in[2] + x_in[4] + x_in[6];
            x_out[1] = x_in[1] + x_in[3] + x_in[5] + x_in[7];
            x_out[2] = x_in[0] - x_in[3] - x_in[4] + x_in[7];
            x_out[3] = -x_in[1] - x_in[2] + x_in[5] + x_in[6];
            x_out[4] = x_in[0] + x_in[3] - x_in[4] - x_in[7];
            x_out[5] = x_in[1] - x_in[2] - x_in[5] + x_in[6];
            x_out[6] = x_in[0] - x_in[2] + x_in[4] - x_in[6];
            x_out[7] = -x_in[1] + x_in[3] - x_in[5] + x_in[7];
    }

    if (radix == 5)
    {
        // TODO
    }

    if (radix == 7)
    {
       // TODO
    }
}

// This function can be used on the first level of fft real, when there are no twiddle factors
template<typename T, size_t radix, size_t arch> void fft_real_1d_one_level(const T *data_in, T *data_out, hhfft::StepInfoReal<T> &step_info)
{
    // Must be the first level of DIT
    assert (step.info.stride == 1);
    size_t repeats = step_info.repeats;

    T x_temp_in[2*radix];
    T x_temp_out[2*radix];

    for (size_t i = 0; i < repeats; i++)
    {
        size_t index = i*radix;
        auto dir = step_info.directions[i];

        // Read in the values used in this step
        for (size_t j = 0; j < radix; j++)
        {
            // Input is always real!
            x_temp_in[2*j + 0] = data_in[index + j];
            x_temp_in[2*j + 1] = 0;
        }

        // Multiply with coefficients
        multiply_coeff_real<T,radix>(x_temp_in, x_temp_out);

        // radix is even (2 or 4). Direction is not needed to be taken into account.
        if (radix%2 == 0)
        {
            data_out[index + 0] = x_temp_out[0];
            data_out[index + 1] = x_temp_out[radix];
            for (size_t j = 2; j < radix; j++)
            {
                data_out[index + j] = x_temp_out[j];
            }
        } else // radix is odd
        {
            data_out[index + 0] = x_temp_out[0]; // First one is real

            for (size_t j = 1; j < radix; j+=2)
            {
                if (dir) // direction normal
                {
                    data_out[index + j + 0] = x_temp_out[j + 0];
                    data_out[index + j + 1] = x_temp_out[j + 1];
                } else // direction inverted
                {
                    data_out[index + j + 0] = x_temp_out[radix - j];
                    data_out[index + j + 1] = x_temp_out[radix + 2 - j];
                }
            }
        }
    }
}


// This function can be used when stride is even i.e. there are two real numbers in the beginning of a stride
template<typename T, size_t radix, size_t arch> void fft_real_1d_one_level_stride_even(const T *data_in, T *data_out, hhfft::StepInfoReal<T> &step_info)
{
    size_t stride = step_info.stride;
    size_t repeats = step_info.repeats;

    int dir_out = 1;
    for (size_t i = 0; i < repeats; i++)
    {
        size_t index = i*radix*stride;

        // The first two values in each stride are real, and must be taken into account together
        {
            T x0_temp_in[2*radix], x1_temp_in[2*radix];
            T x0_temp_out[2*radix], x1_temp_out[2*radix];

            for (size_t j = 0; j < radix; j++)
            {
                T x0_r = data_in[index + j*stride + 0];
                T x0_i = 0;
                T x1_r = data_in[index + j*stride + 1];
                T x1_i = 0;

                // only x1 needs to be multiplied with twiddle factors
                T w1_r = step_info.twiddle_factors[2*(j*stride + stride/2) + 0];
                T w1_i = step_info.twiddle_factors[2*(j*stride + stride/2) + 1];

                x0_temp_in[2*j + 0] = x0_r;
                x0_temp_in[2*j + 1] = x0_i;
                x1_temp_in[2*j + 0] = w1_r*x1_r - w1_i*x1_i;
                x1_temp_in[2*j + 1] = w1_i*x1_r + w1_r*x1_i;
            }

            // Does direction affect the multiplication here?
            multiply_coeff_real<T,radix>(x0_temp_in, x0_temp_out);
            multiply_coeff_real<T,radix>(x1_temp_in, x1_temp_out);

            // Real part from first
            data_out[index] = x0_temp_out[0];

            // only half is written            
            for (size_t j = 1; j < radix/2; j++)
            {                
                if (dir_out)
                {
                    data_out[index + 2*j*stride + 0] = x0_temp_out[2*j + 0];
                    data_out[index + 2*j*stride + 1] = x0_temp_out[2*j + 1];
                } else
                {
                    data_out[index + 2*j*stride + 0] = x0_temp_out[2*(radix/2 - j) + 0];
                    data_out[index + 2*j*stride + 1] = x0_temp_out[2*(radix/2 - j) + 1];
                }
            }

            // Real part from half way + 1
            data_out[index + 1] = x0_temp_out[radix];

            // only half is written            
            for (size_t j = 0; j < radix/2; j++)
            {
                if (dir_out)
                {
                    data_out[index + stride + 2*j*stride + 0] = x1_temp_out[2*j + 0];
                    data_out[index + stride + 2*j*stride + 1] = x1_temp_out[2*j + 1];
                } else
                {
                    data_out[index + stride + 2*j*stride + 0] = x1_temp_out[2*(radix/2 - j - 1) + 0];
                    data_out[index + stride + 2*j*stride + 1] = x1_temp_out[2*(radix/2 - j - 1) + 1];
                }
            }
        }

        // Rest of the values represent complex numbers
        for (size_t k = 1; k < stride/2; k++)
        {            
            T x_temp_in[2*radix];
            T x_temp_out[2*radix];
            int dir_in = (i*radix + 1)&1;

            // Read in the values used in this step and multiply them with twiddle factors
            for (size_t j = 0; j < radix; j++)
            {
                // direction affects in which direction the inputs are read
                T x_r, x_i;
                if (dir_in)
                {
                    x_r = data_in[index + 2*k + j*stride + 0];
                    x_i = data_in[index + 2*k + j*stride + 1];
                } else
                {
                    x_r = data_in[index + 2*(stride/2 - k) + j*stride + 0];
                    x_i = data_in[index + 2*(stride/2 - k) + j*stride + 1];
                }

                T w_r = step_info.twiddle_factors[2*(j*stride + k) + 0];
                T w_i = step_info.twiddle_factors[2*(j*stride + k) + 1];

                x_temp_in[2*j + 0] = w_r*x_r - w_i*x_i;
                x_temp_in[2*j + 1] = w_i*x_r + w_r*x_i;

                // TESTING
                //std::cout << "k = " << k << ", j = " << j << ", x_in = " << x_r << ", " << x_i << ", w = " << w_r << ", " << w_i << ", x_temp = " << x_temp_in[2*j + 0] << ", " << x_temp_in[2*j + 1] << std::endl;

                dir_in = dir_in^1;
            }

            multiply_coeff_real2<T,radix>(x_temp_in, x_temp_out);

            // reverse the output order if required
            // TODO this could be done already in multiply_coeff
            if (!dir_out)
            {
                for (size_t j = 0; j < radix/2; j++)
                {
                    T x_r = x_temp_out[2*j + 0]; x_temp_out[2*j + 0] = x_temp_out[2*(radix - j - 1) + 0]; x_temp_out[2*(radix - j - 1) + 0] = x_r;
                    T x_i = x_temp_out[2*j + 1]; x_temp_out[2*j + 1] = x_temp_out[2*(radix - j - 1) + 1]; x_temp_out[2*(radix - j - 1) + 1] = x_i;
                }
            }

            // save output taking the directions into account
            dir_in = (i*radix + 1)&1;
            for (size_t j = 0; j < radix; j++)
            {                                
                T x_r = x_temp_out[2*j + 0];
                T x_i = x_temp_out[2*j + 1];

                if (dir_in)
                {
                    data_out[index + 2*k + j*stride + 0] = x_r;
                    data_out[index + 2*k + j*stride + 1] = x_i;
                } else
                {
                    data_out[index + 2*(stride/2 - k) + j*stride + 0] = x_r;
                    data_out[index + 2*(stride/2 - k) + j*stride + 1] = x_i;
                }

                dir_in = dir_in^1;
            }
        }
        dir_out = dir_out^1;
    }
}

}

#endif // HHFFT_1D_PLAIN_REAL_IMPL_H
