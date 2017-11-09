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

// TODO scaling for inverse fft should be combined here!
// TODO inverse version is also needed! Reordering should be done at the same time as inverse version!
template<typename T, size_t arch> void dht_1d_to_fft(const T *data_in, T *data_out, hhfft::StepInfoReal<T> &step_info)
{
    size_t n = step_info.repeats;

    data_out[0] = data_in[0];
    data_out[1] = data_in[n/2];

    for (size_t i = 1; i < n/2; i++)
    {
        T a = data_in[i];
        T b = data_in[n - i];
        data_out[2*i + 0] = 0.5*(b + a);
        data_out[2*i + 1] = 0.5*(b - a);
    }   
}

// NOTE this is out of place reordering!
template<typename T, size_t arch, bool forward> void dht_1d_reorder(const T *data_in, T *data_out, hhfft::StepInfoReal<T> &step_info)
{
    size_t n = step_info.repeats;    
    uint32_t *reorder_table = step_info.reorder_table;

    // Needed only in idht. Equal to 1/N
    // TODO it would be better to do the multiplication in the last step when data is again reordered to
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

// NOTE this is out of place reordering!
template<typename T, size_t arch, bool forward> void dht_1d_reorder_dht_to_fft(const T *data_in, T *data_out, hhfft::StepInfoReal<T> &step_info)
{
    std::cout << "dht_1d_reorder_dht_to_fft. forward = " << forward << std::endl;

    size_t n = step_info.repeats;
    uint32_t *reorder_table = step_info.reorder_table;

    // Needed only in idht. Equal to 1/N
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

    // TODO add also the conversion dht -> fft conversion here! (Reordering table needs to be a bit different)
}

template<typename T> T cas(const T x)
{
    return cos(x) + sin(x);
}

template<typename T, size_t radix> void multiply_coeff(const T *x_in, T *x_out)
{
    if (radix == 2)
    {
        x_out[0] = x_in[0] + x_in[1];
        x_out[1] = x_in[0] - x_in[1];
    }

    if (radix == 3)
    {
        const T k1 = cas(2*M_PI*1.0/3.0);
        const T k2 = -cas(2*M_PI*2.0/3.0);

        x_out[0] = x_in[0] + x_in[1] + x_in[2];
        x_out[1] = x_in[0] + k1*x_in[1] - k2*x_in[2];
        x_out[2] = x_in[0] - k2*x_in[1] + k1*x_in[2];
    }

    if (radix == 4)
    {                        
        x_out[0] = x_in[0] + x_in[1] + x_in[2] + x_in[3];
        x_out[1] = x_in[0] + x_in[1] - x_in[2] - x_in[3];
        x_out[2] = x_in[0] - x_in[1] + x_in[2] - x_in[3];
        x_out[3] = x_in[0] - x_in[1] - x_in[2] + x_in[3];
    }

    if (radix == 5)
    {
        const T k1 =  cas(2*M_PI*1.0/5.0);
        const T k2 = -cas(2*M_PI*2.0/5.0);
        const T k3 = -cas(2*M_PI*3.0/5.0);
        const T k4 = -cas(2*M_PI*4.0/5.0);

        x_out[0] = x_in[0] + x_in[1] + x_in[2] + x_in[3] + x_in[4];
        x_out[1] = x_in[0] + k1*x_in[1] - k2*x_in[2] - k3*x_in[3] - k4*x_in[4];
        x_out[2] = x_in[0] - k2*x_in[1] - k4*x_in[2] + k1*x_in[3] - k3*x_in[4];
        x_out[3] = x_in[0] - k3*x_in[1] + k1*x_in[2] - k4*x_in[3] - k2*x_in[4];
        x_out[4] = x_in[0] - k4*x_in[1] - k3*x_in[2] - k2*x_in[3] + k1*x_in[4];
    }

    if (radix == 7)
    {
        const T k1 =  cas(2*M_PI*1.0/7.0);
        const T k2 =  cas(2*M_PI*2.0/7.0);
        const T k3 = -cas(2*M_PI*3.0/7.0);
        const T k4 = -cas(2*M_PI*4.0/7.0);
        const T k5 = -cas(2*M_PI*5.0/7.0);
        const T k6 = -cas(2*M_PI*6.0/7.0);

        x_out[0] = x_in[0] + x_in[1] + x_in[2] + x_in[3] + x_in[4] + x_in[5] + x_in[6];
        x_out[1] = x_in[0] + k1*x_in[1] + k2*x_in[2] - k3*x_in[3] - k4*x_in[4] - k5*x_in[5] - k6*x_in[6];
        x_out[2] = x_in[0] + k2*x_in[1] - k4*x_in[2] - k6*x_in[3] + k1*x_in[4] - k3*x_in[5] - k5*x_in[6];
        x_out[3] = x_in[0] - k3*x_in[1] - k6*x_in[2] + k2*x_in[3] - k5*x_in[4] + k1*x_in[5] - k4*x_in[6];
        x_out[4] = x_in[0] - k4*x_in[1] + k1*x_in[2] - k5*x_in[3] + k2*x_in[4] - k6*x_in[5] - k3*x_in[6];
        x_out[5] = x_in[0] - k5*x_in[1] - k3*x_in[2] + k1*x_in[3] - k6*x_in[4] - k4*x_in[5] + k2*x_in[6];
        x_out[6] = x_in[0] - k6*x_in[1] - k5*x_in[2] - k4*x_in[3] - k3*x_in[4] + k2*x_in[5] + k1*x_in[6];
    }
}

// This function can be used on the first level, when there are no twiddle factors
template<typename T, size_t radix, size_t arch, bool reorder> void dht_1d_one_level(const T *data_in, T *data_out, hhfft::StepInfoReal<T> &step_info)
{
    size_t stride = step_info.stride;
    size_t repeats = step_info.repeats;
    uint32_t *reorder_table = step_info.reorder_table;

    T x_temp_in[radix];
    T x_temp_out[radix];

    for (size_t i = 0; i < repeats; i++)
    {
        for (size_t k = 0; k < stride; k++)
        {
            // Read in the values used in this step
            for (size_t j = 0; j < radix; j++)
            {
                size_t i2 = i*radix*stride + j*stride + k;
                if (reorder)
                {
                    i2 = reorder_table[i2];
                }
                x_temp_in[j] = data_in[i2];
            }

            // Multiply with coefficients
            multiply_coeff<T,radix>(x_temp_in, x_temp_out);

            // Write the values used in this step
            for (size_t j = 0; j < radix; j++)
            {
                data_out[i*radix*stride + j*stride + k] = x_temp_out[j];
            }
        }
    }    
}

// This function is to be used when there are cos/sin factors for DIT
template<typename T, size_t radix, size_t arch> void dht_1d_one_level_twiddle(const T *data_in, T *data_out, hhfft::StepInfoReal<T> &step_info)
{
    size_t stride = step_info.stride;
    size_t repeats = step_info.repeats;

    T x_temp_in[radix];
    T x_temp_in2[radix];
    T x_temp_out[radix];
    T x_temp_out2[radix];

    for (size_t i = 0; i < repeats; i++)
    {
        // First one, no cos/sin factors
        for (size_t j = 0; j < radix; j++)
            x_temp_in[j] = data_in[i*radix*stride + j*stride];
        multiply_coeff<T,radix>(x_temp_in, x_temp_out);
        for (size_t j = 0; j < radix; j++)
            data_out[i*radix*stride + j*stride] = x_temp_out[j];

        // Loop over most, start from top and bottom
        for (size_t k = 1; k < (stride+1)/2; k++)
        {
            size_t k2 = stride - k;

            // It is assumed that first cos/sin factors are always (1, 0)
            x_temp_in[0] =  data_in[i*radix*stride + k];
            x_temp_in2[0] = data_in[i*radix*stride + k2];

            // Read in the values used in this step and multiply them with cos/sin factors
            for (size_t j = 1; j < radix; j++)
            {
                T x_1 = data_in[i*radix*stride + j*stride + k];
                T x_2 = data_in[i*radix*stride + j*stride + k2];
                T c1 = step_info.cos_factors[j*stride + k];
                T s1 = step_info.sin_factors[j*stride + k];
                T c2 = step_info.cos_factors[j*stride + k2];
                T s2 = step_info.sin_factors[j*stride + k2];

                // TESTING
                std::cout << "x_1 = " << x_1 << ", x_2 = " << x_2 << std::endl;
                std::cout << "c1 = " << c1 << ", c2 = " << c2 <<  ", s1 = " << s1 << ", s2 = " << s2 << std::endl;

                x_temp_in[j]  = c1*x_1 + s1*x_2;
                x_temp_in2[j] = c2*x_2 + s2*x_1;
            }

            multiply_coeff<T,radix>(x_temp_in, x_temp_out);
            multiply_coeff<T,radix>(x_temp_in2, x_temp_out2);

            // Write the values used in this step
            for (size_t j = 0; j < radix; j++)
            {
                data_out[i*radix*stride + j*stride + k] = x_temp_out[j];
                data_out[i*radix*stride + j*stride + k2] = x_temp_out2[j];
            }
        }

        // If stride is even, there is still one more to go
        if (stride%2 == 0)
        {
            size_t k = stride/2;
            x_temp_in[0] =  data_in[i*radix*stride + k];

            // Read in the values used in this step and multiply them with cos/sin factors
            for (size_t j = 1; j < radix; j++)
            {
                T x_1 = data_in[i*radix*stride + j*stride + k];
                T c1 = step_info.cos_factors[j*stride + k];
                T s1 = step_info.sin_factors[j*stride + k];

                x_temp_in[j]  = (c1+s1)*x_1;
            }

            multiply_coeff<T,radix>(x_temp_in, x_temp_out);

            // Write the values used in this step
            for (size_t j = 0; j < radix; j++)
            {
                data_out[i*radix*stride + j*stride + k] = x_temp_out[j];
            }
        }
    }
}

// This function can be used on the first level, when there are no twiddle factors
template<typename T, size_t radix, size_t arch> void dht_1d_one_level_DIF(const T *data_in, T *data_out, hhfft::StepInfoReal<T> &step_info)
{
    std::cout << "dht_1d_one_level_DIF. Radix = " << radix << std::endl;
    size_t stride = step_info.stride;
    size_t repeats = step_info.repeats;

    T x_temp_in[radix];
    T x_temp_out[radix];

    for (size_t i = 0; i < repeats; i++)
    {
        for (size_t k = 0; k < stride; k++)
        {
            // Read in the values used in this step
            for (size_t j = 0; j < radix; j++)
            {
                size_t i2 = i*radix*stride + j*stride + k;
                x_temp_in[j] = data_in[i2];
            }

            // Multiply with coefficients
            multiply_coeff<T,radix>(x_temp_in, x_temp_out);

            // Write the values used in this step
            for (size_t j = 0; j < radix; j++)
            {
                data_out[i*radix*stride + j*stride + k] = x_temp_out[j];
            }
        }
    }
}

// This function is to be used when there are cos/sin factors for DIF
template<typename T, size_t radix, size_t arch> void dht_1d_one_level_twiddle_DIF(const T *data_in, T *data_out, hhfft::StepInfoReal<T> &step_info)
{
    std::cout << "dht_1d_one_level_twiddle_DIF. Radix = " << radix << std::endl;

    size_t stride = step_info.stride;
    size_t repeats = step_info.repeats;

    T x_temp_in[radix];
    T x_temp_in2[radix];
    T x_temp_out[radix];
    T x_temp_out2[radix];

    for (size_t i = 0; i < repeats; i++)
    {
        // First one, no cos/sin factors
        for (size_t j = 0; j < radix; j++)
            x_temp_in[j] = data_in[i*radix*stride + j*stride];
        multiply_coeff<T,radix>(x_temp_in, x_temp_out);
        for (size_t j = 0; j < radix; j++)
            data_out[i*radix*stride + j*stride] = x_temp_out[j];

        // Loop over most, start from top and bottom
        for (size_t k = 1; k < (stride+1)/2; k++)
        {
            size_t k2 = stride - k;

            // Read in the values used in this step and multiply with DHT coeffs
            for (size_t j = 0; j < radix; j++)
            {
                x_temp_in[j] = data_in[i*radix*stride + j*stride + k];
                x_temp_in2[j] = data_in[i*radix*stride + j*stride + k2];
            }
            multiply_coeff<T,radix>(x_temp_in, x_temp_out);
            multiply_coeff<T,radix>(x_temp_in2, x_temp_out2);

            // It is assumed that first cos/sin factors are always (1, 0)
            data_out[i*radix*stride + k] = x_temp_out[0];
            data_out[i*radix*stride + k2] = x_temp_out2[0];

            // Multiply with cos/sin factors and write the output
            for (size_t j = 1; j < radix; j++)
            {
                T t_1 = x_temp_out[j];
                T t_2 = x_temp_out2[j];
                T c1 = step_info.cos_factors[j*stride + k];
                T s1 = step_info.sin_factors[j*stride + k];
                T c2 = step_info.cos_factors[j*stride + k2];
                T s2 = step_info.sin_factors[j*stride + k2];

                std::cout << "t_1 = " << t_1 << ", t_2 = " << t_2 << std::endl;
                std::cout << "c1 = " << c1 << ", c2 = " << c2 <<  ", s1 = " << s1 << ", s2 = " << s2 << std::endl;
                T t = s1*s2 - c1*c2;
                T a1 = c2/t, a2 = c1/t, b1 = -s1/t, b2 = -s2/t;
                std::cout << "t = " << t << ", a1 = " << a1 << ", a2 = " << a2 << ", b1 = " << b1 << ", b2 = " << b2 << std::endl;

                // Does not work for radix = 2
                //data_out[i*radix*stride + j*stride + k] = c1*t_1 + s1*t_2;
                //data_out[i*radix*stride + j*stride + k2] = c2*t_2 + s2*t_1;

                // This should make more sense, but actually there is a problem with division by zero
                data_out[i*radix*stride + j*stride + k] = a1*t_1 + b1*t_2;
                data_out[i*radix*stride + j*stride + k2] = a2*t_2 + b2*t_1;
            }
        }

        // If stride is even, there is still one more to go
        if (stride%2 == 0)
        {
            size_t k = stride/2;
            x_temp_in[0] = data_in[i*radix*stride + k];

            // Read in the values used in this step and multiply with DHT coeffs
            for (size_t j = 1; j < radix; j++)
            {
                x_temp_in[j] = data_in[i*radix*stride + j*stride + k];
            }
            multiply_coeff<T,radix>(x_temp_in, x_temp_out);

            // It is assumed that first cos/sin factors are always (1, 0)
            data_out[i*radix*stride + k] = x_temp_out[0];

            // Multiply with cos/sin factors and write the output
            for (size_t j = 1; j < radix; j++)
            {
                T c1 = step_info.cos_factors[j*stride + k];
                T s1 = step_info.sin_factors[j*stride + k];

                data_out[i*radix*stride + j*stride + k] = (c1+s1)*x_temp_out[j];
            }
        }        
    }
}

}

#endif // HHFFT_1D_PLAIN_REAL_IMPL_H
