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

#include "../common/hhfft_common_avx_f.h"
#include "../raders/raders_avx_f.h"

using namespace hhfft;

// In-place reordering "swap"
void fft_1d_complex_reorder_in_place_avx_f(const float *, float *data_out,const hhfft::StepInfo<float> &step_info)
{
    size_t n = step_info.reorder_table_inplace_size;
    uint32_t *reorder_table = step_info.reorder_table_inplace;

    for (size_t i = 0; i < n; i++)
    {
        size_t ind1 = i + 1; // First one has been omitted!
        size_t ind2 = reorder_table[i];

        // Swap two floats at a time
        ComplexF x_in0 = load_F(data_out + 2*ind1);
        ComplexF x_in1 = load_F(data_out + 2*ind2);
        store_F(x_in0, data_out + 2*ind2);
        store_F(x_in1, data_out + 2*ind1);
    }
}

template<bool scale> void fft_1d_complex_reorder_avx_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info)
{
    size_t n = step_info.radix*step_info.repeats;
    uint32_t *reorder_table = step_info.reorder_table;

    // Needed only in ifft. Equal to 1/N
    float k = step_info.norm_factor;
    ComplexF k64 = broadcast32_F(k);

    for (size_t i = 0; i < n; i++)
    {
        size_t i2 = reorder_table[i];
        if (scale)
        {
            ComplexF x_in = k64*load_F(data_in + 2*i2);
            store_F(x_in, data_out + 2*i);
        } else
        {
            ComplexF x_in = load_F(data_in + 2*i2);
            store_F(x_in, data_out + 2*i);
        }
    }
}

// To be used when stride is 1.
template<RadixType radix_type>
    inline __attribute__((always_inline)) void fft_1d_complex_avx_f_internal_stride1(const float *data_in, float *data_out, float *data_raders, const hhfft::RadersF &raders, size_t repeats)
{
    size_t i = 0;
    size_t radix = get_actual_radix<radix_type>(raders);

    // First use 256-bit variables
    {
        ComplexF4 x_temp_in[radix_type];
        ComplexF4 x_temp_out[radix_type];
        for (i = 0; i+3 < repeats; i+=4)
        {
            // Initialize raders data with zeros
            init_coeff_F4<radix_type>(data_raders, raders);

            // Copy input data from four memory locations.
            for (size_t j = 0; j < radix; j++)
            {
                size_t ind0 = i*radix + j;
                size_t ind1 = i*radix + 1*radix + j;
                size_t ind2 = i*radix + 2*radix + j;
                size_t ind3 = i*radix + 3*radix + j;
                ComplexF4 x = load_four_64_F4(data_in + 2*ind0, data_in + 2*ind1, data_in + 2*ind2, data_in + 2*ind3);
                set_value_F4<radix_type>(x_temp_in, data_raders, j, raders, x);
            }

            // Multiply with coefficients
            multiply_coeff_forward_F4<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

            // Save output to four memory locations.
            for (size_t j = 0; j < radix; j++)
            {
                size_t ind0 = i*radix + j;
                size_t ind1 = i*radix + 1*radix + j;
                size_t ind2 = i*radix + 2*radix + j;
                size_t ind3 = i*radix + 3*radix + j;
                ComplexF4 x = get_value_F4<radix_type>(x_temp_out, data_raders, j, raders);
                store_four_64_F4(x, data_out + 2*ind0, data_out + 2*ind1, data_out + 2*ind2, data_out + 2*ind3);
            }
        }
    }

    // Then, if necessary, use 128-bit variables
    if (i + 1 < repeats)
    {
        ComplexF2 x_temp_in[radix_type];
        ComplexF2 x_temp_out[radix_type];

        // Initialize raders data with zeros
        init_coeff_F2<radix_type>(data_raders, raders);

        // Copy input data from two memory locations.
        for (size_t j = 0; j < radix; j++)
        {
            size_t ind0 = i*radix + j;
            size_t ind1 = i*radix + radix + j;
            ComplexF2 x = load_two_64_F2(data_in + 2*ind0, data_in + 2*ind1);
            set_value_F2<radix_type>(x_temp_in, data_raders, j, raders, x);
        }

        // Multiply with coefficients
        multiply_coeff_forward_F2<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

        // Save output to two memory locations.
        for (size_t j = 0; j < radix; j++)
        {
            size_t ind0 = i*radix + j;
            size_t ind1 = i*radix + radix + j;
            ComplexF2 x = get_value_F2<radix_type>(x_temp_out, data_raders, j, raders);
            store_two_64_F2(x, data_out + 2*ind0, data_out + 2*ind1);
        }
        i+=2;
    }


    // Then, if necessary, use 64-bit variables
    if (i < repeats)
    {
        ComplexF x_temp_in[radix_type];
        ComplexF x_temp_out[radix_type];

        // Initialize raders data with zeros
        init_coeff_F<radix_type>(data_raders, raders);

        // Copy input data (squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            size_t ind = i*radix + j;
            ComplexF x = load_F(data_in + 2*ind);
            set_value_F<radix_type>(x_temp_in, data_raders, j, raders, x);
        }

        // Multiply with coefficients
        multiply_coeff_forward_F<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

        // Copy output data (un-squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            ComplexF x = get_value_F<radix_type>(x_temp_out, data_raders, j, raders);
            store_F(x, data_out + 2*i*radix + 2*j);
        }
    }
}

// To be used when stride is 1 and reordering is needed
template<RadixType radix_type, bool forward>
    inline __attribute__((always_inline)) void fft_1d_complex_avx_f_internal_stride1_reorder(const float *data_in, float *data_out, size_t repeats, const hhfft::StepInfo<float> &step_info)
{
    size_t i = 0;
    uint32_t *reorder_table = step_info.reorder_table;
    float k = step_info.norm_factor;
    const hhfft::RadersF &raders = *step_info.raders;
    size_t radix = get_actual_radix<radix_type>(raders);
    size_t reorder_table_size = step_info.reorder_table_size;

    // Amount of Raders memory needed depends on repeats
    float *data_raders = nullptr;    
    if (repeats > 1)
        data_raders = allocate_raders_F2<radix_type>(raders);
    else
        data_raders = allocate_raders_F<radix_type>(raders);

    // NOTE 256-bit variables are not used here, as they are typically slower due to packing and unpacking
    // First use 128-bit variables
    ComplexF2 x_temp_in[radix_type];
    ComplexF2 x_temp_out[radix_type];
    for (i = 0; i+1 < repeats; i+=2)
    {
        // Initialize raders data with zeros
        init_coeff_F2<radix_type>(data_raders, raders);

        for (size_t j = 0; j < radix; j++)
        {
            size_t i2 = i*radix + j;
            size_t ind0, ind1;            
            if (forward)
            {
                ind0 = reorder_table[i2];
                ind1 = reorder_table[i2 + radix];                
            } else
            {
                ind0 = reorder_table[reorder_table_size - i2 - 1];
                ind1 = reorder_table[reorder_table_size - i2 - radix - 1];                
            }
            ComplexF2 x = load_two_64_F2(data_in + 2*ind0, data_in + 2*ind1);
            set_value_F2<radix_type>(x_temp_in, data_raders, j, raders, x);
        }

        // Multiply with coefficients
        multiply_coeff_forward_F2<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

        // Save output to two memory locations.
        for (size_t j = 0; j < radix; j++)
        {
            size_t ind0 = i*radix + j;
            size_t ind1 = i*radix + radix + j;
            ComplexF2 x = get_value_F2<radix_type>(x_temp_out, data_raders, j, raders);
            if (!forward)
            {
                x = k*x;
            }
            store_two_64_F2(x, data_out + 2*ind0, data_out + 2*ind1);
        }
    }

    // Then, if necessary, use 64-bit variables
    if (i < repeats)
    {
        // Initialize raders data with zeros
        init_coeff_F<radix_type>(data_raders, raders);

        ComplexF x_temp_in[radix_type];
        ComplexF x_temp_out[radix_type];

        // Copy input data taking reordering into account
        for (size_t j = 0; j < radix; j++)
        {
            size_t i2 = i*radix + j;

            if (forward)
            {
                size_t ind = reorder_table[i2];
                ComplexF x = load_F(data_in + 2*ind);
                set_value_F<radix_type>(x_temp_in, data_raders, j, raders, x);
            } else
            {
                size_t ind = reorder_table[reorder_table_size - i2 - 1];
                ComplexF x = k*load_F(data_in + 2*ind);
                set_value_F<radix_type>(x_temp_in, data_raders, j, raders, x);
            }
        }

        // Multiply with coefficients
        multiply_coeff_forward_F<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

        // Save output to two memory locations.
        for (size_t j = 0; j < radix; j++)
        {
            ComplexF x = get_value_F<radix_type>(x_temp_out, data_raders, j, raders);
            store_F(x, data_out + 2*i*radix + 2*j);
        }
    }

    // Free temporary memory
    free_raders_F<radix_type>(raders, data_raders);
}

// To be used when stride is more than 1 and reordering is done after fft
template<RadixType radix_type>
    inline __attribute__((always_inline)) void fft_1d_complex_avx_f_internal_striden_reorder_forward(const float *data_in, float *data_out, size_t stride, const hhfft::StepInfo<float> &step_info)
{
    const hhfft::RadersF &raders = *step_info.raders;
    size_t radix = get_actual_radix<radix_type>(raders);
    uint32_t *reorder_table = step_info.reorder_table;

    // Amount of Raders memory needed depends on stride
    float *data_raders = nullptr;
    if (stride > 3)
        data_raders = allocate_raders_F4<radix_type>(raders);
    else if (stride > 1)
        data_raders = allocate_raders_F2<radix_type>(raders);
    else
        data_raders = allocate_raders_F<radix_type>(raders);

    // First use 256-bit variables
    size_t i = 0;
    for (i = 0; i+3 < stride; i+=4)
    {
        ComplexF4 x_temp_in[radix_type];
        ComplexF4 x_temp_out[radix_type];

        // Initialize raders data with zeros
        init_coeff_F4<radix_type>(data_raders, raders);

        // Copy input data
        for (size_t j = 0; j < radix; j++)
        {
            size_t i2 = j*stride + i;
            ComplexF4 x = load_F4(data_in + 2*i2);
            set_value_F4<radix_type>(x_temp_in, data_raders, j, raders, x);
        }

        // Multiply with coefficients
        multiply_coeff_forward_F4<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

        // Save output taking reordering into account
        for (size_t j = 0; j < radix; j++)
        {
            ComplexF4 x = get_value_F4<radix_type>(x_temp_out, data_raders, j, raders);
            size_t i2 = j*stride + i;
            size_t ind0 = reorder_table[i2];
            size_t ind1 = reorder_table[i2 + 1];
            size_t ind2 = reorder_table[i2 + 2];
            size_t ind3 = reorder_table[i2 + 3];
            store_four_64_F4(x, data_out + 2*ind0, data_out + 2*ind1, data_out + 2*ind2, data_out + 2*ind3);
        }
    }

    // Then, if necessary, use 128-bit variables
    if (i+1 < stride)
    {
        ComplexF2 x_temp_in[radix_type];
        ComplexF2 x_temp_out[radix_type];

        // Initialize raders data with zeros
        init_coeff_F2<radix_type>(data_raders, raders);

        // Copy input data
        for (size_t j = 0; j < radix; j++)
        {
            size_t i2 = j*stride + i;
            ComplexF2 x = load_F2(data_in + 2*i2);
            set_value_F2<radix_type>(x_temp_in, data_raders, j, raders, x);
        }

        // Multiply with coefficients
        multiply_coeff_forward_F2<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

        // Save output taking reordering into account
        for (size_t j = 0; j < radix; j++)
        {
            ComplexF2 x = get_value_F2<radix_type>(x_temp_out, data_raders, j, raders);
            size_t i2 = j*stride + i;
            size_t ind0 = reorder_table[i2];
            size_t ind1 = reorder_table[i2 + 1];
            store_two_64_F2(x, data_out + 2*ind0, data_out + 2*ind1);
        }
    }

    // Then, if necessary, use 64-bit variables
    if (i < stride)
    {
        ComplexF x_temp_in[radix_type];
        ComplexF x_temp_out[radix_type];

        // Initialize raders data with zeros
        init_coeff_F<radix_type>(data_raders, raders);

        // Copy input data
        for (size_t j = 0; j < radix; j++)
        {
            size_t i2 = j*stride + i;
            ComplexF x = load_F(data_in + 2*i2);
            set_value_F<radix_type>(x_temp_in, data_raders, j, raders, x);
        }

        // Multiply with coefficients
        multiply_coeff_forward_F<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

        // Save output taking reordering into account
        for (size_t j = 0; j < radix; j++)
        {
            ComplexF x = get_value_F<radix_type>(x_temp_out, data_raders, j, raders);
            size_t i2 = j*stride + i;
            size_t ind = reorder_table[i2];
            store_F(x, data_out + 2*ind);
        }
    }

    // Free temporary memory
    free_raders_F<radix_type>(raders, data_raders);


    /*
    // TESTING
    // Amount of Raders memory needed depends on stride
    float *data_raders = nullptr;
    if (stride > 1)
        data_raders = allocate_raders_F2<radix_type>(raders);
    else
        data_raders = allocate_raders_F<radix_type>(raders);

    // First use 128-bit variables
    size_t i = 0;
    for (i = 0; i+1 < stride; i+=2)
    {
        ComplexF2 x_temp_in[radix_type];
        ComplexF2 x_temp_out[radix_type];

        // Initialize raders data with zeros
        init_coeff_F2<radix_type>(data_raders, raders);

        // Copy input data
        for (size_t j = 0; j < radix; j++)
        {
            size_t i2 = j*stride + i;
            ComplexF2 x = load_F2(data_in + 2*i2);
            set_value_F2<radix_type>(x_temp_in, data_raders, j, raders, x);
        }

        // Multiply with coefficients
        multiply_coeff_forward_F2<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

        // Save output taking reordering into account
        for (size_t j = 0; j < radix; j++)
        {
            ComplexF2 x = get_value_F2<radix_type>(x_temp_out, data_raders, j, raders);
            size_t i2 = j*stride + i;
            size_t ind0 = reorder_table[i2];
            size_t ind1 = reorder_table[i2 + 1];
            store_two_64_F2(x, data_out + 2*ind0, data_out + 2*ind1);
        }
    }

    // Then, if necessary, use 64-bit variables
    if (i < stride)
    {
        ComplexF x_temp_in[radix_type];
        ComplexF x_temp_out[radix_type];

        // Initialize raders data with zeros
        init_coeff_F<radix_type>(data_raders, raders);

        // Copy input data
        for (size_t j = 0; j < radix; j++)
        {
            size_t i2 = j*stride + i;
            ComplexF x = load_F(data_in + 2*i2);
            set_value_F<radix_type>(x_temp_in, data_raders, j, raders, x);
        }

        // Multiply with coefficients
        multiply_coeff_forward_F<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

        // Save output taking reordering into account
        for (size_t j = 0; j < radix; j++)
        {
            ComplexF x = get_value_F<radix_type>(x_temp_out, data_raders, j, raders);
            size_t i2 = j*stride + i;
            size_t ind = reorder_table[i2];
            store_F(x, data_out + 2*ind);
        }
    }

    // Free temporary memory
    free_raders_F<radix_type>(raders, data_raders);
    */
}

// To be used when stride is more than 1 and reordering is done after ifft
template<RadixType radix_type>
    inline __attribute__((always_inline)) void fft_1d_complex_avx_f_internal_striden_reorder_inverse(const float *data_in, float *data_out, size_t stride, const hhfft::StepInfo<float> &step_info)
{
    // TODO use ComplexF2 and ComplexF4

    const hhfft::RadersF &raders = *step_info.raders;
    size_t radix = get_actual_radix<radix_type>(raders);
    uint32_t *reorder_table = step_info.reorder_table;
    ComplexF k = broadcast32_F(step_info.norm_factor);
    size_t reorder_table_size = step_info.reorder_table_size;

    ComplexF x_temp_in[radix_type];
    ComplexF x_temp_out[radix_type];

    // Allocate memory for Rader's algorithm if needed
    float *data_raders = allocate_raders_F<radix_type>(raders);

    size_t i = 0;
    for (i = 0; i < stride - 1; i++)
    {
        // Initialize raders data with zeros
        init_coeff_F<radix_type>(data_raders, raders);

        // Copy input data
        for (size_t j = 0; j < radix; j++)
        {
            size_t i2 = j*stride + i + 1;
            ComplexF x = load_F(data_in + 2*i2);
            set_value_F<radix_type>(x_temp_in, data_raders, radix - j - 1, raders, x);
        }

        // Multiply with coefficients
        multiply_coeff_forward_F<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

        // Save output taking reordering into account
        for (size_t j = 0; j < radix; j++)
        {
            ComplexF x = k*get_value_F<radix_type>(x_temp_out, data_raders, radix - j - 1, raders);
            size_t i2 = j*stride + i + 1;
            size_t ind = reorder_table[reorder_table_size - i2 - 1];
            store_F(x, data_out + 2*ind);
        }
    }

    // Last value in stride is special case
    {
        // Initialize raders data with zeros
        init_coeff_F<radix_type>(data_raders, raders);

        // Copy input data, taking the special case into account
        ComplexF x = load_F(data_in + 0);
        set_value_F<radix_type>(x_temp_in, data_raders, 0, raders, x);
        for (size_t j = 0; j < radix - 1; j++)
        {
            size_t i2 = j*stride + i + 1;
            ComplexF x = load_F(data_in + 2*i2);
            set_value_F<radix_type>(x_temp_in, data_raders, radix - j - 1, raders, x);
        }

        // Multiply with coefficients
        multiply_coeff_forward_F<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

        // Save output taking reordering into account
        for (size_t j = 0; j < radix; j++)
        {
            ComplexF x = k*get_value_F<radix_type>(x_temp_out, data_raders, radix - j - 1, raders);
            size_t i2 = j*stride + i + 1;
            size_t ind = reorder_table[reorder_table_size - i2 - 1];
            store_F(x, data_out + 2*ind);
        }
    }

    // Free temporary memory
    free_raders_F<radix_type>(raders, data_raders);
}

template<RadixType radix_type, SizeType stride_type>
    inline __attribute__((always_inline)) void fft_1d_complex_avx_f_internal(const float *data_in, float *data_out, float *data_raders, const hhfft::RadersF &raders, size_t stride)
{
    size_t k = 0;
    size_t radix = get_actual_radix<radix_type>(raders);

    // First use 256-bit variables
    {
        ComplexF4 x_temp_in[radix_type];
        ComplexF4 x_temp_out[radix_type];

        for (k = 0; k+1 < stride; k+=4)
        {
            // Initialize raders data with zeros
            init_coeff_F4<radix_type>(data_raders, raders);

            // Copy input data (squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                ComplexF4 x = load_F4(data_in + 2*k + 2*j*stride);
                set_value_F4<radix_type>(x_temp_in, data_raders, j, raders, x);
            }

            // Multiply with coefficients
            multiply_coeff_forward_F4<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

            // Copy output data (un-squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                ComplexF4 x = get_value_F4<radix_type>(x_temp_out, data_raders, j, raders);
                store_F4(x, data_out + 2*k + 2*j*stride);
            }
        }
    }

    // Then, if necassery, use 128-bit variables
    if (k+1 < stride)
    {
        ComplexF2 x_temp_in[radix_type];
        ComplexF2 x_temp_out[radix_type];

        // Initialize raders data with zeros
        init_coeff_F2<radix_type>(data_raders, raders);

        // Copy input data (squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            ComplexF2 x = load_F2(data_in + 2*k + 2*j*stride);
            set_value_F2<radix_type>(x_temp_in, data_raders, j, raders, x);
        }

        // Multiply with coefficients
        multiply_coeff_forward_F2<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

        // Copy output data (un-squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            ComplexF2 x = get_value_F2<radix_type>(x_temp_out, data_raders, j, raders);
            store_F2(x, data_out + 2*k + 2*j*stride);
        }

        k+=2;
    }

    // NOTE slightly more performance could be gain by this
    //if (strid_type is divisible by 2)
    //    return;

    // Then, if necassery, use 64-bit variables
    if (k < stride)
    {
        // Initialize raders data with zeros
        init_coeff_F<radix_type>(data_raders, raders);

        ComplexF x_temp_in[radix_type];
        ComplexF x_temp_out[radix_type];

        // Copy input data (squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            ComplexF x = load_F(data_in + 2*k + 2*j*stride);
            set_value_F<radix_type>(x_temp_in, data_raders, j, raders, x);
        }

        // Multiply with coefficients
        multiply_coeff_forward_F<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

        // Copy output data (un-squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            ComplexF x = get_value_F<radix_type>(x_temp_out, data_raders, j, raders);
            store_F(x, data_out + 2*k + 2*j*stride);
        }
    }
}


template<RadixType radix_type, SizeType stride_type>
    inline __attribute__((always_inline)) void fft_1d_complex_twiddle_dit_avx_f_internal(const float *data_in, float *data_out, const float *twiddle_factors, float *data_raders, const hhfft::RadersF &raders, size_t stride)
{
    size_t k = 0;
    size_t radix = get_actual_radix<radix_type>(raders);

    // First use 256-bit variables
    {
        for (k = 0; k+3 < stride; k+=4)
        {            
            // Initialize raders data with zeros
            init_coeff_F4<radix_type>(data_raders, raders);

            ComplexF4 x_temp_in[radix_type];
            ComplexF4 x_temp_out[radix_type];
            ComplexF4 twiddle_temp[radix_type];

            // Copy input data and multiply with twiddle factors
            for (size_t j = 0; j < radix; j++)
            {
                size_t j2 = 2*k + 2*j*stride;
                ComplexF4 x = load_F4(data_in + j2);
                ComplexF4 w = load_F4(twiddle_factors + j2);
                set_value_twiddle_F4<radix_type>(x_temp_in, data_raders, twiddle_temp, j, raders, x, w);
            }

            // Multiply with coefficients
            multiply_twiddle_F4<radix_type,true>(x_temp_in, x_temp_in, twiddle_temp);
            multiply_coeff_forward_F4<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

            // Copy output data (un-squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                size_t j2 = 2*k + 2*j*stride;
                ComplexF4 x = get_value_F4<radix_type>(x_temp_out, data_raders, j, raders);
                store_F4(x, data_out + j2);
            }
        }
    }    

    // Then, if necassery, use 128-bit variables
    if (k+1 < stride)
    {
        // Initialize raders data with zeros
        init_coeff_F2<radix_type>(data_raders, raders);

        ComplexF2 x_temp_in[radix_type];
        ComplexF2 x_temp_out[radix_type];
        ComplexF2 twiddle_temp[radix_type];

        // Copy input data and multiply with twiddle factors
        for (size_t j = 0; j < radix; j++)
        {
            size_t j2 = 2*k + 2*j*stride;
            ComplexF2 x = load_F2(data_in + j2);
            ComplexF2 w = load_F2(twiddle_factors + j2);
            set_value_twiddle_F2<radix_type>(x_temp_in, data_raders, twiddle_temp, j, raders, x, w);
        }

        // Multiply with coefficients
        multiply_twiddle_F2<radix_type,true>(x_temp_in, x_temp_in, twiddle_temp);
        multiply_coeff_forward_F2<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

        // Copy output data (un-squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            size_t j2 = 2*k + 2*j*stride;
            ComplexF2 x = get_value_F2<radix_type>(x_temp_out, data_raders, j, raders);
            store_F2(x, data_out + j2);
        }

        k += 2;
    }

    //NOTE slightly more performance could be gain by this
    //if (stride_type is divisible by 2)
    //    return;

    // Then, if necassery, use 64-bit variables
    if (k < stride)
    {
        // Initialize raders data with zeros
        init_coeff_F<radix_type>(data_raders, raders);

        ComplexF x_temp_in[radix_type];
        ComplexF x_temp_out[radix_type];
        ComplexF twiddle_temp[radix_type];

        // Copy input data (squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            size_t j2 = 2*k + 2*j*stride;
            ComplexF x = load_F(data_in + j2);
            ComplexF w = load_F(twiddle_factors + j2);
            set_value_twiddle_F<radix_type>(x_temp_in, data_raders, twiddle_temp, j, raders, x, w);
        }

        // Multiply with coefficients
        multiply_twiddle_F<radix_type,true>(x_temp_in, x_temp_in, twiddle_temp);
        multiply_coeff_forward_F<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

        // Copy output data (un-squeeze)
        for (size_t j = 0; j < radix; j++)
        {
            size_t j2 = 2*k + 2*j*stride;
            ComplexF x = get_value_F<radix_type>(x_temp_out, data_raders, j, raders);
            store_F(x, data_out + j2);
        }
    }
}

template<RadixType radix_type, SizeType stride_type>
    void fft_1d_complex_avx_f(const float *data_in, float *data_out, const hhfft::StepInfo<float> &step_info)
{
    size_t stride = get_size<stride_type>(step_info.stride);
    size_t repeats = step_info.repeats;
    const hhfft::RadersF &raders = *step_info.raders;
    size_t radix = get_actual_radix<radix_type>(raders);

    float *data_raders = nullptr;

    // Implementation for stride == 1
    if (stride_type == SizeType::Size1)
    {
        // Amount of Raders memory needed depends on repeats
        if (repeats > 3)
            data_raders = allocate_raders_F4<radix_type>(raders);
        else if (repeats > 1)
            data_raders = allocate_raders_F2<radix_type>(raders);
        else
            data_raders = allocate_raders_F<radix_type>(raders);

        fft_1d_complex_avx_f_internal_stride1<radix_type>(data_in, data_out, data_raders, raders, repeats);
    } else
    {
        // Amount of Raders memory needed depends on stride
        if (stride > 3)
            data_raders = allocate_raders_F4<radix_type>(raders);
        else if (stride > 1)
            data_raders = allocate_raders_F2<radix_type>(raders);
        else
            data_raders = allocate_raders_F<radix_type>(raders);

        // Other stride types
        for (size_t i = 0; i < repeats; i++)
        {
            fft_1d_complex_avx_f_internal<radix_type,stride_type>
                    (data_in + 2*i*radix*stride, data_out + 2*i*radix*stride, data_raders, raders, stride);
        }
    }

    // Free temporary memory
    free_raders_F<radix_type>(raders, data_raders);
}

// Reordering and fft-step combined
template<RadixType radix_type, SizeType stride_type, bool forward>
    void fft_1d_complex_reorder2_avx_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info)
{
    size_t repeats = step_info.repeats;
    size_t stride = get_size<stride_type>(step_info.stride);

    // If stride == 1, first reorder, then fft
    if (stride_type == SizeType::Size1)
    {
        fft_1d_complex_avx_f_internal_stride1_reorder<radix_type, forward>(data_in, data_out, repeats, step_info);
    }
    else
    {
        // If stride > 1, first fft, then reorder. This is more efficient way.
        if (forward)
            fft_1d_complex_avx_f_internal_striden_reorder_forward<radix_type>(data_in, data_out, stride, step_info);
        else
            fft_1d_complex_avx_f_internal_striden_reorder_inverse<radix_type>(data_in, data_out, stride, step_info);
    }
}

template<RadixType radix_type, SizeType stride_type>
    void fft_1d_complex_twiddle_dit_avx_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info)
{
    size_t stride = get_size<stride_type>(step_info.stride);
    size_t repeats = step_info.repeats;
    const hhfft::RadersF &raders = *step_info.raders;
    size_t radix = get_actual_radix<radix_type>(raders);

    // Amount of Raders memory needed depends on stride
    float *data_raders = nullptr;
    if (stride > 3)
        data_raders = allocate_raders_F4<radix_type>(raders);
    else if (stride > 1)
        data_raders = allocate_raders_F2<radix_type>(raders);
    else
        data_raders = allocate_raders_F<radix_type>(raders);

    for (size_t i = 0; i < repeats; i++)
    {
        fft_1d_complex_twiddle_dit_avx_f_internal<radix_type,stride_type>
                (data_in + 2*i*radix*stride, data_out + 2*i*radix*stride, step_info.twiddle_factors, data_raders, raders, stride);
    }

    // Free temporary memory
    free_raders_F<radix_type>(*step_info.raders, data_raders);
}

void fft_1d_complex_convolution_avx_f(const float *data_in0, const float *data_in1, float *data_out, size_t n)
{
    size_t i;

    // First use 256-bit variables
    for (i = 0; i+6 < 2*n; i+=8)
    {
        ComplexF4 x_in0 = load_F4(data_in0 + i);
        ComplexF4 x_in1 = load_F4(data_in1 + i);
        ComplexF4 x_out = mul_F4(x_in0, x_in1);
        store_F4(x_out, data_out + i);
    }

    // Then, if necassery, use 128-bit variables
    if (i+2 < 2*n)
    {
        ComplexF2 x_in0 = load_F2(data_in0 + i);
        ComplexF2 x_in1 = load_F2(data_in1 + i);
        ComplexF2 x_out = mul_F2(x_in0, x_in1);
        store_F2(x_out, data_out + i);
        i+=4;
    }

    // Then, if necassery, use 64-bit variables
    if (i < 2*n)
    {
        ComplexF x_in0 = load_F(data_in0 + i);
        ComplexF x_in1 = load_F(data_in1 + i);
        ComplexF x_out = mul_F(x_in0, x_in1);
        store_F(x_out, data_out + i);
    }
}

// fft for small sizes (1,2,3,4,5,7,8) where only one level is needed
template<size_t n, bool forward> void fft_1d_complex_1level_avx_f(const float *data_in, float *data_out, const hhfft::StepInfo<float> &)
{
    ComplexF k = broadcast32_F(1.0/n);

    if (n == 1)
    {
        ComplexF x = load_F(data_in);
        store_F(x, data_out);
    } else
    {
        ComplexF x_temp_in[n];
        ComplexF x_temp_out[n];

        // Copy input data
        for (size_t i = 0; i < n; i++)
        {
            x_temp_in[i] = load_F(data_in + 2*i);
        }

        // Multiply with coefficients
        multiply_coeff_F<n,forward>(x_temp_in, x_temp_out);

        // Copy output data
        for (size_t i = 0; i < n; i++)
        {
            if(forward)
                store_F(x_temp_out[i], data_out + 2*i);
            else
                store_F(k*x_temp_out[i], data_out + 2*i);
        }
    }
}

// fft for small sizes where two levels are needed n = n1*n2;
template<size_t n1, size_t n2, bool forward> void fft_1d_complex_2level_avx_f(const float *data_in, float *data_out, const hhfft::StepInfo<float> &step_info)
{
    ComplexF k = broadcast32_F(1.0/(n1*n2));
    ComplexF2 k2 = broadcast32_F2(1.0/(n1*n2));

    float x_temp[2*n1*n2];

    // First level
    {
        // First use 128-bit variables
        size_t i = 0;
        for (i = 0; i + 1 < n2; i+=2)
        {
            ComplexF2 x_temp_in[n1];
            ComplexF2 x_temp_out[n1];

            // Copy input data
            for (size_t j = 0; j < n1; j++)
            {
                ComplexF2 x = load_F2(data_in + 2*(j*n2 + i));

                if (!forward)
                {
                    x = k2*x;
                }

                x_temp_in[j] = x;
            }

            // Multiply with coefficients
            multiply_coeff_F2<n1,forward>(x_temp_in, x_temp_out);

            // Store output data
            for (size_t j = 0; j < n1; j++)
            {
                store_two_64_F2(x_temp_out[j], x_temp + 2*(i*n1 + j), x_temp + 2*((i+1)*n1 + j));
            }
        }

        // Then use 64-bit variables
        if (i < n2)
        {
            ComplexF x_temp_in[n1];
            ComplexF x_temp_out[n1];

            // Copy input data
            for (size_t j = 0; j < n1; j++)
            {
                ComplexF x = load_F(data_in + 2*(j*n2 + i));

                if (!forward)
                {
                    x = k*x;
                }

                x_temp_in[j] = x;
            }

            // Multiply with coefficients
            multiply_coeff_F<n1,forward>(x_temp_in, x_temp_out);

            // Store output data
            for (size_t j = 0; j < n1; j++)
            {
                store_F(x_temp_out[j], x_temp + 2*(i*n1 + j));
            }
        }
    }

    // Second level
    {
         // First use 128-bit variables
        size_t i = 0;
        for (i = 0; i + 1 < n1; i+=2)
        {
            ComplexF2 x_temp_in[n2];
            ComplexF2 x_temp_out[n2];
            ComplexF2 twiddle[n2];

            // Copy input data and twiddle factors
            for (size_t j = 0; j < n2; j++)
            {
                size_t j2 = j*n1 + i;
                x_temp_in[j] = load_F2(x_temp + 2*j2);
                twiddle[j] = load_F2(step_info.twiddle_factors + 2*j2);
            }

            // Multiply with coefficients
            multiply_twiddle_F2<n2,forward>(x_temp_in, x_temp_in, twiddle);
            multiply_coeff_F2<n2,forward>(x_temp_in, x_temp_out);

            // Copy output data
            for (size_t j = 0; j < n2; j++)
            {
                store_F2(x_temp_out[j], data_out + 2*(j*n1 + i));
            }
        }

        // Then use 64-bit variables
        if (i < n1)
        {
            ComplexF x_temp_in[n2];
            ComplexF x_temp_out[n2];
            ComplexF twiddle[n2];

            // Copy input data and twiddle factors
            for (size_t j = 0; j < n2; j++)
            {
                size_t j2 = j*n1 + i;
                x_temp_in[j] = load_F(x_temp + 2*j2);
                twiddle[j] = load_F(step_info.twiddle_factors + 2*j2);
            }

            // Multiply with coefficients
            multiply_twiddle_F<n2,forward>(x_temp_in, x_temp_in, twiddle);
            multiply_coeff_F<n2,forward>(x_temp_in, x_temp_out);

            // Copy output data
            for (size_t j = 0; j < n2; j++)
            {
                store_F(x_temp_out[j], data_out + 2*(j*n1 + i));
            }
        }
    }
}


// For problems that need only one level Rader's
template<bool forward> void fft_1d_complex_1level_raders_avx_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info)
{
    size_t n = step_info.radix;
    ComplexF k = broadcast32_F(1.0/n);

    // Allocate memory for Rader's algorithm
    const hhfft::RadersF &raders = *step_info.raders;
    float *data_raders = allocate_raders_F<Raders>(raders);

    // Initialize raders data with zeros
    init_coeff_F<Raders>(data_raders, raders);

    // Copy input data (squeeze)
    for (size_t i = 0; i < n; i++)
    {
        ComplexF x = load_F(data_in + 2*i);
        if (forward)
        {
            set_value_F<Raders>(nullptr, data_raders, i, raders, x);
        } else
        {
            set_value_inverse_F<Raders>(nullptr, data_raders, i, raders, x);
        }
    }

    // Multiply with coefficients
    multiply_coeff_forward_F<Raders>(nullptr, nullptr, data_raders, raders);

    // Copy input data (un-squeeze)
    for (size_t i = 0; i < n; i++)
    {
        ComplexF x = get_value_F<Raders>(nullptr, data_raders, i, raders);

        if(forward)
            store_F(x, data_out + 2*i);
        else
            store_F(k*x, data_out + 2*i);
    }

    // Free temporary memory
    free_raders_F<Raders>(raders, data_raders);
}

// Instantiations of the functions defined in this class
template void fft_1d_complex_reorder_avx_f<false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_reorder_avx_f<true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template void fft_1d_complex_reorder2_avx_f<Raders, SizeType::Size1, false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_reorder2_avx_f<Raders, SizeType::Size1, true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_reorder2_avx_f<Radix2, SizeType::Size1, false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_reorder2_avx_f<Radix2, SizeType::Size1, true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_reorder2_avx_f<Radix3, SizeType::Size1, false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_reorder2_avx_f<Radix3, SizeType::Size1, true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_reorder2_avx_f<Radix4, SizeType::Size1, false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_reorder2_avx_f<Radix4, SizeType::Size1, true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_reorder2_avx_f<Radix5, SizeType::Size1, false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_reorder2_avx_f<Radix5, SizeType::Size1, true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_reorder2_avx_f<Radix6, SizeType::Size1, false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_reorder2_avx_f<Radix6, SizeType::Size1, true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_reorder2_avx_f<Radix7, SizeType::Size1, false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_reorder2_avx_f<Radix7, SizeType::Size1, true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_reorder2_avx_f<Radix8, SizeType::Size1, false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_reorder2_avx_f<Radix8, SizeType::Size1, true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_reorder2_avx_f<Raders, SizeType::SizeN, false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_reorder2_avx_f<Raders, SizeType::SizeN, true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_reorder2_avx_f<Radix2, SizeType::SizeN, false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_reorder2_avx_f<Radix2, SizeType::SizeN, true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_reorder2_avx_f<Radix3, SizeType::SizeN, false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_reorder2_avx_f<Radix3, SizeType::SizeN, true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_reorder2_avx_f<Radix4, SizeType::SizeN, false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_reorder2_avx_f<Radix4, SizeType::SizeN, true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_reorder2_avx_f<Radix5, SizeType::SizeN, false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_reorder2_avx_f<Radix5, SizeType::SizeN, true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_reorder2_avx_f<Radix6, SizeType::SizeN, false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_reorder2_avx_f<Radix6, SizeType::SizeN, true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_reorder2_avx_f<Radix7, SizeType::SizeN, false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_reorder2_avx_f<Radix7, SizeType::SizeN, true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_reorder2_avx_f<Radix8, SizeType::SizeN, false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_reorder2_avx_f<Radix8, SizeType::SizeN, true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template void fft_1d_complex_avx_f<Raders, SizeType::Size1>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_avx_f<Radix2, SizeType::Size1>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_avx_f<Radix3, SizeType::Size1>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_avx_f<Radix4, SizeType::Size1>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_avx_f<Radix5, SizeType::Size1>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_avx_f<Radix6, SizeType::Size1>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_avx_f<Radix7, SizeType::Size1>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_avx_f<Radix8, SizeType::Size1>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_avx_f<Raders, SizeType::SizeN>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_avx_f<Radix2, SizeType::SizeN>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_avx_f<Radix3, SizeType::SizeN>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_avx_f<Radix4, SizeType::SizeN>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_avx_f<Radix5, SizeType::SizeN>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_avx_f<Radix6, SizeType::SizeN>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_avx_f<Radix7, SizeType::SizeN>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_avx_f<Radix8, SizeType::SizeN>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template void fft_1d_complex_twiddle_dit_avx_f<Raders, SizeType::Size1>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_twiddle_dit_avx_f<Radix2, SizeType::Size1>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_twiddle_dit_avx_f<Radix3, SizeType::Size1>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_twiddle_dit_avx_f<Radix4, SizeType::Size1>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_twiddle_dit_avx_f<Radix5, SizeType::Size1>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_twiddle_dit_avx_f<Radix6, SizeType::Size1>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_twiddle_dit_avx_f<Radix7, SizeType::Size1>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_twiddle_dit_avx_f<Radix8, SizeType::Size1>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_twiddle_dit_avx_f<Raders, SizeType::SizeN>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_twiddle_dit_avx_f<Radix2, SizeType::SizeN>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_twiddle_dit_avx_f<Radix3, SizeType::SizeN>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_twiddle_dit_avx_f<Radix4, SizeType::SizeN>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_twiddle_dit_avx_f<Radix5, SizeType::SizeN>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_twiddle_dit_avx_f<Radix6, SizeType::SizeN>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_twiddle_dit_avx_f<Radix7, SizeType::SizeN>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_twiddle_dit_avx_f<Radix8, SizeType::SizeN>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template void fft_1d_complex_1level_avx_f<1, false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_1level_avx_f<2, false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_1level_avx_f<3, false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_1level_avx_f<4, false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_1level_avx_f<5, false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_1level_avx_f<6, false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_1level_avx_f<7, false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_1level_avx_f<8, false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_1level_avx_f<1, true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_1level_avx_f<2, true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_1level_avx_f<3, true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_1level_avx_f<4, true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_1level_avx_f<5, true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_1level_avx_f<6, true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_1level_avx_f<7, true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_1level_avx_f<8, true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template void fft_1d_complex_2level_avx_f<3, 3, true>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<3, 3, false>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<2, 5, true>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<2, 5, false>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<2, 6, true>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<2, 6, false>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<2, 7, true>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<2, 7, false>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<3, 5, true>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<3, 5, false>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<4, 4, true>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<4, 4, false>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<3, 6, true>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<3, 6, false>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<4, 5, true>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<4, 5, false>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<3, 7, true>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<3, 7, false>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<4, 6, true>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<4, 6, false>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<5, 5, true>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<5, 5, false>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<4, 7, true>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<4, 7, false>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<5, 6, true>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<5, 6, false>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<4, 8, true>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<4, 8, false>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<5, 7, true>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<5, 7, false>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<6, 6, true>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<6, 6, false>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<5, 8, true>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<5, 8, false>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<6, 7, true>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<6, 7, false>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<6, 8, true>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<6, 8, false>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<7, 7, true>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<7, 7, false>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<7, 8, true>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<7, 8, false>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<8, 8, true>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);
template void fft_1d_complex_2level_avx_f<8, 8, false>(const float *data_in, float *data_out, const hhfft::StepInfo<float> &);

template void fft_1d_complex_1level_raders_avx_f<true>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template void fft_1d_complex_1level_raders_avx_f<false>(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
