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

// This header contains inline functions used in Rader's algorithm and wrappers that allow using it

#include "raders_d.h"
#include "../common/hhfft_1d_complex_avx_common_d.h"
#include "../raders/raders_sse2_d.h"


template<hhfft::RadixType radix_type> inline double *allocate_raders_D2(const hhfft::RadersD &raders)
{
    if (radix_type == hhfft::RadixType::Raders)
    {
        // Two separate vectors are allocated
        return raders.allocate_memory(2);
    } else
    {
        return nullptr;
    }
}

template<hhfft::RadixType radix_type> inline void free_raders_D2(const hhfft::RadersD &raders, double *data)
{
    if (radix_type == hhfft::RadixType::Raders)
    {
        raders.free_memory(data);
    }
}

// Fill input vector with zeros
template<hhfft::RadixType radix_type> inline void init_coeff_D2(double *x, const hhfft::RadersD &raders)
{
    if (radix_type == hhfft::RadixType::Raders)
    {
        size_t n = raders.n;
        size_t n_data = raders.n_data_size;

        // Two separate vectors are initialized
        for (size_t j = 0; j < 2; j++)
        {
            for (size_t i = 0; i < 2*(n+2); i++)
            {
                x[j*n_data + i] = 0.0;
            }
        }
    }
}

// Write one complex number to input when performing fft
template<hhfft::RadixType radix_type> inline void set_value_D2(ComplexD2 *data_in, double *data_raders, size_t index, const hhfft::RadersD &raders, ComplexD2 x)
{
    if (radix_type == hhfft::RadixType::Raders)
    {
        size_t n  = raders.n;
        size_t n_data = raders.n_data_size;

        // Divide input data into two complex numbers
        ComplexD x0, x1;
        divide_two_128_D2(x, x0, x1);

        // Sum up the values and store it to extra space in the end        
        ComplexD sum0 = load_D(data_raders + 2*n) + x0;
        ComplexD sum1 = load_D(data_raders + n_data + 2*n) + x1;
        store_D(sum0, data_raders + 2*n);
        store_D(sum1, data_raders + n_data + 2*n);

        // Store values separately
        const uint32_t *reorder_table_raders_inverse = raders.reorder_table_raders_inverse.data();        
        size_t i2 = reorder_table_raders_inverse[index];
        store_D(x0, data_raders + 2*i2);
        store_D(x1, data_raders + n_data + 2*i2);
    } else
    {        
        data_in[index] = x;
    }
}

// Multiply one complex number and write it to input when performing fft
template<hhfft::RadixType radix_type> inline void set_value_multiply_twiddle_D2(ComplexD2 *data_in, double *data_raders, size_t index, const hhfft::RadersD &raders, ComplexD2 x, ComplexD2 w)
{
    // Multiply with twiddle factors
    ComplexD2 xw = mul_D2(w,x);

    if (radix_type == hhfft::RadixType::Raders)
    {
        size_t n  = raders.n;
        size_t n_data = raders.n_data_size;

        // Divide input data into two complex numbers
        ComplexD x0, x1;
        divide_two_128_D2(xw, x0, x1);

        // Sum up the values and store it to extra space in the end
        ComplexD sum0 = load_D(data_raders + 2*n) + x0;
        ComplexD sum1 = load_D(data_raders + n_data + 2*n) + x1;
        store_D(sum0, data_raders + 2*n);
        store_D(sum1, data_raders + n_data + 2*n);

        const uint32_t *reorder_table_raders_inverse = raders.reorder_table_raders_inverse.data();
        size_t i2 = reorder_table_raders_inverse[index];
        store_D(x0, data_raders + 2*i2);
        store_D(x1, data_raders + n_data + 2*i2);
    } else
    {
        data_in[index] = xw;
    }
}

// Write one complex number to input when performing ifft
template<hhfft::RadixType radix_type> inline void set_value_inverse_D2(ComplexD2 *data_in, double *data_raders, size_t index, const hhfft::RadersD &raders, ComplexD2 x)
{
    if (radix_type == hhfft::RadixType::Raders)
    {
        // Sum up the values and store it to extra space in the end
        size_t n  = raders.n;
        size_t n_org  = raders.n_org;
        size_t n_data = raders.n_data_size;

        // Divide input data into two complex numbers
        ComplexD x0, x1;
        divide_two_128_D2(x, x0, x1);

        // Sum up the values and store it to extra space in the end
        ComplexD sum0 = load_D(data_raders + 2*n) + x0;
        ComplexD sum1 = load_D(data_raders + n_data + 2*n) + x1;
        store_D(sum0, data_raders + 2*n);
        store_D(sum1, data_raders + n_data + 2*n);

        const uint32_t *reorder_table_raders_inverse = raders.reorder_table_raders_inverse.data();
        size_t i2 = n + 1;

        if (index == 0)
        {
            i2 = reorder_table_raders_inverse[index];
        } else
        {
            i2 = reorder_table_raders_inverse[n_org - index];
        }

        store_D(x0, data_raders + 2*i2);
        store_D(x1, data_raders + n_data + 2*i2);
    } else
    {
        data_in[index] = x;
    }
}

// Read one complex number to input
template<hhfft::RadixType radix_type> inline ComplexD2 get_value_D2(ComplexD2 *data_out, double *data_raders, size_t index, const hhfft::RadersD &raders)
{
    if (radix_type == hhfft::RadixType::Raders)
    {
        size_t n_data = raders.n_data_size;
        const uint32_t *reorder_table_raders_inverse = raders.reorder_table_raders_inverse2.data();
        size_t i2 = reorder_table_raders_inverse[index];

        // Load values separately and combine them
        return load_two_128_D2(data_raders + 2*i2, data_raders + n_data + 2*i2);
    } else
    {
        return data_out[index];
    }
}

// Do the actual Raders algorithm
template<hhfft::RadixType radix_type> inline void multiply_coeff_forward_D2(const ComplexD2 *x_in, ComplexD2 *x_out, double *data_raders, const hhfft::RadersD &raders)
{
    if (radix_type == hhfft::RadixType::Raders)
    {
        size_t n = raders.n;
        size_t n_org = raders.n_org;
        size_t n_data = raders.n_data_size;

        // Two separate vectors are processed
        for (size_t j = 0; j < 2; j++)
        {
            // First input is stored two the extra space in the end
            ComplexD x0 = load_D(data_raders + j*n_data + 2*n + 2);

            // FFT
            raders.fft(data_raders + j*n_data);

            // Convolution
            // NOTE it might be possible to do this using avx-commands, but would it actually be any faster?
            const double *fft_b = raders.fft_b.data();
            for(size_t i = 0; i < n; i++)
            {
                ComplexD x1 = load_D(data_raders + j*n_data + 2*i);
                ComplexD x2 = load_D(fft_b + 2*i);
                ComplexD x3 = mul_D(x1,x2);
                store_D(x3, data_raders + j*n_data + 2*i);
            }

            // IFFT
            raders.ifft(data_raders + j*n_data);

            // Add first value to others
            double k = raders.scale;
            for (size_t i = 0; i < n_org - 1; i++)
            {
                ComplexD res = x0 + k*load_D(data_raders + j*n_data + 2*i);
                store_D(res, data_raders + j*n_data + 2*i);
            }
        }

    } else
    {
        multiply_coeff_D2<radix_type,true>(x_in, x_out);
    }
}
