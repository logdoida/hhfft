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
#include "../common/hhfft_common_avx_f.h"

template<hhfft::RadixType radix_type> inline size_t get_actual_radix(const hhfft::RadersF &raders)
{
    if (radix_type == hhfft::RadixType::Raders)
    {
        return raders.n_org;
    } else
    {
        return radix_type;
    }
}

//////////////////////////////////// ComplexF /////////////////////////////////////////////


template<hhfft::RadixType radix_type> inline float *allocate_raders_F(const hhfft::RadersF &raders)
{
    if (radix_type == hhfft::RadixType::Raders)
    {
        return raders.allocate_memory();
    } else
    {
        return nullptr;
    }
}

template<hhfft::RadixType radix_type> inline void free_raders_F(const hhfft::RadersF &raders, float *data)
{
    if (radix_type == hhfft::RadixType::Raders)
    {
        raders.free_memory(data);
    }
}

// Fill input vector with zeros
template<hhfft::RadixType radix_type> inline void init_coeff_F(float *x, const hhfft::RadersF &raders)
{
    if (radix_type == hhfft::RadixType::Raders)
    {
        size_t n = raders.n;
        for (size_t i = 0; i < 2*(n+2); i++)
        {
            x[i] = 0.0;
        }
    }
}

// Write one complex number to input when performing fft
template<hhfft::RadixType radix_type> inline void set_value_F(ComplexF *data_in, float *data_raders, size_t index, const hhfft::RadersF &raders, ComplexF x)
{
    if (radix_type == hhfft::RadixType::Raders)
    {
        // Sum up the values and store it to extra space in the end
        size_t n  = raders.n;
        ComplexF sum = load_F(data_raders + 2*n) + x;
        store_F(sum, data_raders + 2*n);

        const uint32_t *reorder_table_raders_inverse = raders.reorder_table_raders_inverse.data();
        size_t i2 = reorder_table_raders_inverse[index];
        store_F(x, data_raders + 2*i2);
    } else
    {
        data_in[index] = x;
    }
}

// Multiply one complex number and write it to input when performing fft
template<hhfft::RadixType radix_type> inline void set_value_twiddle_F(ComplexF *data_in, float *data_raders, ComplexF *data_twiddle, size_t index, const hhfft::RadersF &raders, ComplexF x, ComplexF w)
{
    if (radix_type == hhfft::RadixType::Raders)
    {
        // Multiply with twiddle factors. First twiddle factor is assumed to be (1+0i)
        ComplexF xw = mul_F(w,x);

        // Sum up the values and store it to extra space in the end
        size_t n  = raders.n;
        ComplexF sum = load_F(data_raders + 2*n) + xw;
        store_F(sum, data_raders + 2*n);

        const uint32_t *reorder_table_raders_inverse = raders.reorder_table_raders_inverse.data();
        size_t i2 = reorder_table_raders_inverse[index];
        store_F(xw, data_raders + 2*i2);
    } else
    {
        data_twiddle[index] = w;
        data_in[index] = x;
    }
}

// Write one complex number to input when performing ifft
template<hhfft::RadixType radix_type> inline void set_value_inverse_F(ComplexF *data_in, float *data_raders, size_t index, const hhfft::RadersF &raders, ComplexF x)
{
    if (radix_type == hhfft::RadixType::Raders)
    {
        // Sum up the values and store it to extra space in the end
        size_t n  = raders.n;
        size_t n_org  = raders.n_org;
        ComplexF sum = load_F(data_raders + 2*n) + x;
        store_F(sum, data_raders + 2*n);

        const uint32_t *reorder_table_raders_inverse = raders.reorder_table_raders_inverse.data();
        size_t i2 = n + 1;

        if (index == 0)
        {
            i2 = reorder_table_raders_inverse[index];
        } else
        {
            i2 = reorder_table_raders_inverse[n_org - index];
        }

        store_F(x, data_raders + 2*i2);
    } else
    {
        data_in[index] = x;
    }
}

// Read one complex number to input
template<hhfft::RadixType radix_type> inline ComplexF get_value_F(ComplexF *data_out, float *data_raders, size_t index, const hhfft::RadersF &raders)
{
    if (radix_type == hhfft::RadixType::Raders)
    {
        const uint32_t *reorder_table_raders_inverse = raders.reorder_table_raders_inverse2.data();
        size_t i2 = reorder_table_raders_inverse[index];

        return load_F(data_raders + 2*i2);
    } else
    {
        return data_out[index];
    }
}

// Read one complex number to input. Used for real odd
template<hhfft::RadixType radix_type> inline ComplexF get_value_real_odd_forward_F(ComplexF *data_out, float *data_raders, size_t index, const hhfft::RadersF &raders)
{
    size_t radix = get_actual_radix<radix_type>(raders);

    if ((index & 1) == 0)
    {
        ComplexF x = get_value_F<radix_type>(data_out, data_raders, index/2, raders);
        return x;
    } else
    {
        ComplexF x = get_value_F<radix_type>(data_out, data_raders, radix - index/2 - 1, raders);
        return conj_F(x); // conjugate
    }
}

// Do the actual Raders algorithm
template<hhfft::RadixType radix_type> inline void multiply_coeff_forward_F(const ComplexF *x_in, ComplexF *x_out, float *data_raders, const hhfft::RadersF &raders)
{
    if (radix_type == hhfft::RadixType::Raders)
    {
        size_t n = raders.n;
        size_t n_org = raders.n_org;

        // First input is stored two the extra space in the end
        ComplexF x0 = load_F(data_raders + 2*n + 2);

        // FFT
        raders.fft(data_raders);

        // Convolution
        const float *fft_b = raders.fft_b.data();
        for(size_t i = 0; i < n; i++)
        {
            ComplexF x1 = load_F(data_raders + 2*i);
            ComplexF x2 = load_F(fft_b + 2*i);
            ComplexF x3 = mul_F(x1,x2);
            store_F(x3, data_raders + 2*i);
        }

        // IFFT
        raders.ifft(data_raders);

        // Add first value to others
        ComplexF k = broadcast32_F(raders.scale);
        for (size_t i = 0; i < n_org - 1; i++)
        {
            ComplexF res = x0 + k*load_F(data_raders + 2*i);
            store_F(res, data_raders + 2*i);
        }

    } else
    {
        multiply_coeff_F<radix_type,true>(x_in, x_out);
    }
}

//////////////////////////////////// ComplexF2 /////////////////////////////////////////////

template<hhfft::RadixType radix_type> inline float *allocate_raders_F2(const hhfft::RadersF &raders)
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

template<hhfft::RadixType radix_type> inline void free_raders_F2(const hhfft::RadersF &raders, float *data)
{
    if (radix_type == hhfft::RadixType::Raders)
    {
        raders.free_memory(data);
    }
}

// Fill input vector with zeros
template<hhfft::RadixType radix_type> inline void init_coeff_F2(float *x, const hhfft::RadersF &raders)
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
template<hhfft::RadixType radix_type> inline void set_value_F2(ComplexF2 *data_in, float *data_raders, size_t index, const hhfft::RadersF &raders, ComplexF2 x)
{
    if (radix_type == hhfft::RadixType::Raders)
    {
        size_t n  = raders.n;
        size_t n_data = raders.n_data_size;

        // Divide input data into two complex numbers
        ComplexF x0, x1;
        divide_two_64_F2(x, x0, x1);

        // Sum up the values and store it to extra space in the end
        ComplexF sum0 = load_F(data_raders + 2*n) + x0;
        ComplexF sum1 = load_F(data_raders + n_data + 2*n) + x1;
        store_F(sum0, data_raders + 2*n);
        store_F(sum1, data_raders + n_data + 2*n);

        // Store values separately
        const uint32_t *reorder_table_raders_inverse = raders.reorder_table_raders_inverse.data();
        size_t i2 = reorder_table_raders_inverse[index];
        store_F(x0, data_raders + 2*i2);
        store_F(x1, data_raders + n_data + 2*i2);
    } else
    {
        data_in[index] = x;
    }
}

// Multiply one complex number and write it to input when performing fft
template<hhfft::RadixType radix_type, bool copy_twiddle = true> inline void set_value_twiddle_F2(ComplexF2 *data_in, float *data_raders, ComplexF2 *data_twiddle, size_t index, const hhfft::RadersF &raders, ComplexF2 x, ComplexF2 w)
{
    if (radix_type == hhfft::RadixType::Raders)
    {
        size_t n  = raders.n;
        size_t n_data = raders.n_data_size;

        // Multiply with twiddle factors
        ComplexF2 xw = mul_F2(w,x);

        // Divide input data into two complex numbers
        ComplexF x0, x1;
        divide_two_64_F2(xw, x0, x1);

        // Sum up the values and store it to extra space in the end
        ComplexF sum0 = load_F(data_raders + 2*n) + x0;
        ComplexF sum1 = load_F(data_raders + n_data + 2*n) + x1;
        store_F(sum0, data_raders + 2*n);
        store_F(sum1, data_raders + n_data + 2*n);

        const uint32_t *reorder_table_raders_inverse = raders.reorder_table_raders_inverse.data();
        size_t i2 = reorder_table_raders_inverse[index];
        store_F(x0, data_raders + 2*i2);
        store_F(x1, data_raders + n_data + 2*i2);
    } else
    {
        // For optimization reasons, twiddle factors might not be copied here
        if(copy_twiddle)
        {
            data_twiddle[index] = w;
        }
        data_in[index] = x;
    }
}

// Write one complex number to input when performing ifft
template<hhfft::RadixType radix_type> inline void set_value_inverse_F2(ComplexF2 *data_in, float *data_raders, size_t index, const hhfft::RadersF &raders, ComplexF2 x)
{
    if (radix_type == hhfft::RadixType::Raders)
    {
        // Sum up the values and store it to extra space in the end
        size_t n  = raders.n;
        size_t n_org  = raders.n_org;
        size_t n_data = raders.n_data_size;

        // Divide input data into two complex numbers
        ComplexF x0, x1;
        divide_two_64_F2(x, x0, x1);

        // Sum up the values and store it to extra space in the end
        ComplexF sum0 = load_F(data_raders + 2*n) + x0;
        ComplexF sum1 = load_F(data_raders + n_data + 2*n) + x1;
        store_F(sum0, data_raders + 2*n);
        store_F(sum1, data_raders + n_data + 2*n);

        const uint32_t *reorder_table_raders_inverse = raders.reorder_table_raders_inverse.data();
        size_t i2 = n + 1;

        if (index == 0)
        {
            i2 = reorder_table_raders_inverse[index];
        } else
        {
            i2 = reorder_table_raders_inverse[n_org - index];
        }

        store_F(x0, data_raders + 2*i2);
        store_F(x1, data_raders + n_data + 2*i2);
    } else
    {
        data_in[index] = x;
    }
}

// Read one complex number to input
template<hhfft::RadixType radix_type> inline ComplexF2 get_value_F2(ComplexF2 *data_out, float *data_raders, size_t index, const hhfft::RadersF &raders)
{
    if (radix_type == hhfft::RadixType::Raders)
    {
        size_t n_data = raders.n_data_size;
        const uint32_t *reorder_table_raders_inverse = raders.reorder_table_raders_inverse2.data();
        size_t i2 = reorder_table_raders_inverse[index];

        // Load values separately and combine them
        return load_two_64_F2(data_raders + 2*i2, data_raders + n_data + 2*i2);
    } else
    {
        return data_out[index];
    }
}

// Read one complex number to input. Used for real odd
template<hhfft::RadixType radix_type> inline ComplexF2 get_value_real_odd_forward_F2(ComplexF2 *data_out, float *data_raders, size_t index, const hhfft::RadersF &raders)
{
    size_t radix = get_actual_radix<radix_type>(raders);

    if ((index & 1) == 0)
    {
        ComplexF2 x = get_value_F2<radix_type>(data_out, data_raders, index/2, raders);
        return x;
    } else
    {
        ComplexF2 x = get_value_F2<radix_type>(data_out, data_raders, radix - index/2 - 1, raders);
        return conj_F2(x); // conjugate
    }
}

// Do the actual Raders algorithm
template<hhfft::RadixType radix_type> inline void multiply_coeff_forward_F2(const ComplexF2 *x_in, ComplexF2 *x_out, float *data_raders, const hhfft::RadersF &raders)
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
            ComplexF x0 = load_F(data_raders + j*n_data + 2*n + 2);

            // FFT
            raders.fft(data_raders + j*n_data);

            // Convolution
            // NOTE it might be possible to do this using avx-commands, but would it actually be any faster?
            const float *fft_b = raders.fft_b.data();
            for(size_t i = 0; i < n; i++)
            {
                ComplexF x1 = load_F(data_raders + j*n_data + 2*i);
                ComplexF x2 = load_F(fft_b + 2*i);
                ComplexF x3 = mul_F(x1,x2);
                store_F(x3, data_raders + j*n_data + 2*i);
            }

            // IFFT
            raders.ifft(data_raders + j*n_data);

            // Add first value to others
            float k = raders.scale;
            for (size_t i = 0; i < n_org - 1; i++)
            {
                ComplexF res = x0 + k*load_F(data_raders + j*n_data + 2*i);
                store_F(res, data_raders + j*n_data + 2*i);
            }
        }

    } else
    {
        multiply_coeff_F2<radix_type,true>(x_in, x_out);
    }
}
