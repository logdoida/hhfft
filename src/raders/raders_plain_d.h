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
#include "../common/hhfft_common_plain_d.h"

template<hhfft::RadixType radix_type> inline size_t get_actual_radix(const hhfft::RadersD &raders)
{
    if (radix_type == hhfft::RadixType::Raders)
    {
        return raders.n_org;
    } else
    {
        return radix_type;
    }
}

//////////////////////////////////// ComplexD /////////////////////////////////////////////


template<hhfft::RadixType radix_type> inline double *allocate_raders_D(const hhfft::RadersD &raders)
{
    if (radix_type == hhfft::RadixType::Raders)
    {
        return raders.allocate_memory();
    } else
    {
        return nullptr;
    }
}

template<hhfft::RadixType radix_type> inline void free_raders_D(const hhfft::RadersD &raders, double *data)
{
    if (radix_type == hhfft::RadixType::Raders)
    {
        raders.free_memory(data);
    }
}

// Fill input vector with zeros
template<hhfft::RadixType radix_type> inline void init_coeff_D(double *x, const hhfft::RadersD &raders)
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
template<hhfft::RadixType radix_type> inline void set_value_D(ComplexD *data_in, double *data_raders, size_t index, const hhfft::RadersD &raders, ComplexD x)
{
    if (radix_type == hhfft::RadixType::Raders)
    {
        // Sum up the values and store it to extra space in the end
        size_t n  = raders.n;
        ComplexD sum = load_D(data_raders + 2*n) + x;
        store_D(sum, data_raders + 2*n);

        const uint32_t *reorder_table_raders_inverse = raders.reorder_table_raders_inverse.data();
        size_t i2 = reorder_table_raders_inverse[index];
        store_D(x, data_raders + 2*i2);
    } else
    {
        data_in[index] = x;
    }
}

// Multiply one complex number and write it to input when performing fft
template<hhfft::RadixType radix_type> inline void set_value_twiddle_D(ComplexD *data_in, double *data_raders, ComplexD *data_twiddle, size_t index, const hhfft::RadersD &raders, ComplexD x, ComplexD w)
{
    if (radix_type == hhfft::RadixType::Raders)
    {
        // Multiply with twiddle factors. First twiddle factor is assumed to be (1+0i)
        ComplexD xw = mul_D(w,x);

        // Sum up the values and store it to extra space in the end
        size_t n  = raders.n;
        ComplexD sum = load_D(data_raders + 2*n) + xw;
        store_D(sum, data_raders + 2*n);

        const uint32_t *reorder_table_raders_inverse = raders.reorder_table_raders_inverse.data();
        size_t i2 = reorder_table_raders_inverse[index];
        store_D(xw, data_raders + 2*i2);
    } else
    {
        data_twiddle[index] = w;
        data_in[index] = x;
    }
}

// Write one complex number to input when performing ifft
template<hhfft::RadixType radix_type> inline void set_value_inverse_D(ComplexD *data_in, double *data_raders, size_t index, const hhfft::RadersD &raders, ComplexD x)
{
    if (radix_type == hhfft::RadixType::Raders)
    {
        // Sum up the values and store it to extra space in the end
        size_t n  = raders.n;
        size_t n_org  = raders.n_org;
        ComplexD sum = load_D(data_raders + 2*n) + x;
        store_D(sum, data_raders + 2*n);

        const uint32_t *reorder_table_raders_inverse = raders.reorder_table_raders_inverse.data();
        size_t i2 = n + 1;

        if (index == 0)
        {
            i2 = reorder_table_raders_inverse[index];
        } else
        {
            i2 = reorder_table_raders_inverse[n_org - index];
        }

        store_D(x, data_raders + 2*i2);
    } else
    {
        data_in[index] = x;
    }
}

// Read one complex number to input
template<hhfft::RadixType radix_type> inline ComplexD get_value_D(ComplexD *data_out, double *data_raders, size_t index, const hhfft::RadersD &raders)
{
    if (radix_type == hhfft::RadixType::Raders)
    {
        const uint32_t *reorder_table_raders_inverse = raders.reorder_table_raders_inverse2.data();
        size_t i2 = reorder_table_raders_inverse[index];

        return load_D(data_raders + 2*i2);
    } else
    {
        return data_out[index];
    }
}

// Read one complex number to input. Used for real odd
template<hhfft::RadixType radix_type> inline ComplexD get_value_real_odd_forward_D(ComplexD *data_out, double *data_raders, size_t index, const hhfft::RadersD &raders)
{
    size_t radix = get_actual_radix<radix_type>(raders);

    if ((index & 1) == 0)
    {
        ComplexD x = get_value_D<radix_type>(data_out, data_raders, index/2, raders);
        return x;
    } else
    {
        ComplexD x = get_value_D<radix_type>(data_out, data_raders, radix - index/2 - 1, raders);
        return conj_D(x); // conjugate
    }
}

// Do the actual Raders algorithm
template<hhfft::RadixType radix_type> inline void multiply_coeff_forward_D(const ComplexD *x_in, ComplexD *x_out, double *data_raders, const hhfft::RadersD &raders)
{
    if (radix_type == hhfft::RadixType::Raders)
    {
        size_t n = raders.n;
        size_t n_org = raders.n_org;

        // First input is stored to the extra space in the end
        ComplexD x0 = load_D(data_raders + 2*n + 2);

        // FFT
        raders.fft(data_raders);

        // Convolution
        const double *fft_b = raders.fft_b.data();
        for(size_t i = 0; i < n; i++)
        {
            ComplexD x1 = load_D(data_raders + 2*i);
            ComplexD x2 = load_D(fft_b + 2*i);
            ComplexD x3 = mul_D(x1,x2);
            store_D(x3, data_raders + 2*i);
        }

        // IFFT
        raders.ifft(data_raders);

        // Add first value to others
        ComplexD k = broadcast64_D(raders.scale);
        for (size_t i = 0; i < n_org - 1; i++)
        {
            ComplexD res = x0 + k*load_D(data_raders + 2*i);
            store_D(res, data_raders + 2*i);
        }

    } else
    {
        multiply_coeff_D<radix_type,true>(x_in, x_out);
    }
}
