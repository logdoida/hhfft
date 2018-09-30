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
#include "../common/hhfft_1d_complex_plain_common_d.h"

template<hhfft::RadixType radix_type> inline double *allocate_raders(const hhfft::RadersD &raders)
{
    if (radix_type == hhfft::RadixType::Raders)
    {
        return raders.allocate_memory();
    } else
    {
        return nullptr;
    }
}

template<hhfft::RadixType radix_type> inline void free_raders(const hhfft::RadersD &raders, double *data)
{
    if (radix_type == hhfft::RadixType::Raders)
    {
        raders.free_memory(data);
    }
}

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

// Fill input vector with zeros
template<hhfft::RadixType radix_type> inline void init_coeff(double *x, const hhfft::RadersD &raders)
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
template<hhfft::RadixType radix_type> inline void set_value(double *data_in, double *data_raders, size_t index, const hhfft::RadersD &raders, double re, double im)
{
    if (radix_type == hhfft::RadixType::Raders)
    {        
        // Sum up the values and store it to extra space in the end
        size_t n  = raders.n;
        data_raders[2*n + 0] += re;
        data_raders[2*n + 1] += im;

        const uint32_t *reorder_table_raders_inverse = raders.reorder_table_raders_inverse.data();        
        size_t i2 = reorder_table_raders_inverse[index];
        data_raders[2*i2 + 0] = re;
        data_raders[2*i2 + 1] = im;
    } else
    {
        data_in[2*index + 0] = re;
        data_in[2*index + 1] = im;
    }
}

// Multiply one complex number and write it to input when performing fft
template<hhfft::RadixType radix_type> inline void set_value_twiddle(double *data_in, double *data_raders, double *twiddle_factors, size_t index, const hhfft::RadersD &raders, double re, double im, double w_re, double w_im)
{    
    if (radix_type == hhfft::RadixType::Raders)
    {        
        // Multiply with twiddle factors
        double re2 = w_re*re - w_im*im;
        double im2 = w_im*re + w_re*im;

        // Sum up the values and store it to extra space in the end
        size_t n  = raders.n;
        data_raders[2*n + 0] += re2;
        data_raders[2*n + 1] += im2;

        const uint32_t *reorder_table_raders_inverse = raders.reorder_table_raders_inverse.data();
        size_t i2 = reorder_table_raders_inverse[index];
        data_raders[2*i2 + 0] = re2;
        data_raders[2*i2 + 1] = im2;
    } else
    {
        twiddle_factors[2*index + 0] = w_re;
        twiddle_factors[2*index + 1] = w_im;
        data_in[2*index + 0] = re;
        data_in[2*index + 1] = im;
    }
}

// Write one complex number to input when performing ifft
template<hhfft::RadixType radix_type> inline void set_value_inverse(double *data_in, double *data_raders, size_t index, const hhfft::RadersD &raders, double re, double im)
{
    if (radix_type == hhfft::RadixType::Raders)
    {
        // Sum up the values and store it to extra space in the end
        size_t n  = raders.n;
        size_t n_org  = raders.n_org;
        data_raders[2*n + 0] += re;
        data_raders[2*n + 1] += im;

        const uint32_t *reorder_table_raders_inverse = raders.reorder_table_raders_inverse.data();
        size_t i2 = n + 1;

        if (index == 0)
        {
            i2 = reorder_table_raders_inverse[index];
        } else
        {
            i2 = reorder_table_raders_inverse[n_org - index];
        }

        data_raders[2*i2 + 0] = re;
        data_raders[2*i2 + 1] = im;
    } else
    {
        data_in[2*index + 0] = re;
        data_in[2*index + 1] = im;
    }
}

// Read one complex number to input
template<hhfft::RadixType radix_type> inline void get_value(double *data_out, double *data_raders, size_t index, const hhfft::RadersD &raders, double &re, double &im)
{    
    if (radix_type == hhfft::RadixType::Raders)
    {        
        const uint32_t *reorder_table_raders_inverse = raders.reorder_table_raders_inverse2.data();
        size_t i2 = reorder_table_raders_inverse[index];

        re = data_raders[2*i2 + 0];
        im = data_raders[2*i2 + 1];
    } else
    {
        re = data_out[2*index + 0];
        im = data_out[2*index + 1];
    }
}

// Read one complex number to input. Used for real odd
template<hhfft::RadixType radix_type> inline void get_value_real_odd_forward(double *data_out, double *data_raders, size_t index, const hhfft::RadersD &raders, double &re, double &im)
{
    size_t radix = get_actual_radix<radix_type>(raders);

    if ((index & 1) == 0)
    {
        get_value<radix_type>(data_out, data_raders, index/2, raders, re, im);
    } else
    {
        get_value<radix_type>(data_out, data_raders, radix - index/2 - 1, raders, re, im);
        im = -im; // conjugate
    }
}

// Do the actual Raders algorithm
template<hhfft::RadixType radix_type> inline void multiply_coeff_forward(const double *x_in, double *x_out, double *data_raders, const hhfft::RadersD &raders)
{
    if (radix_type == hhfft::RadixType::Raders)
    {
        size_t n = raders.n;
        size_t n_org = raders.n_org;

        // First input is stored two the extra space in the end
        double re_0 = data_raders[2*n + 2];
        double im_0 = data_raders[2*n + 3];

        // FFT
        raders.fft(data_raders);

        // Convolution
        const double *fft_b = raders.fft_b.data();
        for(size_t i = 0; i < n; i++)
        {
            double re1 = data_raders[2*i + 0];
            double im1 = data_raders[2*i + 1];
            double re2 = fft_b[2*i + 0];
            double im2 = fft_b[2*i + 1];

            data_raders[2*i + 0] = re1*re2 - im1*im2;
            data_raders[2*i + 1] = re1*im2 + im1*re2;
        }

        // IFFT
        raders.ifft(data_raders);

        // Add first value to others
        double k = raders.scale;        
        for (size_t i = 0; i < n_org - 1; i++)
        {
            data_raders[2*i + 0] = re_0 + k*data_raders[2*i + 0];
            data_raders[2*i + 1] = im_0 + k*data_raders[2*i + 1];
        }
    } else
    {
        multiply_coeff<radix_type,true>(x_in, x_out);
    }
}
