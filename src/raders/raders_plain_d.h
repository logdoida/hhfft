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

// This header contains inline functions used in Rader's method

#include "raders_d.h"
#include "../common/hhfft_1d_complex_plain_common_d.h"

// Fill input vector with zeros
template<size_t radix> inline void init_coeff(double *x, hhfft::RadersD &raders)
{
    if (radix == 0)
    {
        size_t n = raders.n;
        for (size_t i = 0; i < 2*(n+2); i++)
        {
            x[i] = 0.0;
        }
    }
}

// Write one complex number to input when performing fft
template<size_t radix> inline void set_value(double *data_in, double *data_raders, size_t index, const hhfft::RadersD &raders, double re, double im)
{
    if (radix == 0)
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

// Write one complex number to input when performing ifft
template<size_t radix> inline void set_value_inverse(double *data_in, double *data_raders, size_t index, const hhfft::RadersD &raders, double re, double im)
{
    if (radix == 0)
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
template<size_t radix> inline void get_value(double *data_out, double *data_raders, size_t index, const hhfft::RadersD &raders, double &re, double &im)
{
    if (radix == 0)
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

// Do the actual Raders algorithm
template<size_t radix> inline void multiply_coeff_forward(const double *x_in, double *x_out, double *data_raders, const hhfft::RadersD &raders)
{
    if (radix == 0)
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
        for (size_t i = 0; i < n_org; i++)
        {
            data_raders[2*i + 0] = re_0 + k*data_raders[2*i + 0];
            data_raders[2*i + 1] = im_0 + k*data_raders[2*i + 1];
        }

    } else
    {
        multiply_coeff<radix,true>(x_in, x_out);
    }


}
