/*
*   Copyright Jouko Kalmari 2017-2019
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

// This header contains some small portions of functions that can be usable in all different implementations

#ifndef HHFFT_COMMON_COMPLEX_D
#define HHFFT_COMMON_COMPLEX_D

// Functions that can be used in any implementation

template<hhfft::RadixType radix_type, bool forward> inline __attribute__((always_inline))
   void fft_common_complex_stride1_reorder_d(const double *data_in, double *data_out, double *data_raders, uint32_t *reorder_table, ComplexD k, size_t radix, size_t reorder_table_size, const hhfft::RadersD &raders, size_t i)
{
    // Initialize raders data with zeros
    init_coeff_D<radix_type>(data_raders, raders);

    ComplexD x_temp_in[radix_type];
    ComplexD x_temp_out[radix_type];

    // Copy input data taking reordering into account
    for (size_t j = 0; j < radix; j++)
    {
        size_t i2 = i*radix + j;

        if (forward)
        {
            size_t ind = reorder_table[i2];
            ComplexD x = load_D(data_in + 2*ind);
            set_value_D<radix_type>(x_temp_in, data_raders, j, raders, x);
        } else
        {
            size_t ind = reorder_table[reorder_table_size - i2 - 1];
            ComplexD x = k*load_D(data_in + 2*ind);
            set_value_D<radix_type>(x_temp_in, data_raders, j, raders, x);
        }
    }

    // Multiply with coefficients
    multiply_coeff_forward_D<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

    // Save output to two memory locations.
    for (size_t j = 0; j < radix; j++)
    {
        ComplexD x = get_value_D<radix_type>(x_temp_out, data_raders, j, raders);
        store_D(x, data_out + 2*i*radix + 2*j);
    }
}

template<hhfft::RadixType radix_type> inline __attribute__((always_inline))
void fft_common_complex_d(const double *data_in, double *data_out, double *data_raders, size_t radix, size_t stride, const hhfft::RadersD &raders)
{
    // Initialize raders data with zeros
    init_coeff_D<radix_type>(data_raders, raders);

    ComplexD x_temp_in[radix_type];
    ComplexD x_temp_out[radix_type];

    // Copy input data (squeeze)
    for (size_t j = 0; j < radix; j++)
    {
        ComplexD x = load_D(data_in + 2*j*stride);
        set_value_D<radix_type>(x_temp_in, data_raders, j, raders, x);
    }

    // Multiply with coefficients
    multiply_coeff_forward_D<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

    // Copy output data (un-squeeze)
    for (size_t j = 0; j < radix; j++)
    {
        ComplexD x = get_value_D<radix_type>(x_temp_out, data_raders, j, raders);
        store_D(x, data_out + 2*j*stride);
    }
}

template<hhfft::RadixType radix_type> inline __attribute__((always_inline))
void fft_common_complex_twiddle_d(const double *data_in, double *data_out, double *data_raders, const double *twiddle_factors, size_t radix, size_t stride, const hhfft::RadersD &raders)
{
    // Initialize raders data with zeros
    init_coeff_D<radix_type>(data_raders, raders);

    ComplexD x_temp_in[radix_type];
    ComplexD x_temp_out[radix_type];
    ComplexD twiddle_temp[radix_type];

    // Copy input data (squeeze)
    for (size_t j = 0; j < radix; j++)
    {
        ComplexD x = load_D(data_in + 2*j*stride);
        ComplexD w = load_D(twiddle_factors + 2*j*stride);
        set_value_twiddle_D<radix_type>(x_temp_in, data_raders, twiddle_temp, j, raders, x, w);
    }

    // Multiply with coefficients
    multiply_twiddle_D<radix_type,true>(x_temp_in, x_temp_in, twiddle_temp);
    multiply_coeff_forward_D<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

    // Copy output data (un-squeeze)
    for (size_t j = 0; j < radix; j++)
    {
        ComplexD x = get_value_D<radix_type>(x_temp_out, data_raders, j, raders);
        store_D(x, data_out + 2*j*stride);
    }
}


// Functions that can be used if compiled with avx, avx512f etc
#ifdef HHFFT_COMMON_AVX_D

template<hhfft::RadixType radix_type, bool forward> inline __attribute__((always_inline))
   void fft_common_complex_stride1_reorder_d2(const double *data_in, double *data_out, double *data_raders, uint32_t *reorder_table, ComplexD2 k, size_t radix, size_t reorder_table_size, const hhfft::RadersD &raders, size_t i)
{
    ComplexD2 x_temp_in[radix_type];
    ComplexD2 x_temp_out[radix_type];

    // Initialize raders data with zeros
    init_coeff_D2<radix_type>(data_raders, raders);

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
        ComplexD2 x = load_two_128_D2(data_in + 2*ind0, data_in + 2*ind1);
        set_value_D2<radix_type>(x_temp_in, data_raders, j, raders, x);
    }

    // Multiply with coefficients
    multiply_coeff_forward_D2<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

    // Save output to two memory locations.
    for (size_t j = 0; j < radix; j++)
    {
        size_t ind0 = i*radix + j;
        size_t ind1 = i*radix + radix + j;
        ComplexD2 x = get_value_D2<radix_type>(x_temp_out, data_raders, j, raders);
        if (!forward)
        {
            x = k*x;
        }
        store_two_128_D2(x, data_out + 2*ind0, data_out + 2*ind1);
    }
}


template<hhfft::RadixType radix_type> inline __attribute__((always_inline))
void fft_common_complex_d2(const double *data_in, double *data_out, double *data_raders, size_t radix, size_t stride, const hhfft::RadersD &raders)
{
    ComplexD2 x_temp_in[radix_type];
    ComplexD2 x_temp_out[radix_type];

    // Initialize raders data with zeros
    init_coeff_D2<radix_type>(data_raders, raders);

    // Copy input data (squeeze)
    for (size_t j = 0; j < radix; j++)
    {
        ComplexD2 x = load_D2(data_in + 2*j*stride);
        set_value_D2<radix_type>(x_temp_in, data_raders, j, raders, x);
    }

    // Multiply with coefficients
    multiply_coeff_forward_D2<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

    // Copy output data (un-squeeze)
    for (size_t j = 0; j < radix; j++)
    {
        ComplexD2 x = get_value_D2<radix_type>(x_temp_out, data_raders, j, raders);
        store_D2(x, data_out + 2*j*stride);
    }
}


template<hhfft::RadixType radix_type> inline __attribute__((always_inline))
void fft_common_complex_twiddle_d2(const double *data_in, double *data_out, double *data_raders, const double *twiddle_factors, size_t radix, size_t stride, const hhfft::RadersD &raders)
{
    // Initialize raders data with zeros
    init_coeff_D2<radix_type>(data_raders, raders);

    ComplexD2 x_temp_in[radix_type];
    ComplexD2 x_temp_out[radix_type];
    ComplexD2 twiddle_temp[radix_type];

    // Copy input data and multiply with twiddle factors
    for (size_t j = 0; j < radix; j++)
    {
        ComplexD2 x = load_D2(data_in + 2*j*stride);
        ComplexD2 w = load_D2(twiddle_factors + 2*j*stride);
        set_value_twiddle_D2<radix_type>(x_temp_in, data_raders, twiddle_temp, j, raders, x, w);
    }

    // Multiply with coefficients
    multiply_twiddle_D2<radix_type,true>(x_temp_in, x_temp_in, twiddle_temp);
    multiply_coeff_forward_D2<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

    // Copy output data (un-squeeze)
    for (size_t j = 0; j < radix; j++)
    {
        ComplexD2 x = get_value_D2<radix_type>(x_temp_out, data_raders, j, raders);
        store_D2(x, data_out + 2*j*stride);
    }
}

#endif

#endif
