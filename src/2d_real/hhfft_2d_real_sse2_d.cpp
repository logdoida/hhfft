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

#include "architecture.h"
#include "step_info.h"
#include <stdlib.h>
#include <assert.h>
#include <cmath>

#include "../common/hhfft_common_sse2_d.h"
#include "../raders/raders_sse2_d.h"

using namespace hhfft;

void fft_2d_real_reorder_rows_in_place_sse2_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info)
{
    size_t n = step_info.stride; // number of rows
    size_t m = step_info.size; // number of columns
    size_t reorder_table_size = step_info.reorder_table_inplace_size;
    uint32_t *reorder_table = step_info.reorder_table_inplace;

    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < reorder_table_size; j++)
        {
            size_t ind1 = j + 1; // First one has been omitted!
            size_t ind2 = reorder_table[j];

            ComplexD temp1 = load_D(data_in + 2*i*m + 2*ind1);
            ComplexD temp2 = load_D(data_in + 2*i*m + 2*ind2);
            store_D(temp1, data_out + 2*i*m + 2*ind2);
            store_D(temp2, data_out + 2*i*m + 2*ind1);
        }
    }
}

// Recalculate first column and add last column
void fft_2d_complex_to_complex_packed_first_column_sse2_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info)
{
    //const double *packing_table = step_info.twiddle_factors;
    size_t n = step_info.repeats; // n = number of original real rows
    size_t m = step_info.size;    // m = number of original real columns

    size_t m2 = m + 2;

    // First copy data so that row size increases from m/2 to m/2 + 1
    for (size_t i = 0; i < n; i++)
    {
        size_t i2 = n - i - 1;

        for (size_t j = 0; j < m; j+=2)
        {
            size_t j2 = m - j - 2;
            ComplexD x = load_D(data_out + i2*m + j2);
            store_D(x, data_out + i2*m2 + j2);
        }
    }

    // Re-calculate first and last columns
    // First
    {
        double x_r = data_in[0];
        double x_i = data_in[1];
        data_out[0] = x_r + x_i;
        data_out[1] = 0.0;
        data_out[m + 0] = x_r - x_i;
        data_out[m + 1] = 0.0;
    }

    // Middle
    if (n%2 == 0)
    {
        double x_r = data_in[n/2*m2 + 0];
        double x_i = data_in[n/2*m2 + 1];
        data_out[n/2*m2 + 0] = x_r + x_i;
        data_out[n/2*m2 + 1] = 0.0;
        data_out[n/2*m2 + m + 0] = x_r - x_i;
        data_out[n/2*m2 + m + 1] = 0.0;
    }

    // Others
    for (size_t i = 1; i < (n+1)/2; i++)
    {
        size_t i2 = n - i;
        double x0_r = data_in[i*m2 + 0];
        double x0_i = data_in[i*m2 + 1];
        double x1_r = data_in[i2*m2 + 0];
        double x1_i = data_in[i2*m2 + 1];

        double t1 = 0.5*(x0_r + x1_r);
        double t2 = 0.5*(x0_r - x1_r);
        double t3 = 0.5*(x0_i + x1_i);
        double t4 = 0.5*(x0_i - x1_i);

        data_out[i*m2 + 0] = t1 + t3;
        data_out[i*m2 + 1] = t4 - t2;
        data_out[i*m2 + m + 0] = t1 - t3;
        data_out[i*m2 + m + 1] = t2 + t4;
        data_out[i2*m2 + 0] = t1 + t3;
        data_out[i2*m2 + 1] = -(t4 - t2);
        data_out[i2*m2 + m + 0] = t1 - t3;
        data_out[i2*m2 + m + 1] = -(t2 + t4);
    }
}


template<bool forward>
    void fft_2d_complex_to_complex_packed_sse2_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info)
{
    const double *packing_table = step_info.twiddle_factors;
    size_t n = step_info.repeats; // n = number of original real rows
    size_t m = step_info.size;    // m = number of original real columns

    if (m%4 == 0)
    {
        for (size_t i = 0; i < n; i++)
        {
            ComplexD x_in = load_D(data_in + i*m + m/2);
            ComplexD x_out = change_sign_D(x_in, const1_128);
            store_D(x_out, data_out + i*m + m/2);
        }
    }

    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 2; j < m/2; j+=2)
        {
            ComplexD sssc = load_D(packing_table + j);
            if (!forward)
            {
                sssc = change_sign_D(sssc, const1_128);
            }

            ComplexD x0_in = load_D(data_out + i*m + j);
            ComplexD x1_in = load_D(data_out + i*m + (m-j));

            ComplexD temp0 = x0_in + change_sign_D(x1_in, const2_128);
            ComplexD temp1 = mul_D(sssc, temp0);

            ComplexD x0_out = temp1 + x0_in;
            ComplexD x1_out = change_sign_D(temp1, const2_128) + x1_in;

            store_D(x0_out, data_out + i*m + j);
            store_D(x1_out, data_out + i*m + (m-j));
        }
    }
}

template<RadixType radix_type>
void fft_2d_real_reorder2_inverse_sse2_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info)
{
    size_t m = step_info.stride;  // number of columns
    size_t m2 = m + 1;            // number of columns in input    
    size_t repeats = step_info.repeats;
    uint32_t *reorder_table_columns = step_info.reorder_table;
    size_t reorder_table_size = step_info.reorder_table_size;
    ComplexD norm_factor = broadcast64_D(step_info.norm_factor);
    const hhfft::RadersD &raders = *step_info.raders;
    size_t radix = get_actual_radix<radix_type>(raders);

    // Allocate memory for Rader's algorithm if needed
    double *data_raders = allocate_raders_D<radix_type>(raders);

    for (size_t i = 0; i < repeats; i++)
    {
        // The first column is calculated from first and last column
        // k = 0
        {
            // Initialize raders data with zeros
            init_coeff_D<radix_type>(data_raders, raders);

            ComplexD x_temp_in[radix_type];
            ComplexD x_temp_out[radix_type];

            // Copy input data (squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                size_t j1 = i*radix + j;
                size_t j2 = reorder_table_columns[reorder_table_size - j1 - 1];

                double x0_r = data_in[2*j2*m2 + 0];
                double x0_i = data_in[2*j2*m2 + 1];
                double x1_r = data_in[2*j2*m2 + 2*m + 0];
                double x1_i = data_in[2*j2*m2 + 2*m + 1];

                double t1 = 0.5*(x0_r + x1_r);
                double t2 = 0.5*(x1_i - x0_i);
                double t3 = 0.5*(x0_r - x1_r);
                double t4 = 0.5*(x0_i + x1_i);

                ComplexD x = norm_factor*load_D(t1 + t2, t3 + t4);
                set_value_D<radix_type>(x_temp_in, data_raders, j, raders, x);
            }

            // Multiply with coefficients
            multiply_coeff_forward_D<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

            // Copy output data (un-squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                ComplexD x = get_value_D<radix_type>(x_temp_out, data_raders, j, raders);
                store_D(x, data_out + 2*i*radix*m + 2*j*m);
            }
        }

        for (size_t k = 1; k < m; k++)
        {
            // Initialize raders data with zeros
            init_coeff_D<radix_type>(data_raders, raders);

            ComplexD x_temp_in[radix_type];
            ComplexD x_temp_out[radix_type];

            // Copy input data (squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                size_t j1 = i*radix + j;                
                size_t j2 = reorder_table_columns[reorder_table_size - j1 - 1];                
                ComplexD x = norm_factor*load_D(data_in + 2*j2*m2 + 2*k);
                set_value_D<radix_type>(x_temp_in, data_raders, j, raders, x);

            }

            // Multiply with coefficients
            multiply_coeff_forward_D<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

            // Copy output data (un-squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                ComplexD x = get_value_D<radix_type>(x_temp_out, data_raders, j, raders);
                store_D(x, data_out + 2*i*radix*m + 2*j*m + 2*k);
            }
        }
    }

    // Free temporary memory
    free_raders_D<radix_type>(raders, data_raders);
}

//////////////////////// Odd number of columns ////////////////////////////

// Combine reordering and first row wise FFT
template<RadixType radix_type> void fft_2d_real_reorder2_odd_rows_forward_sse2_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info)
{
    const hhfft::RadersD &raders = *step_info.raders;
    size_t radix = get_actual_radix<radix_type>(raders);
    size_t n = step_info.size;
    size_t m = step_info.repeats * radix;  // row size in input
    size_t m2 = m+1; // row size in output
    size_t repeats = step_info.repeats;
    uint32_t *reorder_table_columns = step_info.reorder_table;
    uint32_t *reorder_table_rows = step_info.reorder_table2;

    // Allocate memory for Rader's algorithm if needed
    double *data_raders = allocate_raders_D<radix_type>(raders);

    // FFT and reordering
    for (size_t i = 0; i < n; i++)
    {
        size_t i2 = reorder_table_columns[i];

        // As the row size increases by one, set first value to be zero
        data_out[i*m2] = 0;

        bool dir_out = true;
        for (size_t j = 0; j < repeats; j++)
        {
            // Initialize raders data with zeros
            init_coeff_D<radix_type>(data_raders, raders);

            ComplexD x_temp_in[radix_type];
            ComplexD x_temp_out[radix_type];

            // Copy input data taking reordering into account
            for (size_t k = 0; k < radix; k++)
            {
                size_t j2 = reorder_table_rows[j*radix + k];

                ComplexD x = load_real_D(data_in + i2*m + j2);
                set_value_D<radix_type>(x_temp_in, data_raders, k, raders, x);
            }

            // Multiply with coefficients
            multiply_coeff_forward_D<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

            // Save only about half of the output
            // First/ last one is real
            if (dir_out) // direction normal
            {
                ComplexD x0 = get_value_D<radix_type>(x_temp_out, data_raders, 0, raders);
                store_real_D(x0, data_out + i*m2 + j*radix + 1);

                for (size_t k = 1; k < radix/2 + 1; k++)
                {
                    ComplexD x = get_value_D<radix_type>(x_temp_out, data_raders, k, raders);
                    store_D(x, data_out + i*m2 + j*radix + 2*k);
                }
            } else // direction inverted
            {
                ComplexD x0 = get_value_D<radix_type>(x_temp_out, data_raders, 0, raders);
                store_real_D(x0, data_out + i*m2 + j*radix + radix);

                for (size_t k = 1; k < radix/2 + 1; k++)
                {
                    ComplexD x = get_value_D<radix_type>(x_temp_out, data_raders, k, raders);
                    store_D(x, data_out + i*m2 + j*radix + radix - 2*k);
                }
            }
            dir_out = !dir_out;
        }
    }

    // Free temporary memory
    free_raders_D<radix_type>(raders, data_raders);
}


// Calculates first ifft step for the first column and saves it to a temporary variable
template<RadixType radix_type> void fft_2d_real_odd_rows_reorder_first_column_sse2_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info)
{
    size_t m2 = 2*step_info.stride; // row size in input
    size_t repeats = step_info.repeats;
    uint32_t *reorder_table_columns = step_info.reorder_table;
    size_t reorder_table_columns_size = step_info.reorder_table_size;
    double k = step_info.norm_factor;
    const hhfft::RadersD &raders = *step_info.raders;
    size_t radix = get_actual_radix<radix_type>(raders);

    // Allocate memory for Rader's algorithm if needed
    double *data_raders = allocate_raders_D<radix_type>(raders);

    for (size_t i = 0; i < repeats; i++)
    {
        // Initialize raders data with zeros
        init_coeff_D<radix_type>(data_raders, raders);

        ComplexD x_temp_in[radix_type];
        ComplexD x_temp_out[radix_type];

        // Copy input data taking reordering and scaling into account
        for (size_t j = 0; j < radix; j++)
        {
            size_t i1 = i*radix + j;
            size_t i2 = reorder_table_columns[reorder_table_columns_size - i1 - 1];
            ComplexD x = load_D(data_in + i2*m2)*k;
            set_value_D<radix_type>(x_temp_in, data_raders, j, raders, x);
        }

        // Multiply with coefficients
        multiply_coeff_forward_D<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

        // save output
        for (size_t j = 0; j < radix; j++)
        {
            ComplexD x = get_value_D<radix_type>(x_temp_out, data_raders, j, raders);
            store_D(x, data_out + 2*(i*radix + j));
        }
    }

    // Free temporary memory
    free_raders_D<radix_type>(raders, data_raders);
}

// Reordering row- and columnwise, and first IFFT-step combined
template<RadixType radix_type> void fft_2d_real_odd_rows_reorder_columns_sse2_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info)
{    
    size_t m2 = step_info.size; // row size in input
    size_t m = 2*m2 - 1;        // row size originally
    size_t repeats = step_info.repeats;
    uint32_t *reorder_table_columns = step_info.reorder_table;
    uint32_t *reorder_table_rows = step_info.reorder_table2;
    size_t reorder_table_columns_size = step_info.reorder_table_size;
    double norm_factor = step_info.norm_factor;
    const hhfft::RadersD &raders = *step_info.raders;
    size_t radix = get_actual_radix<radix_type>(raders);

    // Allocate memory for Rader's algorithm if needed
    double *data_raders = allocate_raders_D<radix_type>(raders);

    for (size_t i = 0; i < repeats; i++)
    {
        // all columns, skip the first one as it is processed in temp variable
        for (size_t j = 1; j < m2; j++)
        {
            // Initialize raders data with zeros
            init_coeff_D<radix_type>(data_raders, raders);

            size_t j2 = m - reorder_table_rows[j];

            // For some of the columns the output should be conjugated
            // This is achieved by conjugating input and changing its order: conj(ifft(x)) = ifft(conj(swap(x))
            // Also ifft(x) = fft(swap(x))!
            bool conj = false;
            if (j2 > m/2)
            {
                j2 = m - j2;
                conj = true;
            }

            ComplexD x_temp_in[radix_type];
            ComplexD x_temp_out[radix_type];

            // Copy input data taking reordering and scaling into account
            for (size_t k = 0; k < radix; k++)
            {
                size_t i2;
                if (conj)
                {
                    i2 = reorder_table_columns[i*radix + k];
                } else
                {
                    i2 = reorder_table_columns[reorder_table_columns_size - i*radix - k - 1];
                }

                ComplexD x = load_D(data_in + i2*2*m2 + 2*j2)*norm_factor;

                if (conj)
                {
                    set_value_D<radix_type>(x_temp_in, data_raders, k, raders, conj_D(x));
                } else
                {
                    set_value_D<radix_type>(x_temp_in, data_raders, k, raders, x);
                }
            }

            // Multiply with coefficients
            multiply_coeff_forward_D<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

            // save output. The row size is decreased to m2-1 so that all data fits
            for (size_t k = 0; k < radix; k++)
            {
                ComplexD x = get_value_D<radix_type>(x_temp_out, data_raders, k, raders);
                store_D(x, data_out + 2*(i*radix + k)*(m2-1) + 2*j - 2);
            }
        }
    }

    // Free temporary memory
    free_raders_D<radix_type>(raders, data_raders);
}

// Instantiations of the functions defined in this class
template void fft_2d_complex_to_complex_packed_sse2_d<false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_to_complex_packed_sse2_d<true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template void fft_2d_real_reorder2_inverse_sse2_d<Raders>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_sse2_d<Radix2>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_sse2_d<Radix3>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_sse2_d<Radix4>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_sse2_d<Radix5>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_sse2_d<Radix6>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_sse2_d<Radix7>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_sse2_d<Radix8>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_sse2_d<Radix11>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_sse2_d<Radix13>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template void fft_2d_real_reorder2_odd_rows_forward_sse2_d<Raders>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_odd_rows_forward_sse2_d<Radix3>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_odd_rows_forward_sse2_d<Radix5>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_odd_rows_forward_sse2_d<Radix7>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_odd_rows_forward_sse2_d<Radix11>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_odd_rows_forward_sse2_d<Radix13>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template void fft_2d_real_odd_rows_reorder_first_column_sse2_d<Raders>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_first_column_sse2_d<Radix2>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_first_column_sse2_d<Radix3>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_first_column_sse2_d<Radix4>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_first_column_sse2_d<Radix5>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_first_column_sse2_d<Radix6>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_first_column_sse2_d<Radix7>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_first_column_sse2_d<Radix8>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_first_column_sse2_d<Radix11>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_first_column_sse2_d<Radix13>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template void fft_2d_real_odd_rows_reorder_columns_sse2_d<Raders>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_columns_sse2_d<Radix2>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_columns_sse2_d<Radix3>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_columns_sse2_d<Radix4>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_columns_sse2_d<Radix5>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_columns_sse2_d<Radix6>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_columns_sse2_d<Radix7>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_columns_sse2_d<Radix8>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_columns_sse2_d<Radix11>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_columns_sse2_d<Radix13>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
