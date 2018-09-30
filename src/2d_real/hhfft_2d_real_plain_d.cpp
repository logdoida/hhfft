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

#include "../common/hhfft_1d_complex_plain_common_d.h"
#include "../raders/raders_plain_d.h"

using namespace hhfft;

void fft_2d_real_reorder_rows_in_place_plain_d(const double *, double *data_out,const hhfft::StepInfo<double> &step_info)
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

            double r_temp = data_out[2*i*m + 2*ind1+0];
            double c_temp = data_out[2*i*m + 2*ind1+1];
            data_out[2*i*m + 2*ind1+0] = data_out[2*i*m + 2*ind2+0];
            data_out[2*i*m + 2*ind1+1] = data_out[2*i*m + 2*ind2+1];
            data_out[2*i*m + 2*ind2+0] = r_temp;
            data_out[2*i*m + 2*ind2+1] = c_temp;
        }
    }
}

// Recalculate first column and add last column
void fft_2d_complex_to_complex_packed_first_column_plain_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info)
{
    //const double *packing_table = step_info.twiddle_factors;
    size_t n = step_info.repeats; // n = number of original real rows
    size_t m = step_info.size;    // m = number of original real columns

    size_t m2 = m + 2;

    // First copy data so that row size increases from m/2 to m/2 + 1
    for (size_t i = 0; i < n; i++)
    {
        size_t i2 = n - i - 1;
        for (size_t j = 0; j < m; j++)
        {
            size_t j2 = m - j - 1;
            data_out[i2*m2 + j2] = data_in[i2*m + j2];
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
    void fft_2d_complex_to_complex_packed_plain_d(const double *, double *data_out,const hhfft::StepInfo<double> &step_info)
{        
    const double *packing_table = step_info.twiddle_factors;
    size_t n = step_info.repeats; // n = number of original real rows
    size_t m = step_info.size;    // m = number of original real columns

    if (m%4 == 0)
    {
        for (size_t i = 0; i < n; i++)
        {
            double x_r = data_out[i*m + m/2 + 0];
            double x_i = data_out[i*m + m/2 + 1];

            data_out[i*m + m/2 + 0] =  x_r;
            data_out[i*m + m/2 + 1] = -x_i;
        }
    }

    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 2; j < m/2; j+=2)
        {
            double ss = -packing_table[j + 0];
            double sc = packing_table[j + 1];

            if (forward)
            {
                sc = -sc;
            }

            double x0_r = data_out[i*m + j + 0];
            double x0_i = data_out[i*m + j + 1];
            double x1_r = data_out[i*m + (m-j) + 0];
            double x1_i = data_out[i*m + (m-j) + 1];

            double temp0 = -ss*(x0_r - x1_r) + sc*(x0_i + x1_i);
            double temp1 = -sc*(x0_r - x1_r) - ss*(x0_i + x1_i);

            data_out[i*m + j + 0]     = temp0 + x0_r;
            data_out[i*m + j + 1]     = temp1 + x0_i;
            data_out[i*m + (m-j) + 0] = -temp0 + x1_r;
            data_out[i*m + (m-j) + 1] = temp1 + x1_i;
        }
    }
}

template<RadixType radix_type>
void fft_2d_real_reorder2_inverse_plain_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info)
{
    size_t m = step_info.stride;  // number of columns
    size_t m2 = m + 1;            // number of columns in input
    size_t repeats = step_info.repeats;    
    uint32_t *reorder_table_columns = step_info.reorder_table;
    size_t reorder_table_size = step_info.reorder_table_size;
    double norm_factor = step_info.norm_factor;
    const hhfft::RadersD &raders = *step_info.raders;
    size_t radix = get_actual_radix<radix_type>(raders);

    // Allocate memory for Rader's algorithm if needed
    double *data_raders = allocate_raders<radix_type>(raders);

    double x_temp_in[2*radix_type];
    double x_temp_out[2*radix_type];

    for (size_t i = 0; i < repeats; i++)
    {
        // The first column is calculated from first and last column
        {
            // Initialize raders data with zeros
            init_coeff<radix_type>(data_raders, raders);

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

                double re = t1 + t2;
                double im = t3 + t4;
                set_value<radix_type>(x_temp_in, data_raders, j, raders, re, im);
            }

            // Multiply with coefficients
            multiply_coeff_forward<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

            // Copy output data (un-squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                double re,im;
                get_value<radix_type>(x_temp_out, data_raders, j, raders, re, im);
                data_out[2*i*radix*m + 2*j*m + 0] = norm_factor*re;
                data_out[2*i*radix*m + 2*j*m + 1] = norm_factor*im;
            }
        }

        // do other columns
        for (size_t k = 1; k < m; k++)
        {
            // Initialize raders data with zeros
            init_coeff<radix_type>(data_raders, raders);

            // Copy input data (squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                size_t j1 = i*radix + j;                
                size_t j2 = reorder_table_columns[reorder_table_size - j1 - 1];

                double re = data_in[2*j2*m2 + 2*k + 0];
                double im = data_in[2*j2*m2 + 2*k + 1];
                set_value<radix_type>(x_temp_in, data_raders, j, raders, re, im);
            }

            // Multiply with coefficients
            multiply_coeff_forward<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

            // Copy output data (un-squeeze)
            for (size_t j = 0; j < radix; j++)
            {
                double re,im;
                get_value<radix_type>(x_temp_out, data_raders, j, raders, re, im);
                data_out[2*i*radix*m + 2*j*m + 2*k + 0] = norm_factor*re;
                data_out[2*i*radix*m + 2*j*m + 2*k + 1] = norm_factor*im;
            }
        }
    }

    // Free temporary memory
    free_raders<radix_type>(raders, data_raders);
}


//////////////////////// Odd number of columns ////////////////////////////

// Combine reordering and first row wise FFT
template<RadixType radix_type> void fft_2d_real_reorder2_odd_rows_forward_plain_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info)
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
    double *data_raders = allocate_raders<radix_type>(raders);

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
            init_coeff<radix_type>(data_raders, raders);

            double x_temp_in[2*radix_type];
            double x_temp_out[2*radix_type];

            // Copy input data taking reordering into account
            for (size_t k = 0; k < radix; k++)
            {
                size_t j2 = reorder_table_rows[j*radix + k];                
                double re = data_in[i2*m + j2];
                double im = 0;
                set_value<radix_type>(x_temp_in, data_raders, k, raders, re, im);
            }

            // Multiply with coefficients
            multiply_coeff_forward<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

            // Save only about half of the output
            // First/ last one is real
            if (dir_out) // direction normal
            {
                double re,im;
                get_value<radix_type>(x_temp_out, data_raders, 0, raders, re, im);
                data_out[i*m2 + j*radix + 1] = re;

                for (size_t k = 1; k < radix/2 + 1; k++)
                {
                    double re,im;
                    get_value<radix_type>(x_temp_out, data_raders, k, raders, re, im);
                    data_out[i*m2 + j*radix + 2*k] = re;
                    data_out[i*m2 + j*radix + 2*k + 1] = im;
                }
            } else // direction inverted
            {
                double re,im;
                get_value<radix_type>(x_temp_out, data_raders, 0, raders, re, im);
                data_out[i*m2 + j*radix + radix] = re;

                for (size_t k = 1; k < radix/2 + 1; k++)
                {
                    double re,im;
                    get_value<radix_type>(x_temp_out, data_raders, k, raders, re, im);
                    data_out[i*m2 + j*radix + radix - 2*k] = re;
                    data_out[i*m2 + j*radix + radix - 2*k + 1] = im;
                }
            }
            dir_out = !dir_out;
        }
    }

    // Free temporary memory
    free_raders<radix_type>(raders, data_raders);
}


// Calculates first ifft step for the first column and saves it to a temporary variable
template<RadixType radix_type> void fft_2d_real_odd_rows_reorder_first_column_plain_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info)
{    
    size_t m2 = 2*step_info.stride; // row size in input
    size_t repeats = step_info.repeats;
    uint32_t *reorder_table_columns = step_info.reorder_table;
    size_t reorder_table_columns_size = step_info.reorder_table_size;
    double k = step_info.norm_factor;
    const hhfft::RadersD &raders = *step_info.raders;
    size_t radix = get_actual_radix<radix_type>(raders);

    // Allocate memory for Rader's algorithm if needed
    double *data_raders = allocate_raders<radix_type>(raders);

    for (size_t i = 0; i < repeats; i++)
    {
        // Initialize raders data with zeros
        init_coeff<radix_type>(data_raders, raders);

        double x_temp_in[2*radix_type];
        double x_temp_out[2*radix_type];

        // Copy input data taking reordering and scaling into account
        for (size_t j = 0; j < radix; j++)
        {
            size_t i1 = i*radix + j;
            size_t i2 = reorder_table_columns[reorder_table_columns_size - i1 - 1];            
            double re = data_in[i2*m2 + 0] * k;
            double im = data_in[i2*m2 + 1] * k;
            set_value<radix_type>(x_temp_in, data_raders, j, raders, re, im);
        }

        // Multiply with coefficients
        multiply_coeff_forward<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

        // save output
        for (size_t j = 0; j < radix; j++)
        {
            double re,im;
            get_value<radix_type>(x_temp_out, data_raders, j, raders, re, im);
            data_out[2*(i*radix + j) + 0] = re;
            data_out[2*(i*radix + j) + 1] = im;
        }
    }

    // Free temporary memory
    free_raders<radix_type>(raders, data_raders);
}

// Reordering row- and columnwise, and first IFFT-step combined
template<RadixType radix_type> void fft_2d_real_odd_rows_reorder_columns_plain_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info)
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
    double *data_raders = allocate_raders<radix_type>(raders);

    for (size_t i = 0; i < repeats; i++)
    {
        // all columns, skip the first one as it is processed in temp variable
        for (size_t j = 1; j < m2; j++)
        {
            // Initialize raders data with zeros
            init_coeff<radix_type>(data_raders, raders);

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

            double x_temp_in[2*radix_type];
            double x_temp_out[2*radix_type];

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

                double re = data_in[i2*2*m2 + 2*j2 + 0] * norm_factor;
                double im = data_in[i2*2*m2 + 2*j2 + 1] * norm_factor;

                if (conj)
                {                    
                    im = -im;
                }
                set_value<radix_type>(x_temp_in, data_raders, k, raders, re, im);
            }

            // Multiply with coefficients
            multiply_coeff_forward<radix_type>(x_temp_in, x_temp_out, data_raders, raders);

            // save output. The row size is decreased to m2-1 so that all data fits
            for (size_t k = 0; k < radix; k++)
            {
                double re,im;
                get_value<radix_type>(x_temp_out, data_raders, k, raders, re, im);
                data_out[2*(i*radix + k)*(m2-1) + 2*j - 2] = re;
                data_out[2*(i*radix + k)*(m2-1) + 2*j - 1] = im;
            }
        }
    }

    // Free temporary memory
    free_raders<radix_type>(raders, data_raders);
}


// Instantiations of the functions defined in this class
template void fft_2d_complex_to_complex_packed_plain_d<false>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_complex_to_complex_packed_plain_d<true>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template void fft_2d_real_reorder2_inverse_plain_d<Raders>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_plain_d<Radix2>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_plain_d<Radix3>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_plain_d<Radix4>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_plain_d<Radix5>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_plain_d<Radix6>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_plain_d<Radix7>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_inverse_plain_d<Radix8>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template void fft_2d_real_reorder2_odd_rows_forward_plain_d<Raders>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_odd_rows_forward_plain_d<Radix3>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_odd_rows_forward_plain_d<Radix5>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_reorder2_odd_rows_forward_plain_d<Radix7>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template void fft_2d_real_odd_rows_reorder_first_column_plain_d<Raders>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_first_column_plain_d<Radix2>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_first_column_plain_d<Radix3>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_first_column_plain_d<Radix4>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_first_column_plain_d<Radix5>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_first_column_plain_d<Radix6>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_first_column_plain_d<Radix7>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_first_column_plain_d<Radix8>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template void fft_2d_real_odd_rows_reorder_columns_plain_d<Raders>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_columns_plain_d<Radix2>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_columns_plain_d<Radix3>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_columns_plain_d<Radix4>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_columns_plain_d<Radix5>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_columns_plain_d<Radix6>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_columns_plain_d<Radix7>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template void fft_2d_real_odd_rows_reorder_columns_plain_d<Radix8>(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
