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

#include <iostream>
#include <cmath>
#include <algorithm>
#include <assert.h>

#include "architecture.h"
#include "utilities.h"

#include "hhfft_2d_real_d.h"

#include "1d_complex/hhfft_1d_complex_d.h"
#include "2d_complex/hhfft_2d_complex_d.h"
#include "1d_real/hhfft_1d_real_setter_d.h"
#include "2d_real/hhfft_2d_real_setter_d.h"

using namespace hhfft;
using hhfft::HHFFT_2D_REAL_D;

// True if dif should be used
static const bool use_dif = false;

double* HHFFT_2D_REAL_D::allocate_memory()
{
    // For real data only n*m doubles are needed
    // For complex data 2*((n/2)+1)*m doubles are needed
    return (double *) allocate_aligned_memory(2*((n/2)+1)*m*sizeof(double));
}

void HHFFT_2D_REAL_D::free_memory(double *data)
{
    free(data);
}

// Does the planning step
HHFFT_2D_REAL_D::HHFFT_2D_REAL_D(size_t n, size_t m, InstructionSet instruction_set)
{
    this->n = n;
    this->m = m;

    // This limitation comes from using uint32 in reorder table
    if ((n >= (1ul << 32ul)) || (m >= (1ul << 32ul)))
    {
        throw(std::runtime_error("HHFFT error: maximum size for the fft size is 2^32 - 1!"));
    }

    if ((n == 1) || (m == 1))
    {
        // TODO add a support to small radices with a single pass dft
        throw(std::runtime_error("HHFFT error: fft size must be larger than 1!"));
    }

    if (n == 2)
    {
        // TODO add a support to small radices
        throw(std::runtime_error("HHFFT error: fft size n must be larger than 2!"));
    }

    // Define instruction set if needed
    if (instruction_set == InstructionSet::automatic)
    {
        instruction_set = hhfft::get_best_instruction_set();
    }

    if (n%2 == 0)
    {
        plan_even(instruction_set);
    } else
    {
        // NOTE there might be need for different version for n odd, m even etc
        plan_odd(instruction_set);
    }
}

void HHFFT_2D_REAL_D::plan_odd(InstructionSet instruction_set)
{
    throw(std::runtime_error("HHFFT error: odd fft size not supported!"));
}

void HHFFT_2D_REAL_D::plan_even(InstructionSet instruction_set)
{
    // FFT for columns is done using complex fft of size n/2
    size_t n_complex = n/2;

    // Calculate factorization
    std::vector<size_t> N_columns = calculate_factorization(n_complex, use_dif);
    std::vector<size_t> N_rows = calculate_factorization(m, use_dif);

    // TESTING print factorization    
    //for (size_t i = 0; i < N_columns.size(); i++)  { std::cout << N_columns[i] << " ";} std::cout << std::endl;
    //for (size_t i = 0; i < N_rows.size(); i++)  { std::cout << N_rows[i] << " ";} std::cout << std::endl;

    // First calculate the reorder table
    reorder_table_columns = calculate_reorder_table(N_columns);
    reorder_table_rows = calculate_reorder_table(N_rows);

    // Then in-place version of the reorder table
    reorder_table_in_place_columns = calculate_reorder_table_in_place(reorder_table_columns);
    reorder_table_in_place_rows = calculate_reorder_table_in_place(reorder_table_rows);

    // TESTING print reorder tables
    //std::cout << "reorder = " << std::endl;
    //for (auto r: reorder_table)  { std::cout << r << " ";} std::cout << std::endl;
    //std::cout << "reorder_table_in_place = " << std::endl;
    //for (auto r: reorder_table_in_place)  { std::cout << r << " ";} std::cout << std::endl;

    // Calculate twiddle factors
    // NOTE that a portion of these are always one and they could be removed to decrease memory requirements.

    // Add packing factors
    AlignedVector<double> packing_factors = hhfft::calculate_packing_factors(n);
    twiddle_factors_columns.push_back(packing_factors);

    for (size_t i = 1; i < N_columns.size(); i++)
    {
        AlignedVector<double> w;
        if (use_dif)        
            w = calculate_twiddle_factors_DIF(i, N_columns);
        else
            w = calculate_twiddle_factors_DIT(i, N_columns);
        twiddle_factors_columns.push_back(w);
        //print_complex_vector(w.data(), w.size()/2);
    }
    twiddle_factors_rows.push_back(AlignedVector<double>()); // No twiddle factors are needed before the first fft-level
    for (size_t i = 1; i < N_rows.size(); i++)
    {
        AlignedVector<double> w;
        if (use_dif)
            w = calculate_twiddle_factors_DIF(i, N_rows);
        else
            w = calculate_twiddle_factors_DIT(i, N_rows);
        twiddle_factors_rows.push_back(w);
        //print_complex_vector(w.data(), w.size()/2);
    }

    if (use_dif)
    {
        throw(std::runtime_error("HHFFT error: DIF not yet supported on 2D"));
    }
    else
    {
        // FFT
        ///////// FFT column-wise /////////////

        // Two rows are suffled into one and columns-reordered
        {
            hhfft::StepInfoD step;
            step.data_type_in = hhfft::StepDataType::data_in;
            step.data_type_out = hhfft::StepDataType::data_out;
            step.reorder_table = reorder_table_columns.data();
            step.reorder_table_inplace = reorder_table_in_place_columns.data(); // It is possible that data_in = data_out!
            step.repeats = reorder_table_in_place_columns.size();
            step.stride = n_complex;
            step.size = m;
            step.dif = false;
            HHFFT_2D_Real_D_set_reorder_column_function(step, instruction_set);
            forward_steps.push_back(step);
        }

        // Put first fft step
        // NOTE actually 1D fft is used as no twiddle factors are involved
        {
            hhfft::StepInfoD step;
            step.radix = N_columns[0];
            step.stride = m;
            step.repeats = n_complex / step.radix;
            step.data_type_in = hhfft::StepDataType::data_out;
            step.data_type_out = hhfft::StepDataType::data_out;
            step.dif = false;
            HHFFT_2D_Complex_D_set_function_columns(step, instruction_set);
            forward_steps.push_back(step);            
        }

        // then put rest fft steps combined with twiddle factor
        for (size_t i = 1; i < N_columns.size(); i++)
        {
            hhfft::StepInfoD step;
            hhfft::StepInfoD &step_prev = forward_steps.back();
            step.radix = N_columns[i];
            step.stride = step_prev.stride * step_prev.radix;
            if (i == 1)
                step.stride = step.stride / m;
            step.repeats = step_prev.repeats / step.radix;
            step.size = m;
            step.data_type_in = hhfft::StepDataType::data_out;
            step.data_type_out = hhfft::StepDataType::data_out;
            step.twiddle_factors = twiddle_factors_columns[i].data();
            step.dif = false;
            HHFFT_2D_Complex_D_set_function_columns(step, instruction_set);
            forward_steps.push_back(step);            
        }

        // Convert from complex packed to complex. One new row is added.
        {
            hhfft::StepInfo<double> step;
            step.repeats = n;
            step.size = m;
            step.twiddle_factors = twiddle_factors_columns[0].data();
            step.data_type_in = hhfft::StepDataType::data_out;
            step.data_type_out = hhfft::StepDataType::data_out;
            step.forward = true;
            HHFFT_2D_Real_D_set_complex_to_complex_packed_function(step, instruction_set);
            forward_steps.push_back(step);
        }

        ///////// FFT row-wise ////////////

        // Put reordering step if needed
        if (N_rows.size() > 1)
        {
            // TODO can these two reordering steps be combined?
            hhfft::StepInfoD step;
            step.data_type_in = hhfft::StepDataType::data_out;
            step.data_type_out = hhfft::StepDataType::data_out;
            step.reorder_table = nullptr; // Reordering is always done in-place
            step.reorder_table_inplace = reorder_table_in_place_rows.data();
            step.repeats = reorder_table_in_place_rows.size();
            step.stride = m;
            step.size = n_complex + 1;
            step.dif = false;
            HHFFT_2D_Complex_D_set_function_rows(step, instruction_set);
            forward_steps.push_back(step);
        }

        // Put first fft step
        // NOTE actually 1D fft is used as no twiddle factors are involved
        {
            hhfft::StepInfoD step;
            step.radix = N_rows[0];
            step.stride = 1;
            step.repeats = m * (n_complex + 1) / step.radix;
            step.data_type_in = hhfft::StepDataType::data_out;
            step.data_type_out = hhfft::StepDataType::data_out;
            step.dif = false;
            HHFFT_2D_Complex_D_set_function_rows(step, instruction_set);
            forward_steps.push_back(step);
        }

        // then put rest fft steps combined with twiddle factor
        for (size_t i = 1; i < N_rows.size(); i++)
        {
            hhfft::StepInfoD step;
            hhfft::StepInfoD &step_prev = forward_steps.back();
            step.radix = N_rows[i];
            step.stride = step_prev.stride * step_prev.radix;
            step.repeats = step_prev.repeats / step.radix;
            step.size = 1;
            step.data_type_in = hhfft::StepDataType::data_out;
            step.data_type_out = hhfft::StepDataType::data_out;
            step.twiddle_factors = twiddle_factors_rows[i].data();
            step.dif = false;
            HHFFT_2D_Complex_D_set_function_rows(step, instruction_set);
            forward_steps.push_back(step);
        }


        // IFFT
        ///////// IFFT row-wise ////////////
        // Last row is processed in a temporary array
        temp_data_size = 2*m;

        // Reorder row-wise
        {
            // TODO can these two reordering steps be combined?
            hhfft::StepInfoD step;
            step.data_type_in = hhfft::StepDataType::data_in;
            step.data_type_out = hhfft::StepDataType::data_out;
            step.reorder_table = reorder_table_rows.data();
            step.reorder_table_inplace = reorder_table_in_place_rows.data();
            step.repeats = reorder_table_in_place_rows.size();
            step.stride = m;
            step.size = n_complex;
            step.dif = false;
            step.norm_factor = 1.0 / (n_complex*m);
            step.forward = false;
            HHFFT_2D_Complex_D_set_function_rows(step, instruction_set);
            inverse_steps.push_back(step);
        }        

        // Reorder last row to temp array
        {
            hhfft::StepInfoD step;
            step.data_type_in = hhfft::StepDataType::data_in;
            step.start_index_in = n*m;
            step.data_type_out = hhfft::StepDataType::temp_data;
            step.reorder_table = reorder_table_rows.data();
            step.reorder_table_inplace = nullptr; // In-place not possible
            step.stride = m;
            step.norm_factor = 1.0 / (n_complex*m);
            step.forward = false;
            step.dif = use_dif;
            HHFFT_1D_Complex_D_set_function(step, instruction_set);
            inverse_steps.push_back(step);
        }

        // Put first fft step
        // NOTE actually 1D fft is used as no twiddle factors are involved
        {
            hhfft::StepInfoD step;
            step.radix = N_rows[0];
            step.stride = 1;
            step.repeats = m * n_complex / step.radix;
            step.data_type_in = hhfft::StepDataType::data_out;
            step.data_type_out = hhfft::StepDataType::data_out;
            step.forward = false;
            step.dif = false;
            HHFFT_2D_Complex_D_set_function_rows(step, instruction_set);
            inverse_steps.push_back(step);
        }

        // then put rest fft steps combined with twiddle factor
        for (size_t i = 1; i < N_rows.size(); i++)
        {
            hhfft::StepInfoD step;
            hhfft::StepInfoD &step_prev = inverse_steps.back();
            step.radix = N_rows[i];
            step.stride = step_prev.stride * step_prev.radix;
            step.repeats = step_prev.repeats / step.radix;
            step.size = 1;
            step.data_type_in = hhfft::StepDataType::data_out;
            step.data_type_out = hhfft::StepDataType::data_out;
            step.twiddle_factors = twiddle_factors_rows[i].data();
            step.forward = false;
            step.dif = false;
            HHFFT_2D_Complex_D_set_function_rows(step, instruction_set);
            inverse_steps.push_back(step);
        }

        // Put first fft step for last row
        {
            hhfft::StepInfoD step;
            step.radix = N_rows[0];
            step.stride = 1;
            step.repeats = m / step.radix;
            step.data_type_in = hhfft::StepDataType::temp_data;
            step.data_type_out = hhfft::StepDataType::temp_data;
            step.forward = false;
            step.dif = false;
            HHFFT_1D_Complex_D_set_function(step, instruction_set);
            inverse_steps.push_back(step);
        }

        // then put rest fft steps combined with twiddle factor for last row
        for (size_t i = 1; i < N_rows.size(); i++)
        {
            hhfft::StepInfoD step;
            hhfft::StepInfoD &step_prev = inverse_steps.back();
            step.radix = N_rows[i];
            step.stride = step_prev.stride * step_prev.radix;
            step.repeats = step_prev.repeats / step.radix;
            step.data_type_in = hhfft::StepDataType::temp_data;
            step.data_type_out = hhfft::StepDataType::temp_data;
            step.twiddle_factors = twiddle_factors_rows[i].data();
            step.forward = false;
            step.dif = false;
            HHFFT_1D_Complex_D_set_function(step, instruction_set);
            inverse_steps.push_back(step);
        }        

        // Convert from complex to complex complex packed. The temp row disappears here
        {
            hhfft::StepInfo<double> step;
            step.repeats = n;
            step.size = m;
            step.twiddle_factors = twiddle_factors_columns[0].data();
            step.data_type_in = hhfft::StepDataType::temp_data; // NOTE!
            step.data_type_out = hhfft::StepDataType::data_out;
            step.forward = false;
            HHFFT_2D_Real_D_set_complex_to_complex_packed_function(step, instruction_set);
            inverse_steps.push_back(step);
        }

        ///////// IFFT column-wise /////////////

        // Reorder columns if needed
        if (N_columns.size() > 1)
        {
            hhfft::StepInfoD step;
            step.data_type_in = hhfft::StepDataType::data_out;
            step.data_type_out = hhfft::StepDataType::data_out;
            step.reorder_table = nullptr; // This is always in-place
            step.reorder_table_inplace = reorder_table_in_place_columns.data();
            step.repeats = reorder_table_in_place_columns.size();
            step.stride = n_complex;
            step.size = m;
            step.norm_factor = 1.0; // normalization has been performed earlier
            step.forward = false;
            step.dif = false;
            HHFFT_2D_Complex_D_set_function_columns(step, instruction_set);
            inverse_steps.push_back(step);
        }        

        // Put first fft step
        // NOTE actually 1D fft is used as no twiddle factors are involved
        {
            hhfft::StepInfoD step;
            step.radix = N_columns[0];
            step.stride = m;
            step.repeats = n_complex / step.radix;
            step.data_type_in = hhfft::StepDataType::data_out;
            step.data_type_out = hhfft::StepDataType::data_out;
            step.forward = false;
            step.dif = false;
            HHFFT_2D_Complex_D_set_function_columns(step, instruction_set);
            inverse_steps.push_back(step);
        }

        // then put rest fft steps combined with twiddle factors
        for (size_t i = 1; i < N_columns.size(); i++)
        {
            hhfft::StepInfoD step;
            hhfft::StepInfoD &step_prev = inverse_steps.back();
            step.radix = N_columns[i];
            step.stride = step_prev.stride * step_prev.radix;
            if (i == 1)
                step.stride = step.stride / m;
            step.repeats = step_prev.repeats / step.radix;
            step.size = m;
            step.data_type_in = hhfft::StepDataType::data_out;
            step.data_type_out = hhfft::StepDataType::data_out;
            step.twiddle_factors = twiddle_factors_columns[i].data();
            step.forward = false;
            step.dif = false;
            HHFFT_2D_Complex_D_set_function_columns(step, instruction_set);
            inverse_steps.push_back(step);
        }

        // One row is suffled into two
        {
            hhfft::StepInfoD step;
            step.data_type_in = hhfft::StepDataType::data_out;
            step.data_type_out = hhfft::StepDataType::data_out;
            step.stride = n_complex;
            step.size = m;            
            step.dif = false;
            step.forward = false;
            HHFFT_2D_Real_D_set_reorder_column_function(step, instruction_set);
            inverse_steps.push_back(step);
        }        
    }

    // Set the convolution function
    convolution_function = HHFFT_1D_Complex_D_set_convolution_function(instruction_set);
}

void HHFFT_2D_REAL_D::fft(const double *in, double *out)
{
    // Allocate some extra space if needed    
    hhfft::AlignedVector<double> temp_data(temp_data_size);

    // Put all possible input/output data sources here
    const double *data_in[3] = {in, out, temp_data.data()};
    double *data_out[3] = {nullptr, out, temp_data.data()};

    for (auto &step: forward_steps)
    {
        step.step_function(data_in[step.data_type_in] + step.start_index_in, data_out[step.data_type_out] + step.start_index_out, step);

        // TESTING print        
        //print_complex_matrix(data_out[step.data_type_out], n/2+1, m);
    }
}

void HHFFT_2D_REAL_D::ifft(const double *in, double *out)
{
    // Allocate some extra space if needed    
    hhfft::AlignedVector<double> temp_data(temp_data_size);

    // Put all possible input/output data sources here
    const double *data_in[3] = {in, out, temp_data.data()};
    double *data_out[3] = {nullptr, out, temp_data.data()};

    for (auto &step: inverse_steps)
    {
        step.step_function(data_in[step.data_type_in] + step.start_index_in, data_out[step.data_type_out] + step.start_index_out, step);

        // TESTING print
        //if (step.data_type_out == hhfft::StepDataType::temp_data)
        //    print_complex_matrix(data_out[step.data_type_out], temp_data_size/2, 1);
        //else
        //    print_complex_matrix(data_out[step.data_type_out], n/2, m);
    }
}


// Calculates convolution in Fourier space
void HHFFT_2D_REAL_D::convolution(const double *in1, const double *in2, double *out)
{
    convolution_function(in1, in2, out, (n/2+1)*m);
}

// Prints contents of a matrix that has nxm complex numbers (2*n*m doubles)
void HHFFT_2D_REAL_D::print_complex_matrix(const double *data, size_t n, size_t m)
{
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < m; j++)
        {
            double real = data[2*i*m + 2*j];
            double imag = data[2*i*m + 2*j+1];
            if (imag >= 0.0)
                std::cout << real << "+" << imag << "i  ";
            else
                std::cout << real << imag << "i  ";
        }
        std::cout << "; ";
    }

    std::cout << std::endl;
}

// Prints contents of a matrix that has nxm rael numbers (n*m doubles)
void HHFFT_2D_REAL_D::print_real_matrix(const double *data, size_t n, size_t m)
{
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < m; j++)
        {
            double real = data[i*m + j];
           std::cout << real << "  ";
        }
        std::cout << "; ";
    }

    std::cout << std::endl;
}
