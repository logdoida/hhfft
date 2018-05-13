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

#include "hhfft_2d_d.h"

#include "1d_complex/hhfft_1d_complex_d.h"
#include "2d_complex/hhfft_2d_complex_d.h"

using namespace hhfft;
using hhfft::HHFFT_2D_D;

// True if dif should be used
static const bool use_dif = false;

double* HHFFT_2D_D::allocate_memory()
{
    return (double *) allocate_aligned_memory(2*n*m*sizeof(double));
}

void HHFFT_2D_D::free_memory(double *data)
{
    free(data);
}

// Does the planning step
HHFFT_2D_D::HHFFT_2D_D(size_t n, size_t m, InstructionSet instruction_set)
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

    // Define instruction set if needed
    if (instruction_set == InstructionSet::automatic)
    {
        instruction_set = hhfft::get_best_instruction_set();
    }

    // Calculate factorization
    std::vector<size_t> N_columns = calculate_factorization(n, use_dif);
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
    // NOTE if n==m, only one set of twiddle factors could be used
    twiddle_factors_columns.push_back(AlignedVector<double>()); // No twiddle factors are needed before the first fft-level
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

    // This helps to create the inverse steps    
    std::vector<bool> forward_step_columns;

    if (use_dif)
    {
        throw(std::runtime_error("HHFFT error: DIF not yet supported on 2D"));
    }
    else
    {
        ///////// FFT column-wise /////////////

        // Put reordering step if needed
        // TODO it is not possible to skip this step because of scaling required in the inverse step!
        //if (N_columns.size() > 1)
        //{
            // TODO reordering and first fft step could be combined
            hhfft::StepInfoD step1;
            step1.data_type_in = hhfft::StepDataType::data_in;
            step1.data_type_out = hhfft::StepDataType::data_out;
            step1.reorder_table = reorder_table_columns.data();
            step1.reorder_table_inplace = reorder_table_in_place_columns.data(); // It is possible that data_in = data_out!
            step1.repeats = reorder_table_in_place_columns.size();
            step1.stride = n;
            step1.size = m;
            //step1.norm_factor = 1.0/(double(n*m));
            step1.dif = false;
            HHFFT_2D_Complex_D_set_function_columns(step1, instruction_set);
            forward_steps.push_back(step1);
            forward_step_columns.push_back(true);
        //}

        // Put first fft step
        // NOTE actually 1D fft is used as no twiddle factors are involved
        hhfft::StepInfoD step2;
        step2.radix = N_columns[0];
        step2.stride = m;
        step2.repeats = n / step2.radix;
        step2.data_type_in = hhfft::StepDataType::data_out;
        //if (forward_steps.size() == 0)
        //    step2.data_type_in = hhfft::StepDataType::data_in;
        step2.data_type_out = hhfft::StepDataType::data_out;
        step2.dif = false;
        HHFFT_2D_Complex_D_set_function_columns(step2, instruction_set);
        forward_steps.push_back(step2);
        forward_step_columns.push_back(true);

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
            forward_step_columns.push_back(true);
        }

        ///////// FFT row-wise ////////////

        // Put reordering step if needed
        if (N_rows.size() > 1)
        {
            // TODO can these two reordering steps be combined?
            hhfft::StepInfoD step3;
            step3.data_type_in = hhfft::StepDataType::data_out;
            step3.data_type_out = hhfft::StepDataType::data_out;
            step3.reorder_table = nullptr; // Reordering is always done in-place
            step3.reorder_table_inplace = reorder_table_in_place_rows.data();
            step3.repeats = reorder_table_in_place_rows.size();
            step3.stride = m;
            step3.size = n;
            step3.norm_factor = 1.0; // Scaling has been done in the first reordering step!
            step3.dif = false;
            HHFFT_2D_Complex_D_set_function_rows(step3, instruction_set);
            forward_steps.push_back(step3);
            forward_step_columns.push_back(false);
        }

        // Put first fft step
        // NOTE actually 1D fft is used as no twiddle factors are involved
        hhfft::StepInfoD step4;
        step4.radix = N_rows[0];
        step4.stride = 1;
        step4.repeats = m * n / step4.radix;
        step4.data_type_in = hhfft::StepDataType::data_out;
        step4.data_type_out = hhfft::StepDataType::data_out;
        step4.dif = false;
        HHFFT_2D_Complex_D_set_function_rows(step4, instruction_set);
        forward_steps.push_back(step4);
        forward_step_columns.push_back(false);

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
            forward_step_columns.push_back(false);
        }
    }

    // Make the inverse steps. They are otherwise the same, but different version of function is called    
    for (size_t i = 0; i < forward_steps.size(); i++)
    {
        auto step = forward_steps[i];

        step.forward = false;

        // Scaling is done in the first step
        if (i == 0)
        {
            step.norm_factor = 1.0/(double(n*m));
        }

        if (forward_step_columns[i])
        {
            HHFFT_2D_Complex_D_set_function_columns(step, instruction_set);
        } else
        {
            HHFFT_2D_Complex_D_set_function_rows(step, instruction_set);
        }

        inverse_steps.push_back(step);
    }

    // Set the convolution function
    convolution_function = HHFFT_1D_Complex_D_set_convolution_function(instruction_set);
}

void HHFFT_2D_D::fft(const double *in, double *out)
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
        //print_complex_matrix(data_out[step.data_type_out], n, m);
    }
}

void HHFFT_2D_D::ifft(const double *in, double *out)
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
        //print_complex_matrix(data_out[step.data_type_out], n, m);
    }
}

// Calculates convolution in Fourier space
void HHFFT_2D_D::convolution(const double *in1, const double *in2, double *out)
{
    convolution_function(in1, in2, out, n*m);
}

// Prints contents of a matrix that has nxm complex numbers (2*n*m doubles)
void HHFFT_2D_D::print_complex_matrix(const double *data, size_t n, size_t m)
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
