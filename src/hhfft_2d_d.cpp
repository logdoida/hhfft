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
#include "hhfft_1d_d.h"

#include "1d_complex/hhfft_1d_complex_d.h"
#include "2d_complex/hhfft_2d_complex_d.h"

using namespace hhfft;
using hhfft::HHFFT_2D_D;

double* HHFFT_2D_D::allocate_memory() const
{
    return (double *) allocate_aligned_memory(2*n*m*sizeof(double));
}

void HHFFT_2D_D::free_memory(double *data)
{
    free(data);
}

void HHFFT_2D_D::set_radix_raders(size_t radix, StepInfoD &step, InstructionSet instruction_set)
{
    if (radix > 8)
    {
        // Use Rader's algorithm instead
        raders.push_back(std::unique_ptr<RadersD>(new RadersD(radix, instruction_set)));
        step.raders = raders.back().get();
    }
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

    // Define instruction set if needed
    if (instruction_set == InstructionSet::automatic)
    {
        instruction_set = hhfft::get_best_instruction_set();
    }

    // Set the convolution function
    convolution_function = HHFFT_1D_Complex_D_set_convolution_function(instruction_set);

    if ((n == 1) || (m == 1))
    {
        // Use 1d fft to calculate the transformation
        if (n == 1)
        {
            plan_vector(m, instruction_set);
        } else
        {
            plan_vector(n, instruction_set);
        }
        return;
    }

    // Calculate factorization
    std::vector<size_t> N_columns = calculate_factorization(n);
    std::vector<size_t> N_rows = calculate_factorization(m);

    // TESTING print factorization    
    //for (size_t i = 0; i < N_columns.size(); i++)  { std::cout << N_columns[i] << " ";} std::cout << std::endl;
    //for (size_t i = 0; i < N_rows.size(); i++)  { std::cout << N_rows[i] << " ";} std::cout << std::endl;

    // First calculate the reorder table
    reorder_table_columns = calculate_reorder_table(N_columns);
    reorder_table_rows = calculate_reorder_table(N_rows);

    // Add extra values to the end for ifft reordering
    append_reorder_table(reorder_table_columns, n/N_columns.back());
    append_reorder_table(reorder_table_rows, m/N_rows.back());

    // TESTING print reorder tables    
    //std::cout << "reorder_table_columns = " << std::endl;
    //for (auto r: reorder_table_columns)  { std::cout << r << " ";} std::cout << std::endl;
    //std::cout << "reorder_table_rows = " << std::endl;
    //for (auto r: reorder_table_rows)  { std::cout << r << " ";} std::cout << std::endl;
    //std::cout << "reorder_table_in_place_columns = " << std::endl;
    //for (auto r: reorder_table_in_place_columns)  { std::cout << r << " ";} std::cout << std::endl;
    //std::cout << "reorder_table_in_place_rows = " << std::endl;
    //for (auto r: reorder_table_in_place_rows)  { std::cout << r << " ";} std::cout << std::endl;

    // Calculate twiddle factors
    // NOTE that a portion of these are always one and they could be removed to decrease memory requirements.
    // NOTE if n==m, only one set of twiddle factors could be used
    twiddle_factors_columns.push_back(AlignedVector<double>()); // No twiddle factors are needed before the first fft-level
    for (size_t i = 1; i < N_columns.size(); i++)
    {
        AlignedVector<double> w = calculate_twiddle_factors_DIT<double>(i, N_columns);
        twiddle_factors_columns.push_back(w);
        //print_complex_vector(w.data(), w.size()/2);
    }
    twiddle_factors_rows.push_back(AlignedVector<double>()); // No twiddle factors are needed before the first fft-level
    for (size_t i = 1; i < N_rows.size(); i++)
    {
        AlignedVector<double> w;
        w = calculate_twiddle_factors_DIT<double>(i, N_rows);
        twiddle_factors_rows.push_back(w);
        //print_complex_vector(w.data(), w.size()/2);
    }

    // This helps to create the inverse steps    
    std::vector<bool> forward_step_columns;

    ///////// FFT column-wise /////////////

    // Reordering row- and columnwise, and first FFT-step combined here. And scaling in ifft
    {
        hhfft::StepInfoD step;
        step.data_type_in = hhfft::StepDataType::data_in;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.reorder_table = reorder_table_columns.data();
        step.reorder_table_size = reorder_table_columns.size();
        step.reorder_table2 = reorder_table_rows.data();
        step.reorder_table2_size = reorder_table_rows.size();
        step.stride = n;
        step.size = m;
        set_radix_raders(N_columns[0], step, instruction_set);
        step.radix = N_columns[0];
        step.repeats = n / step.radix;
        HHFFT_2D_Complex_D_set_function_columns(step, instruction_set);
        forward_steps.push_back(step);
        forward_step_columns.push_back(true);
    }

    // then put rest fft steps combined with twiddle factor
    for (size_t i = 1; i < N_columns.size(); i++)
    {
        hhfft::StepInfoD step;
        hhfft::StepInfoD &step_prev = forward_steps.back();
        set_radix_raders(N_columns[i], step, instruction_set);
        step.radix = N_columns[i];
        step.stride = step_prev.stride * step_prev.radix;
        if (i == 1)
            step.stride = step_prev.size * step_prev.radix / m;
        step.repeats = step_prev.repeats / step.radix;
        step.size = m;
        step.data_type_in = hhfft::StepDataType::data_out;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.twiddle_factors = twiddle_factors_columns[i].data();
        HHFFT_2D_Complex_D_set_function_columns(step, instruction_set);
        forward_steps.push_back(step);
        forward_step_columns.push_back(true);
    }

    ///////// FFT row-wise ////////////

    // Put first fft step
    // NOTE actually 1D fft is used as no twiddle factors are involved
    {
        hhfft::StepInfoD step;
        set_radix_raders(N_rows[0], step, instruction_set);
        step.radix = N_rows[0];
        step.stride = 1;
        step.repeats = m * n / step.radix;
        step.data_type_in = hhfft::StepDataType::data_out;
        step.data_type_out = hhfft::StepDataType::data_out;
        HHFFT_2D_Complex_D_set_function_rows(step, instruction_set);
        forward_steps.push_back(step);
        forward_step_columns.push_back(false);
    }

    // then put rest fft steps combined with twiddle factor
    for (size_t i = 1; i < N_rows.size(); i++)
    {
        hhfft::StepInfoD step;
        hhfft::StepInfoD &step_prev = forward_steps.back();
        set_radix_raders(N_rows[i], step, instruction_set);
        step.radix = N_rows[i];
        step.stride = step_prev.stride * step_prev.radix;
        step.repeats = step_prev.repeats / step.radix;
        step.size = 1;
        step.data_type_in = hhfft::StepDataType::data_out;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.twiddle_factors = twiddle_factors_rows[i].data();
        HHFFT_2D_Complex_D_set_function_rows(step, instruction_set);
        forward_steps.push_back(step);
        forward_step_columns.push_back(false);
    }


    // Make the inverse steps. They are otherwise the same, but different version of function is called    
    for (size_t i = 0; i < forward_steps.size(); i++)
    {
        auto step = forward_steps[i];

        // Reordering + scaling is done in the first step
        if (i == 0)
        {
            step.norm_factor = 1.0/(double(n*m));
            step.forward = false;
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
}

void HHFFT_2D_D::plan_vector(size_t n, InstructionSet instruction_set)
{
    HHFFT_1D_D fft_1d(n, instruction_set);

    // Copy/move data from the 1d plan
    temp_data_size = fft_1d.temp_data_size;
    reorder_table_rows = std::move(fft_1d.reorder_table);    
    forward_steps = std::move(fft_1d.forward_steps);
    inverse_steps = std::move(fft_1d.inverse_steps);
    twiddle_factors_rows = std::move(fft_1d.twiddle_factors);
    raders = std::move(fft_1d.raders);
}

void HHFFT_2D_D::fft(const double *in, double *out) const
{
    // If there is just one step, run it directly
    if (forward_steps.size() == 1)
    {
        forward_steps[0].step_function(in,out,forward_steps[0]);
        return;
    }

    // If transform is made in-place, copy input to a temporary variable
    hhfft::AlignedVector<double> temp_data_in;
    if (in == out)
    {
        temp_data_in.resize(2*n*m);
        std::copy(in, in + 2*n*m, temp_data_in.data());
        in = temp_data_in.data();
    }

    // Allocate some extra space if needed    
    hhfft::AlignedVector<double> temp_data(temp_data_size);

    // Put all possible input/output data sources here
    const double *data_in[3] = {in, out, temp_data.data()};
    double *data_out[3] = {nullptr, out, temp_data.data()};

    // Run all the steps
    for (auto &step: forward_steps)
    {
        step.step_function(data_in[step.data_type_in] + step.start_index_in, data_out[step.data_type_out] + step.start_index_out, step);

        // TESTING print        
        //print_complex_matrix(data_out[step.data_type_out], n, m);
    }
}

void HHFFT_2D_D::ifft(const double *in, double *out) const
{
    // If there is just one step, run it directly
    if (inverse_steps.size() == 1)
    {
        inverse_steps[0].step_function(in,out,inverse_steps[0]);
        return;
    }

    // If transform is made in-place, copy input to a temporary variable
    hhfft::AlignedVector<double> temp_data_in;
    if (in == out)
    {
        temp_data_in.resize(2*n*m);
        std::copy(in, in + 2*n*m, temp_data_in.data());
        in = temp_data_in.data();
    }

    // Allocate some extra space if needed
    hhfft::AlignedVector<double> temp_data(temp_data_size);

    // Put all possible input/output data sources here
    const double *data_in[3] = {in, out, temp_data.data()};
    double *data_out[3] = {nullptr, out, temp_data.data()};

    // Run all the steps
    for (auto &step: inverse_steps)
    {
        step.step_function(data_in[step.data_type_in] + step.start_index_in, data_out[step.data_type_out] + step.start_index_out, step);

        // TESTING print
        //print_complex_matrix(data_out[step.data_type_out], n, m);
    }
}

// Calculates convolution in Fourier space
void HHFFT_2D_D::convolution(const double *in1, const double *in2, double *out) const
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
