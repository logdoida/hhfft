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
#include "hhfft_1d_real_d.h"

#include "1d_complex/hhfft_1d_complex_setter.h"
#include "2d_complex/hhfft_2d_complex_setter.h"
#include "1d_real/hhfft_1d_real_setter.h"
#include "2d_real/hhfft_2d_real_setter_d.h"

using namespace hhfft;
using hhfft::HHFFT_2D_REAL_D;

double* HHFFT_2D_REAL_D::allocate_memory() const
{
    // For real data only n*m doubles are needed
    // For complex data n*2*((m/2)+1) doubles are needed
    return (double *) allocate_aligned_memory(n*2*((m/2)+1)*sizeof(double));
}

void HHFFT_2D_REAL_D::free_memory(double *data)
{
    free(data);
}

void HHFFT_2D_REAL_D::set_radix_raders(size_t radix, StepInfoD &step, InstructionSet instruction_set)
{
    if (radix > 8)
    {
        // Use Rader's algorithm instead
        raders.push_back(std::unique_ptr<RadersD>(new RadersD(radix, instruction_set)));
        step.raders = raders.back().get();
    }
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

    // Define instruction set if needed
    if (instruction_set == InstructionSet::automatic)
    {
        instruction_set = hhfft::get_best_instruction_set();
    }

    // Set the convolution function
    convolution_function = HHFFT_1D_Complex_set_convolution_function<double>(instruction_set);

    if ((n == 1) || (m == 1))
    {
        // Use 1d fft to calculate the transformation
        if (n == 1)
        {
            plan_vector(m, instruction_set, false);
        } else
        {
            plan_vector(n, instruction_set, true);
        }
        return;
    }

    if (m%2 == 0)
    {
        plan_even(instruction_set);
    } else
    {        
        plan_odd(instruction_set);
    }
}

// This function is used as the final step when m = 1
void fft_2d_real_one_column_conj(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info)
{
    size_t n = step_info.size;

    // On odd FFTs the first real value needs to be moved one position in some cases
    if (step_info.forward)
    {
        data_out[0] = data_in[1];
        data_out[1] = 0;
    }

    for (size_t i = 0; i < (n-1)/2; i++)
    {
        size_t i1 = 2*(n/2+1+i);
        size_t i2 = 2*((n-1)/2-i);

        data_out[i1 + 0] = data_in[i2 + 0];
        data_out[i1 + 1] = -data_in[i2 + 1];
    }
}

void HHFFT_2D_REAL_D::plan_vector(size_t n, InstructionSet instruction_set, bool is_column)
{
    HHFFT_1D_REAL_D fft_1d_real(n, instruction_set);

    // Copy/move data from the 1d plan
    temp_data_size = fft_1d_real.temp_data_size;
    reorder_table_rows = std::move(fft_1d_real.reorder_table);    
    forward_steps = std::move(fft_1d_real.forward_steps);
    inverse_steps = std::move(fft_1d_real.inverse_steps);
    twiddle_factors_rows = std::move(fft_1d_real.twiddle_factors);
    reorder_table_ifft_odd_rows = std::move(fft_1d_real.reorder_table_ifft_odd);
    reorder_table_columns = std::move(fft_1d_real.reorder_table_inverse);
    raders = std::move(fft_1d_real.raders);

    if (is_column)
    {
        // Copy and conjugate
        hhfft::StepInfoD step;
        step.size = n;
        step.step_function = fft_2d_real_one_column_conj;
        step.forward = (forward_steps.size() > 1) && ((n & 1) == 1); // Determine in which cases the first real value is in wrong position
        forward_steps.push_back(step);
    }
}

void HHFFT_2D_REAL_D::plan_odd(InstructionSet instruction_set)
{
    // Calculate factorization
    std::vector<size_t> N_columns = calculate_factorization(n);
    std::vector<size_t> N_rows = calculate_factorization(m);

    // First calculate the reorder table
    reorder_table_columns = calculate_reorder_table(N_columns);
    reorder_table_rows = calculate_reorder_table(N_rows);

    // Add extra values to the end for ifft reordering
    append_reorder_table(reorder_table_columns, n/N_columns.back());

    // Calculate reorder table for ifft
    reorder_table_ifft_odd_rows = calculate_reorder_table_ifft_odd(reorder_table_rows, N_rows);

    // TESTING print reorder tables
    //std::cout << "reorder_table_columns = " << std::endl;
    //for (auto r: reorder_table_columns)  { std::cout << r << " ";} std::cout << std::endl;
    //std::cout << "reorder_table_ifft_odd_rows = " << std::endl;
    //for (auto r: reorder_table_ifft_odd_rows)  { std::cout << r << " ";} std::cout << std::endl;

    // Calculate twiddle factors
    twiddle_factors_rows.push_back(AlignedVector<double>()); // No twiddle factors are needed before the first fft-level
    for (size_t i = 1; i < N_rows.size(); i++)
    {
        AlignedVector<double> w;
        w = calculate_twiddle_factors_DIT<double>(i, N_rows);
        twiddle_factors_rows.push_back(w);
        //print_complex_vector(w.data(), w.size()/2);
    }
    twiddle_factors_columns.push_back(AlignedVector<double>()); // No twiddle factors are needed before the first fft-level
    for (size_t i = 1; i < N_columns.size(); i++)
    {
        AlignedVector<double> w;
        w = calculate_twiddle_factors_DIT<double>(i, N_columns);
        twiddle_factors_columns.push_back(w);
        //print_complex_vector(w.data(), w.size()/2);
    }

    // FFT
    ///////// FFT row-wise ////////////
    // Reordering row- and columnwise, and first FFT-step combined here.
    {
        hhfft::StepInfoD step;
        step.data_type_in = hhfft::StepDataType::data_in;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.reorder_table = reorder_table_columns.data();
        step.reorder_table2 = reorder_table_rows.data();
        set_radix_raders(N_rows[0], step, instruction_set);
        step.radix = N_rows[0];
        step.stride = 1;
        step.repeats = m / step.radix;
        step.size = n;        
        HHFFT_2D_Real_D_odd_set_function_rows(step, instruction_set);
        forward_steps.push_back(step);
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
        step.size = n;
        step.data_type_in = hhfft::StepDataType::data_out;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.twiddle_factors = twiddle_factors_rows[i].data();
        HHFFT_2D_Real_D_odd_set_function_rows(step, instruction_set);
        forward_steps.push_back(step);
    }

    // convert first column from [0 r] to [r 0]
    {
        hhfft::StepInfoD step;
        step.data_type_in = hhfft::StepDataType::data_out;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.radix = 1;
        step.stride = n;
        step.size = m;
        HHFFT_2D_Real_D_odd_set_function_rows(step, instruction_set);
        forward_steps.push_back(step);
    }

    ///////// FFT column-wise /////////////
    size_t m2 = (m+1)/2;

    // First FFT step
    // NOTE 1D fft is used as no twiddle factors are involved
    {
        hhfft::StepInfoD step;
        step.data_type_in = hhfft::StepDataType::data_out;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.stride = m2;
        set_radix_raders(N_columns[0], step, instruction_set);
        step.radix = N_columns[0];
        step.repeats = n / step.radix;
        HHFFT_1D_Complex_set_function<double>(step, instruction_set);
        forward_steps.push_back(step);
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
            step.stride = step_prev.radix;
        step.repeats = step_prev.repeats / step.radix;
        step.size = m2;
        step.data_type_in = hhfft::StepDataType::data_out;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.twiddle_factors = twiddle_factors_columns[i].data();
        HHFFT_2D_Complex_set_function_columns<double>(step, instruction_set);
        forward_steps.push_back(step);
    }


    ///////// IFFT column-wise ////////////
    // Process first column in temp variable
    {
        hhfft::StepInfoD step;
        step.data_type_in = hhfft::StepDataType::data_in;
        step.data_type_out = hhfft::StepDataType::temp_data;
        step.reorder_table = reorder_table_columns.data();        
        step.reorder_table_size = reorder_table_columns.size();
        step.stride = m2;
        set_radix_raders(N_columns[0], step, instruction_set);
        step.radix = N_columns[0];
        step.repeats = n / step.radix;
        step.forward = false;
        step.norm_factor = 1.0/(n*m);
        HHFFT_2D_Real_D_odd_set_function_columns(step, instruction_set);
        inverse_steps.push_back(step);
        temp_data_size = 2*n;
    }

    // other fft steps for first column
    for (size_t i = 1; i < N_columns.size(); i++)
    {
        hhfft::StepInfoD step;
        hhfft::StepInfoD &step_prev = inverse_steps.back();
        set_radix_raders(N_columns[i], step, instruction_set);
        step.radix = N_columns[i];
        if (i == 1)
        {
            step.stride = step_prev.radix;
        } else
        {
            step.stride = step_prev.stride * step_prev.radix;
        }
        step.repeats = step_prev.repeats / step.radix;
        step.data_type_in = hhfft::StepDataType::temp_data;
        step.data_type_out = hhfft::StepDataType::temp_data;
        step.twiddle_factors = twiddle_factors_columns[i].data();
        step.forward = true;
        HHFFT_1D_Complex_set_function<double>(step, instruction_set);
        inverse_steps.push_back(step);
    }


    // Process other columns
    // Reordering row- and columnwise, and first IFFT-step combined here.
    {
        hhfft::StepInfoD step;
        step.data_type_in = hhfft::StepDataType::data_in;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.reorder_table = reorder_table_columns.data();
        step.reorder_table_size = reorder_table_columns.size();
        step.reorder_table2 = reorder_table_ifft_odd_rows.data();
        set_radix_raders(N_columns[0], step, instruction_set);
        step.radix = N_columns[0];
        step.stride = 1;
        step.repeats = n / step.radix;
        step.size = m2;
        step.forward = false;
        step.norm_factor = 1.0/(n*m);
        HHFFT_2D_Real_D_odd_set_function_columns(step, instruction_set);
        inverse_steps.push_back(step);
    }

    // then put rest fft steps combined with twiddle factor
    for (size_t i = 1; i < N_columns.size(); i++)
    {
        hhfft::StepInfoD step;
        hhfft::StepInfoD &step_prev = inverse_steps.back();
        step.data_type_in = hhfft::StepDataType::data_out;
        step.data_type_out = hhfft::StepDataType::data_out;
        set_radix_raders(N_columns[i], step, instruction_set);
        step.radix = N_columns[i];
        step.stride = step_prev.stride * step_prev.radix;
        step.repeats = step_prev.repeats / step.radix;
        step.size = m2 - 1; // First column is in temp variable!
        step.twiddle_factors = twiddle_factors_columns[i].data();
        step.forward = true;
        HHFFT_2D_Complex_set_function_columns<double>(step, instruction_set);
        inverse_steps.push_back(step);
    }

    // combine first column and the rest
    {
        hhfft::StepInfoD step;
        step.data_type_in = hhfft::StepDataType::temp_data;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.radix = 1;
        step.stride = n;
        step.size = m;
        step.forward = false;
        HHFFT_2D_Real_D_odd_set_function_columns(step, instruction_set);
        inverse_steps.push_back(step);
    }

    ///////// IFFT row-wise /////////////

    // First ifft step
    {
        hhfft::StepInfoD step;
        step.data_type_in = hhfft::StepDataType::data_out;
        step.data_type_out = hhfft::StepDataType::data_out;
        set_radix_raders(N_rows[0], step, instruction_set);
        step.radix = N_rows[0];
        step.stride = 1;
        step.repeats = (m / step.radix + 1)/2;
        step.size = n;
        step.forward = false;
        HHFFT_2D_Real_D_odd_set_function_rows(step, instruction_set);
        inverse_steps.push_back(step);
    }

    // Rest ifft steps combined with twiddle factor
    for (size_t i = 1; i < N_rows.size(); i++)
    {
        hhfft::StepInfoD step;
        hhfft::StepInfoD &step_prev = inverse_steps.back();
        set_radix_raders(N_rows[i], step, instruction_set);
        step.radix = N_rows[i];
        step.stride = step_prev.stride * step_prev.radix;
        step.repeats = ((2*step_prev.repeats - 1) / step.radix + 1)/2;
        step.size = n;
        step.data_type_in = hhfft::StepDataType::data_out;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.twiddle_factors = twiddle_factors_rows[i].data();
        step.forward = false;
        HHFFT_2D_Real_D_odd_set_function_rows(step, instruction_set);
        inverse_steps.push_back(step);
    }
}

void HHFFT_2D_REAL_D::plan_even(InstructionSet instruction_set)
{
    // FFT for columns is done using complex fft of size n/2
    size_t m_complex = m/2;

    // Calculate factorization
    std::vector<size_t> N_columns = calculate_factorization(n);
    std::vector<size_t> N_rows = calculate_factorization(m_complex);

    // TESTING print factorization    
    //for (size_t i = 0; i < N_columns.size(); i++)  { std::cout << N_columns[i] << " ";} std::cout << std::endl;
    //for (size_t i = 0; i < N_rows.size(); i++)  { std::cout << N_rows[i] << " ";} std::cout << std::endl;

    // First calculate the reorder table
    reorder_table_columns = calculate_reorder_table(N_columns);
    reorder_table_rows = calculate_reorder_table(N_rows);

    // Add extra values to the end for ifft reordering
    append_reorder_table(reorder_table_columns, n/N_columns.back());

    // Calculate reorder table in place "inverted" as input is actually reordered instead of calling ifft
    std::vector<uint32_t> reorder_table_rows_temp(m_complex);
    for (size_t i = 1; i < m_complex; i++)
    {
        reorder_table_rows_temp[i] = m_complex - reorder_table_rows[i];
    }
    reorder_table_in_place_rows = calculate_reorder_table_in_place(reorder_table_rows_temp);

    // TESTING print reorder tables
    //std::cout << "reorder_table_columns = " << std::endl;
    //for (auto r: reorder_table_columns)  { std::cout << r << " ";} std::cout << std::endl;
    //std::cout << "reorder_table_in_place = " << std::endl;
    //for (auto r: reorder_table_in_place)  { std::cout << r << " ";} std::cout << std::endl;
    //std::cout << "reorder_table_rows = " << std::endl;
    //for (auto r: reorder_table_rows)  { std::cout << r << " ";} std::cout << std::endl;
    //std::cout << "reorder_table_in_place_rows = " << std::endl;
    //for (auto r: reorder_table_in_place_rows)  { std::cout << r << " ";} std::cout << std::endl;

    // Add packing factors
    AlignedVector<double> packing_factors = hhfft::calculate_packing_factors<double>(m);
    twiddle_factors_rows.push_back(packing_factors);

    // Calculate twiddle factors
    for (size_t i = 1; i < N_rows.size(); i++)
    {
        AlignedVector<double> w;
        w = calculate_twiddle_factors_DIT<double>(i, N_rows);
        twiddle_factors_rows.push_back(w);
        //print_complex_vector(w.data(), w.size()/2);
    }
    twiddle_factors_columns.push_back(AlignedVector<double>()); // No twiddle factors are needed before the first fft-level
    for (size_t i = 1; i < N_columns.size(); i++)
    {
        AlignedVector<double> w;
        w = calculate_twiddle_factors_DIT<double>(i, N_columns);
        twiddle_factors_columns.push_back(w);
        //print_complex_vector(w.data(), w.size()/2);
    }

    // FFT
    ///////// FFT row-wise ////////////
    if (m_complex == 1)
    {
        // Only reordering of columns is required
        hhfft::StepInfoD step;
        step.data_type_in = hhfft::StepDataType::data_in;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.reorder_table = reorder_table_columns.data();
        step.radix = 1;
        step.repeats = n;
        HHFFT_1D_Complex_set_reorder_function<double>(step, instruction_set);
        forward_steps.push_back(step);

    } else {
        // Reordering row- and columnwise, and first FFT-step combined here.
        hhfft::StepInfoD step;
        step.data_type_in = hhfft::StepDataType::data_in;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.reorder_table = reorder_table_columns.data();
        step.reorder_table2 = reorder_table_rows.data();        
        set_radix_raders(N_rows[0], step, instruction_set);
        step.radix = N_rows[0];
        step.stride = 1;
        step.repeats = m_complex / step.radix;
        step.size = n;
        HHFFT_2D_Complex_set_function_rows<double>(step, instruction_set);
        forward_steps.push_back(step);
    }

    // then put rest fft steps combined with twiddle factor
    // this is actually using 1D complex FFT
    for (size_t i = 1; i < N_rows.size(); i++)
    {
        hhfft::StepInfoD step;
        hhfft::StepInfoD &step_prev = forward_steps.back();
        set_radix_raders(N_rows[i], step, instruction_set);
        step.radix = N_rows[i];
        step.stride = step_prev.stride * step_prev.radix;
        step.repeats = step_prev.repeats / step.radix;
        if (i == 1)
            step.repeats *= n;
        step.size = 1;
        step.data_type_in = hhfft::StepDataType::data_out;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.twiddle_factors = twiddle_factors_rows[i].data();
        HHFFT_1D_Complex_set_function<double>(step, instruction_set);
        forward_steps.push_back(step);        
    }

    // Convert from complex packed to complex.
    {
        hhfft::StepInfo<double> step;
        step.repeats = n;
        step.size = m;
        step.twiddle_factors = twiddle_factors_rows[0].data();
        step.data_type_in = hhfft::StepDataType::data_out;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.forward = true;
        HHFFT_2D_Real_D_set_complex_to_complex_packed_function(step, instruction_set);
        forward_steps.push_back(step);
    }

    ///////// FFT column-wise /////////////    
    // First FFT step
    // NOTE 1D fft is used as no twiddle factors are involved
    {
        hhfft::StepInfoD step;
        step.data_type_in = hhfft::StepDataType::data_out;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.stride = m_complex;
        step.radix = N_columns[0];
        set_radix_raders(N_columns[0], step, instruction_set);
        step.repeats = n / step.radix;
        HHFFT_1D_Complex_set_function<double>(step, instruction_set);
        forward_steps.push_back(step);        
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
            step.stride = step_prev.radix;
        step.repeats = step_prev.repeats / step.radix;
        step.size = m_complex;
        step.data_type_in = hhfft::StepDataType::data_out;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.twiddle_factors = twiddle_factors_columns[i].data();
        HHFFT_2D_Complex_set_function_columns<double>(step, instruction_set);
        forward_steps.push_back(step);
    }

    // Add new column to the end, and recalculate first and last column
    {
        hhfft::StepInfoD step;
        step.repeats = n;
        step.size = m;
        step.data_type_in = hhfft::StepDataType::data_out;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.forward = true;
        HHFFT_2D_Real_D_set_complex_to_complex_packed_first_column_function(step, instruction_set);
        forward_steps.push_back(step);
    }

    ///////// IFFT column-wise /////////////

    // Reordering column-wise and first FFT-step combined. Also, the extra column from the end is removed
    {
        hhfft::StepInfoD step;
        step.data_type_in = hhfft::StepDataType::data_in;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.reorder_table = reorder_table_columns.data();
        step.reorder_table_size = reorder_table_columns.size();
        step.stride = m_complex;
        step.size = n;
        set_radix_raders(N_columns[0], step, instruction_set);
        step.radix = N_columns[0];
        step.repeats = n / step.radix;
        step.norm_factor = 1.0 / (m_complex*n);
        step.forward = false;
        HHFFT_2D_Real_D_set_function(step, instruction_set);
        inverse_steps.push_back(step);
    }

    // then put rest fft steps combined with twiddle factors
    for (size_t i = 1; i < N_columns.size(); i++)
    {
        hhfft::StepInfoD step;
        hhfft::StepInfoD &step_prev = inverse_steps.back();
        set_radix_raders(N_columns[i], step, instruction_set);
        step.radix = N_columns[i];
        step.stride = step_prev.stride * step_prev.radix;
        if (i == 1)
            step.stride = step.stride / m_complex;
        step.repeats = step_prev.repeats / step.radix;
        step.size = m_complex;
        step.data_type_in = hhfft::StepDataType::data_out;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.twiddle_factors = twiddle_factors_columns[i].data();
        step.forward = true;
        HHFFT_2D_Complex_set_function_columns<double>(step, instruction_set);
        inverse_steps.push_back(step);
    }

    // Convert from complex to complex complex packed.
    {
        hhfft::StepInfo<double> step;
        step.repeats = n;
        step.size = m;
        step.twiddle_factors = twiddle_factors_rows[0].data();
        step.data_type_in = hhfft::StepDataType::data_out;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.forward = false;
        HHFFT_2D_Real_D_set_complex_to_complex_packed_function(step, instruction_set);
        inverse_steps.push_back(step);
    }

    ///////// IFFT row-wise ////////////

    // Reorder rows in-place
    // NOTE if N_rows.size() == 1, it would be enough just to swap the ordering from 0 1 2 3 ... to 0 ... 3 2 1
    if (reorder_table_in_place_rows.size() > 0)
    {
        hhfft::StepInfoD step;
        step.data_type_in = hhfft::StepDataType::data_out;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.reorder_table_inplace = reorder_table_in_place_rows.data();
        step.reorder_table_inplace_size = reorder_table_in_place_rows.size();
        step.stride = n;
        step.size = m_complex;
        step.forward = false;
        HHFFT_2D_Real_D_set_function(step, instruction_set);
        inverse_steps.push_back(step);
    }

    // First FFT step if needed
    // NOTE 1D fft is used as no twiddle factors are involved
    if (m_complex > 1)
    {
        hhfft::StepInfoD step;
        step.data_type_in = hhfft::StepDataType::data_out;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.stride = 1;
        set_radix_raders(N_rows[0], step, instruction_set);
        step.radix = N_rows[0];
        step.repeats = n*m_complex / step.radix;
        step.forward = true;
        HHFFT_1D_Complex_set_function<double>(step, instruction_set);
        inverse_steps.push_back(step);
    }

    // then put rest fft steps combined with twiddle factor
    for (size_t i = 1; i < N_rows.size(); i++)
    {
        hhfft::StepInfoD step;
        hhfft::StepInfoD &step_prev = inverse_steps.back();
        set_radix_raders(N_rows[i], step, instruction_set);
        step.radix = N_rows[i];
        step.stride = step_prev.stride * step_prev.radix;
        step.repeats = step_prev.repeats / step.radix;
        step.data_type_in = hhfft::StepDataType::data_out;
        step.data_type_out = hhfft::StepDataType::data_out;
        step.twiddle_factors = twiddle_factors_rows[i].data();
        step.forward = true;
        HHFFT_1D_Complex_set_function<double>(step, instruction_set);
        inverse_steps.push_back(step);
    }
}

void HHFFT_2D_REAL_D::fft(const double *in, double *out) const
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
        temp_data_in.resize(n*m);
        std::copy(in, in + n*m, temp_data_in.data());
        in = temp_data_in.data();
    }

    // Allocate some extra space if needed    
    hhfft::AlignedVector<double> temp_data(temp_data_size);

    // Put all possible input/output data sources here
    const double *data_in[3] = {in, out, temp_data.data()};
    double *data_out[3] = {nullptr, out, temp_data.data()};

    for (auto &step: forward_steps)
    {
        // Run all the steps
        step.step_function(data_in[step.data_type_in] + step.start_index_in, data_out[step.data_type_out] + step.start_index_out, step);

        // TESTING print        
        //print_complex_matrix(data_out[step.data_type_out], n, m/2);
        //print_complex_matrix(data_out[step.data_type_out], n, m/2 + 1);
    }    

    // On odd FFTs the first real value needs to be moved one position if there is only one row
    // Note that for m = 1 same thing is done in a special function
    if (n == 1 && ((m & 1) == 1))
    {
        out[0] = out[1];
        out[1] = 0;
    }
}

void HHFFT_2D_REAL_D::ifft(const double *in, double *out) const
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
        size_t nn = 2*n*((m/2)+1);
        temp_data_in.resize(nn);
        std::copy(in, in + nn, temp_data_in.data());
        in = temp_data_in.data();        
    }

    // Allocate some extra space if needed    
    hhfft::AlignedVector<double> temp_data(temp_data_size);

    // Put all possible input/output data sources here
    const double *data_in[3] = {in, out, temp_data.data()};
    double *data_out[3] = {nullptr, out, temp_data.data()};

    for (auto &step: inverse_steps)
    {
        // Run all the steps
        step.step_function(data_in[step.data_type_in] + step.start_index_in, data_out[step.data_type_out] + step.start_index_out, step);

        // TESTING print        
        //print_complex_matrix(data_out[step.data_type_out], n, m/2);
        //print_real_matrix(data_out[step.data_type_out], n, m);
    }
}


// Calculates convolution in Fourier space
void HHFFT_2D_REAL_D::convolution(const double *in1, const double *in2, double *out) const
{
    convolution_function(in1, in2, out, n*(m/2+1));
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
