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

#include "hhfft_2d_real_setter_d.h"
#include "../aligned_arrays.h"
#include <stdexcept>
#include <assert.h>
#include <cmath>

#include <iostream> //TESTING

using namespace hhfft;

// Shuffles one row into two rows
void fft_2d_real_reorder_columns_inverse_inplace_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    // It is assumed that this is always in-place as this is the last step
    assert (data_in == data_out);

    size_t n = step_info.stride;
    size_t m = step_info.size;

    // Only shuffle is needed
    hhfft::AlignedVector<double> temp(m);
    double* temp_data = temp.data();
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < m; j++)
        {
            data_out[2*i*m + j] = data_in[2*i*m + 2*j + 0];
            temp_data[j]        = data_in[2*i*m + 2*j + 1];
        }

        for (size_t j = 0; j < m; j++)
        {
            data_out[2*i*m + m + j] = temp_data[j];
        }
    }
}


void hhfft::HHFFT_2D_Real_D_set_reorder_column_function(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{
    if (!step_info.forward)
    {        
        step_info.step_function = fft_2d_real_reorder_columns_inverse_inplace_d;
    }

    if (step_info.step_function == nullptr)
    {
        throw(std::runtime_error("HHFFT error: Unable to set a function!"));
    }
}


// Actual implementations are in different .cpp-files
template<bool forward> void fft_2d_complex_to_complex_packed_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template<bool forward> void fft_2d_complex_to_complex_packed_sse2_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template<bool forward> void fft_2d_complex_to_complex_packed_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template<bool forward> void fft_2d_complex_to_complex_packed_avx512_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);


template<bool forward> void set_instruction_set_d(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{

#ifdef HHFFT_COMPILED_WITH_AVX512F
    if (instruction_set == hhfft::InstructionSet::avx512f)
    {
        
    }
#endif

#ifdef HHFFT_COMPILED_WITH_AVX
    if (instruction_set == hhfft::InstructionSet::avx)
    {
       step_info.step_function = fft_2d_complex_to_complex_packed_avx_d<forward>;
    }
#endif

    if (instruction_set == hhfft::InstructionSet::sse2)
    {
        step_info.step_function = fft_2d_complex_to_complex_packed_sse2_d<forward>;
    }

    if (instruction_set == hhfft::InstructionSet::none)
    {
        step_info.step_function = fft_2d_complex_to_complex_packed_plain_d<forward>;
    }      
}

// This set pointer to correct functions
void hhfft::HHFFT_2D_Real_D_set_complex_to_complex_packed_function(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{
    step_info.step_function = nullptr;

    if (step_info.forward)
       set_instruction_set_d<true>(step_info, instruction_set);
    else
       set_instruction_set_d<false>(step_info, instruction_set);

    if (step_info.step_function == nullptr)
    {
        throw(std::runtime_error("HHFFT error: Unable to set a function!"));
    }
}



void fft_2d_complex_to_complex_packed_first_column_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
void fft_2d_complex_to_complex_packed_first_column_sse2_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
void fft_2d_complex_to_complex_packed_first_column_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

void set_instruction_set_first_column_d(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{

#ifdef HHFFT_COMPILED_WITH_AVX512F
    if (instruction_set == hhfft::InstructionSet::avx512f)
    {

    }
#endif

#ifdef HHFFT_COMPILED_WITH_AVX
    if (instruction_set == hhfft::InstructionSet::avx)
    {
       step_info.step_function = fft_2d_complex_to_complex_packed_first_column_avx_d;
    }
#endif

    if (instruction_set == hhfft::InstructionSet::sse2)
    {
        step_info.step_function = fft_2d_complex_to_complex_packed_first_column_sse2_d;
    }

    if (instruction_set == hhfft::InstructionSet::none)
    {
        step_info.step_function = fft_2d_complex_to_complex_packed_first_column_plain_d;
    }
}

// This set pointer to correct functions
void hhfft::HHFFT_2D_Real_D_set_complex_to_complex_packed_first_column_function(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{
    step_info.step_function = nullptr;

    if (step_info.forward)
       set_instruction_set_first_column_d(step_info, instruction_set);

    if (step_info.step_function == nullptr)
    {
        throw(std::runtime_error("HHFFT error: Unable to set a function!"));
    }
}



// Reorder and do FFT
template<size_t radix> void fft_2d_real_reorder2_inverse_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template<size_t radix> void fft_2d_real_reorder2_inverse_sse2_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template<size_t radix> void fft_2d_real_reorder2_inverse_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);


template<size_t radix> void set_instruction_set_2d_real_d(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{

#ifdef HHFFT_COMPILED_WITH_AVX512F
    if (instruction_set == hhfft::InstructionSet::avx512f)
    {

    }
#endif

#ifdef HHFFT_COMPILED_WITH_AVX
    if (instruction_set == hhfft::InstructionSet::avx)
    {        
        step_info.step_function = fft_2d_real_reorder2_inverse_avx_d<radix>;
    }
#endif

    if (instruction_set == hhfft::InstructionSet::sse2)
    {        
        step_info.step_function = fft_2d_real_reorder2_inverse_sse2_d<radix>;
    }

    if (instruction_set == hhfft::InstructionSet::none)
    {
        step_info.step_function = fft_2d_real_reorder2_inverse_plain_d<radix>;
    }
}

void set_radix_2d_real_d(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{
    size_t radix = step_info.radix;

    if (radix == 2)
    {
        set_instruction_set_2d_real_d<2>(step_info, instruction_set);
    } if (radix == 3)
    {
        set_instruction_set_2d_real_d<3>(step_info, instruction_set);
    } if (radix == 4)
    {
        set_instruction_set_2d_real_d<4>(step_info, instruction_set);
    } if (radix == 5)
    {
        set_instruction_set_2d_real_d<5>(step_info, instruction_set);
    } if (radix == 6)
    {
        set_instruction_set_2d_real_d<6>(step_info, instruction_set);
    } if (radix == 7)
    {
        set_instruction_set_2d_real_d<7>(step_info, instruction_set);
    } if (radix == 8)
    {
       set_instruction_set_2d_real_d<8>(step_info, instruction_set);
    }
}

void fft_2d_real_reorder_rows_in_place_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
void fft_2d_real_reorder_rows_in_place_sse2_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
void fft_2d_real_reorder_rows_in_place_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

void set_instruction_set_2d_reorder_rows_d(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{

#ifdef HHFFT_COMPILED_WITH_AVX512F
    if (instruction_set == hhfft::InstructionSet::avx512f)
    {

    }
#endif

#ifdef HHFFT_COMPILED_WITH_AVX
    if (instruction_set == hhfft::InstructionSet::avx)
    {
        step_info.step_function = fft_2d_real_reorder_rows_in_place_avx_d;
    }
#endif

    if (instruction_set == hhfft::InstructionSet::sse2)
    {
        step_info.step_function = fft_2d_real_reorder_rows_in_place_sse2_d;
    }

    if (instruction_set == hhfft::InstructionSet::none)
    {
        step_info.step_function = fft_2d_real_reorder_rows_in_place_plain_d;
    }
}


// Used for setting the first step where reordering and first fft are combined
void hhfft::HHFFT_2D_Real_D_set_function(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{
    step_info.step_function = nullptr;

    if (step_info.radix == 1 && step_info.reorder_table_inplace != nullptr)
    {
        set_instruction_set_2d_reorder_rows_d(step_info, instruction_set);
    }

    if (step_info.radix != 1 && !step_info.forward)
    {
        set_radix_2d_real_d(step_info, instruction_set);
    }

    if (step_info.step_function == nullptr)
    {
        throw(std::runtime_error("HHFFT error: Unable to set a function!"));
    }
}
