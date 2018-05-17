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

#include "hhfft_1d_real_setter_d.h"
#include <stdexcept>
#include <assert.h>
#include <cmath>

#include <iostream> //TESTING

using namespace hhfft;

// Actual implementations are in different .cpp-files
template<bool forward> void fft_1d_complex_to_complex_packed_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template<bool forward> void fft_1d_complex_to_complex_packed_sse2_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template<bool forward> void fft_1d_complex_to_complex_packed_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template<bool forward> void fft_1d_complex_to_complex_packed_avx512_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);


// Combines complex-packed-to-complex and reordering
void fft_1d_complex_to_complex_packed_ifft_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
void fft_1d_complex_to_complex_packed_ifft_sse2_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
void fft_1d_complex_to_complex_packed_ifft_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);


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
        step_info.step_function = fft_1d_complex_to_complex_packed_avx_d<forward>;
    }
#endif

    if (instruction_set == hhfft::InstructionSet::sse2)
    {
        step_info.step_function = fft_1d_complex_to_complex_packed_sse2_d<forward>;
    }

    if (instruction_set == hhfft::InstructionSet::none)
    {       
        step_info.step_function = fft_1d_complex_to_complex_packed_plain_d<forward>;
    }      
}

void set_instruction2_d(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{

#ifdef HHFFT_COMPILED_WITH_AVX512F
    if (instruction_set == hhfft::InstructionSet::avx512f)
    {

    }
#endif

#ifdef HHFFT_COMPILED_WITH_AVX
    if (instruction_set == hhfft::InstructionSet::avx)
    {
        step_info.step_function = fft_1d_complex_to_complex_packed_ifft_avx_d;
    }
#endif

    if (instruction_set == hhfft::InstructionSet::sse2)
    {
        step_info.step_function = fft_1d_complex_to_complex_packed_ifft_sse2_d;
    }

    if (instruction_set == hhfft::InstructionSet::none)
    {
        step_info.step_function = fft_1d_complex_to_complex_packed_ifft_plain_d;
    }
}

// This set pointer to correct functions
void hhfft::HHFFT_1D_Real_D_set_complex_to_complex_packed_function(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{
    step_info.step_function = nullptr;

    if (step_info.reorder_table == nullptr)
    {
        if (step_info.forward)
           set_instruction_set_d<true>(step_info, instruction_set);
        else
           set_instruction_set_d<false>(step_info, instruction_set);
    } else
    {
        set_instruction2_d(step_info, instruction_set);
    }

    if (step_info.step_function == nullptr)
    {
        throw(std::runtime_error("HHFFT error: Unable to set a function!"));
    }
}

