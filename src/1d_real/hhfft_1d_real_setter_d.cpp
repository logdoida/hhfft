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

using namespace hhfft;

////////////////////////////////////////// Even sizes /////////////////////////////////////////////7////

// Actual implementations are in different .cpp-files
template<bool forward> void fft_1d_complex_to_complex_packed_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template<bool forward> void fft_1d_complex_to_complex_packed_sse2_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template<bool forward> void fft_1d_complex_to_complex_packed_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template<bool forward> void fft_1d_complex_to_complex_packed_avx512_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

// Combines complex-packed-to-complex and reordering
void fft_1d_complex_to_complex_packed_ifft_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
void fft_1d_complex_to_complex_packed_ifft_sse2_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
void fft_1d_complex_to_complex_packed_ifft_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

// Small single level FFT
template<size_t n, bool forward> void fft_1d_real_1level_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template<size_t n, bool forward> void fft_1d_real_1level_sse2_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

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

////////////////////////////////////////// Odd sizes /////////////////////////////////////////////7////

template<size_t radix> void fft_1d_real_first_level_forward_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template<size_t radix> void fft_1d_real_first_level_forward_sse2_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template<size_t radix> void fft_1d_real_first_level_forward_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template<size_t radix> void fft_1d_real_first_level_inverse_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template<size_t radix> void fft_1d_real_first_level_inverse_sse2_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template<size_t radix> void fft_1d_real_first_level_inverse_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template<size_t radix> void fft_1d_real_one_level_forward_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template<size_t radix> void fft_1d_real_one_level_forward_sse2_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template<size_t radix> void fft_1d_real_one_level_forward_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template<size_t radix> void fft_1d_real_one_level_inverse_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template<size_t radix> void fft_1d_real_one_level_inverse_sse2_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);
template<size_t radix> void fft_1d_real_one_level_inverse_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template<size_t radix> void set_instruction_odd_first_level_d(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{

#ifdef HHFFT_COMPILED_WITH_AVX512F
    if (instruction_set == hhfft::InstructionSet::avx512f)
    {

    }
#endif

#ifdef HHFFT_COMPILED_WITH_AVX
    if (instruction_set == hhfft::InstructionSet::avx)
    {
        /*
        if(step_info.forward)
            step_info.step_function = fft_1d_real_first_level_forward_avx_d<radix>;
        else
            step_info.step_function = fft_1d_real_first_level_inverse_avx_d<radix>;
            */
    }
#endif

    if (instruction_set == hhfft::InstructionSet::sse2)
    {
        /*
        if(step_info.forward)
            step_info.step_function = fft_1d_real_first_level_forward_sse2_d<radix>;
        else
            step_info.step_function = fft_1d_real_first_level_inverse_sse2_d<radix>;
            */
    }

    if (instruction_set == hhfft::InstructionSet::none)
    {
        if(step_info.forward)
            step_info.step_function = fft_1d_real_first_level_forward_plain_d<radix>;
        else
            step_info.step_function = fft_1d_real_first_level_inverse_plain_d<radix>;
    }
}

void set_instruction_odd_first_level_radix_d(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{
    if(step_info.radix == 3)
        set_instruction_odd_first_level_d<3>(step_info, instruction_set);
    else if(step_info.radix == 5)
        set_instruction_odd_first_level_d<5>(step_info, instruction_set);
    else if(step_info.radix == 7)
        set_instruction_odd_first_level_d<7>(step_info, instruction_set);
}

template<size_t radix> void set_instruction_odd_other_level_d(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{

#ifdef HHFFT_COMPILED_WITH_AVX512F
    if (instruction_set == hhfft::InstructionSet::avx512f)
    {

    }
#endif

#ifdef HHFFT_COMPILED_WITH_AVX
    if (instruction_set == hhfft::InstructionSet::avx)
    {        
        //step_info.step_function = fft_1d_real_one_level_forward_avx_d<radix>;
    }
#endif

    if (instruction_set == hhfft::InstructionSet::sse2)
    {        
        //step_info.step_function = fft_1d_real_one_level_forward_sse2_d<radix>;
    }

    if (instruction_set == hhfft::InstructionSet::none)
    {
        if(step_info.forward)
            step_info.step_function = fft_1d_real_one_level_forward_plain_d<radix>;
        else
            step_info.step_function = fft_1d_real_one_level_inverse_plain_d<radix>;
    }
}

void set_instruction_odd_other_level_radix_d(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{
    if(step_info.radix == 3)
        set_instruction_odd_other_level_d<3>(step_info, instruction_set);
    else if(step_info.radix == 5)
        set_instruction_odd_other_level_d<5>(step_info, instruction_set);
    else if(step_info.radix == 7)
        set_instruction_odd_other_level_d<7>(step_info, instruction_set);
}

void hhfft::HHFFT_1D_Real_D_odd_set_function(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{
    // First level of odd FFT/IFFT
    if (step_info.stride == 1 && step_info.reorder_table != nullptr)
    {
        set_instruction_odd_first_level_radix_d(step_info, instruction_set);
    }

    // Other levels of odd FFT/IFFT
    if (step_info.stride != 1 && step_info.reorder_table == nullptr)
    {
        set_instruction_odd_other_level_radix_d(step_info, instruction_set);
    }

    if (step_info.step_function == nullptr)
    {
        throw(std::runtime_error("HHFFT error: Unable to set a function!"));
    }
}


//////////////////////////////// Small one level functions ////////////////////////////////////////////

// n = 1!
void fft_1d_real_n1_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    data_out[0] = data_in[0];
    data_out[1] = 0;
}


template<size_t n> void set_small_function_instruction_set_d(StepInfoD &step_info, hhfft::InstructionSet instruction_set, bool forward)
{
#ifdef HHFFT_COMPILED_WITH_AVX
    if (instruction_set == hhfft::InstructionSet::avx)
    {        
        if(forward)
            step_info.step_function = fft_1d_real_1level_avx_d<n,true>;
        else
            step_info.step_function = fft_1d_real_1level_avx_d<n,false>;
        return;        
    }
#endif

    if (instruction_set == hhfft::InstructionSet::sse2)
    {
        if(forward)
            step_info.step_function =  fft_1d_real_1level_sse2_d<n,true>;
        else
            step_info.step_function =  fft_1d_real_1level_sse2_d<n,false>;
        return;
    }

    // This is needed in all architectures
    if(n == 1)
    {
        step_info.step_function = fft_1d_real_n1_d;
    }
}

void hhfft::HHFFT_1D_Real_D_set_small_function(StepInfoD &step_info, size_t n, bool forward, hhfft::InstructionSet instruction_set)
{
    step_info.step_function = nullptr;

    if(n == 2)
    {
        set_small_function_instruction_set_d<2>(step_info, instruction_set, forward);
    } else if(n == 4)
    {
        set_small_function_instruction_set_d<4>(step_info, instruction_set, forward);
    } else if(n == 6)
    {
        set_small_function_instruction_set_d<6>(step_info, instruction_set, forward);
    } else if(n == 8)
    {
        set_small_function_instruction_set_d<8>(step_info, instruction_set, forward);
    } else if(n == 10)
    {
        set_small_function_instruction_set_d<10>(step_info, instruction_set, forward);
    } else if(n == 12)
    {
        set_small_function_instruction_set_d<12>(step_info, instruction_set, forward);
    } else if(n == 14)
    {
        set_small_function_instruction_set_d<14>(step_info, instruction_set, forward);
    } else if(n == 16)
    {
        set_small_function_instruction_set_d<16>(step_info, instruction_set, forward);
    }

    return;
}
