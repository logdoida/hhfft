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

#include "hhfft_1d_real_setter.h"
#include <stdexcept>
#include <assert.h>
#include <cmath>

using namespace hhfft;

////////////////////////////////////////// Even sizes /////////////////////////////////////////////7////

// Actual implementations are in different .cpp-files
template<bool forward> void fft_1d_complex_to_complex_packed_plain_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template<bool forward> void fft_1d_complex_to_complex_packed_sse2_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template<bool forward> void fft_1d_complex_to_complex_packed_avx_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template<bool forward> void fft_1d_complex_to_complex_packed_avx512_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

// Combines complex-packed-to-complex and reordering
void fft_1d_complex_to_complex_packed_ifft_plain_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
void fft_1d_complex_to_complex_packed_ifft_sse2_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
void fft_1d_complex_to_complex_packed_ifft_avx_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template<bool forward> void set_instruction_set_f(StepInfoF &step_info, hhfft::InstructionSet instruction_set)
{
/*
#ifdef HHFFT_COMPILED_WITH_AVX512F
    if (instruction_set == hhfft::InstructionSet::avx512f)
    {
        
    }
#endif
*/

#ifdef HHFFT_COMPILED_WITH_AVX
    if (instruction_set == hhfft::InstructionSet::avx)
    {
        step_info.step_function = fft_1d_complex_to_complex_packed_avx_f<forward>;
    }
#endif

    if (instruction_set == hhfft::InstructionSet::sse2)
    {
        step_info.step_function = fft_1d_complex_to_complex_packed_sse2_f<forward>;
    }

    if (instruction_set == hhfft::InstructionSet::none)
    {       
        step_info.step_function = fft_1d_complex_to_complex_packed_plain_f<forward>;
    }      
}

void set_instruction2_f(StepInfoF &step_info, hhfft::InstructionSet instruction_set)
{
/*
#ifdef HHFFT_COMPILED_WITH_AVX512F
    if (instruction_set == hhfft::InstructionSet::avx512f)
    {

    }
#endif
*/

#ifdef HHFFT_COMPILED_WITH_AVX
    if (instruction_set == hhfft::InstructionSet::avx)
    {
        step_info.step_function = fft_1d_complex_to_complex_packed_ifft_avx_f;
    }
#endif

    if (instruction_set == hhfft::InstructionSet::sse2)
    {
        step_info.step_function = fft_1d_complex_to_complex_packed_ifft_sse2_f;
    }

    if (instruction_set == hhfft::InstructionSet::none)
    {
        step_info.step_function = fft_1d_complex_to_complex_packed_ifft_plain_f;
    }
}

namespace hhfft
{
    // This set pointer to correct functions
    template<> void HHFFT_1D_Real_set_complex_to_complex_packed_function<float>(StepInfoF &step_info, hhfft::InstructionSet instruction_set)
    {
        step_info.step_function = nullptr;

        if (step_info.reorder_table == nullptr)
        {
            if (step_info.forward)
               set_instruction_set_f<true>(step_info, instruction_set);
            else
               set_instruction_set_f<false>(step_info, instruction_set);
        } else
        {
            set_instruction2_f(step_info, instruction_set);
        }

        if (step_info.step_function == nullptr)
        {
            throw(std::runtime_error("HHFFT error: Unable to set a function!"));
        }
    }
}

////////////////////////////////////////// Odd sizes /////////////////////////////////////////////7////

template<RadixType radix_type> void fft_1d_real_first_level_forward_avx_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template<RadixType radix_type> void fft_1d_real_first_level_forward_sse2_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template<RadixType radix_type> void fft_1d_real_first_level_forward_plain_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template<RadixType radix_type> void fft_1d_real_first_level_inverse_avx_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template<RadixType radix_type> void fft_1d_real_first_level_inverse_sse2_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template<RadixType radix_type> void fft_1d_real_first_level_inverse_plain_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template<RadixType radix_type> void fft_1d_real_one_level_forward_avx_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template<RadixType radix_type> void fft_1d_real_one_level_forward_sse2_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template<RadixType radix_type> void fft_1d_real_one_level_forward_plain_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template<RadixType radix_type> void fft_1d_real_one_level_inverse_avx_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template<RadixType radix_type> void fft_1d_real_one_level_inverse_sse2_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template<RadixType radix_type> void fft_1d_real_one_level_inverse_plain_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template<RadixType radix_type> void set_instruction_odd_first_level_f(StepInfoF &step_info, hhfft::InstructionSet instruction_set)
{
/*
#ifdef HHFFT_COMPILED_WITH_AVX512F
    if (instruction_set == hhfft::InstructionSet::avx512f)
    {

    }
#endif
*/

#ifdef HHFFT_COMPILED_WITH_AVX
    if (instruction_set == hhfft::InstructionSet::avx)
    {       
        if(step_info.forward)
            step_info.step_function = fft_1d_real_first_level_forward_avx_f<radix_type>;
        else
            step_info.step_function = fft_1d_real_first_level_inverse_avx_f<radix_type>;
    }
#endif

    if (instruction_set == hhfft::InstructionSet::sse2)
    {        
        if(step_info.forward)
            step_info.step_function = fft_1d_real_first_level_forward_sse2_f<radix_type>;
        else
            step_info.step_function = fft_1d_real_first_level_inverse_sse2_f<radix_type>;
    }

    if (instruction_set == hhfft::InstructionSet::none)
    {
        if(step_info.forward)
            step_info.step_function = fft_1d_real_first_level_forward_plain_f<radix_type>;
        else
            step_info.step_function = fft_1d_real_first_level_inverse_plain_f<radix_type>;
    }
}

void set_instruction_odd_first_level_radix_f(StepInfoF &step_info, hhfft::InstructionSet instruction_set)
{
    if(step_info.radix == 3)
        set_instruction_odd_first_level_f<Radix3>(step_info, instruction_set);
    else if(step_info.radix == 5)
        set_instruction_odd_first_level_f<Radix5>(step_info, instruction_set);
    else if(step_info.radix == 7)
        set_instruction_odd_first_level_f<Radix7>(step_info, instruction_set);
    else if(step_info.radix > 8)
        set_instruction_odd_first_level_f<Raders>(step_info, instruction_set);
}

template<RadixType radix_type> void set_instruction_odd_other_level_f(StepInfoF &step_info, hhfft::InstructionSet instruction_set)
{
/*
#ifdef HHFFT_COMPILED_WITH_AVX512F
    if (instruction_set == hhfft::InstructionSet::avx512f)
    {

    }
#endif
*/

#ifdef HHFFT_COMPILED_WITH_AVX
    if (instruction_set == hhfft::InstructionSet::avx)
    {        
        if(step_info.forward)
            step_info.step_function = fft_1d_real_one_level_forward_avx_f<radix_type>;
        else
            step_info.step_function = fft_1d_real_one_level_inverse_avx_f<radix_type>;
    }
#endif

    if (instruction_set == hhfft::InstructionSet::sse2)
    {
        if(step_info.forward)
            step_info.step_function = fft_1d_real_one_level_forward_sse2_f<radix_type>;
        else
            step_info.step_function = fft_1d_real_one_level_inverse_sse2_f<radix_type>;
    }

    if (instruction_set == hhfft::InstructionSet::none)
    {
        if(step_info.forward)
            step_info.step_function = fft_1d_real_one_level_forward_plain_f<radix_type>;
        else
            step_info.step_function = fft_1d_real_one_level_inverse_plain_f<radix_type>;
    }
}

void set_instruction_odd_other_level_radix_f(StepInfoF &step_info, hhfft::InstructionSet instruction_set)
{
    if(step_info.radix == 3)
        set_instruction_odd_other_level_f<Radix3>(step_info, instruction_set);
    else if(step_info.radix == 5)
        set_instruction_odd_other_level_f<Radix5>(step_info, instruction_set);
    else if(step_info.radix == 7)
        set_instruction_odd_other_level_f<Radix7>(step_info, instruction_set);
    else if(step_info.radix > 8)
        set_instruction_odd_other_level_f<Raders>(step_info, instruction_set);
}

namespace hhfft
{
    template<> void HHFFT_1D_Real_odd_set_function<float>(StepInfoF &step_info, hhfft::InstructionSet instruction_set)
    {
        // First level of odd FFT/IFFT
        if (step_info.stride == 1 && step_info.reorder_table != nullptr)
        {
            set_instruction_odd_first_level_radix_f(step_info, instruction_set);
        }

        // Other levels of odd FFT/IFFT
        if (step_info.stride != 1 && step_info.reorder_table == nullptr)
        {
            set_instruction_odd_other_level_radix_f(step_info, instruction_set);
        }

        if (step_info.step_function == nullptr)
        {
            throw(std::runtime_error("HHFFT error: Unable to set a function!"));
        }
    }
}


//////////////////////////////// Small one level functions ////////////////////////////////////////////

// Small single level FFT
template<size_t n, bool forward> void fft_1d_real_1level_avx_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template<size_t n, bool forward> void fft_1d_real_1level_sse2_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template<size_t n, bool forward> void fft_1d_real_1level_plain_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template<bool forward> void fft_1d_real_1level_raders_plain_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template<bool forward> void fft_1d_real_1level_raders_sse2_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template<bool forward> void fft_1d_real_1level_raders_avx_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template<size_t n> void set_small_function_instruction_set_real_f(StepInfoF &step_info, hhfft::InstructionSet instruction_set, bool forward)
{
#ifdef HHFFT_COMPILED_WITH_AVX
    if (instruction_set == hhfft::InstructionSet::avx)
    {
        if(forward)
            step_info.step_function = fft_1d_real_1level_avx_f<n,true>;
        else
            step_info.step_function = fft_1d_real_1level_avx_f<n,false>;
        return;        
    }
#endif

    if (instruction_set == hhfft::InstructionSet::sse2)
    {
        if(forward)
            step_info.step_function = fft_1d_real_1level_sse2_f<n,true>;
        else
            step_info.step_function = fft_1d_real_1level_sse2_f<n,false>;
        return;
    }

    if (instruction_set == hhfft::InstructionSet::none)
    {
        if(forward)
            step_info.step_function = fft_1d_real_1level_plain_f<n,true>;
        else
            step_info.step_function = fft_1d_real_1level_plain_f<n,false>;
        return;
    }
}


namespace hhfft
{

    template<> void HHFFT_1D_Real_set_small_function<float>(StepInfoF &step_info, size_t n, bool forward, hhfft::InstructionSet instruction_set)
    {
        step_info.step_function = nullptr;

        if(n == 1)
        {
            set_small_function_instruction_set_real_f<1>(step_info, instruction_set, forward);
        }
        else if(n == 2)
        {
            set_small_function_instruction_set_real_f<2>(step_info, instruction_set, forward);
        } else if(n == 3)
        {
            set_small_function_instruction_set_real_f<3>(step_info, instruction_set, forward);
        } else if(n == 4)
        {
            set_small_function_instruction_set_real_f<4>(step_info, instruction_set, forward);
        } else if(n == 5)
        {
            set_small_function_instruction_set_real_f<5>(step_info, instruction_set, forward);
        } else if(n == 6)
        {
            set_small_function_instruction_set_real_f<6>(step_info, instruction_set, forward);
        } else if(n == 7)
        {
            set_small_function_instruction_set_real_f<7>(step_info, instruction_set, forward);
        } else if(n == 8)
        {
            set_small_function_instruction_set_real_f<8>(step_info, instruction_set, forward);
        } else if(n == 10)
        {
            set_small_function_instruction_set_real_f<10>(step_info, instruction_set, forward);
        } else if(n == 12)
        {
            set_small_function_instruction_set_real_f<12>(step_info, instruction_set, forward);
        } else if(n == 14)
        {
            set_small_function_instruction_set_real_f<14>(step_info, instruction_set, forward);
        } else if(n == 16)
        {
            set_small_function_instruction_set_real_f<16>(step_info, instruction_set, forward);
        }

        return;
    }


    template<> void HHFFT_1D_Real_set_1level_raders_function<float>(StepInfoF &step_info, bool forward, hhfft::InstructionSet instruction_set)
    {
    #ifdef HHFFT_COMPILED_WITH_AVX
        if (instruction_set == hhfft::InstructionSet::avx)
        {
            if(forward)
                step_info.step_function = fft_1d_real_1level_raders_avx_f<true>;
            else
                step_info.step_function = fft_1d_real_1level_raders_avx_f<false>;
            return;
        }
    #endif
        if (instruction_set == hhfft::InstructionSet::sse2)
        {
            if(forward)
                step_info.step_function = fft_1d_real_1level_raders_sse2_f<true>;
            else
                step_info.step_function = fft_1d_real_1level_raders_sse2_f<false>;
            return;
        }

        if (instruction_set == hhfft::InstructionSet::none)
        {
            if(forward)
                step_info.step_function = fft_1d_real_1level_raders_plain_f<true>;
            else
                step_info.step_function = fft_1d_real_1level_raders_plain_f<false>;
            return;
        }
    }
}
