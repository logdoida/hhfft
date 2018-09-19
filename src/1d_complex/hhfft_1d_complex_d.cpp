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

#include "hhfft_1d_complex_d.h"
#include <stdexcept>
#include <assert.h>
#include <cmath>


using namespace hhfft;

// Actual implementations are in different .cpp-files
// No twiddle factors
template<RadixType radix_type>
    void fft_1d_complex_plain_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template<RadixType radix_type, SizeType stride_type>
    void fft_1d_complex_sse2_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template<RadixType radix_type, SizeType stride_type>
    void fft_1d_complex_avx_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

//template<size_t radix, SizeType stride_type>
//    void fft_1d_complex_avx512_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

// DIT
template<RadixType radix_type>
    void fft_1d_complex_twiddle_dit_plain_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template<RadixType radix_type, SizeType stride_type>
    void fft_1d_complex_twiddle_dit_sse2_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template<RadixType radix_type, SizeType stride_type>
    void fft_1d_complex_twiddle_dit_avx_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

//template<size_t radix, SizeType stride_type>
//    void fft_1d_complex_twiddle_dit_avx512_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

// Reorder in-place
void fft_1d_complex_reorder_in_place_plain_d(const double *, double *data_out,const hhfft::StepInfo<double> &step_info);
void fft_1d_complex_reorder_in_place_sse2_d(const double *, double *data_out,const hhfft::StepInfo<double> &step_info);
void fft_1d_complex_reorder_in_place_avx_d(const double *, double *data_out,const hhfft::StepInfo<double> &step_info);

// Reordering
template<bool scale>
    void fft_1d_complex_reorder_plain_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template<bool scale>
    void fft_1d_complex_reorder_sse2_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template<bool scale>
    void fft_1d_complex_reorder_avx_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

// Reorder and do FFT
template<RadixType radix_type, bool forward>
void fft_1d_complex_reorder2_plain_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template<RadixType radix_type, SizeType stride_type, bool forward>
void fft_1d_complex_reorder2_sse2_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template<RadixType radix_type, SizeType stride_type, bool forward>
void fft_1d_complex_reorder2_avx_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

// Convolution
void fft_1d_complex_convolution_plain_d(const double *in1, const double *in2, double *out, size_t n);
void fft_1d_complex_convolution_sse2_d(const double *in1, const double *in2, double *out, size_t n);
void fft_1d_complex_convolution_avx_d(const double *in1, const double *in2, double *out, size_t n);

// Small single level FFT
template<size_t n, bool forward> void fft_1d_complex_1level_avx_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template<size_t n, bool forward> void fft_1d_complex_1level_sse2_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template<size_t n, bool forward> void fft_1d_complex_1level_plain_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template<bool forward> void fft_1d_complex_1level_raders_plain_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template<bool forward> void fft_1d_complex_1level_raders_sse2_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);
template<bool forward> void fft_1d_complex_1level_raders_avx_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);


template<RadixType radix_type, SizeType stride_type, bool forward> void set_instruction_set_d(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{

#ifdef HHFFT_COMPILED_WITH_AVX512F
    if (instruction_set == hhfft::InstructionSet::avx512f)
    {
        if (step_info.twiddle_factors == nullptr)
        {
            // Check if reordering should be supported
            if(step_info.reorder_table == nullptr)
                step_info.step_function = fft_1d_complex_avx512_d<radix, stride_type>;
            else if(step_info.stride == 1)
            {                
                step_info.step_function = fft_1d_complex_reorder2_avx512_d<radix, stride_type, forward>;
            }
        } else
        {
            if (step_info.dif)
                step_info.step_function = fft_1d_complex_twiddle_dif_avx512_d<radix, stride_type, forward>;
            else
                step_info.step_function = fft_1d_complex_twiddle_dit_avx512_d<radix, stride_type>;
        }
    }
#endif

#ifdef HHFFT_COMPILED_WITH_AVX
    if (instruction_set == hhfft::InstructionSet::avx)
    {        
        if (step_info.twiddle_factors == nullptr)
        {
            // Check if reordering should be supported
            if(step_info.reorder_table == nullptr)
                step_info.step_function = fft_1d_complex_avx_d<radix_type, stride_type>;
            else if(step_info.stride == 1)
            {
                step_info.step_function = fft_1d_complex_reorder2_avx_d<radix_type, SizeType::Size1, forward>;
            }
        } else
        {
            step_info.step_function = fft_1d_complex_twiddle_dit_avx_d<radix_type, stride_type>;
        }        
    }
#endif

    if (instruction_set == hhfft::InstructionSet::sse2)
    {        
        if (step_info.twiddle_factors == nullptr)
        {
            // Check if reordering should be supported
            if(step_info.reorder_table == nullptr)
                step_info.step_function = fft_1d_complex_sse2_d<radix_type, stride_type>;
            else if(step_info.stride == 1)
            {                
                step_info.step_function = fft_1d_complex_reorder2_sse2_d<radix_type, SizeType::Size1, forward>;
            }
        } else
        {
            step_info.step_function = fft_1d_complex_twiddle_dit_sse2_d<radix_type, stride_type>;
        }        
    }  

    if (instruction_set == hhfft::InstructionSet::none)
    {
        // NOTE plain is more or less an unused reference implementation, as sse2 should always be supported
        if (step_info.twiddle_factors == nullptr)
        {
           // Check if reordering should be supported
           if(step_info.reorder_table == nullptr)
               step_info.step_function = fft_1d_complex_plain_d<radix_type>;
           else if(step_info.stride == 1)
           {
               step_info.step_function = fft_1d_complex_reorder2_plain_d<radix_type, forward>;
           }
        } else
        {
            step_info.step_function = fft_1d_complex_twiddle_dit_plain_d<radix_type>;
        }
    }  
}

// These functions set different template parameters one at time
template<RadixType radix_type, SizeType stride_type> void set_forward_d(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{
    if (step_info.forward)
    {
        set_instruction_set_d<radix_type, stride_type, true>(step_info, instruction_set);
    } else
    {
        set_instruction_set_d<radix_type, stride_type, false>(step_info, instruction_set);
    }
}

template<RadixType radix_type> void set_stride_type_d(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{
    // Knowing something about the stride at compile time can help to optimize some cases
    size_t stride = step_info.stride;

    // Only stride = 1 and N are currently used
    if (stride == 1)
    {
        set_forward_d<radix_type, SizeType::Size1>(step_info, instruction_set);
    } else
    {
        set_forward_d<radix_type, SizeType::SizeN>(step_info, instruction_set);
    }

    /*
    size_t stride = step_info.stride;

    // Knowing the stride at compile time can help the compiler to optimze
    if (stride == 1)
    {
        set_forward_d<radix, StrideType::Stride1>(step_info, instruction_set);
    } else if (stride == 2)
    {
        set_forward_d<radix, StrideType::Stride2>(step_info, instruction_set);
    } else if (stride == 4)
    {
        set_forward_d<radix, StrideType::Stride4>(step_info, instruction_set);
    } else if (stride%4 == 0)
    {
        set_forward_d<radix, StrideType::Stride4N>(step_info, instruction_set);
    } else if (stride%2 == 0)
    {
        set_forward_d<radix, StrideType::Stride2N>(step_info, instruction_set);
    } else
    {
        set_forward_d<radix, StrideType::StrideN>(step_info, instruction_set);
    }
    */
}

void set_radix_d(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{
    size_t radix = step_info.radix;

    if (radix > 8)  // Rader's algorithm (radix can be something else)
    {
        set_stride_type_d<Raders>(step_info, instruction_set);
    } else if (radix == 2)
    {
        set_stride_type_d<Radix2>(step_info, instruction_set);
    } else if (radix == 3)
    {
        set_stride_type_d<Radix3>(step_info, instruction_set);
    } else if (radix == 4)
    {
        set_stride_type_d<Radix4>(step_info, instruction_set);
    } else if (radix == 5)
    {
        set_stride_type_d<Radix5>(step_info, instruction_set);
    } else if (radix == 6)
    {
        set_stride_type_d<Radix6>(step_info, instruction_set);
    } else if (radix == 7)
    {
        set_stride_type_d<Radix7>(step_info, instruction_set);
    } else if (radix == 8)
    {
        set_stride_type_d<Radix8>(step_info, instruction_set);
    }
}


template<bool forward> void set_reorder_instruction_set_d(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{

#ifdef HHFFT_COMPILED_WITH_AVX512F
    if (instruction_set == hhfft::InstructionSet::avx512f)
    {
        if (step_info.reorder_table != nullptr)
            step_info.step_function = fft_1d_complex_reorder_avx512_d<forward>;
        // TODO in-place
    }
#endif

#ifdef HHFFT_COMPILED_WITH_AVX
    if (instruction_set == hhfft::InstructionSet::avx)
    {
        if (step_info.reorder_table != nullptr)
            step_info.step_function = fft_1d_complex_reorder_avx_d<forward>;
        if (step_info.reorder_table_inplace != nullptr)
            step_info.step_function = fft_1d_complex_reorder_in_place_avx_d;
    }
#endif

    if (instruction_set == hhfft::InstructionSet::sse2)
    {
        if (step_info.reorder_table != nullptr)
            step_info.step_function = fft_1d_complex_reorder_sse2_d<forward>;
        if (step_info.reorder_table_inplace != nullptr)
            step_info.step_function = fft_1d_complex_reorder_in_place_sse2_d;
    }

    if (instruction_set == hhfft::InstructionSet::none)
    {        
        if (step_info.reorder_table != nullptr)
            step_info.step_function = fft_1d_complex_reorder_plain_d<forward>;
        if (step_info.reorder_table_inplace != nullptr)
            step_info.step_function = fft_1d_complex_reorder_in_place_plain_d;
    }
}

void hhfft::HHFFT_1D_Complex_D_set_reorder_function(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{
    step_info.step_function = nullptr;

    if (step_info.radix == 1 && (step_info.reorder_table != nullptr || step_info.reorder_table_inplace != nullptr))
    {
        // Add reordering step
        if (step_info.forward)
            set_reorder_instruction_set_d<true>(step_info, instruction_set);
        else
            set_reorder_instruction_set_d<false>(step_info, instruction_set);
    }

    if (step_info.step_function == nullptr)
    {
        throw(std::runtime_error("HHFFT error: Unable to set a function!"));
    }
}

void hhfft::HHFFT_1D_Complex_D_set_function(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{
    step_info.step_function = nullptr;    

    // Add a fft step
    set_radix_d(step_info, instruction_set);

    if (step_info.step_function == nullptr)
    {
        throw(std::runtime_error("HHFFT error: Unable to set a function!"));
    }
}

// This returns a pointer to correct convolution function based on instruction set
void (*hhfft::HHFFT_1D_Complex_D_set_convolution_function(hhfft::InstructionSet instruction_set))(const double *, const double *, double *, size_t)
{
#ifdef HHFFT_COMPILED_WITH_AVX512F
    if (instruction_set == hhfft::InstructionSet::avx512f)
    {
        return fft_1d_complex_convolution_avx512_d;
    }
#endif

#ifdef HHFFT_COMPILED_WITH_AVX
    if (instruction_set == hhfft::InstructionSet::avx)
    {
        return fft_1d_complex_convolution_avx_d;
    }
#endif

    if (instruction_set == hhfft::InstructionSet::sse2)
    {
        return fft_1d_complex_convolution_sse2_d;
    }

    return fft_1d_complex_convolution_plain_d;
}


template<size_t n> void set_small_function_instruction_set_d(StepInfoD &step_info, hhfft::InstructionSet instruction_set, bool forward)
{
#ifdef HHFFT_COMPILED_WITH_AVX
    if (instruction_set == hhfft::InstructionSet::avx)
    {
        if(forward)
            step_info.step_function = fft_1d_complex_1level_avx_d<n,true>;
        else
            step_info.step_function = fft_1d_complex_1level_avx_d<n,false>;
        return;
    }
#endif

    if (instruction_set == hhfft::InstructionSet::sse2)
    {
        if(forward)
            step_info.step_function =  fft_1d_complex_1level_sse2_d<n,true>;
        else
            step_info.step_function =  fft_1d_complex_1level_sse2_d<n,false>;
        return;
    }

    if (instruction_set == hhfft::InstructionSet::none)
    {
        if(forward)
            step_info.step_function =  fft_1d_complex_1level_plain_d<n,true>;
        else
            step_info.step_function =  fft_1d_complex_1level_plain_d<n,false>;
        return;
    }
}

void hhfft::HHFFT_1D_Complex_D_set_small_function(StepInfoD &step_info, size_t n, bool forward, hhfft::InstructionSet instruction_set)
{
    step_info.step_function = nullptr;

    if(n == 1)
    {
        set_small_function_instruction_set_d<1>(step_info, instruction_set, forward);
    } else if(n == 2)
    {
        set_small_function_instruction_set_d<2>(step_info, instruction_set, forward);
    } else if(n == 3)
    {
        set_small_function_instruction_set_d<3>(step_info, instruction_set, forward);
    } else if(n == 4)
    {
        set_small_function_instruction_set_d<4>(step_info, instruction_set, forward);
    } else if(n == 5)
    {
        set_small_function_instruction_set_d<5>(step_info, instruction_set, forward);
    } else if(n == 6)
    {
        set_small_function_instruction_set_d<6>(step_info, instruction_set, forward);
    } else if(n == 7)
    {
        set_small_function_instruction_set_d<7>(step_info, instruction_set, forward);
    } else if(n == 8)
    {
        set_small_function_instruction_set_d<8>(step_info, instruction_set, forward);
    }

    return;
}

void hhfft::HHFFT_1D_Complex_D_set_1level_raders_function(StepInfoD &step_info, bool forward, hhfft::InstructionSet instruction_set)
{
#ifdef HHFFT_COMPILED_WITH_AVX
    if (instruction_set == hhfft::InstructionSet::avx)
    {
        if(forward)
            step_info.step_function = fft_1d_complex_1level_raders_avx_d<true>;
        else
            step_info.step_function = fft_1d_complex_1level_raders_avx_d<false>;
        return;
    }
#endif

    if (instruction_set == hhfft::InstructionSet::sse2)
    {
        if(forward)
            step_info.step_function = fft_1d_complex_1level_raders_sse2_d<true>;
        else
            step_info.step_function = fft_1d_complex_1level_raders_sse2_d<false>;
        return;
    }

    if (instruction_set == hhfft::InstructionSet::none)
    {
        if(forward)
            step_info.step_function = fft_1d_complex_1level_raders_plain_d<true>;
        else
            step_info.step_function = fft_1d_complex_1level_raders_plain_d<false>;
        return;
    }
}
