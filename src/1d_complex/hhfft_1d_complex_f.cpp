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

#include "hhfft_1d_complex_f.h"
#include <stdexcept>
#include <assert.h>
#include <cmath>


using namespace hhfft;

// Actual implementations are in different .cpp-files
// No twiddle factors
template<RadixType radix_type, SizeType stride_type>
    void fft_1d_complex_plain_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template<RadixType radix_type, SizeType stride_type>
    void fft_1d_complex_sse2_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template<RadixType radix_type, SizeType stride_type>
    void fft_1d_complex_avx_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

//template<size_t radix, SizeType stride_type>
//    void fft_1d_complex_avx512_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

// DIT
template<RadixType radix_type, SizeType stride_type>
    void fft_1d_complex_twiddle_dit_plain_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template<RadixType radix_type, SizeType stride_type>
    void fft_1d_complex_twiddle_dit_sse2_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template<RadixType radix_type, SizeType stride_type>
    void fft_1d_complex_twiddle_dit_avx_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

//template<size_t radix, SizeType stride_type>
//    void fft_1d_complex_twiddle_dit_avx512_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

// Reorder in-place
void fft_1d_complex_reorder_in_place_plain_f(const float *, float *data_out,const hhfft::StepInfo<float> &step_info);
void fft_1d_complex_reorder_in_place_sse2_f(const float *, float *data_out,const hhfft::StepInfo<float> &step_info);
void fft_1d_complex_reorder_in_place_avx_f(const float *, float *data_out,const hhfft::StepInfo<float> &step_info);

// Reordering
template<bool scale>
    void fft_1d_complex_reorder_plain_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template<bool scale>
    void fft_1d_complex_reorder_sse2_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template<bool scale>
    void fft_1d_complex_reorder_avx_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

// Reorder and do FFT
template<RadixType radix_type, SizeType stride_type, bool forward>
void fft_1d_complex_reorder2_plain_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template<RadixType radix_type, SizeType stride_type, bool forward>
void fft_1d_complex_reorder2_sse2_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template<RadixType radix_type, SizeType stride_type, bool forward>
void fft_1d_complex_reorder2_avx_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

// Convolution
void fft_1d_complex_convolution_plain_f(const float *in1, const float *in2, float *out, size_t n);
void fft_1d_complex_convolution_sse2_f(const float *in1, const float *in2, float *out, size_t n);
void fft_1d_complex_convolution_avx_f(const float *in1, const float *in2, float *out, size_t n);

// Small single level FFT
template<size_t n, bool forward> void fft_1d_complex_1level_avx_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template<size_t n, bool forward> void fft_1d_complex_1level_sse2_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template<size_t n, bool forward> void fft_1d_complex_1level_plain_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template<size_t n1, size_t n2, bool forward> void fft_1d_complex_2level_avx_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template<size_t n1, size_t n2, bool forward> void fft_1d_complex_2level_sse2_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template<size_t n1, size_t n2, bool forward> void fft_1d_complex_2level_plain_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template<bool forward> void fft_1d_complex_1level_raders_plain_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template<bool forward> void fft_1d_complex_1level_raders_sse2_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template<bool forward> void fft_1d_complex_1level_raders_avx_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);


template<RadixType radix_type, SizeType stride_type, bool forward> void set_instruction_set_f(StepInfoF &step_info, hhfft::InstructionSet instruction_set)
{

    /*
#ifdef HHFFT_COMPILED_WITH_AVX512F
    if (instruction_set == hhfft::InstructionSet::avx512f)
    {
        if (step_info.twiddle_factors == nullptr)
        {
            // Check if reordering should be supported
            if(step_info.reorder_table == nullptr)
                step_info.step_function = fft_1d_complex_avx512_f<radix, stride_type>;
            else if(step_info.stride == 1)
            {                
                step_info.step_function = fft_1d_complex_reorder2_avx512_f<radix, stride_type, forward>;
            }
        } else
        {
            if (step_info.dif)
                step_info.step_function = fft_1d_complex_twiddle_fif_avx512_f<radix, stride_type, forward>;
            else
                step_info.step_function = fft_1d_complex_twiddle_fit_avx512_f<radix, stride_type>;
        }
    }
#endif
*/

#ifdef HHFFT_COMPILED_WITH_AVX
    if (instruction_set == hhfft::InstructionSet::avx)
    {        
        if (step_info.twiddle_factors == nullptr)
        {
            // Check if reordering should be supported
            if(step_info.reorder_table == nullptr)
                step_info.step_function = fft_1d_complex_avx_f<radix_type, stride_type>;
            else
                step_info.step_function = fft_1d_complex_reorder2_avx_f<radix_type, stride_type, forward>;
        } else
        {
            step_info.step_function = fft_1d_complex_twiddle_dit_avx_f<radix_type, stride_type>;
        }        
    }
#endif

    if (instruction_set == hhfft::InstructionSet::sse2)
    {        
        if (step_info.twiddle_factors == nullptr)
        {
            // Check if reordering should be supported
            if(step_info.reorder_table == nullptr)
                step_info.step_function = fft_1d_complex_sse2_f<radix_type, stride_type>;
            else
                step_info.step_function = fft_1d_complex_reorder2_sse2_f<radix_type, stride_type, forward>;
        } else
        {
            step_info.step_function = fft_1d_complex_twiddle_dit_sse2_f<radix_type, stride_type>;
        }        
    }    

    if (instruction_set == hhfft::InstructionSet::none)
    {
        // NOTE plain is more or less an unused reference implementation, as sse2 should always be supported
        if (step_info.twiddle_factors == nullptr)
        {
           // Check if reordering should be supported
           if(step_info.reorder_table == nullptr)
               step_info.step_function = fft_1d_complex_plain_f<radix_type, stride_type>;
           else
               step_info.step_function = fft_1d_complex_reorder2_plain_f<radix_type, stride_type, forward>;
        } else
        {
            step_info.step_function = fft_1d_complex_twiddle_dit_plain_f<radix_type, stride_type>;
        }
    }  
}

// These functions set different template parameters one at time
template<RadixType radix_type, SizeType stride_type> void set_forward_f(StepInfoF &step_info, hhfft::InstructionSet instruction_set)
{
    if (step_info.forward)
    {
        set_instruction_set_f<radix_type, stride_type, true>(step_info, instruction_set);
    } else
    {
        set_instruction_set_f<radix_type, stride_type, false>(step_info, instruction_set);
    }
}

template<RadixType radix_type> void set_stride_type_f(StepInfoF &step_info, hhfft::InstructionSet instruction_set)
{
    // Knowing something about the stride at compile time can help to optimize some cases
    size_t stride = step_info.stride;

    // Only stride = 1 and N are currently used
    if (stride == 1)
    {
        set_forward_f<radix_type, SizeType::Size1>(step_info, instruction_set);
    } else
    {
        set_forward_f<radix_type, SizeType::SizeN>(step_info, instruction_set);
    }

    /*
    size_t stride = step_info.stride;

    // Knowing the stride at compile time can help the compiler to optimze
    if (stride == 1)
    {
        set_forward_f<radix, StrideType::Stride1>(step_info, instruction_set);
    } else if (stride == 2)
    {
        set_forward_f<radix, StrideType::Stride2>(step_info, instruction_set);
    } else if (stride == 4)
    {
        set_forward_f<radix, StrideType::Stride4>(step_info, instruction_set);
    } else if (stride%4 == 0)
    {
        set_forward_f<radix, StrideType::Stride4N>(step_info, instruction_set);
    } else if (stride%2 == 0)
    {
        set_forward_f<radix, StrideType::Stride2N>(step_info, instruction_set);
    } else
    {
        set_forward_f<radix, StrideType::StrideN>(step_info, instruction_set);
    }
    */
}

void set_radix_f(StepInfoF &step_info, hhfft::InstructionSet instruction_set)
{
    size_t radix = step_info.radix;

    if (radix > 8)  // Rader's algorithm (radix can be something else)
    {
        set_stride_type_f<Raders>(step_info, instruction_set);
    } else if (radix == 2)
    {
        set_stride_type_f<Radix2>(step_info, instruction_set);
    } else if (radix == 3)
    {
        set_stride_type_f<Radix3>(step_info, instruction_set);
    } else if (radix == 4)
    {
        set_stride_type_f<Radix4>(step_info, instruction_set);
    } else if (radix == 5)
    {
        set_stride_type_f<Radix5>(step_info, instruction_set);
    } else if (radix == 6)
    {
        set_stride_type_f<Radix6>(step_info, instruction_set);
    } else if (radix == 7)
    {
        set_stride_type_f<Radix7>(step_info, instruction_set);
    } else if (radix == 8)
    {
        set_stride_type_f<Radix8>(step_info, instruction_set);
    }
}


template<bool forward> void set_reorder_instruction_set_f(StepInfoF &step_info, hhfft::InstructionSet instruction_set)
{

/*
#ifdef HHFFT_COMPILED_WITH_AVX512F
    if (instruction_set == hhfft::InstructionSet::avx512f)
    {
        if (step_info.reorder_table != nullptr)
            step_info.step_function = fft_1d_complex_reorder_avx512_f<forward>;
        // TODO in-place
    }
#endif
*/

#ifdef HHFFT_COMPILED_WITH_AVX
    if (instruction_set == hhfft::InstructionSet::avx)
    {
        if (step_info.reorder_table != nullptr)
            step_info.step_function = fft_1d_complex_reorder_avx_f<forward>;
        if (step_info.reorder_table_inplace != nullptr)
            step_info.step_function = fft_1d_complex_reorder_in_place_avx_f;
    }
#endif

    if (instruction_set == hhfft::InstructionSet::sse2)
    {
        if (step_info.reorder_table != nullptr)
            step_info.step_function = fft_1d_complex_reorder_sse2_f<forward>;
        if (step_info.reorder_table_inplace != nullptr)
            step_info.step_function = fft_1d_complex_reorder_in_place_sse2_f;
    }

    if (instruction_set == hhfft::InstructionSet::none)
    {        
        if (step_info.reorder_table != nullptr)
            step_info.step_function = fft_1d_complex_reorder_plain_f<forward>;
        if (step_info.reorder_table_inplace != nullptr)
            step_info.step_function = fft_1d_complex_reorder_in_place_plain_f;
    }
}

void hhfft::HHFFT_1D_Complex_F_set_reorder_function(StepInfoF &step_info, hhfft::InstructionSet instruction_set)
{
    step_info.step_function = nullptr;

    if (step_info.radix == 1 && (step_info.reorder_table != nullptr || step_info.reorder_table_inplace != nullptr))
    {
        // Add reordering step
        if (step_info.forward)
            set_reorder_instruction_set_f<true>(step_info, instruction_set);
        else
            set_reorder_instruction_set_f<false>(step_info, instruction_set);
    }

    if (step_info.step_function == nullptr)
    {
        throw(std::runtime_error("HHFFT error: Unable to set a function!"));
    }
}

void hhfft::HHFFT_1D_Complex_F_set_function(StepInfoF &step_info, hhfft::InstructionSet instruction_set)
{
    step_info.step_function = nullptr;    

    // Add a fft step
    set_radix_f(step_info, instruction_set);

    if (step_info.step_function == nullptr)
    {
        throw(std::runtime_error("HHFFT error: Unable to set a function!"));
    }
}

// This returns a pointer to correct convolution function based on instruction set
void (*hhfft::HHFFT_1D_Complex_F_set_convolution_function(hhfft::InstructionSet instruction_set))(const float *, const float *, float *, size_t)
{
/*
#ifdef HHFFT_COMPILED_WITH_AVX512F
    if (instruction_set == hhfft::InstructionSet::avx512f)
    {
        return fft_1d_complex_convolution_avx512_f;
    }
#endif
*/

#ifdef HHFFT_COMPILED_WITH_AVX
    if (instruction_set == hhfft::InstructionSet::avx)
    {
        return fft_1d_complex_convolution_avx_f;
    }
#endif

    if (instruction_set == hhfft::InstructionSet::sse2)
    {
        return fft_1d_complex_convolution_sse2_f;
    }

    return fft_1d_complex_convolution_plain_f;
}


template<size_t n1, size_t n2> std::vector<size_t> set_small_function_instruction_set_f(StepInfoF &step_info, hhfft::InstructionSet instruction_set, bool forward)
{
#ifdef HHFFT_COMPILED_WITH_AVX
    if (instruction_set == hhfft::InstructionSet::avx)
    {
        if (n2 == 1)
        {
            if(forward)
                step_info.step_function = fft_1d_complex_1level_avx_f<n1,true>;
            else
                step_info.step_function = fft_1d_complex_1level_avx_f<n1,false>;
        } else
        {
            if(forward)
                step_info.step_function = fft_1d_complex_2level_avx_f<n1,n2,true>;
            else
                step_info.step_function = fft_1d_complex_2level_avx_f<n1,n2,false>;
            return {n1,n2};
        }
    }
#endif

    if (instruction_set == hhfft::InstructionSet::sse2)
    {
        if (n2 == 1)
        {
            if(forward)
                step_info.step_function = fft_1d_complex_1level_sse2_f<n1,true>;
            else
                step_info.step_function = fft_1d_complex_1level_sse2_f<n1,false>;
        } else
        {
            if(forward)
                step_info.step_function = fft_1d_complex_2level_sse2_f<n1,n2,true>;
            else
                step_info.step_function = fft_1d_complex_2level_sse2_f<n1,n2,false>;
            return {n1,n2};
        }
    }    

    if (instruction_set == hhfft::InstructionSet::none)
    {
        if (n2 == 1)
        {
            if(forward)
                step_info.step_function = fft_1d_complex_1level_plain_f<n1,true>;
            else
                step_info.step_function = fft_1d_complex_1level_plain_f<n1,false>;
        } else
        {
            if(forward)
                step_info.step_function = fft_1d_complex_2level_plain_f<n1,n2,true>;
            else
                step_info.step_function = fft_1d_complex_2level_plain_f<n1,n2,false>;
            return {n1,n2};
        }        
    }

    return std::vector<size_t>();
}

std::vector<size_t> hhfft::HHFFT_1D_Complex_F_set_small_function(StepInfoF &step_info, size_t n, bool forward, hhfft::InstructionSet instruction_set)
{
    step_info.step_function = nullptr;

    switch (n)
    {
        // One fft level needed
        case 1:
            return set_small_function_instruction_set_f<1,1>(step_info, instruction_set, forward);
        case 2:
            return set_small_function_instruction_set_f<2,1>(step_info, instruction_set, forward);
        case 3:
            return set_small_function_instruction_set_f<3,1>(step_info, instruction_set, forward);
        case 4:
            return set_small_function_instruction_set_f<4,1>(step_info, instruction_set, forward);
        case 5:
            return set_small_function_instruction_set_f<5,1>(step_info, instruction_set, forward);
        case 6:
            return set_small_function_instruction_set_f<6,1>(step_info, instruction_set, forward);
        case 7:
            return set_small_function_instruction_set_f<7,1>(step_info, instruction_set, forward);
        case 8:
            return set_small_function_instruction_set_f<8,1>(step_info, instruction_set, forward);

        // Two fft levels needed            
        case 9:
            return set_small_function_instruction_set_f<3,3>(step_info, instruction_set, forward);
        case 10:
            return set_small_function_instruction_set_f<2,5>(step_info, instruction_set, forward);
        case 12:
            return set_small_function_instruction_set_f<2,6>(step_info, instruction_set, forward);
        case 14:
            return set_small_function_instruction_set_f<2,7>(step_info, instruction_set, forward);
        case 15:
            return set_small_function_instruction_set_f<3,5>(step_info, instruction_set, forward);
        case 16:
            return set_small_function_instruction_set_f<4,4>(step_info, instruction_set, forward);
        case 18:
            return set_small_function_instruction_set_f<3,6>(step_info, instruction_set, forward);
        case 20:
            return set_small_function_instruction_set_f<4,5>(step_info, instruction_set, forward);
        case 21:
            return set_small_function_instruction_set_f<3,7>(step_info, instruction_set, forward);
        case 24:
            return set_small_function_instruction_set_f<4,6>(step_info, instruction_set, forward);
        case 25:
            return set_small_function_instruction_set_f<5,5>(step_info, instruction_set, forward);
        case 28:
            return set_small_function_instruction_set_f<4,7>(step_info, instruction_set, forward);
        case 30:
            return set_small_function_instruction_set_f<5,6>(step_info, instruction_set, forward);
        case 32:
            return set_small_function_instruction_set_f<4,8>(step_info, instruction_set, forward);
        case 35:
            return set_small_function_instruction_set_f<5,7>(step_info, instruction_set, forward);
        case 36:
            return set_small_function_instruction_set_f<6,6>(step_info, instruction_set, forward);
        case 40:
            return set_small_function_instruction_set_f<5,8>(step_info, instruction_set, forward);
        case 42:
            return set_small_function_instruction_set_f<6,7>(step_info, instruction_set, forward);
        case 48:
            return set_small_function_instruction_set_f<6,8>(step_info, instruction_set, forward);
        case 49:
            return set_small_function_instruction_set_f<7,7>(step_info, instruction_set, forward);
        case 56:
            return set_small_function_instruction_set_f<7,8>(step_info, instruction_set, forward);
        case 64:
            return set_small_function_instruction_set_f<8,8>(step_info, instruction_set, forward);

        default:
            break;
    }

    return std::vector<size_t>();
}

void hhfft::HHFFT_1D_Complex_F_set_1level_raders_function(StepInfoF &step_info, bool forward, hhfft::InstructionSet instruction_set)
{
#ifdef HHFFT_COMPILED_WITH_AVX
    if (instruction_set == hhfft::InstructionSet::avx)
    {
        if(forward)
            step_info.step_function = fft_1d_complex_1level_raders_avx_f<true>;
        else
            step_info.step_function = fft_1d_complex_1level_raders_avx_f<false>;
        return;
    }
#endif

    if (instruction_set == hhfft::InstructionSet::sse2)
    {
        if(forward)
            step_info.step_function = fft_1d_complex_1level_raders_sse2_f<true>;
        else
            step_info.step_function = fft_1d_complex_1level_raders_sse2_f<false>;
        return;
    }    

    if (instruction_set == hhfft::InstructionSet::none)
    {
        if(forward)
            step_info.step_function = fft_1d_complex_1level_raders_plain_f<true>;
        else
            step_info.step_function = fft_1d_complex_1level_raders_plain_f<false>;
        return;
    }
}
