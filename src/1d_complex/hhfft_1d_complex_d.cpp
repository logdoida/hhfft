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

#include <immintrin.h>

using namespace hhfft;

// Actual implementations are in different .cpp-files
// No twiddle factors
template<size_t radix, bool forward>
    void fft_1d_complex_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template<size_t radix, SizeType stride_type, bool forward>
    void fft_1d_complex_sse2_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template<size_t radix, SizeType stride_type, bool forward>
    void fft_1d_complex_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template<size_t radix, SizeType stride_type, bool forward>
    void fft_1d_complex_avx512_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

// DIT
template<size_t radix, bool forward>
    void fft_1d_complex_twiddle_dit_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template<size_t radix, SizeType stride_type, bool forward>
    void fft_1d_complex_twiddle_dit_sse2_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template<size_t radix, SizeType stride_type, bool forward>
    void fft_1d_complex_twiddle_dit_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template<size_t radix, SizeType stride_type, bool forward>
    void fft_1d_complex_twiddle_dit_avx512_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

// DIT
template<size_t radix, bool forward>
    void fft_1d_complex_twiddle_dif_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template<size_t radix, SizeType stride_type, bool forward>
    void fft_1d_complex_twiddle_dif_sse2_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template<size_t radix, SizeType stride_type, bool forward>
    void fft_1d_complex_twiddle_dif_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template<size_t radix, SizeType stride_type, bool forward>
    void fft_1d_complex_twiddle_dif_avx512_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

// Reordering
template<bool scale>
    void fft_1d_complex_reorder_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template<bool scale>
    void fft_1d_complex_reorder_sse2_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template<bool scale>
    void fft_1d_complex_reorder_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

// Convolution
void fft_1d_complex_convolution_plain_d(const double *in1, const double *in2, double *out, size_t n);
void fft_1d_complex_convolution_sse2_d(const double *in1, const double *in2, double *out, size_t n);
void fft_1d_complex_convolution_avx_d(const double *in1, const double *in2, double *out, size_t n);


template<size_t radix, SizeType stride_type, bool forward> void set_instruction_set_d(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{

#ifdef HHFFT_COMPILED_WITH_AVX512F
    if (instruction_set == hhfft::InstructionSet::avx512f)
    {
        if (step_info.twiddle_factors == nullptr)
        {
           step_info.step_function = fft_1d_complex_avx512_d<radix, stride_type, forward>;
        } else
        {
            if (step_info.dif)
                step_info.step_function = fft_1d_complex_twiddle_dif_avx512_d<radix, stride_type, forward>;
            else
                step_info.step_function = fft_1d_complex_twiddle_dit_avx512_d<radix, stride_type, forward>;
        }
    }
#endif

#ifdef HHFFT_COMPILED_WITH_AVX
    if (instruction_set == hhfft::InstructionSet::avx)
    {
        if (step_info.twiddle_factors == nullptr)
        {
           step_info.step_function = fft_1d_complex_avx_d<radix, stride_type, forward>;
        } else
        {
            if (step_info.dif)
                step_info.step_function = fft_1d_complex_twiddle_dif_avx_d<radix, stride_type, forward>;
            else
                step_info.step_function = fft_1d_complex_twiddle_dit_avx_d<radix, stride_type, forward>;
        }
    }
#endif

    if (instruction_set == hhfft::InstructionSet::sse2)
    {
        if (step_info.twiddle_factors == nullptr)
        {
           step_info.step_function = fft_1d_complex_sse2_d<radix, stride_type, forward>;
        } else
        {
            if (step_info.dif)
                step_info.step_function = fft_1d_complex_twiddle_dif_sse2_d<radix, stride_type, forward>;
            else
                step_info.step_function = fft_1d_complex_twiddle_dit_sse2_d<radix, stride_type, forward>;
        }
    }  

    if (instruction_set == hhfft::InstructionSet::none)
    {
        // NOTE plain is more or less an unused reference implementation, as sse2 should always be supported
        if (step_info.twiddle_factors == nullptr)
        {
           step_info.step_function = fft_1d_complex_plain_d<radix, forward>;
        } else
        {
            if (step_info.dif)
                step_info.step_function = fft_1d_complex_twiddle_dif_plain_d<radix, forward>;
            else
                step_info.step_function = fft_1d_complex_twiddle_dit_plain_d<radix, forward>;
        }
    }  
}

// These functions set different template parameters one at time
template<size_t radix, SizeType stride_type> void set_forward_d(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{
    if (step_info.forward)
    {
        set_instruction_set_d<radix, stride_type, true>(step_info, instruction_set);
    } else
    {
        set_instruction_set_d<radix, stride_type, false>(step_info, instruction_set);
    }
}

template<size_t radix> void set_stride_type_d(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{
    // Knowing something about the stride at compile time can help to optimize some cases
    size_t stride = step_info.stride;

    // Only stride = 1 and N are currently used
    if (stride == 1)
    {
        set_forward_d<radix, SizeType::Size1>(step_info, instruction_set);
    } else
    {
        set_forward_d<radix, SizeType::SizeN>(step_info, instruction_set);
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

    if (radix == 2)
    {
        set_stride_type_d<2>(step_info, instruction_set);
    } if (radix == 3)
    {
        set_stride_type_d<3>(step_info, instruction_set);
    } if (radix == 4)
    {
        set_stride_type_d<4>(step_info, instruction_set);
    } if (radix == 5)
    {
        set_stride_type_d<5>(step_info, instruction_set);
    } if (radix == 7)
    {
        set_stride_type_d<7>(step_info, instruction_set);
    } if (radix == 8)
    {
        set_stride_type_d<8>(step_info, instruction_set);
    }
}

template<bool scale> void set_reorder_instruction_set_d(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{

#ifdef HHFFT_COMPILED_WITH_AVX512F
    if (instruction_set == hhfft::InstructionSet::avx512f)
    {
        step_info.step_function = fft_1d_complex_reorder_avx512_d<scale>;
    }
#endif

#ifdef HHFFT_COMPILED_WITH_AVX
    if (instruction_set == hhfft::InstructionSet::avx)
    {
        step_info.step_function = fft_1d_complex_reorder_avx_d<scale>;
    }
#endif

    if (instruction_set == hhfft::InstructionSet::sse2)
    {
        step_info.step_function = fft_1d_complex_reorder_sse2_d<scale>;
    }

    if (instruction_set == hhfft::InstructionSet::none)
    {
        // NOTE plain is more or less an unused reference implementation, as sse2 should always be supported
        step_info.step_function = fft_1d_complex_reorder_plain_d<scale>;
    }
}

// This set pointer to correct fft functions based on radix and stride etc
void hhfft::HHFFT_1D_Complex_D_set_function(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{
    step_info.step_function = nullptr;

    if (step_info.reorder_table != nullptr || step_info.reorder_table_inplace != nullptr)
    {
        // Add reordering step
        if (step_info.norm_factor != 1.0)
            set_reorder_instruction_set_d<true>(step_info, instruction_set);
        else
            set_reorder_instruction_set_d<false>(step_info, instruction_set);
    } else
    {
        // Add a fft step
        set_radix_d(step_info, instruction_set);
    }

    if (step_info.step_function == nullptr)
    {
        throw(std::runtime_error("HHFFT error: Unable to set a function!"));
    }
}

// This returns a pointer to correct convolution function based on instruction set
void (*hhfft::HHFFT_1D_Complex_D_set_convolution_function(hhfft::InstructionSet instruction_set))(const double *, const double *, double *, size_t n)
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
