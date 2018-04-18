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

#include <iostream> //TESTING

using namespace hhfft;

// Compiler is not able to optimize this to use sse2! TODO implement different versions (plain/sse2/avx) of this!
// In-place reordering "swap"
template<bool forward> void fft_1d_complex_reorder_in_place_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{    
    size_t n = step_info.repeats;
    uint32_t *reorder_table = step_info.reorder_table_inplace;

    // In-place algorithm
    assert (data_in == data_out);

    for (size_t i = 0; i < n; i++)
    {
        size_t ind1 = i + 1; // First one has been omitted!
        size_t ind2 = reorder_table[i];

        // Swap two doubles at a time
        /*
        __m128d temp1 = _mm_loadu_pd(data_in + 2*ind1);
        __m128d temp2 = _mm_loadu_pd(data_in + 2*ind2);
        _mm_storeu_pd(data_out + 2*ind2, temp1);
        _mm_storeu_pd(data_out + 2*ind1, temp2);
        */

        double r_temp = data_out[2*ind1+0];
        double c_temp = data_out[2*ind1+1];
        data_out[2*ind1+0] = data_out[2*ind2+0];
        data_out[2*ind1+1] = data_out[2*ind2+1];
        data_out[2*ind2+0] = r_temp;
        data_out[2*ind2+1] = c_temp;
    }

    // Scaling needs to be done as a separate step as some data might be copied twice or zero times
    // TODO this is not very efficient. Scaling could be done at some other step (first/last)    
    size_t n2 = step_info.stride;
    if (!forward)
    {
        // Needed only in ifft. Equal to 1/N
        double k = step_info.norm_factor;

        for (size_t i = 0; i < 2*n2; i++)
        {
            data_out[i] *= k;
        }
    }
}

// Compiler is not able to optimize this to use sse2! TODO implement different versions (plain/sse2/avx) of this!
template<bool forward> void fft_1d_complex_reorder_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    std::cout << "fft_1d_reorder" << std::endl;

    // Check, if in-place should be done instead
    if (data_in == data_out)
    {
        fft_1d_complex_reorder_in_place_d<forward>(data_in, data_out, step_info);
        return;
    }

    size_t n = step_info.stride;
    uint32_t *reorder_table = step_info.reorder_table;

    // Needed only in ifft. Equal to 1/N
    double k = step_info.norm_factor;

    for (size_t i = 0; i < n; i++)
    {
        size_t i2 = reorder_table[i];
        if (forward)
        {
            // Compiler is not able to optimize this to use sse2!
            data_out[2*i+0] = data_in[2*i2+0];
            data_out[2*i+1] = data_in[2*i2+1];
        } else
        {
            // Compiler is not able to optimize this to use sse2!
            data_out[2*i+0] = k*data_in[2*i2+0];
            data_out[2*i+1] = k*data_in[2*i2+1];
        }
    }
}


// Actual implementations are in different .cpp-files
// No twiddle factors
template<size_t radix, StrideType stride_type, bool forward>
    void fft_1d_complex_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template<size_t radix, StrideType stride_type, bool forward>
    void fft_1d_complex_sse2_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template<size_t radix, StrideType stride_type, bool forward>
    void fft_1d_complex_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template<size_t radix, StrideType stride_type, bool forward>
    void fft_1d_complex_avx512_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

// DIT
template<size_t radix, StrideType stride_type, bool forward>
    void fft_1d_complex_twiddle_dit_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template<size_t radix, StrideType stride_type, bool forward>
    void fft_1d_complex_twiddle_dit_sse2_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template<size_t radix, StrideType stride_type, bool forward>
    void fft_1d_complex_twiddle_dit_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template<size_t radix, StrideType stride_type, bool forward>
    void fft_1d_complex_twiddle_dit_avx512_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

// DIT
template<size_t radix, StrideType stride_type, bool forward>
    void fft_1d_complex_twiddle_dif_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template<size_t radix, StrideType stride_type, bool forward>
    void fft_1d_complex_twiddle_dif_sse2_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template<size_t radix, StrideType stride_type, bool forward>
    void fft_1d_complex_twiddle_dif_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template<size_t radix, StrideType stride_type, bool forward>
    void fft_1d_complex_twiddle_dif_avx512_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);


template<size_t radix, StrideType stride_type, bool forward> void set_instruction_set_d(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{
    // TESTING to speedup compilation            
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

    /*
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
    */

    /*
    if (step_info.twiddle_factors == nullptr)
    {
       step_info.step_function = fft_1d_complex_plain_d<radix, stride_type, forward>;
    } else
    {
        if (step_info.dif)
            step_info.step_function = fft_1d_complex_twiddle_dif_plain_d<radix, stride_type, forward>;
        else
            step_info.step_function = fft_1d_complex_twiddle_dit_plain_d<radix, stride_type, forward>;
    }
    */


/*
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
    } else if (instruction_set == hhfft::InstructionSet::avx)
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
    } else if (instruction_set == hhfft::InstructionSet::sse2)
    {
        if (step_info.twiddle_factors == nullptr)
        {
           step_info.step_function = fft_1d_complex_sse2_d<radix, stride_type, forward>;
        } else
        {
            if (step_info.dif)
                step_info.step_function = fft_1d_complex_twiddle_dif_sse2_d<radix, stride_type, forward>;
            else
                step_info.step_function = fft_1d_complex_twiddle_dit_sse_d<radix, stride_type, forward>;
        }
    } else
    {
        // NOTE plain is more or less an unused reference implementation, as sse2 should always be supported
        if (step_info.twiddle_factors == nullptr)
        {
           step_info.step_function = fft_1d_complex_plain_d<radix, stride_type, forward>;
        } else
        {
            if (step_info.dif)
                step_info.step_function = fft_1d_complex_twiddle_dif_plain_d<radix, stride_type, forward>;
            else
                step_info.step_function = fft_1d_complex_twiddle_dit_plain_d<radix, stride_type, forward>;
        }
    }
  */
}

// These functions set different template parameters one at time
template<size_t radix, StrideType stride_type> void set_forward_d(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
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
    // TESTING to speed up compilation
    set_forward_d<radix, StrideType::StrideN>(step_info, instruction_set);

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
    }
}

// This set pointer to correct fft functions based on radix and stride etc
void hhfft::HHFFT_1D_Complex_D_set_function(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{
    step_info.step_function = nullptr;

    if (step_info.reorder_table != nullptr || step_info.reorder_table_inplace != nullptr)
    {    
        if (step_info.forward)
            step_info.step_function = fft_1d_complex_reorder_d<true>;
        else
            step_info.step_function = fft_1d_complex_reorder_d<false>;

        return;
    }

    set_radix_d(step_info, instruction_set);

    if (step_info.step_function == nullptr)
    {
        throw(std::runtime_error("HHFFT error: Unable to set a function!"));
    }
}

