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

#include "hhfft_2d_real_setter.h"
#include "../aligned_arrays.h"
#include <stdexcept>
#include <assert.h>
#include <cmath>

using namespace hhfft;

// Shuffles one row into two rows
void fft_2d_real_reorder_columns_inverse_inplace_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info)
{
    // It is assumed that this is always in-place as this is the last step
    assert (data_in == data_out);

    size_t n = step_info.stride;
    size_t m = step_info.size;

    // Only shuffle is needed
    hhfft::AlignedVector<float> temp(m);
    float* temp_data = temp.data();
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

// Change values in the first column from [0 r] to [r 0]
void fft_2d_real_fix_first_column_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info)
{
    // It is assumed that this is always in-place as this is the last step
    assert (data_in == data_out);

    size_t n = step_info.stride;
    size_t m = step_info.size + 1;

    for (size_t i = 0; i < n; i++)
    {
        float r = data_in[i*m + 1];
        data_out[i*m + 0] = r;
        data_out[i*m + 1] = 0;
    }
}

// Combines first column to rest of the columns in ifft odd
void fft_2d_real_odd_rows_combine_columns_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info)
{
    size_t n = step_info.stride;
    size_t m2 = step_info.size;  // row size after this function
    size_t m = m2 - 1;           // row size before this function

    for (size_t i = 0; i < n; i++)
    {
        size_t i2 = n - i - 1;

        // Move normal columns
        for (size_t j = 0; j < m; j++)
        {
            size_t j2 = m - j - 1;
            data_out[i2*m2 + j2 + 1] = data_out[i2*m + j2];
        }

        // Copy real part from first column
        data_out[i2*m2] = data_in[2*i2];
    }
}



namespace hhfft
{
    template<> void HHFFT_2D_Real_set_reorder_column_function<float>(StepInfoF &step_info)
    {
        if (!step_info.forward)
        {
            step_info.step_function = fft_2d_real_reorder_columns_inverse_inplace_f;
        }

        if (step_info.step_function == nullptr)
        {
            throw(std::runtime_error("HHFFT error: Unable to set a function!"));
        }
    }
}


// Actual implementations are in different .cpp-files
template<bool forward> void fft_2d_complex_to_complex_packed_plain_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template<bool forward> void fft_2d_complex_to_complex_packed_sse2_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template<bool forward> void fft_2d_complex_to_complex_packed_avx_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template<bool forward> void fft_2d_complex_to_complex_packed_avx512_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);


template<bool forward> void set_instruction_set_f(StepInfoF &step_info, hhfft::InstructionSet instruction_set)
{

#ifdef HHFFT_COMPILED_WITH_AVX512F
    if (instruction_set == hhfft::InstructionSet::avx512f)
    {
        
    }
#endif

    /*
#ifdef HHFFT_COMPILED_WITH_AVX
    if (instruction_set == hhfft::InstructionSet::avx)
    {
       step_info.step_function = fft_2d_complex_to_complex_packed_avx_f<forward>;
    }
#endif

    if (instruction_set == hhfft::InstructionSet::sse2)
    {
        step_info.step_function = fft_2d_complex_to_complex_packed_sse2_f<forward>;
    }
    */

    if (instruction_set == hhfft::InstructionSet::none)
    {
        step_info.step_function = fft_2d_complex_to_complex_packed_plain_f<forward>;
    }      
}

namespace hhfft
{
    // This set pointer to correct functions
    template<> void HHFFT_2D_Real_set_complex_to_complex_packed_function<float>(StepInfoF &step_info, hhfft::InstructionSet instruction_set)
    {
        step_info.step_function = nullptr;

        if (step_info.forward)
           set_instruction_set_f<true>(step_info, instruction_set);
        else
           set_instruction_set_f<false>(step_info, instruction_set);

        if (step_info.step_function == nullptr)
        {
            throw(std::runtime_error("HHFFT error: Unable to set a function!"));
        }
    }
}


void fft_2d_complex_to_complex_packed_first_column_plain_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
void fft_2d_complex_to_complex_packed_first_column_sse2_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
void fft_2d_complex_to_complex_packed_first_column_avx_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

void set_instruction_set_first_column_f(StepInfoF &step_info, hhfft::InstructionSet instruction_set)
{

#ifdef HHFFT_COMPILED_WITH_AVX512F
    if (instruction_set == hhfft::InstructionSet::avx512f)
    {

    }
#endif

    /*
#ifdef HHFFT_COMPILED_WITH_AVX
    if (instruction_set == hhfft::InstructionSet::avx)
    {
       step_info.step_function = fft_2d_complex_to_complex_packed_first_column_avx_f;
    }
#endif

    if (instruction_set == hhfft::InstructionSet::sse2)
    {
        step_info.step_function = fft_2d_complex_to_complex_packed_first_column_sse2_f;
    }
    */

    if (instruction_set == hhfft::InstructionSet::none)
    {
        step_info.step_function = fft_2d_complex_to_complex_packed_first_column_plain_f;
    }
}


namespace hhfft
{
    // This set pointer to correct functions
    template<> void HHFFT_2D_Real_set_complex_to_complex_packed_first_column_function<float>(StepInfoF &step_info, hhfft::InstructionSet instruction_set)
    {
        step_info.step_function = nullptr;

        if (step_info.forward)
           set_instruction_set_first_column_f(step_info, instruction_set);

        if (step_info.step_function == nullptr)
        {
            throw(std::runtime_error("HHFFT error: Unable to set a function!"));
        }
    }
}



// Reorder and do FFT
template<RadixType radix_type> void fft_2d_real_reorder2_inverse_avx_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template<RadixType radix_type> void fft_2d_real_reorder2_inverse_sse2_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
template<RadixType radix_type> void fft_2d_real_reorder2_inverse_plain_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);


template<RadixType radix_type> void set_instruction_set_2d_real_f(StepInfoF &step_info, hhfft::InstructionSet instruction_set)
{

#ifdef HHFFT_COMPILED_WITH_AVX512F
    if (instruction_set == hhfft::InstructionSet::avx512f)
    {

    }
#endif

    /*
#ifdef HHFFT_COMPILED_WITH_AVX
    if (instruction_set == hhfft::InstructionSet::avx)
    {        
        step_info.step_function = fft_2d_real_reorder2_inverse_avx_f<radix_type>;
    }
#endif

    if (instruction_set == hhfft::InstructionSet::sse2)
    {        
        step_info.step_function = fft_2d_real_reorder2_inverse_sse2_f<radix_type>;
    }
    */


    if (instruction_set == hhfft::InstructionSet::none)
    {
        step_info.step_function = fft_2d_real_reorder2_inverse_plain_f<radix_type>;
    }
}

void set_radix_2d_real_f(StepInfoF &step_info, hhfft::InstructionSet instruction_set)
{
    size_t radix = step_info.radix;

    if (radix > 8)
    {
        set_instruction_set_2d_real_f<Raders>(step_info, instruction_set);
    }
    else if (radix == 2)
    {
        set_instruction_set_2d_real_f<Radix2>(step_info, instruction_set);
    } else if (radix == 3)
    {
        set_instruction_set_2d_real_f<Radix3>(step_info, instruction_set);
    } else if (radix == 4)
    {
        set_instruction_set_2d_real_f<Radix4>(step_info, instruction_set);
    } else if (radix == 5)
    {
        set_instruction_set_2d_real_f<Radix5>(step_info, instruction_set);
    } else if (radix == 6)
    {
        set_instruction_set_2d_real_f<Radix6>(step_info, instruction_set);
    } else if (radix == 7)
    {
        set_instruction_set_2d_real_f<Radix7>(step_info, instruction_set);
    } else if (radix == 8)
    {
       set_instruction_set_2d_real_f<Radix8>(step_info, instruction_set);
    }
}

void fft_2d_real_reorder_rows_in_place_plain_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
void fft_2d_real_reorder_rows_in_place_sse2_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);
void fft_2d_real_reorder_rows_in_place_avx_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

void set_instruction_set_2d_reorder_rows_f(StepInfoF &step_info, hhfft::InstructionSet instruction_set)
{

#ifdef HHFFT_COMPILED_WITH_AVX512F
    if (instruction_set == hhfft::InstructionSet::avx512f)
    {

    }
#endif

    /*
#ifdef HHFFT_COMPILED_WITH_AVX
    if (instruction_set == hhfft::InstructionSet::avx)
    {
        step_info.step_function = fft_2d_real_reorder_rows_in_place_avx_f;
    }
#endif

    if (instruction_set == hhfft::InstructionSet::sse2)
    {
        step_info.step_function = fft_2d_real_reorder_rows_in_place_sse2_f;
    }
    */

    if (instruction_set == hhfft::InstructionSet::none)
    {
        step_info.step_function = fft_2d_real_reorder_rows_in_place_plain_f;
    }
}


namespace hhfft
{
    // Used for setting the first step where reordering and first fft are combined
    template<> void HHFFT_2D_Real_set_function<float>(StepInfoF &step_info, hhfft::InstructionSet instruction_set)
    {
        step_info.step_function = nullptr;

        if (step_info.radix == 1 && step_info.reorder_table_inplace != nullptr)
        {
            set_instruction_set_2d_reorder_rows_f(step_info, instruction_set);
        }

        if (step_info.radix != 1 && !step_info.forward)
        {
            set_radix_2d_real_f(step_info, instruction_set);
        }

        if (step_info.step_function == nullptr)
        {
            throw(std::runtime_error("HHFFT error: Unable to set a function!"));
        }
    }
}


////////////////// Odd number of columns ////////////////////////////

// Reorder both rows and columns and do FFT on rows
template<RadixType radix_type>
void fft_2d_real_reorder2_odd_rows_forward_avx_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template<RadixType radix_type>
void fft_2d_real_reorder2_odd_rows_forward_sse2_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template<RadixType radix_type>
void fft_2d_real_reorder2_odd_rows_forward_plain_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

// These are implemented in hhfft_1d_real_*
template<RadixType radix_type>
void fft_2d_real_odd_rows_forward_plain_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template<RadixType radix_type>
void fft_2d_real_odd_rows_forward_sse2_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template<RadixType radix_type>
void fft_2d_real_odd_rows_forward_avx_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template<RadixType radix_type>
void fft_2d_real_odd_rows_first_level_inverse_plain_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template<RadixType radix_type>
void fft_2d_real_odd_rows_first_level_inverse_sse2_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template<RadixType radix_type>
void fft_2d_real_odd_rows_first_level_inverse_avx_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template<RadixType radix_type>
void fft_2d_real_odd_rows_inverse_plain_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template<RadixType radix_type>
void fft_2d_real_odd_rows_inverse_sse2_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template<RadixType radix_type>
void fft_2d_real_odd_rows_inverse_avx_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);



template<RadixType radix_type> void set_instruction_set_odd_rows_2d_f(StepInfoF &step_info, hhfft::InstructionSet instruction_set)
{

#ifdef HHFFT_COMPILED_WITH_AVX512F
    if (instruction_set == hhfft::InstructionSet::avx512f)
    {

    }
#endif

    /*
#ifdef HHFFT_COMPILED_WITH_AVX
    if (instruction_set == hhfft::InstructionSet::avx)
    {
        if (step_info.forward)
        {
            if (step_info.reorder_table != nullptr && step_info.reorder_table2 != nullptr)
                step_info.step_function = fft_2d_real_reorder2_odd_rows_forward_avx_f<radix_type>;
            else
                step_info.step_function = fft_2d_real_odd_rows_forward_avx_f<radix_type>;
        } else
        {
            if (step_info.stride == 1)
                step_info.step_function = fft_2d_real_odd_rows_first_level_inverse_avx_f<radix_type>;
            else
                step_info.step_function = fft_2d_real_odd_rows_inverse_avx_f<radix_type>;
        }
    }
#endif

    if (instruction_set == hhfft::InstructionSet::sse2)
    {
        if (step_info.forward)
        {
            if (step_info.reorder_table != nullptr && step_info.reorder_table2 != nullptr)
                step_info.step_function = fft_2d_real_reorder2_odd_rows_forward_sse2_f<radix_type>;
            else
                step_info.step_function = fft_2d_real_odd_rows_forward_sse2_f<radix_type>;
        } else
        {
            if (step_info.stride == 1)
                step_info.step_function = fft_2d_real_odd_rows_first_level_inverse_sse2_f<radix_type>;
            else
                step_info.step_function = fft_2d_real_odd_rows_inverse_sse2_f<radix_type>;
        }
    }
    */

    if (instruction_set == hhfft::InstructionSet::none)
    {
        if (step_info.forward)
        {
            if (step_info.reorder_table != nullptr && step_info.reorder_table2 != nullptr)
                step_info.step_function = fft_2d_real_reorder2_odd_rows_forward_plain_f<radix_type>;
            else
                step_info.step_function = fft_2d_real_odd_rows_forward_plain_f<radix_type>;
        } else
        {
            if (step_info.stride == 1)
                step_info.step_function = fft_2d_real_odd_rows_first_level_inverse_plain_f<radix_type>;
            else
                step_info.step_function = fft_2d_real_odd_rows_inverse_plain_f<radix_type>;
        }
    }
}

void set_radix_2d_odd_rows_f(StepInfoF &step_info, hhfft::InstructionSet instruction_set)
{
    size_t radix = step_info.radix;

    if (radix > 8)
    {
        set_instruction_set_odd_rows_2d_f<Raders>(step_info, instruction_set);
    }
    else if (radix == 3)
    {
        set_instruction_set_odd_rows_2d_f<Radix3>(step_info, instruction_set);
    } else if (radix == 5)
    {
        set_instruction_set_odd_rows_2d_f<Radix5>(step_info, instruction_set);
    } else if (radix == 7)
    {
        set_instruction_set_odd_rows_2d_f<Radix7>(step_info, instruction_set);
    }
}

namespace hhfft
{
    template<> void HHFFT_2D_Real_odd_set_function_rows<float>(StepInfoF &step_info, hhfft::InstructionSet instruction_set)
    {
        step_info.step_function = nullptr;

        if (step_info.radix != 1)
        {
            set_radix_2d_odd_rows_f(step_info, instruction_set);
        }

        if (step_info.radix == 1 && step_info.forward)
        {
            step_info.step_function = fft_2d_real_fix_first_column_f;
        }


        if (step_info.step_function == nullptr)
        {
            throw(std::runtime_error("HHFFT error: Unable to set a function!"));
        }
    }
}

// Reorder first column and calculate first FFT
template<RadixType radix_type>
void fft_2d_real_odd_rows_reorder_first_column_avx_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template<RadixType radix_type>
void fft_2d_real_odd_rows_reorder_first_column_sse2_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template<RadixType radix_type>
void fft_2d_real_odd_rows_reorder_first_column_plain_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

// Reorder other columns and calculate first FFT
template<RadixType radix_type>
void fft_2d_real_odd_rows_reorder_columns_avx_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template<RadixType radix_type>
void fft_2d_real_odd_rows_reorder_columns_sse2_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);

template<RadixType radix_type>
void fft_2d_real_odd_rows_reorder_columns_plain_f(const float *data_in, float *data_out,const hhfft::StepInfo<float> &step_info);


template<RadixType radix_type> void set_instruction_set_odd_columns_2d_f(StepInfoF &step_info, hhfft::InstructionSet instruction_set)
{

#ifdef HHFFT_COMPILED_WITH_AVX512F
    if (instruction_set == hhfft::InstructionSet::avx512f)
    {

    }
#endif

    /*
#ifdef HHFFT_COMPILED_WITH_AVX
    if (instruction_set == hhfft::InstructionSet::avx)
    {
        if (step_info.data_type_out == hhfft::StepDataType::temp_data)
        {
            step_info.step_function = fft_2d_real_odd_rows_reorder_first_column_avx_f<radix_type>;
        } else
        {
            step_info.step_function = fft_2d_real_odd_rows_reorder_columns_avx_f<radix_type>;
        }
    }
#endif

    if (instruction_set == hhfft::InstructionSet::sse2)
    {
        if (step_info.data_type_out == hhfft::StepDataType::temp_data)
        {
            step_info.step_function = fft_2d_real_odd_rows_reorder_first_column_sse2_f<radix_type>;
        } else
        {
            step_info.step_function = fft_2d_real_odd_rows_reorder_columns_sse2_f<radix_type>;
        }
    }
    */

    if (instruction_set == hhfft::InstructionSet::none)
    {
        if (step_info.data_type_out == hhfft::StepDataType::temp_data)
        {
            step_info.step_function = fft_2d_real_odd_rows_reorder_first_column_plain_f<radix_type>;
        } else
        {
            step_info.step_function = fft_2d_real_odd_rows_reorder_columns_plain_f<radix_type>;
        }        
    }
}

void set_radix_2d_odd_columns_f(StepInfoF &step_info, hhfft::InstructionSet instruction_set)
{
    size_t radix = step_info.radix;

    if (radix > 8)
    {
        set_instruction_set_odd_columns_2d_f<Raders>(step_info, instruction_set);
    } else if (radix == 2)
    {
        set_instruction_set_odd_columns_2d_f<Radix2>(step_info, instruction_set);
    } else if (radix == 3)
    {
        set_instruction_set_odd_columns_2d_f<Radix3>(step_info, instruction_set);
    } else if (radix == 4)
    {
        set_instruction_set_odd_columns_2d_f<Radix4>(step_info, instruction_set);
    } else if (radix == 5)
    {
        set_instruction_set_odd_columns_2d_f<Radix5>(step_info, instruction_set);
    } else if (radix == 6)
    {
        set_instruction_set_odd_columns_2d_f<Radix6>(step_info, instruction_set);
    } else if (radix == 7)
    {
        set_instruction_set_odd_columns_2d_f<Radix7>(step_info, instruction_set);
    } else if (radix == 8)
    {
        set_instruction_set_odd_columns_2d_f<Radix8>(step_info, instruction_set);
    }
}

namespace hhfft
{
    template<> void HHFFT_2D_Real_odd_set_function_columns<float>(StepInfoF &step_info, hhfft::InstructionSet instruction_set)
    {
        step_info.step_function = nullptr;

        if (step_info.radix != 1)
        {
            set_radix_2d_odd_columns_f(step_info, instruction_set);
        }

        if (step_info.radix == 1)
        {
            step_info.step_function = fft_2d_real_odd_rows_combine_columns_f;
        }

        if (step_info.step_function == nullptr)
        {
            throw(std::runtime_error("HHFFT error: Unable to set a function!"));
        }
    }
}
