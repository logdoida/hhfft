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

#include "../1d_complex/hhfft_1d_complex_setter.h"
#include "hhfft_2d_complex_setter.h"
#include <stdexcept>
#include <assert.h>
#include <cmath>

using namespace hhfft;

// Actual implementations are in different .cpp-files
template<RadixType radix_type, SizeType size_type>
    void fft_2d_complex_column_twiddle_dit_plain_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template<RadixType radix_type, SizeType size_type>
    void fft_2d_complex_column_twiddle_dit_sse2_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template<RadixType radix_type, SizeType size_type>
    void fft_2d_complex_column_twiddle_dit_avx_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template<RadixType radix_type, SizeType size_type>
    void fft_2d_complex_column_twiddle_dit_avx512_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);


// Reorder both rows and columns and do FFT
template<RadixType radix_type, bool forward>
void fft_2d_complex_reorder2_plain_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template<RadixType radix_type, bool forward>
void fft_2d_complex_reorder2_sse2_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template<RadixType radix_type, bool forward>
void fft_2d_complex_reorder2_avx_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);


//////////////////////// column-wise ////////////////////////////////////

template<RadixType radix_type, SizeType size_type, bool forward> void set_instruction_set_columns_2d_d(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{

#ifdef HHFFT_COMPILED_WITH_AVX512F
    if (instruction_set == hhfft::InstructionSet::avx512f)
    {

    }
#endif

#ifdef HHFFT_COMPILED_WITH_AVX
    if (instruction_set == hhfft::InstructionSet::avx)
    {
        if (step_info.reorder_table == nullptr && step_info.reorder_table2 == nullptr && step_info.twiddle_factors != nullptr)
            step_info.step_function = fft_2d_complex_column_twiddle_dit_avx_d<radix_type, size_type>;
        if (step_info.reorder_table != nullptr && step_info.reorder_table2 != nullptr && step_info.twiddle_factors == nullptr)
        {            
            step_info.step_function = fft_2d_complex_reorder2_avx_d<radix_type, forward>;
        }
    }
#endif

    if (instruction_set == hhfft::InstructionSet::sse2)
    {
        if (step_info.reorder_table == nullptr && step_info.reorder_table2 == nullptr && step_info.twiddle_factors != nullptr)
            step_info.step_function = fft_2d_complex_column_twiddle_dit_sse2_d<radix_type, size_type>;
        if (step_info.reorder_table != nullptr && step_info.reorder_table2 != nullptr && step_info.twiddle_factors == nullptr)
        {            
            step_info.step_function = fft_2d_complex_reorder2_sse2_d<radix_type, forward>;
        }
    }  

    if (instruction_set == hhfft::InstructionSet::none)
    {
        if (step_info.reorder_table == nullptr && step_info.reorder_table2 == nullptr && step_info.twiddle_factors != nullptr)
            step_info.step_function = fft_2d_complex_column_twiddle_dit_plain_d<radix_type, size_type>;
        if (step_info.reorder_table != nullptr && step_info.reorder_table2 != nullptr && step_info.twiddle_factors == nullptr)
        {            
            step_info.step_function = fft_2d_complex_reorder2_plain_d<radix_type, forward>;
        }
    }
}

// These functions set different template parameters one at time
template<RadixType radix_type, SizeType size_type> void set_forward_2d_columns_d(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{
    if (step_info.forward)
    {
        set_instruction_set_columns_2d_d<radix_type, size_type, true>(step_info, instruction_set);
    } else
    {
        set_instruction_set_columns_2d_d<radix_type, size_type, false>(step_info, instruction_set);
    }
}

template<RadixType radix_type> void set_stride_type_2d_columns_d(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{
    // Knowing something about the row length at compile time can help to optimize some cases
    // NOTE for row-wise the stride is used instead!
    size_t n_size = step_info.size;

    // Only stride = 1 and N are currently used
    if (n_size == 1)
    {
        set_forward_2d_columns_d<radix_type, SizeType::Size1>(step_info, instruction_set);
    } else
    {
        set_forward_2d_columns_d<radix_type, SizeType::SizeN>(step_info, instruction_set);
    }

    /*
    if (n_size == 1)
    {
        set_forward_2d_d<radix, SizeType::Sizee1>(step_info, instruction_set);
    } else if (n_size == 2)
    {
        set_forward_2d_d<radix, SizeType::Size2>(step_info, instruction_set);
    } else if (n_size == 4)
    {
        set_forward_2d_d<radix, SizeType::Size4>(step_info, instruction_set);
    } else if (n_size%4 == 0)
    {
        set_forward_2d_d<radix, SizeType::Size4N>(step_info, instruction_set);
    } else if (n_size%2 == 0)
    {
        set_forward_2d_d<radix, SizeType::Size2N>(step_info, instruction_set);
    } else
    {
        set_forward_2d_d<radix, SizeType::SizeeN>(step_info, instruction_set);
    }
    */
}

void set_radix_2d_colums_d(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{
    size_t radix = step_info.radix;

    if (radix == 2)
    {
        set_stride_type_2d_columns_d<Radix2>(step_info, instruction_set);
    } else if (radix == 3)
    {
        set_stride_type_2d_columns_d<Radix3>(step_info, instruction_set);
    } else if (radix == 4)
    {
        set_stride_type_2d_columns_d<Radix4>(step_info, instruction_set);
    } else if (radix == 5)
    {
        set_stride_type_2d_columns_d<Radix5>(step_info, instruction_set);
    } else if (radix == 6)
    {
        set_stride_type_2d_columns_d<Radix6>(step_info, instruction_set);
    } else if (radix == 7)
    {
        set_stride_type_2d_columns_d<Radix7>(step_info, instruction_set);
    } else if (radix == 8)
    {
       set_stride_type_2d_columns_d<Radix8>(step_info, instruction_set);
    } else if (radix == 11)
    {
        set_stride_type_2d_columns_d<Radix11>(step_info, instruction_set);
    } else if (radix == 13)
    {
        set_stride_type_2d_columns_d<Radix13>(step_info, instruction_set);
    } else if (radix > 13)
    {
        set_stride_type_2d_columns_d<Raders>(step_info, instruction_set);
    }
}

namespace hhfft
{
    // This set pointer to correct fft functions based on radix and stride etc
    // NOTE currently only reordering steps are needed
    template<> void HHFFT_2D_Complex_set_function_columns<double>(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
    {
        step_info.step_function = nullptr;

        if (step_info.radix != 1)
        {
            if (step_info.twiddle_factors == nullptr
                    && step_info.reorder_table == nullptr && step_info.reorder_table_inplace == nullptr
                    && step_info.reorder_table2 == nullptr && step_info.reorder_table2_inplace == nullptr)
            {
                // 1D FFT is used here instead!
                HHFFT_1D_Complex_set_function<double>(step_info, instruction_set);
            } else
            {
                set_radix_2d_colums_d(step_info, instruction_set);
            }
        }

        if (step_info.step_function == nullptr)
        {
            throw(std::runtime_error("HHFFT error: Unable to set a function!"));
        }
    }
}

///////////////////////// row-wise ////////////////////////////////////


// Reorder both rows and columns and do FFT on rows
template<RadixType radix_type>
void fft_2d_complex_reorder2_rows_forward_avx_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template<RadixType radix_type>
void fft_2d_complex_reorder2_rows_forward_sse2_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);

template<RadixType radix_type>
void fft_2d_complex_reorder2_rows_forward_plain_d(const double *data_in, double *data_out,const hhfft::StepInfo<double> &step_info);


template<RadixType radix_type> void set_instruction_set_rows_2d_d(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{

#ifdef HHFFT_COMPILED_WITH_AVX512F
    if (instruction_set == hhfft::InstructionSet::avx512f)
    {

    }
#endif

#ifdef HHFFT_COMPILED_WITH_AVX
    if (instruction_set == hhfft::InstructionSet::avx)
    {
        if (step_info.forward)
        {
            step_info.step_function = fft_2d_complex_reorder2_rows_forward_avx_d<radix_type>;
        } else
        {
            throw(std::runtime_error("HHFFT error: Unable to set a function!"));
        }
    }
#endif

    if (instruction_set == hhfft::InstructionSet::sse2)
    {
        if (step_info.forward)
        {
            step_info.step_function = fft_2d_complex_reorder2_rows_forward_sse2_d<radix_type>;
        } else
        {
            throw(std::runtime_error("HHFFT error: Unable to set a function!"));
        }
    }  

    if (instruction_set == hhfft::InstructionSet::none)
    {
        if (step_info.forward)
        {
            step_info.step_function = fft_2d_complex_reorder2_rows_forward_plain_d<radix_type>;
        } else
        {
            throw(std::runtime_error("HHFFT error: Unable to set a function!"));
        }
    }
}


void set_radix_2d_rows_d(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{
    size_t radix = step_info.radix;

    if (radix == 2)
    {
        set_instruction_set_rows_2d_d<Radix2>(step_info, instruction_set);
    } else if (radix == 3)
    {
        set_instruction_set_rows_2d_d<Radix3>(step_info, instruction_set);
    } else if (radix == 4)
    {
        set_instruction_set_rows_2d_d<Radix4>(step_info, instruction_set);
    } else if (radix == 5)
    {
        set_instruction_set_rows_2d_d<Radix5>(step_info, instruction_set);
    } else if (radix == 6)
    {
        set_instruction_set_rows_2d_d<Radix6>(step_info, instruction_set);
    } else if (radix == 7)
    {
        set_instruction_set_rows_2d_d<Radix7>(step_info, instruction_set);
    } else if (radix == 8)
    {
       set_instruction_set_rows_2d_d<Radix8>(step_info, instruction_set);
    }  else if (radix == 11)
    {
       set_instruction_set_rows_2d_d<Radix11>(step_info, instruction_set);
    } else if (radix == 13)
    {
       set_instruction_set_rows_2d_d<Radix13>(step_info, instruction_set);
    } else if (radix > 13)
    {
        set_instruction_set_rows_2d_d<Raders>(step_info, instruction_set);
    }
}

namespace hhfft
{
    template<> void HHFFT_2D_Complex_set_function_rows<double>(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
    {
        step_info.step_function = nullptr;

        if (step_info.reorder_table != nullptr || step_info.reorder_table_inplace != nullptr)
        {
            if (step_info.radix != 1)
            {
                set_radix_2d_rows_d(step_info, instruction_set);
                return;
            }
        }

        // 1D FFT is used here instead!
        HHFFT_1D_Complex_set_function<double>(step_info, instruction_set);

        if (step_info.step_function == nullptr)
        {
            throw(std::runtime_error("HHFFT error: Unable to set a function!"));
        }
    }
}

