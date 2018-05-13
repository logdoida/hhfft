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

#include "../1d_complex/hhfft_1d_complex_d.h"
#include "hhfft_2d_complex_d.h"
#include <stdexcept>
#include <assert.h>
#include <cmath>

#include <iostream> //TESTING

using namespace hhfft;

// Compiler is not able to optimize this to use sse2! TODO implement different versions (plain/sse2/avx) of this!
// In-place reordering "swap"
template<bool scale> void fft_2d_complex_reorder_rows_in_place_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    size_t n = step_info.size; // number of rows
    size_t m = step_info.stride; // number of columns
    size_t reorder_table_size = step_info.repeats;
    uint32_t *reorder_table = step_info.reorder_table_inplace;

    // In-place algorithm
    assert (data_in == data_out);

    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < reorder_table_size; j++)
        {
            size_t ind1 = j + 1; // First one has been omitted!
            size_t ind2 = reorder_table[j];

            // Swap two doubles at a time
            /*
            __m128d temp1 = _mm_loadu_pd(data_in + 2*i*m + 2*ind1);
            __m128d temp2 = _mm_loadu_pd(data_in + 2*i*m + 2*ind2);
            _mm_storeu_pd(data_out + 2*i*m + 2*ind2, temp1);
            _mm_storeu_pd(data_out + 2*i*m + 2*ind1, temp2);
            */

            double r_temp = data_out[2*i*m + 2*ind1+0];
            double c_temp = data_out[2*i*m + 2*ind1+1];
            data_out[2*i*m + 2*ind1+0] = data_out[2*i*m + 2*ind2+0];
            data_out[2*i*m + 2*ind1+1] = data_out[2*i*m + 2*ind2+1];
            data_out[2*i*m + 2*ind2+0] = r_temp;
            data_out[2*i*m + 2*ind2+1] = c_temp;
        }
    }

    // Scaling can be needed for real-fft
    double k = step_info.norm_factor;
    if (scale)
    {
        // Needed only in ifft. Equal to 1/N
        double k = step_info.norm_factor;

        for (size_t i = 0; i < 2*m*n; i++)
        {
            data_out[i] *= k;
        }
    }    
}

// TODO implement different versions (plain/sse2/avx) of this?
template<bool scale> void fft_2d_complex_reorder_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    // Check, if in-place should be done instead
    if (data_in == data_out)
    {
        fft_2d_complex_reorder_rows_in_place_d<scale>(data_in, data_out, step_info);
        return;
    }

    size_t n = step_info.size; // number of rows
    size_t m = step_info.stride; // number of columns
    uint32_t *reorder_table = step_info.reorder_table;

    // Needed only in ifft. Equal to 1/(n*m)
    double k = step_info.norm_factor;    

    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < m; j++)
        {
            size_t j2 = reorder_table[j];
            if (scale)
            {
                data_out[2*i*m + 2*j + 0] = k*data_in[2*i*m + 2*j2 + 0];
                data_out[2*i*m + 2*j + 1] = k*data_in[2*i*m + 2*j2 + 1];
            } else {
                data_out[2*i*m + 2*j + 0] = data_in[2*i*m + 2*j2 + 0];
                data_out[2*i*m + 2*j + 1] = data_in[2*i*m + 2*j2 + 1];
            }
        }
    }
}

// TODO implement different versions (plain/sse2/avx) of this?
// In-place reordering "swap"
template<bool scale> void fft_2d_complex_reorder_columns_in_place_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{
    size_t n = step_info.repeats;
    size_t m = step_info.size;
    uint32_t *reorder_table = step_info.reorder_table_inplace;

    // In-place algorithm
    assert (data_in == data_out);

    for (size_t i = 0; i < n; i++)
    {
        size_t ind1 = i + 1; // First one has been omitted!
        size_t ind2 = reorder_table[i];

        for (size_t j = 0; j < 2*m; j++)
        {
            double temp = data_out[2*ind1*m + j];
            data_out[2*ind1*m + j] = data_out[2*ind2*m + j];
            data_out[2*ind2*m + j] = temp;
        }
    }

    // Scaling needs to be done as a separate step as some data might be copied twice or zero times
    // TODO this is not very efficient. Scaling could be done at some other step (first/last)
    // Scaling needs to be done as a separate step as some data might be copied twice or zero times        
    size_t n2 = step_info.stride;
    double k = step_info.norm_factor;
    if (scale)
    {        
        for (size_t i = 0; i < 2*n2*m; i++)
        {
            data_out[i] *= k;
        }
    }    
}

// TODO implement different versions (plain/sse2/avx) of this?
template<bool scale> void fft_2d_complex_reorder_columns_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info)
{    
    // Check, if in-place should be done instead
    if (data_in == data_out)
    {
        fft_2d_complex_reorder_columns_in_place_d<scale>(data_in, data_out, step_info);
        return;
    }

    size_t n = step_info.stride;
    size_t m = step_info.size;
    uint32_t *reorder_table = step_info.reorder_table;

    // Needed only in ifft. Equal to 1/N
    double k = step_info.norm_factor;    

    for (size_t i = 0; i < n; i++)
    {
        size_t i2 = reorder_table[i];

        for (size_t j = 0; j < 2*m; j++)
        {
            if (scale)
                data_out[2*i*m + j] = k*data_in[2*i2*m + j];
            else
                data_out[2*i*m + j] = data_in[2*i2*m + j];
        }
    }
}

// Actual implementations are in different .cpp-files

// DIT, column-wise
template<size_t radix, bool forward>
    void fft_2d_complex_column_twiddle_dit_plain_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template<size_t radix, SizeType size_type, bool forward>
    void fft_2d_complex_column_twiddle_dit_sse2_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template<size_t radix, SizeType size_type, bool forward>
    void fft_2d_complex_column_twiddle_dit_avx_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

template<size_t radix, SizeType size_type, bool forward>
    void fft_2d_complex_column_twiddle_dit_avx512_d(const double *data_in, double *data_out, hhfft::StepInfo<double> &step_info);

// TODO DIF not implemented yet!


//////////////////////// column-wise ////////////////////////////////////

template<size_t radix, SizeType size_type, bool forward> void set_instruction_set_columns_2d_d(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{

#ifdef HHFFT_COMPILED_WITH_AVX512F
    if (instruction_set == hhfft::InstructionSet::avx512f)
    {

    }
#endif

#ifdef HHFFT_COMPILED_WITH_AVX
    if (instruction_set == hhfft::InstructionSet::avx)
    {
        if (step_info.dif)
        {
            //step_info.step_function = fft_2d_complex_column_twiddle_dif_avx_d<radix, size_type, forward>;
        } else
        {
            step_info.step_function = fft_2d_complex_column_twiddle_dit_avx_d<radix, size_type, forward>;
        }
    }
#endif

    if (instruction_set == hhfft::InstructionSet::sse2)
    {
        if (step_info.dif)
        {
                //step_info.step_function = fft_2d_complex_column_twiddle_dif_sse2_d<radix, size_type, forward>;
        } else
        {
            step_info.step_function = fft_2d_complex_column_twiddle_dit_sse2_d<radix, size_type, forward>;
        }
    }

    if (instruction_set == hhfft::InstructionSet::none)
    {
        if (step_info.dif)
        {
                //step_info.step_function = fft_2d_complex_column_twiddle_dif_plain_d<radix, forward>;
        } else
        {
            step_info.step_function = fft_2d_complex_column_twiddle_dit_plain_d<radix, forward>;
        }
    }
}

// These functions set different template parameters one at time
template<size_t radix, SizeType size_type> void set_forward_2d_columns_d(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{
    if (step_info.forward)
    {
        set_instruction_set_columns_2d_d<radix, size_type, true>(step_info, instruction_set);
    } else
    {
        set_instruction_set_columns_2d_d<radix, size_type, false>(step_info, instruction_set);
    }
}

template<size_t radix> void set_stride_type_2d_columns_d(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{
    // Knowing something about the row length at compile time can help to optimize some cases
    // NOTE for row-wise the stride is used instead!
    size_t n_size = step_info.size;

    // Only stride = 1 and N are currently used
    if (n_size == 1)
    {
        set_forward_2d_columns_d<radix, SizeType::Size1>(step_info, instruction_set);
    } else
    {
        set_forward_2d_columns_d<radix, SizeType::SizeN>(step_info, instruction_set);
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
        set_stride_type_2d_columns_d<2>(step_info, instruction_set);
    } if (radix == 3)
    {
        set_stride_type_2d_columns_d<3>(step_info, instruction_set);
    } if (radix == 4)
    {
        set_stride_type_2d_columns_d<4>(step_info, instruction_set);
    } if (radix == 5)
    {
        set_stride_type_2d_columns_d<5>(step_info, instruction_set);
    } if (radix == 7)
    {
        set_stride_type_2d_columns_d<7>(step_info, instruction_set);
    }
}

// This set pointer to correct fft functions based on radix and stride etc
// NOTE currently only reordering steps are needed
void hhfft::HHFFT_2D_Complex_D_set_function_columns(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{
    step_info.step_function = nullptr;

    if (step_info.reorder_table != nullptr || step_info.reorder_table_inplace != nullptr)
    {    
        if (step_info.norm_factor != 1.0)
            step_info.step_function = fft_2d_complex_reorder_columns_d<true>;
        else
            step_info.step_function = fft_2d_complex_reorder_columns_d<false>;

        return;
    }

    if (step_info.twiddle_factors == nullptr)
    {
        // 1D FFT is used here instead!
        HHFFT_1D_Complex_D_set_function(step_info, instruction_set);
    } else
    {
        set_radix_2d_colums_d(step_info, instruction_set);
    }

    if (step_info.step_function == nullptr)
    {
        throw(std::runtime_error("HHFFT error: Unable to set a function!"));
    }
}

///////////////////////// row-wise ////////////////////////////////////

void hhfft::HHFFT_2D_Complex_D_set_function_rows(StepInfoD &step_info, hhfft::InstructionSet instruction_set)
{  
    step_info.step_function = nullptr;

    if (step_info.reorder_table != nullptr || step_info.reorder_table_inplace != nullptr)
    {        
        if (step_info.norm_factor != 1.0)
            step_info.step_function = fft_2d_complex_reorder_d<true>;
        else
            step_info.step_function = fft_2d_complex_reorder_d<false>;

        return;
    }

    if (step_info.dif)
    {
        throw(std::runtime_error("HHFFT error: 2D DIF not supported yet!"));
    }

    if (step_info.twiddle_factors == nullptr)
    {
        // 1D FFT is used here instead!
        HHFFT_1D_Complex_D_set_function(step_info, instruction_set);
    } else
    {
        // 1D FFT is used here instead!
        HHFFT_1D_Complex_D_set_function(step_info, instruction_set);
    }

    if (step_info.step_function == nullptr)
    {
        throw(std::runtime_error("HHFFT error: Unable to set a function!"));
    }
}

