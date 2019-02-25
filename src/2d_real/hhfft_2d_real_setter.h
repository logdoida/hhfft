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

#ifndef HHFFT_2D_REAL_SETTER_H
#define HHFFT_2D_REAL_SETTER_H

#include "step_info.h"
#include "architecture.h"

namespace hhfft
{

// Sets pointer to correct function
template<typename T> void HHFFT_2D_Real_set_reorder_column_function(StepInfo<T> &step_info);

// Sets pointer to correct function
template<typename T> void HHFFT_2D_Real_set_complex_to_complex_packed_function(StepInfo<T> &step_info, hhfft::InstructionSet instruction_set);

// Sets pointer to correct function
template<typename T> void HHFFT_2D_Real_set_complex_to_complex_packed_first_column_function(StepInfo<T> &step_info, hhfft::InstructionSet instruction_set);

// Sets pointer to correct function
template<typename T> void HHFFT_2D_Real_set_function(StepInfo<T> &step_info, hhfft::InstructionSet instruction_set);

// Sets pointer to correct function when row size is odd
template<typename T> void HHFFT_2D_Real_odd_set_function_rows(StepInfo<T> &step_info, hhfft::InstructionSet instruction_set);

// Sets pointer to correct function when row size is odd
template<typename T> void HHFFT_2D_Real_odd_set_function_columns(StepInfo<T> &step_info, hhfft::InstructionSet instruction_set);

}

#endif // HHFFT_2D_REAL_SETTER_H
