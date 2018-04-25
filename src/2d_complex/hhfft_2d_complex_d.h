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

#ifndef HHFFT_2D_COMPLEX_D_H
#define HHFFT_2D_COMPLEX_D_H

#include "step_info.h"
#include "architecture.h"

namespace hhfft
{

// For column-wise operations
void HHFFT_2D_Complex_D_set_function_columns(StepInfoD &step_info, hhfft::InstructionSet instruction_set);

// For row-wise operations
void HHFFT_2D_Complex_D_set_function_rows(StepInfoD &step_info, hhfft::InstructionSet instruction_set);


}

#endif // HHFFT_2D_COMPLEX_H
