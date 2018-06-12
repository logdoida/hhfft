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

#ifndef HHFFT_1D_REAL_SETTER_D_H
#define HHFFT_1D_REAL_SETTER_D_H

#include "step_info.h"
#include "architecture.h"

namespace hhfft
{

// Sets pointer to correct function
void HHFFT_1D_Real_D_set_complex_to_complex_packed_function(StepInfoD &step_info, hhfft::InstructionSet instruction_set);

// Sets pointer to correct odd function
void HHFFT_1D_Real_D_odd_set_function(StepInfoD &step_info, hhfft::InstructionSet instruction_set);

// This sets pointer to a one level fft/ifft function if such exists
void HHFFT_1D_Real_D_set_small_function(StepInfoD &step_info, size_t n, bool forward, hhfft::InstructionSet instruction_set);

}

#endif // HHFFT_1D_REAL_SETTER_D_H
