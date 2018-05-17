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

#ifndef HHFFT_1D_COMPLEX_D_H
#define HHFFT_1D_COMPLEX_D_H

#include "step_info.h"
#include "architecture.h"

namespace hhfft
{

// This sets pointer to correct fft functions based on radix and stride etc
void HHFFT_1D_Complex_D_set_function(StepInfoD &step_info, hhfft::InstructionSet instruction_set);

// This returns a pointer to correct convolution function based on instruction set
void (*HHFFT_1D_Complex_D_set_convolution_function(hhfft::InstructionSet instruction_set))(const double *, const double *, double *, size_t);

// This sets pointer to a one level fft/ifft function if such exists
void HHFFT_1D_Complex_D_set_small_function(StepInfoD &step_info, size_t n, bool forward, hhfft::InstructionSet instruction_set);

}

#endif // HHFFT_1D_COMPLEX_H
