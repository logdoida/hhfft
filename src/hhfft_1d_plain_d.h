/*
*   Copyright Jouko Kalmari 2017
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

#ifndef HHFFT_1D_PLAIN_D_H
#define HHFFT_1D_PLAIN_D_H

#include "step_info.h"

namespace hhfft
{

// This set pointer to correct fft functions based on radix and stride (DIT version)
void HHFFT_1D_Plain_set_function(StepInfoD &step_info);

// DIF version
void HHFFT_1D_Plain_set_function_DIF(StepInfoD &step_info);

}

#endif // HHFFT_1D_PLAIN_H
