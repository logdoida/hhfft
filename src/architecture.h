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

#ifndef HHFFT_ARCHITECTURE
#define HHFFT_ARCHITECTURE

namespace hhfft
{

struct CPUID_info
{
    bool sse2 = false;
    bool avx = false;
    bool avx512f = false;
};

enum InstructionSet{avx512f, avx, sse2, none, automatic};

// This function is used to determine what versions to actually use at run-time
CPUID_info get_supported_instructions();

// Returns the best available instruction set
InstructionSet get_best_instruction_set();

}

# endif
