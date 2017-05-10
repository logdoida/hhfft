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

#ifndef HHFFT_ARCHITECTURE
#define HHFFT_ARCHITECTURE

// These macros are used to determine what is supported in single source file
// NOTE these are not same as HHFFT_COMPILED_WITH_AVX etc that tell what versions are compiled for the whole project
#ifdef __SSE2__
#define HHFFT_SSE2
#endif

#ifdef __AVX__
#define HHFFT_AVX
#endif

#ifdef __AVX512F__
#define HHFFT_AVX512F
#endif

namespace hhfft
{

struct CPUID_info
{
    bool sse2 = false;
    bool avx = false;
    bool avx512f = false;
};

// This function is used to determine what versions to actually use at run-time
CPUID_info get_supported_instructions();

}

# endif
