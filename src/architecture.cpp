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

#include "architecture.h"

bool sse2_supported()
{
#ifdef __GNUC__
    return __builtin_cpu_supports("sse2");
#else
    return false; // Sorry, only GCC (and clang) supported
#endif
}

bool avx_supported()
{
#ifdef __GNUC__
    return __builtin_cpu_supports("avx");
#else
    return false; // Sorry, only GCC (and clang) supported
#endif
}

bool avx512f_supported()
{
#ifdef __GNUC__
    return __builtin_cpu_supports("avx512f");
#else
    return false; // Sorry, only GCC (and clang) supported
#endif
}
