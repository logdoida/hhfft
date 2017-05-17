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

#include <cpuid.h>

hhfft::CPUID_info hhfft::get_supported_instructions()
{
    // TODO this function checks what CPU supports, but in addition there must also be OS support!

    hhfft::CPUID_info si = {};

// Currently support only for gcc
#ifdef __GNUC__

    unsigned int max_basic_level = __get_cpuid_max(0x0, nullptr);

    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (max_basic_level >= 1)
    {
        __get_cpuid(1, &eax, &ebx, &ecx, &edx);

        //si.sse        = (edx & (1 << 25)) != 0; // Not needed
        si.sse2       = (edx & (1 << 26)) != 0;
        //si.sse3       = (ecx & (1 << 0 )) != 0;
        //si.fma        = (ecx & (1 << 12)) != 0;
        //si.sse4_1     = (ecx & (1 << 19)) != 0;
        //si.sse4_2     = (ecx & (1 << 20)) != 0;
        //si.osxsave    = (ecx & (1 << 27)) != 0; // XSAVE enabled by OS (Important!)
        si.avx        = (ecx & (1 << 28)) != 0;

        /*
        std::cout << "sse: " << si.sse << std::endl;
        std::cout << "sse2: " << si.sse2 << std::endl;
        std::cout << "sse3: " << si.sse3 << std::endl;
        std::cout << "fma: " << si.fma << std::endl;
        std::cout << "sse4_1: " << si.sse4_1 << std::endl;
        std::cout << "sse4_2: " << si.sse4_2 << std::endl;
        std::cout << "avx: " << si.avx << std::endl;
        std::cout << "osxsave: " << si.osxsave << std::endl;
        */
    }

    if (max_basic_level >= 7)
    {
        __get_cpuid(7, &eax, &ebx, &ecx, &edx);

        //si.avx2       = (ebx & (1 << 5)) != 0;
        si.avx512f    = (ebx & (1 << 16)) != 0;
        //si.avx512ifma = (ebx & (1 << 21)) != 0;

        /*
        std::cout << "avx2 " << si.avx2 << std::endl;
        std::cout << "avx512f: " << si.avx512f << std::endl;
        std::cout << "avx512ifma: " << si.avx512ifma << std::endl;'
        */
    }

 #endif

    return si;
}
