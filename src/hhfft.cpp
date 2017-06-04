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

#include "hhfft.h"
#include "architecture.h"
#include "hhfft_base.h"
#include "hhfft_avx_d.h"
#include "hhfft_avx512f_d.h"
#include "hhfft_plain_d.h"
#include "hhfft_plain_avx_d.h"
#include "hhfft_plain_avx512f_d.h"


using hhfft::HHFFT_D;

HHFFT_D::HHFFT_D(size_t n, size_t m)
{
    // Select version based on what the compiled library and what the system actully supports
    hhfft = nullptr;

    hhfft::CPUID_info info = hhfft::get_supported_instructions();

#ifdef HHFFT_COMPILED_WITH_AVX512F
    if (info.avx512f && !hhfft)
    {
        hhfft = new HHFFT_Plain_AVX512F_D(n,m);
    }
#endif


#ifdef HHFFT_COMPILED_WITH_AVX
    if (info.avx && !hhfft)
    {        
        hhfft = new HHFFT_AVX_D(n,m); // Hand optimized version
    }
#endif

    // Use the basic version (sse2 on 64bit systems) as a backup
    if (!hhfft)
    {
        hhfft = new HHFFT_Plain_D(n,m);
    }
}

HHFFT_D::~HHFFT_D()
{
    delete hhfft;
}

void HHFFT_D::fft_real(const double *in, double *out)
{
    hhfft->fft_real(in, out);
}

void HHFFT_D::ifft_real(const double *in, double *out)
{
    hhfft->ifft_real(in, out);
}

void HHFFT_D::convolution_real(const double *in1, const double *in2, double *out)
{
    hhfft->convolution_real(in1, in2, out);
}

void HHFFT_D::convolution_real_add(const double *in1, const double *in2, double *out)
{
    hhfft->convolution_real_add(in1, in2, out);
}

bool HHFFT_D::avx_support_on()
{
#ifdef HHFFT_COMPILED_WITH_AVX
    hhfft::CPUID_info info = hhfft::get_supported_instructions();
    return info.avx;
#else
    return false;
#endif
}

bool HHFFT_D::avx512f_support_on()
{
#ifdef HHFFT_COMPILED_WITH_AVX512F
    hhfft::CPUID_info info = hhfft::get_supported_instructions();
    return info.avx512f;
#else
    return false;
#endif
}
