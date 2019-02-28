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

// This program is a simple demonstration on how to use hhfft and to test that it actully works

#include <iostream>
#include <vector>
#include <random>

#include "hhfft_2d_real.h"

using namespace hhfft;

int main()
{    
    // Print some information about how HHFFT works on your system    
    CPUID_info instructions = get_supported_instructions();

    std::cout << "HHFFT supports SSE2 on your computer: " << (instructions.sse2 ? "YES" : "NO") << std::endl;
    std::cout << "HHFFT supports AVX on your computer: " << (instructions.avx ? "YES" : "NO") << std::endl;

    size_t n = 16, m = 32;

    HHFFT_2D_REAL_D hhfft_2d_real(n, m);

    // Allocate data
    double *x       = hhfft_2d_real.allocate_memory(); // n x m real numbers
    double *x_fft   = hhfft_2d_real.allocate_memory(); // n x (m+2)/2 complex numbers
    double *x_ifft  = hhfft_2d_real.allocate_memory(); // n x m real numbers

    // Initialize x with random numbers
    std::mt19937 gen(12345);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (size_t i = 0; i < n*m; i++) x[i] = dis(gen);

    // do FFT
    hhfft_2d_real.fft(x, x_fft);

    // do IFFT
    hhfft_2d_real.ifft(x_fft, x_ifft);

    // Check that the result is correct
    double max_err = 0.0;
    for (size_t i = 0; i < n*m; i++)
    {
        max_err = std::max(max_err, x[i] - x_ifft[i]);
    }
    std::cout << "Maximum error between x and ifft(fft(x)) = " << max_err << std::endl;
    if (max_err < 1e-15)
    {
        std::cout << "Test passed!" << std::endl;
        return 0;
    } else
    {
        std::cout << "Test fails!" << std::endl;
        return 1;
    }

    // Free data
    hhfft_2d_real.free_memory(x);
    hhfft_2d_real.free_memory(x_fft);
    hhfft_2d_real.free_memory(x_ifft);
}

