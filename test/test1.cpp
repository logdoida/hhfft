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

// This program is a simple demonstration on how to use hhfft and to test that it actully works

#include <iostream>
#include <vector>
#include <random>

#include "hhfft.h"

using namespace hhfft;

int main()
{    
    // Print some information about how HHFFT works on your system
    std::cout << "HHFFT supports AVX512f on your computer: " << (HHFFT_D::avx512f_support_on() ? "YES" : "NO") << std::endl;
    std::cout << "HHFFT supports AVX on your computer: " << (HHFFT_D::avx_support_on() ? "YES" : "NO") << std::endl;

    size_t n = 16, m = 32;

    HHFFT_D hhfft(n, m);

    // Initialize data
    std::vector<double> x(n*m); // Real input
    std::vector<double> x_fft((n+2)*m); // Output from fft
    std::vector<double> x_ifft((n+2)*m); // Output from ifft

    // Initialize x with random numbers
    std::mt19937 gen(12345);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (size_t i = 0; i < n*m; i++) x[i] = dis(gen);

    // do FFT
    hhfft.fft_real(x.data(), x_fft.data());

    // do IFFT
    hhfft.ifft_real(x_fft.data(), x_ifft.data());

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
}

