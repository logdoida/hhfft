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

// This program can be used to measure the performance of hhfft

#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <ctime>
#include <cmath>

#include "hhfft_2d_d.h"

using namespace hhfft;

std::clock_t cpu_clock_start;

void start_clock()
{
    // Start the timers to calculate how long whole processing takes
    cpu_clock_start = std::clock();
}

double get_clock()
{
    return (std::clock() - cpu_clock_start) / (double) CLOCKS_PER_SEC;
}

double calc_mflops_real(double time, double n)
{
    return 2.5*n*log2(n)/time/1.0e6;
}

double calc_mflops_complex(double time, double n)
{
    return 5.0*n*log2(n)/time/1.0e6;
}

void measure_perfomance_one_size(size_t n, size_t m)
{
    // Create HHFFT object
    hhfft::HHFFT_2D_D hhfft_2d(n, m);

    // Real input
    double *x = hhfft_2d.allocate_memory();

    // Output from fft
    double *x_fft = hhfft_2d.allocate_memory();

    // Output from ifft
    double *x_ifft = hhfft_2d.allocate_memory();

    // Initialize x with random numbers
    std::mt19937 gen(123);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (size_t i = 0; i < n*m; i++) x[i] = dis(gen);

    // Choose the number of repeats so that each test takes time in the order of one second
    size_t repeats = (size_t) (100e6/(n*m*log2(n*m))) + 10;

    // Do first some warm-up (it takes some time for the SIMD-unit on a CPU start)
    for (size_t i = 0; i < repeats*10; i++)
    {
        hhfft_2d.fft(x, x_fft);
    }

    // Do a number of runs to get the best time
    double time_fft = 1e10, time_ifft = 1e10; // Fastest time

    for (size_t iter = 0; iter < 5; iter++)
    {
        start_clock();
        for (size_t i = 0; i < repeats; i++)
        {
            hhfft_2d.fft(x, x_fft);
        }
        time_fft = std::min(time_fft, get_clock()/repeats);

        start_clock();
        for (size_t i = 0; i < repeats; i++)
        {
            hhfft_2d.ifft(x_fft, x_ifft);
        }
        time_ifft = std::min(time_ifft, get_clock()/repeats);
    }

    std::cout << "n = " << n << ", m = " << m << std::endl;
    std::cout << "FFT:      " << time_fft  << " s, " << calc_mflops_complex(time_fft, n*m)  << " MFLOPS" << std::endl;
    std::cout << "IFFT:     " << time_ifft << " s, " << calc_mflops_complex(time_ifft, n*m) << " MFLOPS" << std::endl;
    std::cout << std::endl;

    hhfft_2d.free_memory(x);
    hhfft_2d.free_memory(x_fft);
    hhfft_2d.free_memory(x_ifft);
}

int main()
{
    std::vector<std::array<size_t,2>> sizes
            = {{8,8}, {8, 16}, {16, 16}, {16, 32}, {32, 32}, {32, 64}, {64, 64}, {64, 128}, {128, 128}, {128, 256}, {256,256}};

    std::cout << "HHFFT performance for 2D complex FFT transformations" << std::endl;

    for (auto s: sizes)
    {
        measure_perfomance_one_size(s[0], s[1]);
    }


    return 0;
}
