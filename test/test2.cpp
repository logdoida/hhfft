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

// This program can be used to measure the performance of hhfft

#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <ctime>
#include <cmath>

#include "hhfft.h"

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

void measure_perfomance_one_size(size_t n, size_t m)
{
    // Real input
    std::vector<double> x(n*m);

    // Output from fft
    std::vector<double> x_fft((n+2)*m);

    // Output from ifft
    std::vector<double> x_ifft((n+2)*m);

    // Initialize x with random numbers
    std::mt19937 gen(123);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (size_t i = 0; i < n*m; i++) x[i] = dis(gen);

    HHFFT_D hhfft(n, m);

    //size_t repeats = 1000;
    // Choose the number of repeats so that each test takes time in the order of one second
    size_t repeats = (size_t) (100e6/(n*m*log2(n*m))) + 10;

    // Do first some warm-up (it takes some time for the SIMD-unit on a CPU start)
    for (size_t i = 0; i < repeats*10; i++)
    {
        hhfft.fft_real(x.data(),x_fft.data());
    }

    // Do a number of runs to get the best time
    double time_fft = 1e10, time_ifft = 1e10; // Fastest time

    for (size_t iter = 0; iter < 5; iter++)
    {
        start_clock();
        for (size_t i = 0; i < repeats; i++)
        {
            hhfft.fft_real(x.data(),x_fft.data());
        }
        time_fft = std::min(time_fft, get_clock()/repeats);

        start_clock();
        for (size_t i = 0; i < repeats; i++)
        {
            hhfft.ifft_real(x_fft.data(),x_ifft.data());
        }
        time_ifft = std::min(time_ifft, get_clock()/repeats);
    }

    std::cout << "n = " << n << ", m = " << m << std::endl;
    std::cout << "fft_real():      " << time_fft  << " s, " << calc_mflops_real(time_fft, n*m)  << " MFLOPS" << std::endl;
    std::cout << "ifft_real():     " << time_ifft << " s, " << calc_mflops_real(time_ifft, n*m) << " MFLOPS" << std::endl;
    std::cout << std::endl;
}

int main()
{
    std::vector<std::array<size_t,2>> sizes
            = {{8,8}, {8, 16}, {16, 16}, {16, 32}, {32, 32}, {32, 64}, {64, 64}, {64, 128}, {128, 128}, {128, 256}, {256,256}};

    for (auto s: sizes)
    {
        measure_perfomance_one_size(s[0], s[1]);
    }


    return 0;
}
