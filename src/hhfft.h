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

#ifndef HHFFT_H
#define HHFFT_H

#include "hhfft_base.h"

namespace hhfft
{

// This is an wrapper that selects correct version to be used
class HHFFT_D
{
public:

    // TODO 1D FFT could also be supported here
    //HHFFT_D(size_t n);

    // Constructs a HHFFT object that can solve FFT problems n x m
    HHFFT_D(size_t n, size_t m);

    ~HHFFT_D();

    // FFT with real input.
    // Input should contain n*m doubles and output (n+2)*m doubles.
    void fft_real(const double *in, double *out);

    // IFFT with real output.
    // Both should contain (n+2)*m doubles, although in output only the first n*m doubles contain relevant values.
    void ifft_real(const double *in, double *out);

    // Convolute two Fourier transformed real arrays. Both inputs and output shall contain (n+2)*m doubles
    void convolution_real(const double *in1, const double *in2, double *out);

    // Convolute two Fourier transformed real arrays and add them to output. Both inputs and output shall contain (n+2)*m doubles
    void convolution_real_add(const double *in1, const double *in2, double *out);


    // These methods are provide information what architecture is supported
    static bool avx_support_on();
    static bool avx512f_support_on();

private:

    HHFFT_Base<double> *hhfft;
};
}

#endif // HHFFT_H
