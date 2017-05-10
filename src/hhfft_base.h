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

#ifndef HHFFT_BASE_H
#define HHFFT_BASE_H

#include <iostream> //TESTING

namespace hhfft
{

// This is an abstract class that is base for all actual classes

template<class T> class HHFFT_Base
{
public:
    HHFFT_Base()
    {

    }

    virtual ~HHFFT_Base()
    {

    }

    virtual void fft_real(const T *in, T *out) = 0;
    virtual void ifft_real(const T *in, T *out) = 0;
    virtual void convolution_real(const T *in1, const T *in2, T *out) = 0;

private:


};
}


#endif // HHFFT_BASE_H
