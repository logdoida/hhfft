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

// This header contains some constants and functions that are used elsewhere

#ifndef HHFFT_COMMON
#define HHFFT_COMMON

// This function can help compiler to optimze the code
template<hhfft::SizeType size_type> inline size_t get_size(size_t size)
{
    if (size_type == hhfft::SizeType::Size1)
    {
        return 1;
    } else if (size_type == hhfft::SizeType::Size2)
    {
        return 2;
    } else if (size_type == hhfft::SizeType::Size4)
    {
        return 4;
    } else
    {
        return size;
    }    
}

#endif
