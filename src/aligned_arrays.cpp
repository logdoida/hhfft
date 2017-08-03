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

#include "aligned_arrays.h"

#include <stdlib.h>
#include <algorithm>
#include <iostream> // TESTING

using namespace hhfft;

void* hhfft::allocate_aligned_memory(size_t num_bytes)
{
    // Alignment in bytes for AVX (TODO other alignments for SSE / AVX512?)
    size_t alignment = 32;

    // aligned_alloc can only be used when num_bytes is multiple of alignment
    if (num_bytes%alignment != 0)
    {
        return malloc(num_bytes);
    }

    return aligned_alloc(alignment, num_bytes);
}

template<typename T> AlignedVector<T>::AlignedVector()
{
    this->n = 0;
    this->array = nullptr;
}

template<typename T> AlignedVector<T>::AlignedVector(size_t n)
{
    this->n = n;
    this->array = (T*) allocate_aligned_memory(n*sizeof(T));
}

template<typename T> AlignedVector<T>::~AlignedVector()
{
    free(array);
}

template<typename T> AlignedVector<T>::AlignedVector(const AlignedVector<T>& other)
{
    this->n = other.n;
    this->array = (T*) allocate_aligned_memory(n*sizeof(T));
    std::copy(other.array, other.array + n, this->array);
}

template<typename T> AlignedVector<T>::AlignedVector(AlignedVector<T>&& other)
{
    this->n = other.n;
    this->array = other.array;

    other.n = 0;
    other.array = nullptr;
}

template<typename T> AlignedVector<T>& AlignedVector<T>::operator=(const AlignedVector<T>& other)
{
    this->n = other.n;
    this->array = (T*) allocate_aligned_memory(n*sizeof(T));
    std::copy(other.array, other.array + n, this->array);

    return *this;
}

template<typename T> AlignedVector<T>& AlignedVector<T>::operator=(AlignedVector<T>&& other)
{
    free(this->array);

    this->n = other.n;
    this->array = other.array;

    other.n = 0;
    other.array = nullptr;

    return *this;
}

template<typename T> T& AlignedVector<T>::operator[](size_t index)
{
    return array[index];
}

template<typename T> T* AlignedVector<T>::data()
{
    return array;
}

// Specialization only for float and double currently supported
template class hhfft::AlignedVector<float>;
template class hhfft::AlignedVector<double>;



