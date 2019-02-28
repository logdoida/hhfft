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

#include "aligned_arrays.h"

#include <algorithm>

using namespace hhfft;

// Alignment in bytes for AVX (TODO other alignments for SSE / AVX512?)
const size_t alignment = 32;

size_t hhfft::calculate_aligned_size(size_t num_bytes)
{    
    return num_bytes/alignment*alignment + alignment;
}

void* hhfft::allocate_aligned_memory(size_t num_bytes, bool allocate_extra)
{    
    // aligned_alloc can only be used when num_bytes is multiple of alignment
    if (num_bytes%alignment != 0)
    {
        // allocate extra if allowed
        if (allocate_extra)
        {
            return aligned_alloc(alignment, calculate_aligned_size(num_bytes));
        } else
        {
            return malloc(num_bytes);
        }
    }

    return aligned_alloc(alignment, num_bytes);
}

template<typename T> void AlignedVector<T>::resize(size_t new_n)
{
    // Allocate new memory
    T* new_array = (T*) allocate_aligned_memory(new_n*sizeof(T), true);

    // Copy data from the old array to the new one
    size_t n_to_copy = std::min(new_n, n);
    for (size_t i = 0; i < n_to_copy; i++)
        new_array[i] = this->array[i];

    // Delete the old memory and set pointers
    free(array);
    array = new_array;
    n = new_n;
}

template<typename T> AlignedVector<T>::AlignedVector(const AlignedVector<T>& other)
{
    this->n = other.n;
    if (n == 0)
    {
        this->array = nullptr;
    } else
    {        
        this->array = (T*) allocate_aligned_memory(n*sizeof(T), true);
        std::copy(other.array, other.array + n, this->array);        
    }
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
    if (n == 0)
    {
        this->array = nullptr;
    } else
    {
        this->array = (T*) allocate_aligned_memory(n*sizeof(T), true);
        std::copy(other.array, other.array + n, this->array);
    }

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

// Specialization only for float and double currently supported
template class hhfft::AlignedVector<float>;
template class hhfft::AlignedVector<double>;



