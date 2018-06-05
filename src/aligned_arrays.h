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

#ifndef HHFFT_ALIGNED_ARRAYS
#define HHFFT_ALIGNED_ARRAYS

#include <stddef.h>

namespace hhfft
{

// Allocates aligned memory
void* allocate_aligned_memory(size_t num_bytes, bool allocate_extra = false);

// A simple vector (1D array) that wraps aligned memory if possible
template<typename T> class AlignedVector
{
public:
    // Constructors
    AlignedVector();
    AlignedVector(size_t n);

    // Destructor
    ~AlignedVector();

    // Copy and move constructors
    AlignedVector(const AlignedVector<T>& other);
    AlignedVector(AlignedVector<T>&& other);

    // Copy and move operators
    AlignedVector<T>& operator=(const AlignedVector<T>& other);
    AlignedVector<T>& operator=(AlignedVector<T>&& other);

    // Resize
    void resize(size_t new_n);

    // Assignment
    T& operator[](size_t index)
    {
        return array[index];
    }

    // Direct access
    T* data()
    {
        return array;
    }

    // Length of the vector
    size_t size()
    {
        return n;
    }

private:
    size_t n;

    T* array;
};

typedef AlignedVector<double> AlignedVectorD;

}

#endif // HHFFT_ALIGNED_ARRAYS
