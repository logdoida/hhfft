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

#ifndef HHFFT_BASE_IMPL_H
#define HHFFT_BASE_IMPL_H

// This class contains actual implementation of hhfft_plain_d hhfft_plain_avx_d etc
// It is not an template class to allow the code to be compiled for different architectures

#include <iostream>
#include <cmath>
#include <assert.h>
#include <complex>
#include <stdexcept>

#define COMPLEX_TYPE std::complex<TYPE>

// Wrapper around raw data
struct MatrixReal
{
    TYPE *data;
    size_t n,m;

    MatrixReal(size_t n, size_t m, TYPE *d)
    {
        this->data = d;
        this->n = n;
        this->m = m;
    }

    TYPE &operator()(size_t i, size_t j)
    {
        return data[i*m+j];
    }
};

struct MatrixRealConst
{
    const TYPE *data;
    size_t n,m;

    MatrixRealConst(size_t n, size_t m, const TYPE *d)
    {
        this->data = d;
        this->n = n;
        this->m = m;
    }

    const TYPE &operator()(size_t i, size_t j)
    {
        return data[i*m+j];
    }
};

struct MatrixComplex
{
    std::complex<TYPE> *data;
    size_t n,m;

    MatrixComplex(size_t n, size_t m, TYPE *d)
    {
        this->data = (std::complex<TYPE> *) d;
        this->n = n;
        this->m = m;
    }

    std::complex<TYPE> &operator()(size_t i, size_t j)
    {
        assert(i < n);
        assert(j < m);
        return data[i*m+j];
    }
};

struct MatrixComplexConst
{
    const std::complex<TYPE> *data;
    size_t n,m;

    MatrixComplexConst(size_t n, size_t m, const TYPE *d)
    {
        this->data = (const std::complex<TYPE> *) d;
        this->n = n;
        this->m = m;
    }

    const std::complex<TYPE> &operator()(size_t i, size_t j)
    {
        assert(i < n && j < m);
        return data[i*m+j];
    }
};


HHFFT_CLASS_NAME::HHFFT_CLASS_NAME(size_t n, size_t m)
{
    this->n = n;
    this->m = m;

    size_t n_bits = (size_t) log2(n);
    size_t m_bits = (size_t) log2(m);

    if ( n != (1u << n_bits) ||  m != (1u << m_bits))
    {
        throw std::runtime_error("HHFFT: dimensions must be powers of two!");
    }
    if (n == 1 || m == 1)
    {
        throw std::runtime_error("HHFFT: dimensions must be at least 2!");
    }

    this->packing_table = calculate_packing_table(n);

    this->factor_table_1 = calculate_factor_table(n);
    this->factor_table_2 = calculate_factor_table(m);

    this->bit_reverse_table_1 = calculate_bit_reverse_table(n, n_bits);
    this->bit_reverse_table_2 = calculate_bit_reverse_table(m, m_bits);

    this->bit_reverse_table_2_inplace = calculate_bit_reverse_table_inplace(this->bit_reverse_table_2);
}

std::vector<TYPE> HHFFT_CLASS_NAME::calculate_factor_table(size_t n)
{
    std::vector<TYPE> table(n-2);

    size_t index = 0;
    while(n > 1)
    {
        for (size_t k=0; k < n/4; k++)
        {
            table[index++] = (TYPE) cos(-2.0*M_PI*double(k)/double(n));
            table[index++] = (TYPE) sin(-2.0*M_PI*double(k)/double(n));
        }
        n = n / 2;
    }    
    return table;
}

std::vector<TYPE> HHFFT_CLASS_NAME::calculate_packing_table(size_t n)
{
    std::vector<TYPE> table(n/2);

    size_t index = 0;
    for (size_t k = 0; k < n/4; k++)
    {
            table[index++] = (TYPE) cos(-2.0*M_PI*double(k + n/4)/double(2*n));
            table[index++] = (TYPE) -sin(-2.0*M_PI*double(k + n/4)/double(2*n));
    }
    return table;
}

std::vector<uint32_t> HHFFT_CLASS_NAME::calculate_bit_reverse_table(size_t n, size_t n_bits)
{
    assert(n == (1u << n_bits));

    std::vector<uint32_t> table(n);

    for (size_t i=0; i < n; i++)
    {
        table[i] = 0;
        for (size_t j = 0; j < n_bits; j++)
        {
            table[i] |= ((i >> j) & 1) << (n_bits - j - 1);
        }
    }
    return table;
}

std::vector<std::array<uint32_t,2>> HHFFT_CLASS_NAME::calculate_bit_reverse_table_inplace(std::vector<uint32_t> &table_in)
{
    size_t n = table_in.size();

    std::vector<std::array<uint32_t,2>> table;
    for (uint32_t i=0; i < n; i++)
    {
        if (table_in[i] > i)
        {
            table.push_back(std::array<uint32_t,2>({i,table_in[i]}));
        }
    }

    return table;
}

// Performs fft columnwise for all columns
template<int FORWARD> inline void fft_columns(MatrixComplex &data, size_t s, const TYPE *factor_table)
{
    size_t n = data.n;
    size_t m = data.m;

    for (size_t i = 0; i < n; i+=s)
    {
        // w = 1 + 0
        for (size_t j = 0; j < m; j++)
        {
            std::complex<TYPE> val1 = data(i,j);
            std::complex<TYPE> val2 = data(i+s/2,j);
            data(i,j) = std::complex<TYPE>(val1.real() + val2.real(), val1.imag() + val2.imag());
            data(i+s/2,j) = std::complex<TYPE>(val1.real() - val2.real(), val1.imag() - val2.imag());
        }

        // w = 0 - i
        for (size_t j = 0; j < m; j++)
        {
            std::complex<TYPE> val1 = data(i+1*s/4,j);
            std::complex<TYPE> val2 = data(i+3*s/4,j);
            data(i+1*s/4,j) = std::complex<TYPE>(val1.real() + FORWARD*val2.imag(), val1.imag() - FORWARD*val2.real());
            data(i+3*s/4,j) = std::complex<TYPE>(val1.real() - FORWARD*val2.imag(), val1.imag() + FORWARD*val2.real());
        }

        // Other cases in two groups
        for (size_t k = 1; k < s/4; k++)
        {
            TYPE w_real = factor_table[2*k];             // cos(-2.0*M_PI*double(k)/double(N));
            TYPE w_imag = FORWARD*factor_table[2*k + 1]; // sin(-2.0*M_PI*double(k)/double(N));

            for (size_t j = 0; j < m; j++)
            {
                std::complex<TYPE> val1 = data(i+k,j);
                std::complex<TYPE> val2 = data(i+k+s/2,j);
                TYPE temp_r = w_real*val2.real() - w_imag*val2.imag();
                TYPE temp_i = w_imag*val2.real() + w_real*val2.imag();
                data(i+k    ,j) = std::complex<TYPE>(val1.real() + temp_r, val1.imag() + temp_i);
                data(i+k+s/2,j) = std::complex<TYPE>(val1.real() - temp_r, val1.imag() - temp_i);
            }
        }

        for (size_t k = s/4+1; k < s/2; k++)
        {
            TYPE w_real =  factor_table[2*(k-s/4) + 1];         // cos(-2.0*M_PI*double(k)/double(N));
            TYPE w_imag = -FORWARD*factor_table[2*(k-s/4)];     // sin(-2.0*M_PI*double(k)/double(N));

            for (size_t j = 0; j < m; j++)
            {
                std::complex<TYPE> val1 = data(i+k,j);
                std::complex<TYPE> val2 = data(i+k+s/2,j);
                TYPE temp_r = w_real*val2.real() - w_imag*val2.imag();
                TYPE temp_i = w_imag*val2.real() + w_real*val2.imag();
                data(i+k    ,j) = std::complex<TYPE>(val1.real() + temp_r, val1.imag() + temp_i);
                data(i+k+s/2,j) = std::complex<TYPE>(val1.real() - temp_r, val1.imag() - temp_i);
            }
        }
    }
}

// Performs a fft for single a row
// FORWARD = 1 (forward) or -1 (inverse)
#ifndef DISABLE_FFT_ROW
template<int FORWARD> inline void fft_row(COMPLEX_TYPE *data, size_t n, size_t s, const TYPE *factor_table)
{
    for (size_t i = 0; i < n; i+=s)
    {
        // w = 1 + 0
        {
            std::complex<TYPE> val1 = data[i];
            std::complex<TYPE> val2 = data[i+s/2];
            data[i] = std::complex<TYPE>(val1.real() + val2.real(), val1.imag() + val2.imag());
            data[i+s/2] = std::complex<TYPE>(val1.real() - val2.real(), val1.imag() - val2.imag());
        }

        // w = 0 - i
        {
            std::complex<TYPE> val1 = data[i+1*s/4];
            std::complex<TYPE> val2 = data[i+3*s/4];
            data[i+1*s/4] = std::complex<TYPE>(val1.real() + FORWARD*val2.imag(), val1.imag() - FORWARD*val2.real());
            data[i+3*s/4] = std::complex<TYPE>(val1.real() - FORWARD*val2.imag(), val1.imag() + FORWARD*val2.real());
        }

        for (size_t k = 1; k < s/4; k++)
        {
            TYPE w_real = factor_table[2*k];             // cos(-2.0*M_PI*double(k)/double(N));
            TYPE w_imag = FORWARD*factor_table[2*k + 1]; // sin(-2.0*M_PI*double(k)/double(N));

            std::complex<TYPE> val1 = data[i+k];
            std::complex<TYPE> val2 = data[i+k+s/2];
            TYPE temp_r = w_real*val2.real() - w_imag*val2.imag();
            TYPE temp_i = w_imag*val2.real() + w_real*val2.imag();
            data[i+k    ] = std::complex<TYPE>(val1.real() + temp_r, val1.imag() + temp_i);
            data[i+k+s/2] = std::complex<TYPE>(val1.real() - temp_r, val1.imag() - temp_i);
        }

        for (size_t k = s/4+1; k < s/2; k++)
        {
            TYPE w_real =  factor_table[2*(k-s/4) + 1];         // cos(-2.0*M_PI*double(k)/double(N));
            TYPE w_imag = -FORWARD*factor_table[2*(k-s/4)];     // sin(-2.0*M_PI*double(k)/double(N));

            std::complex<TYPE> val1 = data[i+k];
            std::complex<TYPE> val2 = data[i+k+s/2];
            TYPE temp_r = w_real*val2.real() - w_imag*val2.imag();
            TYPE temp_i = w_imag*val2.real() + w_real*val2.imag();
            data[i+k    ] = std::complex<TYPE>(val1.real() + temp_r, val1.imag() + temp_i);
            data[i+k+s/2] = std::complex<TYPE>(val1.real() - temp_r, val1.imag() - temp_i);
        }
    }
}
#endif


//

// It is possible to do the conversions in place
// FORWARD = 1 (forward) or -1 (inverse)
template<int FORWARD> inline void fft_complex_to_complex_packed(MatrixComplex &in, MatrixComplex &out, const TYPE *packing_table)
{
    size_t n=0,m=0;

    if (FORWARD == 1)
    {
        n = in.n;
        m = in.m;
        assert (out.n == n+1);
        assert (out.m == m);
    } else
    {
        n = in.n - 1;
        m = in.m;
        assert (out.n == n);
        assert (out.m == m);
    }

    if (FORWARD == 1)
    {
        for (size_t j = 0; j < m; j++)
        {
            std::complex<TYPE> val = in(0,j);            
            out(0,j) = std::complex<TYPE>(val.real() + val.imag(), 0);
            out(n,j) = std::complex<TYPE>(val.real() - val.imag(), 0);
        }
    } else
    {
        for (size_t j = 0; j < m; j++)
        {
           std::complex<TYPE> val1 = in(0,j);
           std::complex<TYPE> val2 = in(n,j);
           out(0,j) = 0.5*std::complex<TYPE>(val1.real() + val2.real(), val1.real() - val2.real());
        }
    }

    for (size_t j = 0; j < m; j++)
    {
        std::complex<TYPE> val = in(n/2,j);
        out(n/2,j) = std::complex<TYPE>(val.real(), -val.imag());
    }

    for (size_t i = 1; i < n/2; i++)
    {
        TYPE c = packing_table[2*i]; // = cos(2.0*M_PI*(double(i) + n/2)/double(4*n));
        TYPE s = FORWARD*packing_table[2*i + 1]; //  = sin(2.0*M_PI*(double(i) + n/2)/double(4*n));

        for (size_t j = 0; j < m; j++)
        {
            std::complex<TYPE> val1 = in(i, j);
            std::complex<TYPE> val2 = in(n - i, j);

            out(i, j) = std::complex<TYPE>(     c*(c*val1.real()  + s*val1.imag()) + s*(s*val2.real() + c*val2.imag()),
                                                c*(-s*val1.real() + c*val1.imag()) + s*(c*val2.real() - s*val2.imag()));
            out(n - i, j) = std::complex<TYPE>(-s*(-s*val1.real() + c*val1.imag()) + c*(c*val2.real() - s*val2.imag()),
                                               -s*(c*val1.real()  + s*val1.imag()) + c*(s*val2.real() + c*val2.imag()));
        }
    }
}

inline void fft_columnwise(MatrixRealConst &m_in, MatrixComplex &m_out, std::vector<uint32_t> &bit_reverse_table_1, std::vector<TYPE> &factor_table_1)
{
    size_t n = m_out.n*2;
    size_t m = m_out.m;

    // Changes data from [a b c d; e f g h] to [a e b f c g d h], re-order rows and does N = 2 FFT if needed
    if (n == 2)
    {
        for (size_t j = 0; j < m; j++)
        {
            m_out(0,j) = std::complex<TYPE>(m_in(0,j), m_in(1,j));
        }
    } else
    {
        for (size_t i = 0; i < n/2; i+=2)
        {
            size_t i2 = bit_reverse_table_1[i*2];
            size_t i3 = bit_reverse_table_1[i*2 + 2];
            for (size_t j = 0; j < m; j++)
            {
                std::complex<TYPE> val1 = std::complex<TYPE>(m_in(i2*2,j), m_in(i2*2 + 1,j));
                std::complex<TYPE> val2 = std::complex<TYPE>(m_in(i3*2,j), m_in(i3*2 + 1,j));
                m_out(i,j) = std::complex<TYPE>(val1.real() + val2.real(), val1.imag() + val2.imag());
                m_out(i+1,j) = std::complex<TYPE>(val1.real() - val2.real(), val1.imag() - val2.imag());
            }
        }
    }

    // Do the rest of the levels starting from N = 4, until N = n/2
    size_t s = 4;
    TYPE *factor_table = factor_table_1.data() + n - 4;
    while (s < n)
    {
        fft_columns<1>(m_out, s, factor_table);

        factor_table = factor_table - s;
        s = s * 2;
    }
}

#ifndef DISABLE_FFT_ROWWISE
inline void fft_rowwise(MatrixComplex &m_out_packed, std::vector<double> &factor_table_2, std::vector<std::array<uint32_t,2>> &bit_reverse_table_2_inplace)
{
    size_t n = (m_out_packed.n-1)*2;
    size_t m = m_out_packed.m;

    // Re-order data (must be done in-place!)
    for (size_t i = 0; i < n/2 + 1; i++)
    {
        // Change only ones that actually need changing
        for (size_t k = 0; k < bit_reverse_table_2_inplace.size(); k++)
        {
            std::array<uint32_t,2> j = bit_reverse_table_2_inplace[k];
            std::complex<TYPE> temp =  m_out_packed(i,j[0]);
            m_out_packed(i,j[0]) = m_out_packed(i,j[1]);
            m_out_packed(i,j[1]) = temp;
        }

        // Start with level N = 2
        for (size_t j = 0; j < m; j+=2)
        {
            std::complex<TYPE> val1 = m_out_packed(i,j);
            std::complex<TYPE> val2 = m_out_packed(i,j+1);
            m_out_packed(i,j) = std::complex<TYPE>(val1.real() + val2.real(), val1.imag() + val2.imag());
            m_out_packed(i,j+1) = std::complex<TYPE>(val1.real() - val2.real(), val1.imag() - val2.imag());
        }

        size_t s = 4;
        TYPE *factor_table2 = factor_table_2.data() + m - 4;
        while (s <= m)
        {
            fft_row<1>((COMPLEX_TYPE *) &m_out_packed(i,0), m, s, factor_table2);
            factor_table2 = factor_table2 - s;
            s = s * 2;
        }
    }
}
#endif

#ifndef DISABLE_FFT_REAL
void HHFFT_CLASS_NAME::fft_real(const TYPE *in, TYPE *out)
{    
    //std::cout << std::endl << "Data in: " << std::endl; print_real_matrix(in, n, m); // TESTING

    MatrixRealConst m_in(n, m, in);
    MatrixComplex m_out(n/2, m, out);
    MatrixComplex m_out_packed(n/2 + 1, m, out);

    // Reorder data so that complex FFT can be performed column-wise
    // Perform FFT column-wise. There are now only n/2 rows!
    fft_columnwise(m_in, m_out, bit_reverse_table_1, factor_table_1);

    // "Pack data" so that the output is half of matrix that would be output of FFT perfomed on actual real data
    // After this step there is one extra row in the data!    
    fft_complex_to_complex_packed<1>(m_out, m_out_packed, packing_table.data());

    // Reorder data and do FFT row-wise, i.e. for each row separately
    fft_rowwise(m_out_packed, factor_table_2, bit_reverse_table_2_inplace);

    //std::cout << std::endl << "fft output:" << std::endl; print_complex_matrix(out, n/2 + 1, m); // TESTING
}
#endif

#ifndef DISABLE_IFFT_ROWWISE
inline void ifft_rowwise(MatrixComplexConst &m_in, MatrixComplex &m_out_packed, std::vector<TYPE> &factor_table_2, std::vector<uint32_t> &bit_reverse_table_2)
{
    size_t n = (m_in.n-1)*2;
    size_t m = m_in.m;

    for (size_t i = 0; i < n/2 + 1; i++)
    {
        // Re-order data and do level N = 2 FFT
        // TODO check if in == out and do this in-place if so!
        for (size_t j = 0; j < m; j+=2)
        {
            std::complex<TYPE> val1 = m_in(i,bit_reverse_table_2[j]);
            std::complex<TYPE> val2 = m_in(i,bit_reverse_table_2[j+1]);
            m_out_packed(i,j) = std::complex<TYPE>(val1.real() + val2.real(), val1.imag() + val2.imag());
            m_out_packed(i,j+1) = std::complex<TYPE>(val1.real() - val2.real(), val1.imag() - val2.imag());
        }

        size_t s = 4;
        TYPE *factor_table2 = factor_table_2.data() + m - 4;
        while (s <= m)
        {
            fft_row<-1>((COMPLEX_TYPE *) &m_out_packed(i,0), m, s, factor_table2);
            factor_table2 = factor_table2 - s;
            s = s * 2;
        }
    }
}
#endif

inline void reorder_2(MatrixComplex &m_out, std::vector<uint32_t> &bit_reverse_table_1)
{
    size_t n = m_out.n*2;
    size_t m = m_out.m;

    for (size_t i = 0; i < n/2; i++)
    {
        size_t i2 = bit_reverse_table_1[i*2];
        if (i2 > i)
        {
            for (size_t j = 0; j < m; j++)
            {
                std::complex<TYPE> temp = m_out(i2,j);
                m_out(i2,j) = m_out(i,j);
                m_out(i,j) = temp;
            }
        }
    }
}

inline void ifft_columnwise(MatrixComplex &m_out, std::vector<TYPE> &factor_table_1)
{
    size_t n = m_out.n*2;
    size_t m = m_out.m;

    // Start with level N = 2, if that is needed (if n=2, n/2 = 1 -> no IFFT needed)
    if (n > 2)
    {
        for (size_t i = 0; i < n/2; i+=2)
        {
            for (size_t j = 0; j < m; j++)
            {
                std::complex<TYPE> val1 = m_out(i,j);
                std::complex<TYPE> val2 = m_out(i+1,j);
                m_out(i,j) = std::complex<TYPE>(val1.real() + val2.real(), val1.imag() + val2.imag());
                m_out(i+1,j) = std::complex<TYPE>(val1.real() - val2.real(), val1.imag() - val2.imag());
            }
        }
    }

    // Do the rest of the levels starting from N = 4, until N = n/2
    size_t s = 4;
    TYPE *factor_table = factor_table_1.data() + n - 4;
    while (s < n)
    {
        fft_columns<-1>(m_out, s, factor_table);

        factor_table = factor_table - s;
        s = s * 2;
    }
}

inline void reorder_3(MatrixReal &m_out_real, MatrixComplex &m_out_packed)
{
    size_t n = m_out_real.n;
    size_t m = m_out_real.m;

    TYPE scaling = 1.0/(0.5*m*n);
    // TODO this could probably be done more efficiently!
    for (size_t i = 0; i < n/2; i++)
    {
        // First copy data to the last row (it is not used anymore)
        for (size_t j = 0; j < m; j++)
        {
            m_out_packed(n/2,j) = m_out_packed(i,j)*scaling;
        }

        // Then copy the data as it should actually be
        for (size_t j = 0; j < m; j++)
        {
            m_out_real(2*i,j) = m_out_packed(n/2,j).real();
            m_out_real(2*i+1,j) = m_out_packed(n/2,j).imag();
        }
    }
}

#ifndef DISABLE_IFFT_REAL
void HHFFT_CLASS_NAME::ifft_real(const TYPE *in, TYPE *out)
{
    MatrixComplexConst m_in(n/2 + 1, m, in);
    MatrixComplex m_out_packed(n/2 + 1, m, out);
    MatrixComplex m_out(n/2, m, out);
    MatrixReal m_out_real(n, m, out);

    // Reorder data and do IFFT rowwise, i.e. for each row separately
    ifft_rowwise(m_in, m_out_packed, factor_table_2, bit_reverse_table_2);

    // After this step there one extra row disappears!
    fft_complex_to_complex_packed<-1>(m_out_packed, m_out, packing_table.data());

    // Again some re-ordering of rows is needed in-place... (could this be combined with some other step?)
    reorder_2(m_out, bit_reverse_table_1);

    // Perform a complex IFFT column-wise.
    ifft_columnwise(m_out, factor_table_1);

    // Re-order data (this is because actual output is real with dimensions n x m) and scale data
    reorder_3(m_out_real, m_out_packed);

    //std::cout << std::endl << "Final output:" << std::endl;  print_real_matrix(out, n, m); // TESTING
}
#endif

// All arrays are complex with dimensions (n/2 + 1) x m
#ifndef DISABLE_CONVOLUTION_REAL
void HHFFT_CLASS_NAME::convolution_real(const TYPE *in1, const TYPE *in2, TYPE *out)
{
    MatrixComplexConst m_in1(n/2 + 1, m, in1);
    MatrixComplexConst m_in2(n/2 + 1, m, in2);
    MatrixComplex m_out(n/2 + 1, m, out);

    for (size_t i = 0; i < n/2 + 1; i++)
    {
        for (size_t j = 0; j < m; j++)
        {
            TYPE r1 = m_in1(i,j).real();
            TYPE i1 = m_in1(i,j).imag();
            TYPE r2 = m_in2(i,j).real();
            TYPE i2 = m_in2(i,j).imag();

            m_out(i,j) = std::complex<TYPE>(r1*r2 - i1*i2, r1*i2 + r2*i1);
        }
    }
}

void HHFFT_CLASS_NAME::convolution_real_add(const TYPE *in1, const TYPE *in2, TYPE *out)
{
    MatrixComplexConst m_in1(n/2 + 1, m, in1);
    MatrixComplexConst m_in2(n/2 + 1, m, in2);
    MatrixComplex m_out(n/2 + 1, m, out);

    for (size_t i = 0; i < n/2 + 1; i++)
    {
        for (size_t j = 0; j < m; j++)
        {
            TYPE r1 = m_in1(i,j).real();
            TYPE i1 = m_in1(i,j).imag();
            TYPE r2 = m_in2(i,j).real();
            TYPE i2 = m_in2(i,j).imag();

            m_out(i,j) += std::complex<TYPE>(r1*r2 - i1*i2, r1*i2 + r2*i1);
        }
    }
}
#endif

// These are for testing purposes
void HHFFT_CLASS_NAME::print_real_matrix(const TYPE *data, size_t n, size_t m)
{
    MatrixRealConst matrix(n, m, data);

    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < m; j++)
        {
            std::cout << matrix(i,j) << "  ";
        }
        std::cout << std::endl;
    }

}

void HHFFT_CLASS_NAME::print_complex_matrix(const TYPE *data, size_t n, size_t m)
{
    MatrixComplexConst matrix(n, m, data);

    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < m; j++)
        {
            TYPE real = matrix(i,j).real();
            TYPE imag = matrix(i,j).imag();
            if (imag >= 0.0)
                std::cout << real << "+" << imag << "i  ";
            else
                std::cout << real << imag << "i  ";
        }
        std::cout << std::endl;
    }
}

#endif // HHFFT_BASE_IMPL_H

