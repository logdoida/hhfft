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
//#include "architecture.h"

// Note: this code can only be compiled with AVX support

#include "hhfft_avx_d.h"

#include <immintrin.h>

using hhfft::HHFFT_AVX_D;

// These functions/methods shall be implemented in this file
#define DISABLE_FFT_ROWWISE
#define DISABLE_FFT_REAL
#define DISABLE_IFFT_ROWWISE
#define DISABLE_IFFT_REAL
//#define DISABLE_FFT_ROW

#define HHFFT_CLASS_NAME HHFFT_AVX_D
#define TYPE double

#include "hhfft_plain_impl.h"

// Performs a forward fft for single a row
const double temp_const1[4] = {1.0, 1.0, 1.0, -1.0};
const double temp_const2[4] = {1.0, -1.0, 1.0, -1.0};
const double temp_const3[4] = {-1.0, 1.0, -1.0, 1.0};
const double temp_const1_inv[4] = {1.0, 1.0, -1.0, 1.0};

// FORWARD = 1 (forward) or -1 (inverse)
template<int FORWARD> inline void fft_row_avx(double *H, size_t n, size_t s, const TYPE *factor_table)
{
    //assert(s >= 4);
    if (s == 4)
    {
        __m256d temp_const1_;
        if (FORWARD == 1)
        {
            temp_const1_ = _mm256_loadu_pd(temp_const1);
        } else
        {
            temp_const1_ = _mm256_loadu_pd(temp_const1_inv);
        }
        for (size_t i = 0; i < 2*n; i+= 2*s)
        {
            __m256d in_0 = _mm256_loadu_pd(H + i);
            __m256d in_1 = _mm256_loadu_pd(H + i + 4);
            in_1 = _mm256_permute_pd(in_1, 2 + 4);
            __m256d temp = _mm256_mul_pd(in_1, temp_const1_);

            // Calculate output
            __m256d out_0 = _mm256_add_pd(in_0, temp);
            __m256d out_1 = _mm256_sub_pd(in_0, temp);

            // Store output
            _mm256_storeu_pd(H + i, out_0);
            _mm256_storeu_pd(H + i + 4, out_1);
        }
        return;
    }

    // When s = 8,16 ...
    __m256d temp_const2_ = _mm256_loadu_pd(temp_const2);
    __m256d temp_const3_ = _mm256_loadu_pd(temp_const3);    

    for (size_t i = 0; i < 2*n; i+= 2*s)
    {
        for (size_t j = 0; j < s/2; j+= 4)
        {
            size_t ii = i + j;
            __m256d w0 = _mm256_loadu_pd(&factor_table[j]);            
            __m256d w1;
            __m256d w2;

            if (FORWARD == 1)
            {
                w1 = _mm256_mul_pd(w0, temp_const2_);
                w2 = _mm256_permute_pd(w0, 1 + 4);
            } else
            {               
                w1 = w0;
                w2 = _mm256_permute_pd(w0, 1 + 4);
                w2 = _mm256_mul_pd(w2, temp_const3_);
            }

            {
                __m256d in0 = _mm256_loadu_pd(H + ii);
                __m256d in1 = _mm256_loadu_pd(H + ii + s);

                __m256d t1 = _mm256_mul_pd(w1, in1);
                __m256d t2 = _mm256_mul_pd(w2, in1);

                __m256d t3 = _mm256_hadd_pd(t1, t2);

                __m256d out0 = _mm256_add_pd(in0, t3);
                __m256d out1 = _mm256_sub_pd(in0, t3);

                _mm256_storeu_pd(H + ii, out0);
                _mm256_storeu_pd(H + ii + s, out1);
            }

            {
                __m256d in0 = _mm256_loadu_pd(H + ii + s/2);
                __m256d in1 = _mm256_loadu_pd(H + ii + s + s/2);                
                __m256d w3;
                __m256d w4;

                if (FORWARD == 1)
                {
                    w3 = w2;
                    w4 = -w1;
                } else
                {
                    w3 = -w2;
                    w4 = w1;
                }

                __m256d t1 = _mm256_mul_pd(w3, in1);
                __m256d t2 = _mm256_mul_pd(w4, in1);

                __m256d t3 = _mm256_hadd_pd(t1, t2);

                __m256d out0 = _mm256_add_pd(in0, t3);
                __m256d out1 = _mm256_sub_pd(in0, t3);

                _mm256_storeu_pd(H + ii + s/2, out0);
                _mm256_storeu_pd(H + ii + s + s/2, out1);
            }
        }
    }
}

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
    }

    // Start with level N = 2    
    std::complex<double> *data = m_out_packed.data;
    for (size_t i = 0; i < (n+2)*m/2; i+=2)
    {
        std::complex<TYPE> val1 = data[i];
        std::complex<TYPE> val2 = data[i+1];
        data[i] = std::complex<TYPE>(val1.real() + val2.real(), val1.imag() + val2.imag());
        data[i+1] = std::complex<TYPE>(val1.real() - val2.real(), val1.imag() - val2.imag());
    }

    // Other levels
    size_t s = 4;
    TYPE *factor_table2 = factor_table_2.data() + m - 4;
    while (s <= m)
    {
        fft_row_avx<1>((double *)data, (n+2)*m/2, s, factor_table2);
        factor_table2 = factor_table2 - s;
        s = s * 2;
    }
}

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
    }


    // Other levels
    size_t s = 4;
    std::complex<double> *data = m_out_packed.data;
    TYPE *factor_table2 = factor_table_2.data() + m - 4;
    while (s <= m)
    {
        fft_row_avx<-1>((double *)data, (n+2)*m/2, s, factor_table2);
        factor_table2 = factor_table2 - s;
        s = s * 2;
    }    
}

void HHFFT_AVX_D::fft_real(const double *in, double *out)
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

void HHFFT_AVX_D::ifft_real(const double *in, double *out)
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
