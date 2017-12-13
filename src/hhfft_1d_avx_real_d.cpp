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

#include "hhfft_1d_avx_real_d.h"
#include "hhfft_1d_plain_real_impl.h"

#include <assert.h>
#include <immintrin.h>

using namespace hhfft;

// Functions to operate data using avx commands

// contains two complex numbers: [r1 i1 r2 i2]
typedef __m256d ComplexD2;

// Read two complex number
inline ComplexD2 load(double r1, double i1, double r2, double i2)
{
    //return _mm256_set_pd(i2,r2,i1,r1); // Why this order?
    return _mm256_setr_pd(r1,i1,r2,i2); // Reversed. Why this order?
}
inline const ComplexD2 load(const double *v)
{
    return _mm256_loadu_pd(v);
}

// Loads same complex number twice: [r i] -> [r i r i]
inline const ComplexD2 broadcast128(const double *v)
{
    // TODO is this safe? Does alignment cause trouble?
    return _mm256_broadcast_pd((const __m128d*) v);
}

// Store a complex number
inline void store(ComplexD2 val, double &r1, double &i1, double &r2, double &i2)
{
    double v[4];
    _mm256_storeu_pd(v, val);
    r1 = val[0]; i1 = val[1]; r2 = val[2]; i2 = val[3];
}
inline void store(ComplexD2 val, double *v)
{
    _mm256_storeu_pd(v, val);
}

// Changes signs of [x1 x2 x3 x4] using [s1 s2 s3 s4]. s should contain only 0.0 and -0.0
// NOTE this seems to actually be bit slower than plain multiplication. Compare!
inline ComplexD2 change_sign(ComplexD2 x, ComplexD2 s)
{
    return _mm256_xor_pd(x,s);
}

// For testing
inline std::ostream& operator<<(std::ostream& os, const ComplexD2 &x)
{
    double v[4];
    store(x, v);
    os << v[0] << ", " << v[1] << ", " << v[2] << ", " << v[3];
    return os;
}

void hhfft::HHFFT_1D_AVX_real_set_function(StepInfoRealD &step_info)
{
    step_info.step_function = nullptr;

    /*
    // If reordering is done as a separate step
    if (step_info.reorder_table != nullptr && step_info.radix == 0)
    {
        // TODO how to use in-place if algorithm if input actually points to output?
        if (step_info.forward)
            step_info.step_function = dht_1d_reorder<double,1,true>;
        else
            step_info.step_function = dht_1d_reorder<double,1,false>;
        return;
    }

    // A dht to fft conversion step
    if (step_info.stride == 0 && step_info.radix == 1)
    {
        step_info.step_function = dht_1d_to_fft<double,1>;
        return;
    }

    if (step_info.cos_factors == nullptr)
    {
        if (step_info.reorder_table != nullptr)
        {
            if (step_info.forward)
                set_dht_1d_one_level<true,true>(step_info);
            else
                set_dht_1d_one_level<false,true>(step_info);
        } else
        {
            if (step_info.forward)
                set_dht_1d_one_level<true,false>(step_info);
            else
                set_dht_1d_one_level<false,false>(step_info);
        }
    } else
    {
        if (step_info.forward)
            set_dht_1d_one_level_twiddle<true>(step_info);
        else
            set_dht_1d_one_level_twiddle<false>(step_info);
    }
    */

    if (step_info.step_function == nullptr)
    {
        throw(std::runtime_error("HHFFT error: Unable to set a function!"));
    }
}




