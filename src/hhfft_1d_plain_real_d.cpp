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

#include "hhfft_1d_plain_real_d.h"
#include "hhfft_1d_plain_real_impl.h"

using namespace hhfft;

template<bool forward> void set_dht_1d_one_level(StepInfoRealD &step_info)
{
    size_t radix = step_info.radix;

    if (radix == 2)
        step_info.step_function = dht_1d_one_level<double,2,0>;
    if (radix == 3)
        step_info.step_function = dht_1d_one_level<double,3,0>;
    if (radix == 4)
        step_info.step_function = dht_1d_one_level<double,4,0>;
    if (radix == 5)
        step_info.step_function = dht_1d_one_level<double,5,0>;
    if (radix == 7)
        step_info.step_function = dht_1d_one_level<double,7,0>;
}

template<bool forward> void set_dht_1d_one_level_twiddle(StepInfoRealD &step_info)
{
    size_t radix = step_info.radix;

    if (radix == 2)
        step_info.step_function = dht_1d_one_level_twiddle<double,2,0>;
    if (radix == 3)
        step_info.step_function = dht_1d_one_level_twiddle<double,3,0>;
    if (radix == 4)
        step_info.step_function = dht_1d_one_level_twiddle<double,4,0>;
    if (radix == 5)
        step_info.step_function = dht_1d_one_level_twiddle<double,5,0>;
    if (radix == 7)
        step_info.step_function = dht_1d_one_level_twiddle<double,7,0>;
}

void hhfft::HHFFT_1D_Plain_real_set_function(StepInfoRealD &step_info)
{
    step_info.step_function = nullptr;

    if (step_info.reorder_table != nullptr)
    {        
        // TODO how to use in-place if algorithm if input actually points to output?
        if (step_info.forward)
            step_info.step_function = dht_1d_reorder<double,0,true>;
        else
            step_info.step_function = dht_1d_reorder<double,0,false>;
        return;
    }

    if (step_info.cos_factors == nullptr)
    {
        if (step_info.forward)
            set_dht_1d_one_level<true>(step_info);
        else
            set_dht_1d_one_level<false>(step_info);
    } else
    {
        if (step_info.forward)
            set_dht_1d_one_level_twiddle<true>(step_info);
        else
            set_dht_1d_one_level_twiddle<false>(step_info);
    }

    if (step_info.step_function == nullptr)
    {
        throw(std::runtime_error("HHFFT error: Unable to set a function!"));
    }
}
