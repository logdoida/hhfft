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

#include "hhfft_1d_plain_d.h"
#include "hhfft_1d_plain_impl.h"

using namespace hhfft;

template<bool forward> void set_fft_1d_one_level(StepInfoD &step_info)
{
    size_t radix = step_info.radix;

    if (radix == 2)
        step_info.step_function = fft_1d_one_level<double,2,0,forward>;
    if (radix == 3)
        step_info.step_function = fft_1d_one_level<double,3,0,forward>;
    if (radix == 4)
        step_info.step_function = fft_1d_one_level<double,4,0,forward>;
    if (radix == 5)
        step_info.step_function = fft_1d_one_level<double,5,0,forward>;
    if (radix == 7)
        step_info.step_function = fft_1d_one_level<double,7,0,forward>;
}

template<bool forward> void set_fft_1d_one_level_twiddle(StepInfoD &step_info)
{
    size_t radix = step_info.radix;

    if (radix == 2)
        step_info.step_function = fft_1d_one_level_twiddle<double,2,0,forward>;
    if (radix == 3)
        step_info.step_function = fft_1d_one_level_twiddle<double,3,0,forward>;
    if (radix == 4)
        step_info.step_function = fft_1d_one_level_twiddle<double,4,0,forward>;
    if (radix == 5)
        step_info.step_function = fft_1d_one_level_twiddle<double,5,0,forward>;
    if (radix == 7)
        step_info.step_function = fft_1d_one_level_twiddle<double,7,0,forward>;
}

void hhfft::HHFFT_1D_Plain_set_function(StepInfoD &step_info)
{
    step_info.step_function = nullptr;

    if (step_info.reorder_table != nullptr)
    {
        if (step_info.forward)
            step_info.step_function = fft_1d_reorder<double,0,true>;
        else
            step_info.step_function = fft_1d_reorder<double,0,false>;
        return;
    }

    if (step_info.twiddle_factors == nullptr)
    {
        if (step_info.forward)
            set_fft_1d_one_level<true>(step_info);
        else
            set_fft_1d_one_level<false>(step_info);
    } else
    {
        if (step_info.forward)
            set_fft_1d_one_level_twiddle<true>(step_info);
        else
            set_fft_1d_one_level_twiddle<false>(step_info);
    }

    if (step_info.step_function == nullptr)
    {
        throw(std::runtime_error("HHFFT error: Unable to set a function!"));
    }
}



