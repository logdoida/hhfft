import hhfft
import numpy as np


# Create a plan for 1D FFT transform with complex input
n = 15
plan_1d_complex = hhfft.Hhfft1dPlan(n)

# Generage some random data and do FFT
x = np.random.rand(n) + 1j*np.random.rand(n)
y = plan_1d_complex.fft(x)

# Perform IFFT and calculate maximum error
z = plan_1d_complex.ifft(y)
err = np.amax(np.abs(x - z))
print('HHFFT 1D complex: err = {:.5}'.format(err))


# Create a plan for 1D FFT transform with real input
n = 15
plan_1d_real = hhfft.Hhfft1dRealPlan(n)

# Generage some random data and do FFT
x = np.random.rand(n)
y = plan_1d_real.fft(x)

# Perform IFFT and calculate maximum error
z = plan_1d_real.ifft(y)
err = np.amax(np.abs(x - z))
print('HHFFT 1D real: err = {:.5}'.format(err))


# Create a plan for 2D FFT transform with complex input
n = 4
m = 6
plan_2d_complex = hhfft.Hhfft2dPlan((n,m))

# Generage some random data and do FFT
x = np.random.rand(n, m) + 1j*np.random.rand(n, m)
y = plan_2d_complex.fft(x)

# Perform IFFT and calculate maximum error
z = plan_2d_complex.ifft(y)
err = np.amax(np.abs(x - z))
print('HHFFT 2D complex: err = {:.5}'.format(err))


# Create a plan for 2D FFT transform with real input
n = 4
m = 6
plan_2d_real = hhfft.Hhfft2dRealPlan((n,m))

# Generage some random data and do FFT
x = np.random.rand(n, m)
y = plan_2d_real.fft(x)

# Perform IFFT and calculate maximum error
z = plan_2d_real.ifft(y)
err = np.amax(np.abs(x - z))
print('HHFFT 2D real: err = {:.5}'.format(err))



# Perform 1D FFT and IFFT complex transform without a plan
# Note that this approach is slower than using a plan, if more that one fft or ifft is used
x = np.random.rand(15) + 1j*np.random.rand(15)
y = hhfft.fft(x)
z = hhfft.ifft(y)
err = np.amax(np.abs(x - z))
print('HHFFT 1D complex (no plan): err = {:.5}'.format(err))


# Perform 2D FFT and IFFT complex transform without a plan
# Note that this approach is slower than using a plan, if more that one fft or ifft is used
x = np.random.rand(4,6) + 1j*np.random.rand(4,6)
y = hhfft.fft2(x)
z = hhfft.ifft2(y)
err = np.amax(np.abs(x - z))
print('HHFFT 2D complex (no plan): err = {:.5}'.format(err))





