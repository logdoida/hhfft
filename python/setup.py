from distutils.core import setup, Extension

module = Extension('hhfft', 
      sources = ['hhfftmodule.cpp'],
      extra_compile_args = ['-std=c++11'],
      libraries = ['hhfft'])

setup (name = 'HHFFT', 
      version = '0.2.2',
      author = 'Jouko Kalmari',
      url = 'https://github.com/logdoida/hhfft/',
      description = 'Fast Fourier Transform (FFT) package',
      classifiers = ['License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)'],
      ext_modules = [module])
