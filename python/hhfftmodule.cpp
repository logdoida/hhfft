/*
*   Copyright Jouko Kalmari 2018
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

#include <Python.h>
#include "structmember.h"

// Deprecated api not used
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include "numpy/utils.h"

// HHFFT includes
#include "hhfft_1d_d.h"
#include "hhfft_1d_real_d.h"
#include "hhfft_2d_d.h"
#include "hhfft_2d_real_d.h"


// HHFFT objects
typedef struct 
{
    PyObject_HEAD
    Py_ssize_t n;
    hhfft::HHFFT_1D_D *hhfft_1d;
} Hhfft1dPlanObject;

typedef struct
{
    PyObject_HEAD
    Py_ssize_t n;
    hhfft::HHFFT_1D_REAL_D *hhfft_real_1d;
} Hhfft1dRealPlanObject;

typedef struct
{
    PyObject_HEAD
    Py_ssize_t n, m;
    hhfft::HHFFT_2D_D *hhfft_2d;
} Hhfft2dPlanObject;

typedef struct
{
    PyObject_HEAD
    Py_ssize_t n, m;
    hhfft::HHFFT_2D_REAL_D *hhfft_real_2d;
} Hhfft2dRealPlanObject;

// New
static PyObject *Hhfft1dPlan_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    Hhfft1dPlanObject *self = (Hhfft1dPlanObject *) type->tp_alloc(type, 0);
    if (self != NULL)
    {
        // initialize object pointer to zero
        self->n = 0;
        self->hhfft_1d = nullptr;
    }
    return (PyObject *) self;
}

static PyObject *Hhfft1dRealPlan_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    Hhfft1dRealPlanObject *self = (Hhfft1dRealPlanObject *) type->tp_alloc(type, 0);
    if (self != NULL)
    {
        // initialize object pointer to zero
        self->n = 0;
        self->hhfft_real_1d = nullptr;
    }
    return (PyObject *) self;
}

static PyObject *Hhfft2dPlan_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    Hhfft2dPlanObject *self = (Hhfft2dPlanObject *) type->tp_alloc(type, 0);
    if (self != NULL)
    {
        // initialize object pointer to zero
        self->n = 0;
        self->m = 0;
        self->hhfft_2d = nullptr;
    }
    return (PyObject *) self;
}

static PyObject *Hhfft2dRealPlan_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    Hhfft2dRealPlanObject *self = (Hhfft2dRealPlanObject *) type->tp_alloc(type, 0);
    if (self != NULL)
    {
        // initialize object pointer to zero
        self->n = 0;
        self->m = 0;
        self->hhfft_real_2d = nullptr;
    }
    return (PyObject *) self;
}

static bool read_check_args(PyObject *args, PyObject *keywords, Py_ssize_t n_dims, Py_ssize_t *n)
{
    // Descriptor is used
    PyArray_Descr* dtype = NULL;
    PyObject* shape = NULL;

    // Read the parameters
    static char *keywordlist[] = {(char *)"shape", (char *)"dtype", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywords, "O|O&", keywordlist, &shape, &PyArray_DescrConverter, &dtype))
    {
        // TODO is this needed?
        //if (dtype != NULL)
        //    Py_DECREF(dtype);

        return false;
    }

    // Read the shape parameter
    bool ok = false;
    if (n_dims == 1)
    {
        // If n_dims = 1, shape must be long object (also tuple with only one long is long!)
        if (PyLong_CheckExact(shape))
        {
            Py_ssize_t val = PyLong_AsSsize_t(shape);
            if (PyErr_Occurred() == NULL)
            {
                ok = true;
                n[0] = val;
            }
        }
    } else
    {
        // If n_dims > 1, shape must be tuple containg correct number of PyLongs
        if (PyTuple_CheckExact(shape))
        {
            Py_ssize_t tuple_size =  PyTuple_Size(shape);
            if (tuple_size == n_dims)
            {
                for (Py_ssize_t i = 0; i < tuple_size; i++)
                {
                    PyObject *item = PyTuple_GetItem(shape, i);
                    if (PyLong_CheckExact(item))
                    {
                        Py_ssize_t val = PyLong_AsSsize_t(item);
                        if (PyErr_Occurred() != NULL)
                        {
                            ok = false;
                            break;
                        }
                        ok = true;
                        n[i] = val;
                    } else
                    {
                        ok = false;
                        break;
                    }
                }
            }
        }
    }

    if (!ok)
    {
        PyErr_SetString(PyExc_ValueError, "shape must be integer or tuple of integers with correct size");

        // Decrement reference counter
        if (dtype != NULL)
            Py_DECREF(dtype);

        return false;
    }

    // dtype checked that its type is NPY_FLOAT64 (only type currently supported)
    // In python code the dtype can set to empty, np.float64, None, 'd', numpy.dtype(np.float64) etc.
    // If float is supported in the future (NPY_FLOAT32) it should be handeld here
    if (dtype != NULL)
    {
        PyArray_Descr *dtype_double = PyArray_DescrFromType(NPY_FLOAT64);
        bool type_double = PyArray_EquivTypes(dtype, dtype_double);

        // Decrement reference counters
        Py_DECREF(dtype);
        Py_DECREF(dtype_double);

        // Error if type is not 'double'
        if (!type_double)
        {
            PyErr_SetString(PyExc_ValueError, "dtype must be equivalent to np.float64");
            return false;
        }
    }

    return true;
}

// Init
static int Hhfft1dPlan_init(Hhfft1dPlanObject *self, PyObject *args, PyObject *keywords)
{
    Py_ssize_t n=0;

    if (!read_check_args(args, keywords, 1, &n))
    {
        return -1;
    }

    // Initialize the HHFFT object
    try
    {
        self->hhfft_1d = new hhfft::HHFFT_1D_D(n);
    } catch (std::runtime_error &e)
    {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return -1;
    }
    self->n = n;

    return 0;
}

static int Hhfft1dRealPlan_init(Hhfft1dRealPlanObject *self, PyObject *args, PyObject *keywords)
{
    Py_ssize_t n=0;

    if (!read_check_args(args, keywords, 1, &n))
    {
        return -1;
    }

    // Initialize the HHFFT object
    try
    {
        self->hhfft_real_1d = new hhfft::HHFFT_1D_REAL_D(n);
    } catch (std::runtime_error &e)
    {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return -1;
    }
    self->n = n;

    return 0;
}

static int Hhfft2dPlan_init(Hhfft2dPlanObject *self, PyObject *args, PyObject *keywords)
{
    Py_ssize_t n[2];

    if (!read_check_args(args, keywords, 2, n))
    {
        return -1;
    }

    // Initialize the HHFFT object
    try
    {
        self->hhfft_2d = new hhfft::HHFFT_2D_D(n[0],n[1]);
    } catch (std::runtime_error &e)
    {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return -1;
    }
    self->n = n[0];
    self->m = n[1];

    return 0;
}

static int Hhfft2dRealPlan_init(Hhfft2dRealPlanObject *self, PyObject *args, PyObject *keywords)
{
    Py_ssize_t n[2];

    if (!read_check_args(args, keywords, 2, n))
    {
        return -1;
    }

    // Initialize the HHFFT object
    try
    {
        self->hhfft_real_2d = new hhfft::HHFFT_2D_REAL_D(n[0],n[1]);
    } catch (std::runtime_error &e)
    {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return -1;
    }
    self->n = n[0];
    self->m = n[1];

    return 0;
}

// Deallocator
static void Hhfft1dPlan_dealloc(Hhfft1dPlanObject *self)
{
    // Delete the hhfft object
    delete self->hhfft_1d;
    self->hhfft_1d = nullptr;
    self->n = 0;
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static void Hhfft1dRealPlan_dealloc(Hhfft1dRealPlanObject *self)
{
    // Delete the hhfft object
    delete self->hhfft_real_1d;
    self->hhfft_real_1d = nullptr;
    self->n = 0;
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static void Hhfft2dPlan_dealloc(Hhfft2dPlanObject *self)
{
    // Delete the hhfft object
    delete self->hhfft_2d;
    self->hhfft_2d = nullptr;
    self->n = 0;
    self->m = 0;
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static void Hhfft2dRealPlan_dealloc(Hhfft2dRealPlanObject *self)
{
    // Delete the hhfft object
    delete self->hhfft_real_2d;
    self->hhfft_real_2d = nullptr;
    self->n = 0;
    self->m = 0;
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static bool read_input(PyObject *args, PyObject *keywords, int type_in, int ndims, npy_intp* dims_in, PyArrayObject **in_arr)
{
    PyObject *in = NULL;

    // Read the parameters
    static char *keywordlist[] = {(char *)"in", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywords, "O", keywordlist, &in))
    {
        return false;
    }

    // Try to convert the input object to a numpy array
    *in_arr = (PyArrayObject *) PyArray_FROM_OTF(in, type_in, NPY_ARRAY_IN_ARRAY);
    if (!*in_arr)
    {
        return false;
    }

    // Check that the number of dimensions of the input array are correct
    const npy_intp *dims = PyArray_DIMS(*in_arr);
    if (PyArray_NDIM(*in_arr) != ndims)
    {
        PyErr_SetString(PyExc_ValueError, "Input dimension number wrong");
        Py_DECREF(*in_arr);
        return false;
    }

    // Copy dimensions
    for (int i = 0; i < ndims; i++)
    {
        dims_in[i] = dims[i];
    }

    // Everything is ok
    return true;
}

static bool create_output(int type_out, int ndims, npy_intp* dims_out, PyArrayObject **out_arr)
{
    // Create an output array
    *out_arr = (PyArrayObject *) PyArray_SimpleNew(ndims, dims_out, type_out);
    if (!*out_arr)
    {
        return false;
    }

    // Everything is ok
    return true;
}

static bool read_inputs_create_output(PyObject *args, PyObject *keywords, int type_in, int type_out, int ndims, npy_intp* dims_in, npy_intp* dims_out, PyArrayObject **in_arr, PyArrayObject **out_arr)
{
    PyObject *in = NULL;

    // Read the parameters
    static char *keywordlist[] = {(char *)"in", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywords, "O", keywordlist, &in))
    {
        return false;
    }

    // Try to convert the input object to a numpy array
    *in_arr = (PyArrayObject *) PyArray_FROM_OTF(in, type_in, NPY_ARRAY_IN_ARRAY);
    if (!*in_arr)
    {
        return false;
    }

    // Check that the number of dimensions of the input array are correct
    const npy_intp *dims = PyArray_DIMS(*in_arr);
    if (PyArray_NDIM(*in_arr) != ndims)
    {
        PyErr_SetString(PyExc_ValueError, "Input dimension number wrong");
        Py_DECREF(*in_arr);
        return false;
    }

    // Check that the dimensions sizes of the input array are correct
    for (int i = 0; i < ndims; i++)
    {
        if (dims[i] != dims_in[i])
        {
            PyErr_SetString(PyExc_ValueError, "Input dimensions wrong");
            Py_DECREF(*in_arr);
            return false;
        }
    }

    // Create an output array
    *out_arr = (PyArrayObject *) PyArray_SimpleNew(ndims, dims_out, type_out);
    if (!*out_arr)
    {
        Py_DECREF(*in_arr);
        return false;
    }

    // Everything is ok
    return true;
}


// FFT 1D
static PyObject *Hhfft1dPlan_fft(Hhfft1dPlanObject *self, PyObject *args, PyObject *keywords)
{
    int type_in = NPY_COMPLEX128;
    int type_out = NPY_COMPLEX128;
    int ndims = 1;
    npy_intp dims_in[1] = {self->n};
    npy_intp dims_out[1] = {self->n};

    PyArrayObject *in_arr = NULL;
    PyArrayObject *out_arr = NULL;

    if(!read_inputs_create_output(args, keywords, type_in, type_out, ndims, dims_in, dims_out, &in_arr, &out_arr))
    {
        return NULL;
    }

    // Do the actual fft
    double *data_in = (double *) PyArray_DATA(in_arr);
    double *data_out = (double *) PyArray_DATA(out_arr);
    self->hhfft_1d->fft(data_in, data_out);

    // Decrement reference counter
    Py_DECREF(in_arr);

    // Return the output array
    return (PyObject *) out_arr;
}

// FFT 1D real
static PyObject *Hhfft1dRealPlan_fft(Hhfft1dRealPlanObject *self, PyObject *args, PyObject *keywords)
{
    int type_in = NPY_DOUBLE;
    int type_out = NPY_COMPLEX128;
    int ndims = 1;
    npy_intp dims_in[1] = {self->n};
    npy_intp dims_out[1] = {(self->n + 2)/2};

    PyArrayObject *in_arr = NULL;
    PyArrayObject *out_arr = NULL;

    if(!read_inputs_create_output(args, keywords, type_in, type_out, ndims, dims_in, dims_out, &in_arr, &out_arr))
    {
        return NULL;
    }

    // Do the actual fft
    double *data_in = (double *) PyArray_DATA(in_arr);
    double *data_out = (double *) PyArray_DATA(out_arr);
    self->hhfft_real_1d->fft(data_in, data_out);

    // Decrement reference counter
    Py_DECREF(in_arr);

    // Return the output array
    return (PyObject *) out_arr;
}

// FFT 2D
static PyObject *Hhfft2dPlan_fft(Hhfft2dPlanObject *self, PyObject *args, PyObject *keywords)
{
    int type_in = NPY_COMPLEX128;
    int type_out = NPY_COMPLEX128;
    int ndims = 2;
    npy_intp dims_in[2] = {self->n,self->m};
    npy_intp dims_out[2] = {self->n,self->m};

    PyArrayObject *in_arr = NULL;
    PyArrayObject *out_arr = NULL;

    if(!read_inputs_create_output(args, keywords, type_in, type_out, ndims, dims_in, dims_out, &in_arr, &out_arr))
    {
        return NULL;
    }

    // Do the actual fft
    double *data_in = (double *) PyArray_DATA(in_arr);
    double *data_out = (double *) PyArray_DATA(out_arr);
    self->hhfft_2d->fft(data_in, data_out);

    // Decrement reference counter
    Py_DECREF(in_arr);

    // Return the output array
    return (PyObject *) out_arr;
}

// FFT 2D real
static PyObject *Hhfft2dRealPlan_fft(Hhfft2dRealPlanObject *self, PyObject *args, PyObject *keywords)
{
    int type_in = NPY_DOUBLE;
    int type_out = NPY_COMPLEX128;
    int ndims = 2;
    npy_intp dims_in[2] = {self->n,self->m};
    npy_intp dims_out[2] = {self->n,(self->m + 2)/2};

    PyArrayObject *in_arr = NULL;
    PyArrayObject *out_arr = NULL;

    if(!read_inputs_create_output(args, keywords, type_in, type_out, ndims, dims_in, dims_out, &in_arr, &out_arr))
    {
        return NULL;
    }

    // Do the actual fft
    double *data_in = (double *) PyArray_DATA(in_arr);
    double *data_out = (double *) PyArray_DATA(out_arr);
    self->hhfft_real_2d->fft(data_in, data_out);

    // Decrement reference counter
    Py_DECREF(in_arr);

    // Return the output array
    return (PyObject *) out_arr;
}

// IFFT 1D
static PyObject *Hhfft1dPlan_ifft(Hhfft1dPlanObject *self, PyObject *args, PyObject *keywords)
{
    int type_in = NPY_COMPLEX128;
    int type_out = NPY_COMPLEX128;
    int ndims = 1;
    npy_intp dims_in[1] = {self->n};
    npy_intp dims_out[1] = {self->n};

    PyArrayObject *in_arr = NULL;
    PyArrayObject *out_arr = NULL;

    if(!read_inputs_create_output(args, keywords, type_in, type_out, ndims, dims_in, dims_out, &in_arr, &out_arr))
    {
        return NULL;
    }

    // Do the actual ifft
    double *data_in = (double *) PyArray_DATA(in_arr);
    double *data_out = (double *) PyArray_DATA(out_arr);
    self->hhfft_1d->ifft(data_in, data_out);

    // Decrement reference counter
    Py_DECREF(in_arr);

    // Return the output array
    return (PyObject *) out_arr;
}

// IFFT 1D real
static PyObject *Hhfft1dRealPlan_ifft(Hhfft1dRealPlanObject *self, PyObject *args, PyObject *keywords)
{
    int type_in = NPY_COMPLEX128;
    int type_out = NPY_DOUBLE;
    int ndims = 1;
    npy_intp dims_in[1] = {(self->n + 2)/2};
    npy_intp dims_out[1] = {self->n};

    PyArrayObject *in_arr = NULL;
    PyArrayObject *out_arr = NULL;

    if(!read_inputs_create_output(args, keywords, type_in, type_out, ndims, dims_in, dims_out, &in_arr, &out_arr))
    {
        return NULL;
    }

    // Do the actual ifft
    double *data_in = (double *) PyArray_DATA(in_arr);
    double *data_out = (double *) PyArray_DATA(out_arr);
    self->hhfft_real_1d->ifft(data_in, data_out);

    // Decrement reference counter
    Py_DECREF(in_arr);

    // Return the output array
    return (PyObject *) out_arr;
}

// IFFT 2D
static PyObject *Hhfft2dPlan_ifft(Hhfft2dPlanObject *self, PyObject *args, PyObject *keywords)
{
    int type_in = NPY_COMPLEX128;
    int type_out = NPY_COMPLEX128;
    int ndims = 2;
    npy_intp dims_in[2] = {self->n,self->m};
    npy_intp dims_out[2] = {self->n,self->m};

    PyArrayObject *in_arr = NULL;
    PyArrayObject *out_arr = NULL;

    if(!read_inputs_create_output(args, keywords, type_in, type_out, ndims, dims_in, dims_out, &in_arr, &out_arr))
    {
        return NULL;
    }

    // Do the actual ifft
    double *data_in = (double *) PyArray_DATA(in_arr);
    double *data_out = (double *) PyArray_DATA(out_arr);
    self->hhfft_2d->ifft(data_in, data_out);

    // Decrement reference counter
    Py_DECREF(in_arr);

    // Return the output array
    return (PyObject *) out_arr;
}

// IFFT 2D real
static PyObject *Hhfft2dRealPlan_ifft(Hhfft2dRealPlanObject *self, PyObject *args, PyObject *keywords)
{
    int type_in = NPY_COMPLEX128;
    int type_out = NPY_DOUBLE;
    int ndims = 2;
    npy_intp dims_in[2] = {self->n,(self->m + 2)/2};
    npy_intp dims_out[2] = {self->n,self->m};

    PyArrayObject *in_arr = NULL;
    PyArrayObject *out_arr = NULL;

    if(!read_inputs_create_output(args, keywords, type_in, type_out, ndims, dims_in, dims_out, &in_arr, &out_arr))
    {
        return NULL;
    }

    // Do the actual fft
    double *data_in = (double *) PyArray_DATA(in_arr);
    double *data_out = (double *) PyArray_DATA(out_arr);
    self->hhfft_real_2d->ifft(data_in, data_out);

    // Decrement reference counter
    Py_DECREF(in_arr);

    // Return the output array
    return (PyObject *) out_arr;
}


//////////// Direct functions (i.e. no existing plan)
// Note that fft real are not supported, as it is not possible to know in ifft what is the correct size!

// FFT 1D directly
static PyObject *fft_1d(PyObject *, PyObject *args, PyObject *keywords)
{
    int type_in = NPY_COMPLEX128;
    int type_out = NPY_COMPLEX128;
    int ndims = 1;
    npy_intp dims_in[1];

    PyArrayObject *in_arr = NULL;
    PyArrayObject *out_arr = NULL;

    // Read the inputs
    if(!read_input(args, keywords, type_in, ndims, dims_in, &in_arr))
    {
        return NULL;
    }

    // Crate output array of same size as input
    if(!create_output(type_out, ndims, dims_in, &out_arr))
    {
        Py_DECREF(in_arr);
        return NULL;
    }

    // Initialize the HHFFT object
    hhfft::HHFFT_1D_D *hhfft_1d = NULL;
    try
    {
        size_t n = dims_in[0];
        hhfft_1d = new hhfft::HHFFT_1D_D(n);
    } catch (std::runtime_error &e)
    {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        Py_DECREF(in_arr);
        return NULL;
    }

    // Do the actual fft
    double *data_in = (double *) PyArray_DATA(in_arr);
    double *data_out = (double *) PyArray_DATA(out_arr);
    hhfft_1d->fft(data_in, data_out);

    // Free the HHFFT object
    delete hhfft_1d;

    // Decrement reference counter
    Py_DECREF(in_arr);

    // Return the output array
    return (PyObject *) out_arr;
}

// IFFT 1D directly
static PyObject *ifft_1d(PyObject *, PyObject *args, PyObject *keywords)
{
    int type_in = NPY_COMPLEX128;
    int type_out = NPY_COMPLEX128;
    int ndims = 1;
    npy_intp dims_in[1];

    PyArrayObject *in_arr = NULL;
    PyArrayObject *out_arr = NULL;

    // Read the inputs
    if(!read_input(args, keywords, type_in, ndims, dims_in, &in_arr))
    {
        return NULL;
    }

    // Crate output array of same size as input
    if(!create_output(type_out, ndims, dims_in, &out_arr))
    {
        Py_DECREF(in_arr);
        return NULL;
    }

    // Initialize the HHFFT object
    hhfft::HHFFT_1D_D *hhfft_1d = NULL;
    try
    {
        size_t n = dims_in[0];
        hhfft_1d = new hhfft::HHFFT_1D_D(n);
    } catch (std::runtime_error &e)
    {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        Py_DECREF(in_arr);
        return NULL;
    }

    // Do the actual fft
    double *data_in = (double *) PyArray_DATA(in_arr);
    double *data_out = (double *) PyArray_DATA(out_arr);
    hhfft_1d->ifft(data_in, data_out);

    // Free the HHFFT object
    delete hhfft_1d;

    // Decrement reference counter
    Py_DECREF(in_arr);

    // Return the output array
    return (PyObject *) out_arr;
}

// FFT 2D directly
static PyObject *fft_2d(PyObject *, PyObject *args, PyObject *keywords)
{
    int type_in = NPY_COMPLEX128;
    int type_out = NPY_COMPLEX128;
    int ndims = 2;
    npy_intp dims_in[2];

    PyArrayObject *in_arr = NULL;
    PyArrayObject *out_arr = NULL;

    // Read the inputs
    if(!read_input(args, keywords, type_in, ndims, dims_in, &in_arr))
    {
        return NULL;
    }

    // Crate output array of same size as input
    if(!create_output(type_out, ndims, dims_in, &out_arr))
    {
        Py_DECREF(in_arr);
        return NULL;
    }

    // Initialize the HHFFT object
    hhfft::HHFFT_2D_D *hhfft_2d = NULL;
    try
    {
        size_t n = dims_in[0];
        size_t m = dims_in[1];
        hhfft_2d = new hhfft::HHFFT_2D_D(n,m);
    } catch (std::runtime_error &e)
    {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        Py_DECREF(in_arr);
        return NULL;
    }

    // Do the actual fft
    double *data_in = (double *) PyArray_DATA(in_arr);
    double *data_out = (double *) PyArray_DATA(out_arr);
    hhfft_2d->fft(data_in, data_out);

    // Free the HHFFT object
    delete hhfft_2d;

    // Decrement reference counter
    Py_DECREF(in_arr);

    // Return the output array
    return (PyObject *) out_arr;
}

// IFFT 2D directly
static PyObject *ifft_2d(PyObject *, PyObject *args, PyObject *keywords)
{
    int type_in = NPY_COMPLEX128;
    int type_out = NPY_COMPLEX128;
    int ndims = 2;
    npy_intp dims_in[2];

    PyArrayObject *in_arr = NULL;
    PyArrayObject *out_arr = NULL;

    // Read the inputs
    if(!read_input(args, keywords, type_in, ndims, dims_in, &in_arr))
    {
        return NULL;
    }

    // Crate output array of same size as input
    if(!create_output(type_out, ndims, dims_in, &out_arr))
    {
        Py_DECREF(in_arr);
        return NULL;
    }

    // Initialize the HHFFT object
    hhfft::HHFFT_2D_D *hhfft_2d = NULL;
    try
    {
        size_t n = dims_in[0];
        size_t m = dims_in[1];
        hhfft_2d = new hhfft::HHFFT_2D_D(n,m);
    } catch (std::runtime_error &e)
    {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        Py_DECREF(in_arr);
        return NULL;
    }

    // Do the actual fft
    double *data_in = (double *) PyArray_DATA(in_arr);
    double *data_out = (double *) PyArray_DATA(out_arr);
    hhfft_2d->ifft(data_in, data_out);

    // Free the HHFFT object
    delete hhfft_2d;

    // Decrement reference counter
    Py_DECREF(in_arr);

    // Return the output array
    return (PyObject *) out_arr;
}

// Members table
static PyMemberDef Hhfft1dPlan_members[] = {
    {(char *) "n", T_PYSSIZET, offsetof(Hhfft1dPlanObject, n), READONLY, (char *) "size of fft"},
    {NULL}
};

static PyMemberDef Hhfft1dRealPlan_members[] = {
    {(char *) "n", T_PYSSIZET, offsetof(Hhfft1dRealPlanObject, n), READONLY, (char *) "size of fft"},
    {NULL}
};

static PyMemberDef Hhfft2dPlan_members[] = {
    {(char *) "n", T_PYSSIZET, offsetof(Hhfft2dPlanObject, n), READONLY, (char *) "size of fft"},
    {(char *) "m", T_PYSSIZET, offsetof(Hhfft2dPlanObject, m), READONLY, (char *) "size of fft"},
    {NULL}
};

static PyMemberDef Hhfft2dRealPlan_members[] = {
    {(char *) "n", T_PYSSIZET, offsetof(Hhfft2dRealPlanObject, n), READONLY, (char *) "size of fft"},
    {(char *) "m", T_PYSSIZET, offsetof(Hhfft2dRealPlanObject, m), READONLY, (char *) "size of fft"},
    {NULL}
};

// Method table
static PyMethodDef Hhfft1dPlan_methods[] = {
    {"fft", (PyCFunction) Hhfft1dPlan_fft, METH_VARARGS | METH_KEYWORDS, "Fast Fourier transform, 1D complex"},
    {"ifft", (PyCFunction) Hhfft1dPlan_ifft, METH_VARARGS | METH_KEYWORDS, "Inverse Fast Fourier transform, 1D complex"},
    {NULL}
};

static PyMethodDef Hhfft1dRealPlan_methods[] = {
    {"fft", (PyCFunction) Hhfft1dRealPlan_fft, METH_VARARGS | METH_KEYWORDS, "Fast Fourier transform, 1D real input"},
    {"ifft", (PyCFunction) Hhfft1dRealPlan_ifft, METH_VARARGS | METH_KEYWORDS, "Inverse Fast Fourier transform, 1D real output"},
    {NULL}
};

static PyMethodDef Hhfft2dPlan_methods[] = {
    {"fft", (PyCFunction) Hhfft2dPlan_fft, METH_VARARGS | METH_KEYWORDS, "Fast Fourier transform, 2D complex"},
    {"ifft", (PyCFunction) Hhfft2dPlan_ifft, METH_VARARGS | METH_KEYWORDS, "Inverse Fast Fourier transform, 2D complex"},
    {NULL}
};

static PyMethodDef Hhfft2dRealPlan_methods[] = {
    {"fft", (PyCFunction) Hhfft2dRealPlan_fft, METH_VARARGS | METH_KEYWORDS, "Fast Fourier transform, 2D real input"},
    {"ifft", (PyCFunction) Hhfft2dRealPlan_ifft, METH_VARARGS | METH_KEYWORDS, "Inverse Fast Fourier transform, 2D real output"},
    {NULL}
};

// Methods to perform fft/ifft directly without making a separate plan first
static PyMethodDef Hhfft_methods[] = {
    {"fft", (PyCFunction) fft_1d, METH_VARARGS | METH_KEYWORDS, "Fast Fourier transform, 1D complex"},
    {"ifft", (PyCFunction) ifft_1d, METH_VARARGS | METH_KEYWORDS, "Inverse Fast Fourier transform, 1D complex"},
    {"fft2", (PyCFunction) fft_2d, METH_VARARGS | METH_KEYWORDS, "Fast Fourier transform, 2D complex"},
    {"ifft2", (PyCFunction) ifft_2d, METH_VARARGS | METH_KEYWORDS, "Inverse Fast Fourier transform, 2D complex"},
    {NULL}
};


// HHFFT type
static PyTypeObject Hhfft1dPlan_type = {
    PyVarObject_HEAD_INIT(NULL, 0)
};

static PyTypeObject Hhfft1dRealPlan_type = {
    PyVarObject_HEAD_INIT(NULL, 0)
};

static PyTypeObject Hhfft2dPlan_type = {
    PyVarObject_HEAD_INIT(NULL, 0)
};

static PyTypeObject Hhfft2dRealPlan_type = {
    PyVarObject_HEAD_INIT(NULL, 0)
};

// Module definition structure
static struct PyModuleDef hhfft_module = {
    PyModuleDef_HEAD_INIT
};

// Initialization function
extern "C" PyMODINIT_FUNC PyInit_hhfft(void)
{   
    // Initialize the Hhfft1dPlan_type
    Hhfft1dPlan_type.tp_name = "hhfft.Hhfft1dPlan";
    Hhfft1dPlan_type.tp_doc = "HHFFT 1D Complex Plan";
    Hhfft1dPlan_type.tp_basicsize = sizeof(Hhfft1dPlanObject);
    Hhfft1dPlan_type.tp_itemsize = 0;
    Hhfft1dPlan_type.tp_flags = Py_TPFLAGS_DEFAULT;
    Hhfft1dPlan_type.tp_new = Hhfft1dPlan_new;
    Hhfft1dPlan_type.tp_init = (initproc) Hhfft1dPlan_init;
    Hhfft1dPlan_type.tp_dealloc = (destructor) Hhfft1dPlan_dealloc;
    Hhfft1dPlan_type.tp_members = Hhfft1dPlan_members;
    Hhfft1dPlan_type.tp_methods = Hhfft1dPlan_methods;

    // Initialize the Hhfft1dRealPlan_type
    Hhfft1dRealPlan_type.tp_name = "hhfft.Hhfft1dRealPlan";
    Hhfft1dRealPlan_type.tp_doc = "HHFFT 1D Complex Plan";
    Hhfft1dRealPlan_type.tp_basicsize = sizeof(Hhfft1dRealPlanObject);
    Hhfft1dRealPlan_type.tp_itemsize = 0;
    Hhfft1dRealPlan_type.tp_flags = Py_TPFLAGS_DEFAULT;
    Hhfft1dRealPlan_type.tp_new = Hhfft1dRealPlan_new;
    Hhfft1dRealPlan_type.tp_init = (initproc) Hhfft1dRealPlan_init;
    Hhfft1dRealPlan_type.tp_dealloc = (destructor) Hhfft1dRealPlan_dealloc;
    Hhfft1dRealPlan_type.tp_members = Hhfft1dRealPlan_members;
    Hhfft1dRealPlan_type.tp_methods = Hhfft1dRealPlan_methods;

    // Initialize the Hhfft2dPlan_type
    Hhfft2dPlan_type.tp_name = "hhfft.Hhfft2dPlan";
    Hhfft2dPlan_type.tp_doc = "HHFFT 2D Complex Plan";
    Hhfft2dPlan_type.tp_basicsize = sizeof(Hhfft2dPlanObject);
    Hhfft2dPlan_type.tp_itemsize = 0;
    Hhfft2dPlan_type.tp_flags = Py_TPFLAGS_DEFAULT;
    Hhfft2dPlan_type.tp_new = Hhfft2dPlan_new;
    Hhfft2dPlan_type.tp_init = (initproc) Hhfft2dPlan_init;
    Hhfft2dPlan_type.tp_dealloc = (destructor) Hhfft2dPlan_dealloc;
    Hhfft2dPlan_type.tp_members = Hhfft2dPlan_members;
    Hhfft2dPlan_type.tp_methods = Hhfft2dPlan_methods;

    // Initialize the Hhfft2dPlan_type
    Hhfft2dRealPlan_type.tp_name = "hhfft.Hhfft2dRealPlan";
    Hhfft2dRealPlan_type.tp_doc = "HHFFT 2D Real Plan";
    Hhfft2dRealPlan_type.tp_basicsize = sizeof(Hhfft2dRealPlanObject);
    Hhfft2dRealPlan_type.tp_itemsize = 0;
    Hhfft2dRealPlan_type.tp_flags = Py_TPFLAGS_DEFAULT;
    Hhfft2dRealPlan_type.tp_new = Hhfft2dRealPlan_new;
    Hhfft2dRealPlan_type.tp_init = (initproc) Hhfft2dRealPlan_init;
    Hhfft2dRealPlan_type.tp_dealloc = (destructor) Hhfft2dRealPlan_dealloc;
    Hhfft2dRealPlan_type.tp_members = Hhfft2dRealPlan_members;
    Hhfft2dRealPlan_type.tp_methods = Hhfft2dRealPlan_methods;

    if (PyType_Ready(&Hhfft1dPlan_type) < 0)
    {
        return NULL;
    }

    if (PyType_Ready(&Hhfft1dRealPlan_type) < 0)
    {
        return NULL;
    }

    if (PyType_Ready(&Hhfft2dPlan_type) < 0)
    {
        return NULL;
    }

    if (PyType_Ready(&Hhfft2dRealPlan_type) < 0)
    {
        return NULL;
    }

    // Initialize the hhfftmodule
    hhfft_module.m_name = "hhfft";
    hhfft_module.m_doc = "HHFFT module";
    hhfft_module.m_size = -1;
    hhfft_module.m_methods = Hhfft_methods;

    PyObject *module = PyModule_Create(&hhfft_module);
    if (module == NULL)
    {
        return NULL;
    }

    Py_INCREF(&Hhfft1dPlan_type);
    Py_INCREF(&Hhfft1dRealPlan_type);
    Py_INCREF(&Hhfft2dPlan_type);
    Py_INCREF(&Hhfft2dRealPlan_type);

    PyModule_AddObject(module, "Hhfft1dPlan", (PyObject *) &Hhfft1dPlan_type);
    PyModule_AddObject(module, "Hhfft1dRealPlan", (PyObject *) &Hhfft1dRealPlan_type);
    PyModule_AddObject(module, "Hhfft2dPlan", (PyObject *) &Hhfft2dPlan_type);
    PyModule_AddObject(module, "Hhfft2dRealPlan", (PyObject *) &Hhfft2dRealPlan_type);

    import_array();

    return module;
}



