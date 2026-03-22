#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <Accelerate/Accelerate.h>

static PyObject *stochastic_ops(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *k_list_obj = NULL;
    PyObject *py_obj = NULL;
    PyObject *vinvz_obj = NULL;
    PyObject *z_obj = NULL;
    PyObject *u_obj = Py_None;
    PyObject *k_seq = NULL;
    PyArrayObject *py_arr = NULL;
    PyArrayObject *vinvz_arr = NULL;
    PyArrayObject *z_arr = NULL;
    PyArrayObject *u_arr = NULL;
    PyArrayObject *first_k_arr = NULL;
    PyArrayObject **k_arrays = NULL;
    PyArrayObject *kpy_out = NULL;
    PyArrayObject *traces_out = NULL;
    PyArrayObject *uku_out = NULL;
    void *tmp_vec = NULL;
    void *tmp_mat = NULL;
    PyObject *result = NULL;
    Py_ssize_t k_count;
    npy_intp n;
    npy_intp s;
    int typenum;

    static char *kwlist[] = {"k_list", "py", "vinvz", "z", "u", NULL};
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOOO|O", kwlist,
            &k_list_obj, &py_obj, &vinvz_obj, &z_obj, &u_obj)) {
        return NULL;
    }

    k_seq = PySequence_Fast(k_list_obj, "k_list must be a sequence");
    if (k_seq == NULL) {
        return NULL;
    }
    k_count = PySequence_Fast_GET_SIZE(k_seq);
    if (k_count < 1) {
        PyErr_SetString(PyExc_ValueError, "k_list must contain at least one matrix");
        goto fail;
    }

    first_k_arr = (PyArrayObject *)PyArray_FROM_OTF(
        PySequence_Fast_GET_ITEM(k_seq, 0), NPY_NOTYPE, NPY_ARRAY_CARRAY_RO);
    if (first_k_arr == NULL) {
        goto fail;
    }
    typenum = PyArray_TYPE(first_k_arr);
    if (typenum != NPY_FLOAT32 && typenum != NPY_FLOAT64) {
        PyErr_SetString(PyExc_TypeError, "Only float32 and float64 arrays are supported");
        goto fail;
    }
    if (PyArray_NDIM(first_k_arr) != 2) {
        PyErr_SetString(PyExc_ValueError, "GRM matrices must be 2-D");
        goto fail;
    }
    n = PyArray_DIM(first_k_arr, 0);
    if (PyArray_DIM(first_k_arr, 1) != n) {
        PyErr_SetString(PyExc_ValueError, "GRM matrices must be square");
        goto fail;
    }

    py_arr = (PyArrayObject *)PyArray_FROM_OTF(py_obj, typenum, NPY_ARRAY_CARRAY_RO);
    vinvz_arr = (PyArrayObject *)PyArray_FROM_OTF(vinvz_obj, typenum, NPY_ARRAY_CARRAY_RO);
    z_arr = (PyArrayObject *)PyArray_FROM_OTF(z_obj, typenum, NPY_ARRAY_CARRAY_RO);
    if (py_arr == NULL || vinvz_arr == NULL || z_arr == NULL) {
        goto fail;
    }
    if (u_obj != Py_None) {
        u_arr = (PyArrayObject *)PyArray_FROM_OTF(u_obj, typenum, NPY_ARRAY_CARRAY_RO);
        if (u_arr == NULL) {
            goto fail;
        }
    }

    if (PyArray_NDIM(py_arr) != 1 || PyArray_DIM(py_arr, 0) != n) {
        PyErr_SetString(PyExc_ValueError, "py must have shape (n,)");
        goto fail;
    }
    if (PyArray_NDIM(vinvz_arr) != 2 || PyArray_DIM(vinvz_arr, 0) != n) {
        PyErr_SetString(PyExc_ValueError, "vinvz must have shape (n, s)");
        goto fail;
    }
    if (PyArray_NDIM(z_arr) != 2 ||
        PyArray_DIM(z_arr, 0) != n ||
        PyArray_DIM(z_arr, 1) != PyArray_DIM(vinvz_arr, 1)) {
        PyErr_SetString(PyExc_ValueError, "z must have shape (n, s) matching vinvz");
        goto fail;
    }
    if (u_arr != NULL &&
        (PyArray_NDIM(u_arr) != 1 || PyArray_DIM(u_arr, 0) != n)) {
        PyErr_SetString(PyExc_ValueError, "u must have shape (n,)");
        goto fail;
    }
    s = PyArray_DIM(vinvz_arr, 1);

    k_arrays = (PyArrayObject **)PyMem_Calloc((size_t)k_count, sizeof(PyArrayObject *));
    if (k_arrays == NULL) {
        PyErr_NoMemory();
        goto fail;
    }
    k_arrays[0] = first_k_arr;
    first_k_arr = NULL;
    for (Py_ssize_t k = 1; k < k_count; ++k) {
        k_arrays[k] = (PyArrayObject *)PyArray_FROM_OTF(
            PySequence_Fast_GET_ITEM(k_seq, k), typenum, NPY_ARRAY_CARRAY_RO);
        if (k_arrays[k] == NULL) {
            goto fail;
        }
        if (PyArray_NDIM(k_arrays[k]) != 2 ||
            PyArray_DIM(k_arrays[k], 0) != n ||
            PyArray_DIM(k_arrays[k], 1) != n) {
            PyErr_SetString(PyExc_ValueError, "All GRM matrices must have shape (n, n)");
            goto fail;
        }
    }

    npy_intp kpy_dims[2] = {n, (npy_intp)k_count};
    npy_intp trace_dims[1] = {(npy_intp)k_count};
    kpy_out = (PyArrayObject *)PyArray_SimpleNew(2, kpy_dims, typenum);
    traces_out = (PyArrayObject *)PyArray_SimpleNew(1, trace_dims, NPY_FLOAT64);
    if (kpy_out == NULL || traces_out == NULL) {
        goto fail;
    }
    if (u_arr != NULL) {
        uku_out = (PyArrayObject *)PyArray_SimpleNew(1, trace_dims, NPY_FLOAT64);
        if (uku_out == NULL) {
            goto fail;
        }
    }

    tmp_vec = PyMem_Malloc((size_t)n * (typenum == NPY_FLOAT32 ? sizeof(float) : sizeof(double)));
    tmp_mat = PyMem_Malloc((size_t)n * (size_t)s *
                           (typenum == NPY_FLOAT32 ? sizeof(float) : sizeof(double)));
    if (tmp_vec == NULL || tmp_mat == NULL) {
        PyErr_NoMemory();
        goto fail;
    }

    Py_BEGIN_ALLOW_THREADS
    if (typenum == NPY_FLOAT32) {
        const float *py_data = (const float *)PyArray_DATA(py_arr);
        const float *vinvz_data = (const float *)PyArray_DATA(vinvz_arr);
        const float *z_data = (const float *)PyArray_DATA(z_arr);
        const float *u_data = u_arr == NULL ? NULL : (const float *)PyArray_DATA(u_arr);
        float *kpy_data = (float *)PyArray_DATA(kpy_out);
        double *traces_data = (double *)PyArray_DATA(traces_out);
        double *uku_data = uku_out == NULL ? NULL : (double *)PyArray_DATA(uku_out);
        float *vec = (float *)tmp_vec;
        float *mat = (float *)tmp_mat;

        for (Py_ssize_t k = 0; k < k_count; ++k) {
            const float *k_data = (const float *)PyArray_DATA(k_arrays[k]);
            double trace_sum = 0.0;

            cblas_ssymv(CblasRowMajor, CblasLower, (int)n,
                        1.0f, k_data, (int)n, py_data, 1, 0.0f, vec, 1);
            for (npy_intp i = 0; i < n; ++i) {
                kpy_data[i * k_count + k] = vec[i];
            }

            cblas_ssymm(CblasRowMajor, CblasLeft, CblasLower,
                        (int)n, (int)s,
                        1.0f, k_data, (int)n, vinvz_data, (int)s,
                        0.0f, mat, (int)s);
            for (npy_intp i = 0; i < n * s; ++i) {
                trace_sum += (double)mat[i] * (double)z_data[i];
            }
            traces_data[k] = trace_sum / (double)s;

            if (u_data != NULL) {
                double uku_sum = 0.0;
                cblas_ssymv(CblasRowMajor, CblasLower, (int)n,
                            1.0f, k_data, (int)n, u_data, 1, 0.0f, vec, 1);
                for (npy_intp i = 0; i < n; ++i) {
                    uku_sum += (double)vec[i] * (double)u_data[i];
                }
                uku_data[k] = uku_sum;
            }
        }
    } else {
        const double *py_data = (const double *)PyArray_DATA(py_arr);
        const double *vinvz_data = (const double *)PyArray_DATA(vinvz_arr);
        const double *z_data = (const double *)PyArray_DATA(z_arr);
        const double *u_data = u_arr == NULL ? NULL : (const double *)PyArray_DATA(u_arr);
        double *kpy_data = (double *)PyArray_DATA(kpy_out);
        double *traces_data = (double *)PyArray_DATA(traces_out);
        double *uku_data = uku_out == NULL ? NULL : (double *)PyArray_DATA(uku_out);
        double *vec = (double *)tmp_vec;
        double *mat = (double *)tmp_mat;

        for (Py_ssize_t k = 0; k < k_count; ++k) {
            const double *k_data = (const double *)PyArray_DATA(k_arrays[k]);
            double trace_sum = 0.0;

            cblas_dsymv(CblasRowMajor, CblasLower, (int)n,
                        1.0, k_data, (int)n, py_data, 1, 0.0, vec, 1);
            for (npy_intp i = 0; i < n; ++i) {
                kpy_data[i * k_count + k] = vec[i];
            }

            cblas_dsymm(CblasRowMajor, CblasLeft, CblasLower,
                        (int)n, (int)s,
                        1.0, k_data, (int)n, vinvz_data, (int)s,
                        0.0, mat, (int)s);
            for (npy_intp i = 0; i < n * s; ++i) {
                trace_sum += mat[i] * z_data[i];
            }
            traces_data[k] = trace_sum / (double)s;

            if (u_data != NULL) {
                double uku_sum = 0.0;
                cblas_dsymv(CblasRowMajor, CblasLower, (int)n,
                            1.0, k_data, (int)n, u_data, 1, 0.0, vec, 1);
                for (npy_intp i = 0; i < n; ++i) {
                    uku_sum += vec[i] * u_data[i];
                }
                uku_data[k] = uku_sum;
            }
        }
    }
    Py_END_ALLOW_THREADS

    if (uku_out == NULL) {
        Py_INCREF(Py_None);
        result = Py_BuildValue("NNN", (PyObject *)kpy_out, (PyObject *)traces_out, Py_None);
        kpy_out = NULL;
        traces_out = NULL;
    } else {
        result = Py_BuildValue("NNN", (PyObject *)kpy_out, (PyObject *)traces_out, (PyObject *)uku_out);
        kpy_out = NULL;
        traces_out = NULL;
        uku_out = NULL;
    }

fail:
    Py_XDECREF(k_seq);
    Py_XDECREF(py_arr);
    Py_XDECREF(vinvz_arr);
    Py_XDECREF(z_arr);
    Py_XDECREF(u_arr);
    Py_XDECREF(first_k_arr);
    if (k_arrays != NULL) {
        for (Py_ssize_t k = 0; k < k_count; ++k) {
            Py_XDECREF(k_arrays[k]);
        }
        PyMem_Free(k_arrays);
    }
    Py_XDECREF(kpy_out);
    Py_XDECREF(traces_out);
    Py_XDECREF(uku_out);
    PyMem_Free(tmp_vec);
    PyMem_Free(tmp_mat);
    return result;
}

static PyMethodDef module_methods[] = {
    {
        "stochastic_ops",
        (PyCFunction)stochastic_ops,
        METH_VARARGS | METH_KEYWORDS,
        "Compute KPy, stochastic traces, and optional u'K_ku terms."
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_greml_accel",
    NULL,
    -1,
    module_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit__greml_accel(void) {
    import_array();
    return PyModule_Create(&moduledef);
}
