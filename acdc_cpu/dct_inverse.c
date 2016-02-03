
#section support_code

#define DTYPE double

bool tensor_same_shape(PyArrayObject* arr1, PyArrayObject* arr2)
{
    if (PyArray_NDIM(arr1) != PyArray_NDIM(arr2))
        return false;
    for (uint i = 0; i < PyArray_NDIM(arr1); ++i)
        if (PyArray_DIMS(arr1)[i] != PyArray_DIMS(arr2)[i])
            return false;
    return true;
}

#section support_code_apply

bool Tensor_run_idct(PyArrayObject* input0, PyArrayObject** output0)
{
    // Validate that the output storage exists and has the same
    // dimension as x.
    if (NULL == *output0 || !(tensor_same_shape(input0, *output0)))
    {
        /* Reference received to invalid output variable.
        Decrease received reference's ref count and allocate new
        output variable */
        Py_XDECREF(*output0);
        *output0 = (PyArrayObject*)PyArray_EMPTY(PyArray_NDIM(input0),
                                                PyArray_DIMS(input0),
                                                TYPENUM_OUTPUT_0,
                                                0);
        if (!*output0) {
            PyErr_Format(PyExc_ValueError,
                        "Could not allocate output storage");
            return false;
        }
    }

    int batch_size;
    int input_size;

    if (PyArray_NDIM(input0) == 1) {
        batch_size = 1;
        input_size = PyArray_DIMS(input0)[0];
    }
    else if (PyArray_NDIM(input0) == 2) {
        batch_size = PyArray_DIMS(input0)[0];
        input_size = PyArray_DIMS(input0)[1];
    }
    else {
        return false;
    }

    // normalize
    for (int i = 0; i < batch_size; ++i) {
        DTYPE_INPUT_0* input_data_ptr = (DTYPE_INPUT_0*)PyArray_DATA(input0);
        DTYPE_INPUT_0* example = input_data_ptr + i * input_size;
        example[0] *= std::sqrt(1.0 / input_size);

        for (int j = 1; j < input_size; ++j) {
            example[j] *= std::sqrt(1.0 / (2.0 * input_size));
        }
    }

    typename fftw<DTYPE>::plan_type fft_plan = fftw<DTYPE>::plan_many_dift_r2r_1d(
        input_size,
        batch_size,
        (DTYPE_INPUT_0*)PyArray_DATA(input0),
        (DTYPE_OUTPUT_0*)PyArray_DATA(*output0),
        FFTW_ESTIMATE);

    fftw<DTYPE>::execute(fft_plan);
    fftw<DTYPE>::destroy_plan(fft_plan);

    return true;
}

int APPLY_SPECIFIC(IDCT_updateOutput)(PyArrayObject* input0,
                                      PyArrayObject** output0) {

    if (!Tensor_run_idct(input0, output0)) {
        PyErr_Format(PyExc_ValueError,
                    "IDCT updateOutput failed");
        return 1;
    }
    return 0;
}

#undef DTYPE
