#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

/**
 * input f_np, a_np
 * output f_np
 */
void function1(py::array_t<double>& f_np, py::array_t<double>& X_np){
   
    py::buffer_info f_buf = f_np.request();
    py::buffer_info X_buf = X_np.request();
    double* f = static_cast<double *>(f_buf.ptr);
    double* X = static_cast<double *>(X_buf.ptr);

    int n1 = X_buf.shape[0];
    int n2 = X_buf.shape[1];

    for(int i=1;i<n2-1;++i){
        for(int j=1;j<n1-1;++j){
            f[i*n1+j] = n1*n1*(X[(i+1)*n1+j]-2.0*X[(i-1)*n1+j]+X[(i-1)*n1+j]) + n2*n2*(X[i*n1+j+1]-2.0*X[i*n1+j]+X[i*n1+j-1]);
        }
    }
}



// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---


PYBIND11_MODULE(screening, m) {
    // optional module docstring
    m.doc() = "pybind11 for screening code";

    m.def("function1", &function1, "example function");
}
