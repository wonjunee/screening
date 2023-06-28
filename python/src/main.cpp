#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>

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

class HelperClass{
public:
// variatbles
    int n1;
    int n2;
    double dx_;
    double dy_;

    std::vector<double> vxx_;
    std::vector<double> vxy_;
    std::vector<double> vyy_;

    HelperClass(py::array_t<double>& phi_np, const double dx, const double dy){
        py::buffer_info phi_buf = phi_np.request();
        this->n1 = phi_buf.shape[0];
        this->n2 = phi_buf.shape[1];

        this->dy_ = dy;
        this->dx_ = dx;

        vxx_.resize(n1*n2);
        vxy_.resize(n1*n2);
        vyy_.resize(n1*n2);
    }


    void print_out() const{
        std::cout << "n1: " << n1 << " n2: " << n2 << " dx: " << dx_ << " dy: " << dy_ << '\n';
    }

    // This function will provide S1(x,) where x and y are n [0,1] double values
    double interpolate_function(double x,double y, std::vector<double>& func) const{
        double indj=fmin(n1-1,fmax(0,x*n1-0.5));
        double indi=fmin(n2-1,fmax(0,y*n2-0.5));

        double lambda1=indj-(int)indj;
        double lambda2=indi-(int)indi;

        double x00 = func[(int)fmin(n2-1,fmax(0,indi))*n1+(int)fmin(n1-1,fmax(0,indj))];
        double x01 = func[(int)fmin(n2-1,fmax(0,indi))*n1+(int)fmin(n1-1,fmax(0,indj+1))];
        double x10 = func[(int)fmin(n2-1,fmax(0,indi+1))*n1+(int)fmin(n1-1,fmax(0,indj))];
        double x11 = func[(int)fmin(n2-1,fmax(0,indi+1))*n1+(int)fmin(n1-1,fmax(0,indj+1))];

        double interpolated_value = (1-lambda1)*(1-lambda2)*x00+(lambda1)*(1-lambda2)*x01
                                   +(1-lambda1)*(lambda2)*x10+(lambda1)*(lambda2)*x11;

        return interpolated_value;  
    }

    void calculate_gradient_vxx(std::vector<double>& vxx, const double* phi, const double dx){
        for(int i=0;i<n2;++i){
            for(int j=0;j<n1;++j){
                int jpp = fmin(n1-1,j+2);
                // int jp  = fmin(n1-1,j+1);
                // int jm  = fmax(0,j-1);
                int jmm = fmax(0,j-2);
                vxx[i*n1+j] = 0.25* (phi[i*n1+jpp] - phi[i*n1+j] - phi[i*n1+j] + phi[i*n1+jmm])/(dx*dx);
            }
        }
    }

    void calculate_gradient_vyy(std::vector<double>& vyy, const double* phi, const double dx){
        for(int i=0;i<n2;++i){
            for(int j=0;j<n1;++j){
                int ipp = fmin(n2-1,i+2);
                // int ip  = fmin(n2-1,i+1);
                // int im  = fmax(0,i-1);
                int imm  = fmax(0,i-2);
                vyy[i*n1+j] = 0.25 * (phi[ipp*n1+j] - phi[i*n1+j] - phi[i*n1+j] + phi[imm*n1+j])/(dx*dx);
            }
        }
    }

    void calculate_gradient_vxy(std::vector<double>& vxy, const double* phi, const double dx){
        for(int i=0;i<n2;++i){
            for(int j=0;j<n1;++j){
                int ip  = fmin(n2-1,i+1);
                int im  = fmax(0,i-1);
                int jp  = (int)fmin(n1-1,j+1);
                int jm  = (int)fmax(0,j-1);
                vxy[i*n1+j] = 0.25 * (phi[ip*n1+jp] - phi[ip*n1+jm] - phi[im*n1+jp] +phi[im*n1+jm])/(dx*dx);
            }
        }
    }


    /**
     * (rhsx, rhsy) = \nabla (\phi - b)
     * (outputx, outputy) = g^{ij}\nabla (\phi - b)
     * 
     * input:
     *  1. np outputx_np
     *  2. np outputy_np
     *  3. np phi_np
     *  4. np psi_np
     *  5. np rhsx_np
     *  6. np rhsy_np 
     */
    void compute_inverse_g(py::array_t<double>& outputx_np, py::array_t<double>& outputy_np, py::array_t<double>& phi_np, py::array_t<double>& psi_np, py::array_t<double>& rhsx_np, py::array_t<double>& rhsy_np){
        py::buffer_info outputx_buf = outputx_np.request();
        py::buffer_info outputy_buf = outputy_np.request();
        py::buffer_info phi_buf = phi_np.request();
        py::buffer_info psi_buf = psi_np.request();
        py::buffer_info rhsx_buf = rhsx_np.request();
        py::buffer_info rhsy_buf = rhsy_np.request();
        double* outputx = static_cast<double *>(outputx_buf.ptr);
        double* outputy = static_cast<double *>(outputy_buf.ptr);
        double* phi = static_cast<double *>(phi_buf.ptr);
        double* psi = static_cast<double *>(psi_buf.ptr);
        double* rhsx = static_cast<double *>(rhsx_buf.ptr);
        double* rhsy = static_cast<double *>(rhsy_buf.ptr);

        calculate_gradient_vxx(vxx_, psi, dx_);
        calculate_gradient_vyy(vyy_, psi, dx_);
        calculate_gradient_vxy(vxy_, psi, dx_);

        // step 1: for each point we will find T(x)
        for(int i=0;i<n2;++i){
            for(int j=0;j<n1;++j){

                double Sy1=(phi[i*n1+static_cast<int>(fmin(0,j+1))]-phi[i*n1+j])/dy_;
                double Sy2=(phi[static_cast<int>(fmin(0,i+1))*n1+j]-phi[i*n1+j])/dy_;

                double vxx_val = - interpolate_function(Sy1,Sy2,vxx_);
                double vyy_val = - interpolate_function(Sy1,Sy2,vyy_);
                double vxy_val = - interpolate_function(Sy1,Sy2,vxy_);

                outputx[i*n1+j] = vxx_val * rhsx[i*n1+j] + vxy_val * rhsy[i*n1+j];
                outputy[i*n1+j] = vxy_val * rhsx[i*n1+j] + vyy_val * rhsy[i*n1+j];

                std::cout << "Outputx: " << outputx[i*n1+j] << '\n';
            }
        }
    }
};
    





// void compute_inverse_g(py::array_t<double>& phi_np){
//     py::buffer_info phi_buf = phi_np.request();
//     double* phi = static_cast<double *>(phi_buf.ptr);

//     int n1 = phi_buf.shape[0];
//     int n2 = phi_buf.shape[1];

//     std::vector<std::vector<int> > directions = {{1,0}, {0,1}, {-1,0}, {0,-1}};

//     // interior
//     for(int i=1;i<n2-1;++i){
//         for(int j=1;j<n1-1;++j){

//         }
//     }
// }


// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---


PYBIND11_MODULE(screening, m) {
    // optional module docstring
    m.doc() = "pybind11 for screening code";

    m.def("function1", &function1, "example function");

    py::class_<HelperClass>(m, "HelperClass")
            .def(py::init<py::array_t<double> &, double, double>()) // py::array_t<double>& phi_np, const double dx, const double dy
            .def("compute_inverse_g", &HelperClass::compute_inverse_g);
}
