#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>

namespace py = pybind11;

/**
 * input f_np, a_np
 * output f_np
 */
void compute_dx(py::array_t<double>& out_np, py::array_t<double>& in_np, double dy){
   
    py::buffer_info out_buf = out_np.request();
    py::buffer_info in_buf = in_np.request();
    double* out = static_cast<double *>(out_buf.ptr);
    double* phi = static_cast<double *>(in_buf.ptr);

    int n1 = in_buf.shape[0];
    int n2 = in_buf.shape[1];

    // interior
    for(int i=0;i<n2;++i){
        for(int j=1;j<n1;++j){
            int jm = j-1;
            out[i*n1+j] = (phi[i*n1+j] - phi[i*n1+jm])/(dy);
        }
    }

    // boundary (using interior points)
    // for(int i=0;i<n2;++i){
    //     int j = n1-1;
    //         int jm = n1-3;
    //         out[i*n1+j] = (phi[i*n1+j] - phi[i*n1+jm])/(2*dy);
    // }

    // boundary (Neumann)
    for(int i=0;i<n2;++i){
        int j = 0;
            out[i*n1+j] = 0;
    }
}
  
  
  
/**
 * input f_np, a_np
 * output f_np
 */
void compute_dy(py::array_t<double>& out_np, py::array_t<double>& in_np, double dy){
   
    py::buffer_info out_buf = out_np.request();
    py::buffer_info in_buf = in_np.request();
    double* out = static_cast<double *>(out_buf.ptr);
    double* phi = static_cast<double *>(in_buf.ptr);

    int n1 = in_buf.shape[0];
    int n2 = in_buf.shape[1];

    // interior
    for(int i=1;i<n2;++i){
        for(int j=0;j<n1;++j){
            int im = i-1;
            out[i*n1+j] = (phi[i*n1+j] - phi[im*n1+j])/(dy);
        }
    }
    // boundary(using interior points)
    // int i = n2-1;
    //     for(int j=0;j<n1;++j){
    //         int im = n2-3;
    //         out[i*n1+j] = (phi[i*n1+j] - phi[im*n1+j])/(2*dy);
    //     }

    // boundary (Neumann)
    int i = 0;
        for(int j=0;j<n1;++j){
            out[i*n1+j] = 0;
        }
}

  
/**
 * input f_np, a_np
 * output f_np
 */
void c_transform_cpp(py::array_t<double>& out_np, py::array_t<double>& phi_np, py::array_t<double>& cost_np){
   
    py::buffer_info out_buf = out_np.request();
    py::buffer_info phi_buf = phi_np.request();
    py::buffer_info cost_buf = cost_np.request();
    double* out = static_cast<double *>(out_buf.ptr);
    double* phi = static_cast<double *>(phi_buf.ptr);
    double* cost = static_cast<double *>(cost_buf.ptr);

    int n1 = phi_buf.shape[0];
    int n2 = phi_buf.shape[1];

    int N = n1*n2;
    // c-transform: calculate phi^c(x) = inf_y phi(y) + c(x,y) => phi^c_i = min_\jb \phi_\jb + c_{i, \jb}
    for(int i=0;i<N;++i){
        double val = phi[0] + cost[i*N+0];
        for(int j=1;j<N;++j){
            val = fmin(val, phi[j] + cost[i*N+j]);
        }
        out[i] = val;
    }
}


  
/**
 * input f_np, a_np
 * output f_np
 */
void c_transform_forward_cpp(py::array_t<double>& out_np, py::array_t<double>& psi_np, py::array_t<double>& cost_np){
   
    py::buffer_info out_buf = out_np.request();
    py::buffer_info psi_buf = psi_np.request();
    py::buffer_info cost_buf = cost_np.request();
    double* out = static_cast<double *>(out_buf.ptr);
    double* psi = static_cast<double *>(psi_buf.ptr);
    double* cost = static_cast<double *>(cost_buf.ptr);

    int n1 = psi_buf.shape[0];
    int n2 = psi_buf.shape[1];

    int N = n1*n2;
    // c-transform: calculate psi^c(x) = max_x psi(x) - c(x,y) => psi^c_\jb = min_i \psi_i - c_{i, \jb}
    for(int j=0;j<N;++j){
        double val = psi[0] - cost[0*N+j];
        for(int i=1;i<N;++i){    
            val = fmax(val, psi[i] - cost[i*N+j]);
        }
        out[j] = val;
    }
}



  
/**
 * input f_np, a_np
 * output f_np
 * (nu: torch.tensor, psi: torch.tensor, phi: torch.tensor, cost: torch.tensor, epsilon: float, dx: float, dy: float)
 */
void approx_push_cpp(py::array_t<double>& out_np, py::array_t<double>& psi_np, py::array_t<double>& phi_np, py::array_t<double>& cost_np, double epsilon, double dx, double dy, int yMax){
   
    py::buffer_info out_buf = out_np.request();
    py::buffer_info psi_buf = psi_np.request();
    py::buffer_info phi_buf = phi_np.request();
    py::buffer_info cost_buf = cost_np.request();
    double* out = static_cast<double *>(out_buf.ptr);
    double* psi = static_cast<double *>(psi_buf.ptr);
    double* phi = static_cast<double *>(phi_buf.ptr);
    double* cost = static_cast<double *>(cost_buf.ptr);

    int n1 = psi_buf.shape[0];
    int n2 = psi_buf.shape[1];

    int N = n1*n2;
    std::vector<double> M(N*N); // M_{i \jb} = exp( (psi_i - phi_\jb - c_{i,\jb})/\epsilon)
    for(int i=0;i<N;++i){
        for(int jb=0;jb<N;++jb){
            M[i*N+jb] = exp( (psi[i] - phi[jb] - cost[i*N+jb]) / epsilon);
        }
    }

    std::vector<double> Msum(N); // Msum_{i} = \sum_{\jb'} M_{i\jb'} dy^2
    for(int i=0;i<N;++i){
        double val = 0;
        for(int jb=0;jb<N;++jb){
            val += M[i*N+jb];
        }
        Msum[i] = val * dy * dy;
    }

    std::vector<double> pi(N*N); // pi_{i \jb} = M_{i \jb} / Msum_{i}
    for(int i=0;i<N;++i){
        for(int j=0;j<N;++j){
            pi[i*N+j] = M[i*N+j] / Msum[i];
        }
    }

    for(int j=0;j<N;++j){
        double val = 0;
        for(int i=0;i<N;++i){
            val += pi[i*N+j];
        }
        out[j] = val * dx * dx;
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

    std::vector< std::vector<int> > stencils_;

    HelperClass(py::array_t<double>& phi_np, const double dx, const double dy){
        py::buffer_info phi_buf = phi_np.request();
        this->n1 = phi_buf.shape[0];
        this->n2 = phi_buf.shape[1];

        this->dy_ = dy;
        this->dx_ = dx;

        vxx_.resize(n1*n2);
        vxy_.resize(n1*n2);
        vyy_.resize(n1*n2);

        // stencils_ = {{0,1}, {2,1}, {1,1}, {1,2}};
        stencils_ = {{0,1}, {3,1}, {2,1}, {3,2}, {1,1}, {2,3}, {1,2}, {1,3}};

        int N = stencils_.size();
        for(int i=0;i<3;++i){
            for(int j=0;j<N;++j){
                stencils_.push_back( {- stencils_[i*N+j][1], stencils_[i*N+j][0]} );
            }
        }
    }


    void print_out() const{
        std::cout << "n1: " << n1 << " n2: " << n2 << " dx: " << dx_ << " dy: " << dy_ << '\n';
    }

    // This function will provide S1(x,) where x and y are n [0,1] double values
    double interpolate_function(double x,double y, std::vector<double>& func) const{
        double indj=fmin(n1-1,fmax(0,x/dy_-0.5));
        double indi=fmin(n2-1,fmax(0,y/dy_-0.5));

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
                // int jpp = fmin(n1-1,j+2);
                int jp  = fmin(n1-1,j+1);
                int jm  = fmax(0,j-1);
                // int jmm = fmax(0,j-2);
                vxx[i*n1+j] = 1.0 * (phi[i*n1+jp] - phi[i*n1+j] - phi[i*n1+j] + phi[i*n1+jm])/(dx*dx);
            }
        }
    }

    void calculate_gradient_vyy(std::vector<double>& vyy, const double* phi, const double dx){
        for(int i=0;i<n2;++i){
            for(int j=0;j<n1;++j){
                // int ipp = fmin(n2-1,i+2);
                int ip  = fmin(n2-1,i+1);
                int im  = fmax(0,i-1);
                // int imm  = fmax(0,i-2);
                vyy[i*n1+j] = 1.0 * (phi[ip*n1+j] - phi[i*n1+j] - phi[i*n1+j] + phi[im*n1+j])/(dx*dx);
            }
        }
    }

    void calculate_gradient_vxy(std::vector<double>& vxy, const double* phi, const double dx){
        for(int i=0;i<n2;++i){
            for(int j=0;j<n1;++j){
                int ip  = fmin(n2-1,i+1);
                int im  = fmax(0,i-1);
                int jp  = fmin(n1-1,j+1);
                int jm  = fmax(0,j-1);
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
                int ind = i*n1+j;
                // int jp = static_cast<int>(fmin(n1-1,j+1));
                // int ip = static_cast<int>(fmin(n2-1,i+1));
                // double Sy1=(phi[i*n1+jp]-phi[ind])/dy_;
                // double Sy2=(phi[ip*n1+j]-phi[ind])/dy_;

                int jm = static_cast<int>(fmax(0,j-1));
                int im = static_cast<int>(fmax(0,i-1));
                double Sy1=(phi[ind]-phi[i*n1+jm])/dy_;
                double Sy2=(phi[ind]-phi[im*n1+j])/dy_;

                // auto save_st = stencils_[0];
                // double val = INT_MIN;
                // for(auto st : stencils_){
                //     int jp = j + st[0];
                //     int ip = i + st[1];
                //     // check if interior
                //     if(jp >= 0 && jp <= n1-1 && ip >= 0 && ip <= n2-1){
                //         double new_val = (phi[ip*n1+jp] - phi[ind])/(dy_ * sqrt(st[0]*st[0] + st[1]*st[1]));
                //         if(new_val > val){
                //             val = new_val;
                //             save_st = st;
                //         }
                //     }
                // }
                // double Sy1 = save_st[0] * fabs(val);
                // double Sy2 = save_st[1] * fabs(val);
                
                double vxx_val = - interpolate_function(Sy1,Sy2,vxx_);
                double vyy_val = - interpolate_function(Sy1,Sy2,vyy_);
                double vxy_val = - interpolate_function(Sy1,Sy2,vxy_);

                outputx[ind] = vxx_val * rhsx[ind] + vxy_val * rhsy[ind];
                outputy[ind] = vxy_val * rhsx[ind] + vyy_val * rhsy[ind];
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
    void compute_inverse_g2(py::array_t<double>& outputx_np, py::array_t<double>& outputy_np, py::array_t<double>& phi_np, py::array_t<double>& psi_np, py::array_t<double>& rhs_np){
        py::buffer_info outputx_buf = outputx_np.request();
        py::buffer_info outputy_buf = outputy_np.request();
        py::buffer_info phi_buf = phi_np.request();
        py::buffer_info psi_buf = psi_np.request();
        py::buffer_info rhs_buf = rhs_np.request();
        
        double* outputx = static_cast<double *>(outputx_buf.ptr);
        double* outputy = static_cast<double *>(outputy_buf.ptr);
        double* phi = static_cast<double *>(phi_buf.ptr);
        double* psi = static_cast<double *>(psi_buf.ptr);
        double* rhs = static_cast<double *>(rhs_buf.ptr);

        // step 1: for each point we will find T(x)
        for(int i=0;i<n2;++i){
            for(int j=0;j<n1;++j){
                int ind = i*n1+j;
                auto xi_sol = stencils_[0];
                double min_val = INT_MAX;
                for(auto xi : stencils_){
                    int jp = j + xi[0];
                    int ip = i + xi[1];
                    // check if interior
                    if(jp >= 0 && jp <= n1-1 && ip >= 0 && ip <= n2-1){
                        double val = 0;
                        for(auto eta : stencils_){
                            int jp = j + eta[0];
                            int ip = i + eta[1];
                            if(jp >= 0 && jp <= n1-1 && ip >= 0 && ip <= n2-1){
                                double tmp = compute_g(xi, eta, phi, i, j) - compute_theta(eta, rhs, i, j);
                                val += tmp*tmp;
                            }else{
                                val += 100;
                            }
                        }
                        if(val < min_val){
                            min_val = val;
                            xi_sol  = xi;
                        }
                    }
                }

                outputx[ind] = xi_sol[0] * dy_;
                outputy[ind] = xi_sol[1] * dy_;
            }
        }
    }

    double compute_theta(std::vector<int>& eta, const double* rhs, int i, int j){
        int jp = j + eta[0];
        int ip = i + eta[1];
        // check if interior
        return (rhs[ip*n1+jp] - rhs[i*n1+j])/(dy_ * sqrt(eta[0]*eta[0] + eta[1]*eta[1]));
    }

    double compute_g(std::vector<int>& xi, std::vector<int>& eta, const double* phi, int i, int j){
        std::vector<int> xi_eta = {xi[0]+eta[0], xi[1]+eta[1]};
        return 0.5 * (compute_delta(xi_eta,phi,i,j) - compute_delta(xi,phi,i,j) - compute_delta(eta,phi,i,j));
    }

    double compute_delta(std::vector<int>& xi, const double* phi, int i, int j){
        double y1 = (j+0.5) * dy_;
        double y2 = (i+0.5) * dy_;

        int ind = i*n1+j;
        int jp = static_cast<int>(fmin(n1-1,j+1));
        int ip = static_cast<int>(fmin(n2-1,i+1));

        double Sy1=(phi[i*n1+jp]-phi[ind])/dy_;
        double Sy2=(phi[ip*n1+j]-phi[ind])/dy_;

        int j_new = j + xi[0];
        int i_new = i + xi[1];
        
        double y1_prime = (j_new + 0.5) * dy_;
        double y2_prime = (i_new + 0.5) * dy_;

        ind = i_new*n1+j_new;
        jp = static_cast<int>(fmin(n1-1,j_new+1));
        ip = static_cast<int>(fmin(n2-1,i_new+1));

        double Sy1_prime=(phi[i_new*n1+jp]-phi[ind])/dy_;
        double Sy2_prime=(phi[ip*n1+j_new]-phi[ind])/dy_;

        return compute_delta_cost(y1_prime, y2_prime, Sy1_prime, Sy2_prime, y1, y2, Sy1, Sy2);
    }

    double compute_delta_cost(double x_prime1, double x_prime2, double y_prime1, double y_prime2, double x1, double x2, double y1, double y2) const{
        double xyprime = - (x1*y_prime1 + x2*y_prime2);
        double xprimey = - (x_prime1*y1 + x_prime2*y2);
        double xy      = - (x1*y1 + x2*y2);
        double xprimeyprime = - (x_prime1*y_prime1 + x_prime2*y_prime2);
        return xyprime + xprimey - xy - xprimeyprime;
    }
};
    



// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---


PYBIND11_MODULE(screening, m) {
    // optional module docstring
    m.doc() = "C++ wrapper for screening code";

    m.def("compute_dx", &compute_dx, "compute gradient x");
    m.def("compute_dy", &compute_dy, "compute gradient y");
    
    m.def("c_transform_cpp", &c_transform_cpp, "compute gradient y");
    m.def("c_transform_forward_cpp", &c_transform_forward_cpp, "compute gradient y");

    m.def("approx_push_cpp", &approx_push_cpp, "approximate pushforward");

    py::class_<HelperClass>(m, "HelperClass")
        .def(py::init<py::array_t<double> &, double, double>()) // py::array_t<double>& phi_np, const double dx, const double dy
        .def("compute_inverse_g", &HelperClass::compute_inverse_g)
        .def("compute_inverse_g2", &HelperClass::compute_inverse_g2);
}
