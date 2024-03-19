#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <future>

namespace py = pybind11;

/**
 * input f_np, a_np
 * output f_np
 */
void compute_dx(py::array_t<double, py::array::c_style | py::array::forcecast> out_np, py::array_t<double, py::array::c_style | py::array::forcecast> in_np, double dy){
   
    py::buffer_info out_buf = out_np.request();
    py::buffer_info in_buf = in_np.request();
    double *out = static_cast<double *>(out_buf.ptr);
    double *phi = static_cast<double *>(in_buf.ptr);

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
void compute_dy(py::array_t<double, py::array::c_style | py::array::forcecast> out_np, py::array_t<double, py::array::c_style | py::array::forcecast> in_np, double dy){
   
    py::buffer_info out_buf = out_np.request();
    py::buffer_info in_buf = in_np.request();
    double *out = static_cast<double *>(out_buf.ptr);
    double *phi = static_cast<double *>(in_buf.ptr);

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

  
double compute_cost(const double dx, const double dy, const int i, const int j, const int n, const int m, double xMin=-1.5, double yMin=-1.5){
    int i1 = i/n;
    int i2 = i%n;

    int j1 = j/m;
    int j2 = j%m;

    double x1 = (i1+0.5)*dx + xMin;
    double x2 = (i2+0.5)*dx + xMin;

    double y1 = (j1+0.5)*dy + yMin;
    double y2 = (j2+0.5)*dy + yMin;

    return 0.5 * ((x1-y1)*(x1-y1) + (x2-y2)*(x2-y2) );
}
/**
 * input f_np, a_np
 * output f_np
 */
void c_transform_cpp(py::array_t<double, py::array::c_style | py::array::forcecast> psi_np, 
                     py::array_t<double, py::array::c_style | py::array::forcecast> phi_np,
                     const double dx, const double dy, const int n, const int m, const double xMin, const double yMin){
   
    py::buffer_info phi_buf = phi_np.request();
    py::buffer_info psi_buf = psi_np.request();
    double *phi = static_cast<double *>(phi_buf.ptr);
    double *psi = static_cast<double *>(psi_buf.ptr);

    int N = n*n;
    int M = m*m;
    // c-transform: calculate psi^c(x) = max_x psi(x) - c(x,y) => psi^c_\jb = min_i \psi_i - c_{i, \jb}
    for(int i=0;i<N;++i){
        int j = 0;
        double cost = compute_cost(dx,dy,i,j,n,m,xMin,yMin); 
        double val = phi[0] + cost;
        for(int j=1;j<M;++j){
            double cost = compute_cost(dx,dy,i,j,n,m,xMin,yMin); 
            val = fmin(val, phi[j] + cost);
        }
        psi[i] = val;
    }
}


  
/**
 * input f_np, a_np
 * output f_np
 */
void c_transform_forward_cpp(py::array_t<double, py::array::c_style | py::array::forcecast> phi_np, 
                             py::array_t<double, py::array::c_style | py::array::forcecast> psi_np,
                             const double dx, const double dy, const int n, const int m){
   
    py::buffer_info phi_buf = phi_np.request();
    py::buffer_info psi_buf = psi_np.request();
    double *phi = static_cast<double *>(phi_buf.ptr);
    double *psi = static_cast<double *>(psi_buf.ptr);

    int N = psi_buf.shape[0];
    int M = phi_buf.shape[0];
    // c-transform: calculate psi^c(x) = max_x psi(x) - c(x,y) => psi^c_\jb = min_i \psi_i - c_{i, \jb}
    for(int j=0;j<M;++j){
        int i = 0;
        double cost = compute_cost(dx,dy,i,j,n,m);
        double val = psi[0] - cost;
        for(int i=1;i<N;++i){
            double cost = compute_cost(dx,dy,i,j,n,m); 
            val = fmax(val, psi[i] - cost);
        }
        phi[j] = val;
    }
}



  
/**
 * input f_np, a_np
 * output f_np
 * (nu: torch.tensor, psi: torch.tensor, phi: torch.tensor, cost: torch.tensor, epsilon: float, dx: float, dy: float)
 */
void approx_push_cpp(py::array_t<double, py::array::c_style | py::array::forcecast> out_np, py::array_t<double, py::array::c_style | py::array::forcecast> psi_np, py::array_t<double, py::array::c_style | py::array::forcecast> phi_np, py::array_t<double, py::array::c_style | py::array::forcecast> cost_np, double epsilon, double dx, double dy, double yMax){
   
    py::buffer_info out_buf = out_np.request();
    py::buffer_info psi_buf = psi_np.request();
    py::buffer_info phi_buf = phi_np.request();
    py::buffer_info cost_buf = cost_np.request();
    double *out = static_cast<double *>(out_buf.ptr);
    double *psi = static_cast<double *>(psi_buf.ptr);
    double *phi = static_cast<double *>(phi_buf.ptr);
    double *cost = static_cast<double *>(cost_buf.ptr);

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

void pushforward_entropic_cpp(
                py::array_t<double, py::array::c_style | py::array::forcecast> nu_np, 
                py::array_t<double, py::array::c_style | py::array::forcecast> mu_np, 
                py::array_t<double, py::array::c_style | py::array::forcecast> psi_np, 
                py::array_t<double, py::array::c_style | py::array::forcecast> phi_np,
                double epsilon, 
                double dx, double dy, double xMin, double yMin, int n, int m){
   
    py::buffer_info nu_buf  = nu_np.request();
    py::buffer_info mu_buf  = mu_np.request();
    py::buffer_info psi_buf = psi_np.request();
    py::buffer_info phi_buf = phi_np.request();
    double *nu = static_cast<double *>(nu_buf.ptr);
    double *mu = static_cast<double *>(mu_buf.ptr);
    double *psi = static_cast<double *>(psi_buf.ptr);
    double *phi = static_cast<double *>(phi_buf.ptr);

    int N = n*n;
    int M = m*m;
    for(int j=0;j<M;++j){
        nu[j] = 0;
        for(int i=0;i<N;++i){
            double cost = compute_cost(dx, dy, i, j, n, m, xMin, yMin);
            nu[j] += exp((psi[i] - phi[j] - cost)/epsilon) * mu[i];
        }
    }
}


/**
 * input f_np, a_np
 * output f_np
 * (nu: torch.tensor, psi: torch.tensor, phi: torch.tensor, cost: torch.tensor, epsilon: float, dx: float, dy: float)
 */
void c_transform_epsilon_cpp(py::array_t<double, py::array::c_style | py::array::forcecast> psi_eps_np, py::array_t<double, py::array::c_style | py::array::forcecast> psi_np, py::array_t<double, py::array::c_style | py::array::forcecast> phi_np, py::array_t<double, py::array::c_style | py::array::forcecast> cost_np, double epsilon, double dx, double dy, double yMax){
   
    py::buffer_info psi_eps_buf = psi_eps_np.request();
    // py::buffer_info psi_buf = psi_np.request();
    py::buffer_info phi_buf = phi_np.request();
    py::buffer_info cost_buf = cost_np.request();
    double *psi_eps = static_cast<double *>(psi_eps_buf.ptr);
    // double *psi = static_cast<double *>(psi_buf.ptr);
    double *phi = static_cast<double *>(phi_buf.ptr);
    double *cost = static_cast<double *>(cost_buf.ptr);

    
    int N = cost_buf.shape[0];
    int M = cost_buf.shape[0];
    // py::print("N",N,"M",M);
    for(int i=0;i<N;++i){
        double val = 0;
        for(int jb=0;jb<M;++jb){
            val += exp( (- phi[jb] - cost[i*M+jb]) / epsilon);
        }
        psi_eps[i] = - epsilon * log(val * dy * dy);
    }
}


double compute_pi_ij(double *psi, double *phi, double *cost, double epsilon, int N, int M, int i, int j){
    return exp( (psi[i] - phi[j] - cost[i*M+j]) / epsilon);
}

/**
 * S(y)   = \argmax_x \psi(x) - c(x,y)
 * phi(y) = \max_x \psi(x) - c(x,y) = \psi(S(y)) - c(S(y),y)
 * Computing S_ind:[0...n_Y-1] -> [0...n_X-1]
*/
void compute_Sy_cpp(py::array_t<int>& out_np, py::array_t<double, py::array::c_style | py::array::forcecast> psi_np, py::array_t<double, py::array::c_style | py::array::forcecast> cost_np){
    py::buffer_info out_buf  = out_np.request();
    py::buffer_info psi_buf  = psi_np.request();
    py::buffer_info cost_buf = cost_np.request();

    int *out     = static_cast<int *>(out_buf.ptr);
    double *psi  = static_cast<double *>(psi_buf.ptr);
    double *cost = static_cast<double *>(cost_buf.ptr);
    
    int n1 = psi_buf.shape[0];
    int n2 = psi_buf.shape[1];

    int N = n1*n2;

    for(int j=0;j<N;++j){
        int target_ind = 0;
        double val         = psi[0] - cost[0*N+j];
        for(int i=1;i<N;++i){
            double new_val = psi[i] - cost[i*N+j];
            if(new_val > val){
                target_ind = i;
                val        = new_val;
            }
        }
        out[j] = target_ind;
    }
    py::print("done");
}

/**
 * c(x,y) = - x . y
*/
double compute_cost(const double x1,const double x2,const double y1,const double y2){
    return - x1*y1 - x2*y2;
}
    



/**
 * a = (y,y') b = (y,y'')
 * $G(y)(a,b)= c(S(y),y')+c(S(y''),y)-c(S(y),y)-c(S(y''),y')$
*/
double compute_G(double *cost, double *phi, int *S_ind, const double dy, const int n, const int N, const int y, const int yp, const int ypp){
    // ------------------------------------------
    // This commented thing is argmax version of computing the delta_c
    // ------------------------------------------
    // int S_y   = S_ind[y];
    // int S_ypp = S_ind[ypp];
    // return cost[S_y*N+yp] + cost[S_ypp*N+y] - cost[S_y*N+y] - cost[S_ypp*N+yp];
    // ------------------------------------------

    // computing S(y) using \nabla \phi
    // note: the backward or forward finite difference does not work (error at the boundary)
    //       Trying the centered difference instead.
    int i,j,im,jm,ip,jp;

    i = y/n;
    j = y%n;

    double y1 = (j+0.5)*dy;
    double y2 = (i+0.5)*dy;

    jm = int(fmax(0,j-1));
    jp = int(fmin(n-1,j-1));
    im = int(fmax(0,i-1));
    ip = int(fmin(n-1,i-1));
    double S_y1 = (phi[i*n+jp] - phi[i*n+jm])/(2*dy);
    double S_y2 = (phi[ip*n+j] - phi[im*n+j])/(2*dy);
    
    i = ypp/n;
    j = ypp%n;
    jm = int(fmax(0,j-1));
    jp = int(fmin(n-1,j-1));
    im = int(fmax(0,i-1));
    ip = int(fmin(n-1,i-1));
    double S_ypp1  = (phi[i*n+jp] - phi[i*n+jm])/(2*dy);
    double S_ypp2  = (phi[ip*n+j] - phi[im*n+j])/(2*dy);

    i = yp/n;
    j = yp%n;
    double yp1 = (j+0.5)*dy;
    double yp2 = (i+0.5)*dy;
    return compute_cost(S_y1,S_y2,yp1,yp2) + compute_cost(S_ypp1,S_ypp2,y1,y2) - compute_cost(S_y1,S_y2,y1,y2) - compute_cost(S_ypp1,S_ypp2,yp1,yp2);
}

/**
 * Computing G(y) which is a |B_y|x|B_y| matrix where B_y is the set of edges coming from y.
*/
void compute_Gy_cpp(py::array_t<double, py::array::c_style | py::array::forcecast> out_np, py::array_t<double, py::array::c_style | py::array::forcecast> cost_np, py::array_t<double, py::array::c_style | py::array::forcecast> phi_np, py::array_t<int>& S_ind_np, py::array_t<int>& edge2target_np, py::array_t<int>& node2edge_np, double dy, int node_ind){
    py::buffer_info out_buf        = out_np.request();
    py::buffer_info cost_buf       = cost_np.request();
    py::buffer_info phi_buf        = phi_np.request();
    py::buffer_info S_ind_buf      = S_ind_np.request();
    py::buffer_info edge2target_buf= edge2target_np.request();
    py::buffer_info node2edge_buf  = node2edge_np.request();

    double *out        = static_cast<double *>(out_buf.ptr);
    double *cost       = static_cast<double *>(cost_buf.ptr);
    double *phi        = static_cast<double *>(phi_buf.ptr);
    int    *S_ind      = static_cast<int *>(S_ind_buf.ptr);
    int    *edge2target= static_cast<int *>(edge2target_buf.ptr);
    int    *node2edge  = static_cast<int *>(node2edge_buf.ptr);
    
    int n = phi_buf.shape[0];
    int N = cost_buf.shape[0];

    int edge_start = node2edge[node_ind];
    int edge_end   = node2edge[node_ind+1];
    int count = 0;
    for(int ind2=edge_start;ind2<edge_end;++ind2){
        int target2 = edge2target[ind2];
        for(int ind1=edge_start;ind1<edge_end;++ind1){
            int target1 = edge2target[ind1];
            double val = compute_G(cost, phi, S_ind, dy, n,  N, node_ind, target1, target2);
            out[count] = val;
            count++;
        }
    }
}



/**
 * $G^{-1}(y)(z-y,z-y)$
 * G(y)(z)  = \delta_c(S(y),y,S(z),z) = c(S(y), z) + c(S(z), y) - c(S(y), y) - c(S(z), z)
 * G(indy)(indz) = C_{i_y, indz} + C_{i_z, indy} - C_{i_y, indy} - C_{i_z, indz}
 * i_y = S_ind(indy), i_z = S_ind(indz)
*/
double compute_G_inv(double *cost, int *S_ind, const int N, const int indy, const int indz){
    int i_y = S_ind[indy];
    int i_z = S_ind[indz];
    double val = cost[i_y*N+indz] + cost[i_z*N+indy] - cost[i_y*N+indy] - cost[i_z*N+indz];
    return 1.0/(val+1e-4); // added 1e-4 for numerical stability../
}



/**
 * input f_np, a_np
 * output f_np
 * (nu: torch.tensor, psi: torch.tensor, phi: torch.tensor, cost: torch.tensor, epsilon: float, dx: float, dy: float)
 */
void compute_nu_and_rho_cpp(py::array_t<double, py::array::c_style | py::array::forcecast> nu_np, py::array_t<double, py::array::c_style | py::array::forcecast> rho_np, py::array_t<double, py::array::c_style | py::array::forcecast> psi_np, py::array_t<double, py::array::c_style | py::array::forcecast> phi_np, py::array_t<double, py::array::c_style | py::array::forcecast> cost_np, py::array_t<double, py::array::c_style | py::array::forcecast> b_np, double epsilon, double dx, double dy, double yMax){
   
    py::buffer_info nu_buf   = nu_np.request();
    py::buffer_info rho_buf  = rho_np.request();
    py::buffer_info psi_buf  = psi_np.request();
    py::buffer_info phi_buf  = phi_np.request();
    py::buffer_info cost_buf = cost_np.request();
    py::buffer_info b_buf    = b_np.request();

    double *nu   = static_cast<double *>(nu_buf.ptr);
    double *rho  = static_cast<double *>(rho_buf.ptr);
    double *psi  = static_cast<double *>(psi_buf.ptr);
    double *phi  = static_cast<double *>(phi_buf.ptr);
    double *cost = static_cast<double *>(cost_buf.ptr);
    double *b    = static_cast<double *>(b_buf.ptr);
    
    int N = psi_buf.shape[0];
    int M = phi_buf.shape[0];

    std::vector<double> plan(N*M);

    for(int i=0;i<N;++i){
        for(int j=0;j<M;++j){
            plan[i*M+j] = exp( (psi[i] - phi[j] - cost[i*M+j]) / epsilon);
        }
    }
    // normalize plan
    double val = 0;
    for(int i=0;i<N*M;++i){
        val += plan[i];
    }
    for(int i=0;i<N*M;++i){
        plan[i] /= val;
    }

    // computing nu
    for(int jb=0;jb<M;++jb){
        double val = 0;
        for(int i=0;i<N;++i){
            val += compute_pi_ij(psi, phi, cost, epsilon, N, M, i, jb);
        }
        nu[jb] = val * dx * dx;
    }

    // defining Q_i := \sum_k (\phi-b)^k \pi^{ik} dy^2
    std::vector<double> Q(N);
    for(int i=0;i<N;++i){
        double val = 0;
        for(int jb=0;jb<M;++jb){
            val += compute_pi_ij(psi, phi, cost, epsilon, N, M, i, jb) * (phi[jb] - b[jb]);
        }
        Q[i] = val * dy * dy;
    }

    // computing rho
    for(int jb=0;jb<M;++jb){
        double sum2 = 0;
        for(int i=0;i<N;++i){
            sum2 += compute_pi_ij(psi, phi, cost, epsilon, N, M, i, jb) * Q[i];
        }
        // rho[jb] = (nu[jb] * (phi[jb] - b[jb]) - sum2*dx*dx)/epsilon;
        rho[jb] = sum2*dx*dx/epsilon;
    }
    // py::print("N:",N,"Hello, World!\n");
}

class HelperClass{
public:
// variatbles
    int n_;
    int m_;
    double dx_;
    double dy_;

    double *vxx_;
    double *vxy_;
    double *vyy_;

    double *x1Map_;
    double *x2Map_;

    double xMin_;
    double yMin_;

    std::vector< std::vector<int> > stencils_;

    HelperClass(
            const double dx, const double dy, const int n, const int m, const double xMin, const double yMin)
            :n_(n), m_(m), dx_(dx), dy_(dy), xMin_(xMin), yMin_(yMin) {
        
        x1Map_ = new double[(n_+1)*(n_+1)];
        x2Map_ = new double[(n_+1)*(n_+1)];
            
        // stencils_ = {{0,1}, {2,1}, {1,1}, {1,2}};
        stencils_ = {{0,1}, {3,1}, {2,1}, {3,2}, {1,1}, {2,3}, {1,2}, {1,3}};

        int N = stencils_.size();
        for(int i=0;i<3;++i){
            for(int j=0;j<N;++j){
                stencils_.push_back( {- stencils_[i*N+j][1], stencils_[i*N+j][0]} );
            }
        }
    }

    ~HelperClass(){
        delete[] x1Map_;
        delete[] x2Map_;
    }

    // This function will provide S1(x,) where x and y are n [0,1] double values
    double interpolate_function_Y(double x,double y, double dx, int n, std::vector<double>& func) const{
        double indj=fmin(n-1,fmax(0,x/dx-0.5));
        double indi=fmin(n-1,fmax(0,y/dx-0.5));

        double lambda1=indj-(int)indj;
        double lambda2=indi-(int)indi;

        double x00 = func[(int)fmin(n-1,fmax(0,indi))*n+(int)fmin(n-1,fmax(0,indj))];
        double x01 = func[(int)fmin(n-1,fmax(0,indi))*n+(int)fmin(n-1,fmax(0,indj+1))];
        double x10 = func[(int)fmin(n-1,fmax(0,indi+1))*n+(int)fmin(n-1,fmax(0,indj))];
        double x11 = func[(int)fmin(n-1,fmax(0,indi+1))*n+(int)fmin(n-1,fmax(0,indj+1))];

        double interpolated_value = (1-lambda1)*(1-lambda2)*x00+(lambda1)*(1-lambda2)*x01
                                   +(1-lambda1)*(lambda2)*x10+(lambda1)*(lambda2)*x11;

        return interpolated_value;  
    }

    // This function will provide S1(x,) where x and y are n [0,1] double values
    double interpolate_function_X(double x,double y, double dx, int n, double *func) const{
        double indj=fmin(n-1,fmax(0,(x-xMin_)/dx-0.5));
        double indi=fmin(n-1,fmax(0,(y-xMin_)/dx-0.5));

        double lambda1=indj-(int)indj;
        double lambda2=indi-(int)indi;

        double x00 = func[(int)fmin(n-1,fmax(0,indi))*n+(int)fmin(n-1,fmax(0,indj))];
        double x01 = func[(int)fmin(n-1,fmax(0,indi))*n+(int)fmin(n-1,fmax(0,indj+1))];
        double x10 = func[(int)fmin(n-1,fmax(0,indi+1))*n+(int)fmin(n-1,fmax(0,indj))];
        double x11 = func[(int)fmin(n-1,fmax(0,indi+1))*n+(int)fmin(n-1,fmax(0,indj+1))];

        double interpolated_value = (1-lambda1)*(1-lambda2)*x00+(lambda1)*(1-lambda2)*x01
                                   +(1-lambda1)*(lambda2)*x10+(lambda1)*(lambda2)*x11;

        return interpolated_value;  
    }

    void calculate_gradient_vxx(std::vector<double>& vxx, const double *psi, const double dx, const int n){
        for(int i=0;i<n;++i){
            for(int j=0;j<n;++j){
                // int jpp = fmin(n-1,j+2);
                int jp  = fmin(n-1,j+1);
                int jm  = fmax(0,j-1);
                // int jmm = fmax(0,j-2);
                vxx[i*n+j] = 1.0 * (psi[i*n+jp] - psi[i*n+j] - psi[i*n+j] + psi[i*n+jm])/(dx*dx);
            }
        }
    }

    void calculate_gradient_vyy(std::vector<double>& vyy, const double *psi, const double dx, const int n){
        for(int i=0;i<n;++i){
            for(int j=0;j<n;++j){
                // int ipp = fmin(n-1,i+2);
                int ip  = fmin(n-1,i+1);
                int im  = fmax(0,i-1);
                // int imm  = fmax(0,i-2);
                vyy[i*n+j] = 1.0 * (psi[ip*n+j] - psi[i*n+j] - psi[i*n+j] + psi[im*n+j])/(dx*dx);
            }
        }
    }

    void calculate_gradient_vxy(std::vector<double>& vxy, const double *psi, const double dx, const int n){
        for(int i=0;i<n;++i){
            for(int j=0;j<n;++j){
                int ip  = fmin(n-1,i+1);
                int im  = fmax(0,i-1);
                int jp  = fmin(n-1,j+1);
                int jm  = fmax(0,j-1);
                vxy[i*n+j] = 0.25 * (psi[ip*n+jp] - psi[ip*n+jm] - psi[im*n+jp] +psi[im*n+jm])/(dx*dx);
            }
        }
    }

    void compute_for_loop(double *outputx, double *outputy, double *phi, double *rhsx, double *rhsy, double dx, double dy, int n, int m, double tau,  const int start_ind, const int end_ind){
        for(int ind=start_ind;ind<end_ind;++ind){
            int i = ind / m;
            int j = ind % m;
            // int jm = static_cast<int>(fmax(0,j-1));
            // int im = static_cast<int>(fmax(0,i-1));
            // int jp = static_cast<int>(fmin(m-1,j+1));
            // int ip = static_cast<int>(fmin(m-1,i+1));
            double y1 = (j+0.5)*dy - yMin_;
            double y2 = (i+0.5)*dy - yMin_;
            // psi(x) - phi(y) = (x-y)^2/(2 * tau)
            // - tau \nabla phi(y) = y - S(y)
            // S(y) = y + \tau \nabla \phi(y)
            // T(x) = x - \tau \nabla \psi(x)
            double Sy1= y1 + tau * rhsx[ind]; // S(y) = y + \tau \nabla \phi(y) where rhsx and rhsy are \phi(y)
            double Sy2= y2 + tau * rhsy[ind];

            double vxx_val = interpolate_function_X(Sy1,Sy2,dx,n,vxx_);
            double vyy_val = interpolate_function_X(Sy1,Sy2,dx,n,vyy_);
            double vxy_val = interpolate_function_X(Sy1,Sy2,dx,n,vxy_);
            
            // g^{\ib \jb} \partial_{\jb}(\phi - b)
            outputx[ind] = (1-tau * vxx_val) * rhsx[ind] + (- tau * vxy_val) * rhsy[ind];
            outputy[ind] = (- tau * vxy_val) * rhsx[ind] + (1-tau * vyy_val) * rhsy[ind];
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
    void compute_inverse_g(
            py::array_t<double, py::array::c_style | py::array::forcecast> outputx_np, 
            py::array_t<double, py::array::c_style | py::array::forcecast> outputy_np, 
            py::array_t<double, py::array::c_style | py::array::forcecast> phi_np, 
            py::array_t<double, py::array::c_style | py::array::forcecast> psi_np, 
            py::array_t<double, py::array::c_style | py::array::forcecast> rhsx_np, 
            py::array_t<double, py::array::c_style | py::array::forcecast> rhsy_np,
            py::array_t<double, py::array::c_style | py::array::forcecast> vxx_np,
            py::array_t<double, py::array::c_style | py::array::forcecast> vyy_np,
            py::array_t<double, py::array::c_style | py::array::forcecast> vxy_np,
            const double dx, const double dy, const int n, const int m, double tau){

        py::buffer_info outputx_buf = outputx_np.request();
        py::buffer_info outputy_buf = outputy_np.request();
        py::buffer_info phi_buf = phi_np.request();
        py::buffer_info psi_buf = psi_np.request();
        py::buffer_info rhsx_buf = rhsx_np.request();
        py::buffer_info rhsy_buf = rhsy_np.request();
        py::buffer_info vxx_buf = vxx_np.request();
        py::buffer_info vyy_buf = vyy_np.request();
        py::buffer_info vxy_buf = vxy_np.request();
        double *outputx = static_cast<double *>(outputx_buf.ptr);
        double *outputy = static_cast<double *>(outputy_buf.ptr);
        double *phi = static_cast<double *>(phi_buf.ptr);
        double *psi = static_cast<double *>(psi_buf.ptr);
        double *rhsx = static_cast<double *>(rhsx_buf.ptr);
        double *rhsy = static_cast<double *>(rhsy_buf.ptr);

        vxx_ = static_cast<double *>(vxx_buf.ptr);
        vyy_ = static_cast<double *>(vyy_buf.ptr);
        vxy_ = static_cast<double *>(vxy_buf.ptr);

        int THREADS_ = std::thread::hardware_concurrency();
        std::vector<std::future<void> > changes(THREADS_); 
        int N_int = m*m;
        for(int th=0;th<THREADS_;++th){
            changes[th] = std::async(std::launch::async, &HelperClass::compute_for_loop, this, outputx, outputy, phi, rhsx, rhsy, dx, dy,  n, m, tau, static_cast<int>(th*N_int/THREADS_), static_cast<int>((th+1)*N_int/THREADS_));
        }
        for(int th=0;th<THREADS_;++th){
            changes[th].get();
        }
    }

    void compute_inverse_g_original(
            py::array_t<double, py::array::c_style | py::array::forcecast> outputx_np, 
            py::array_t<double, py::array::c_style | py::array::forcecast> outputy_np, 
            py::array_t<double, py::array::c_style | py::array::forcecast> phi_np, 
            py::array_t<double, py::array::c_style | py::array::forcecast> psi_np, 
            py::array_t<double, py::array::c_style | py::array::forcecast> rhsx_np, 
            py::array_t<double, py::array::c_style | py::array::forcecast> rhsy_np,
            py::array_t<double, py::array::c_style | py::array::forcecast> vxx_np,
            py::array_t<double, py::array::c_style | py::array::forcecast> vyy_np,
            py::array_t<double, py::array::c_style | py::array::forcecast> vxy_np,
            const double dx, const double dy, const int n, const int m){

        py::buffer_info outputx_buf = outputx_np.request();
        py::buffer_info outputy_buf = outputy_np.request();
        py::buffer_info phi_buf = phi_np.request();
        py::buffer_info psi_buf = psi_np.request();
        py::buffer_info rhsx_buf = rhsx_np.request();
        py::buffer_info rhsy_buf = rhsy_np.request();
        py::buffer_info vxx_buf = vxx_np.request();
        py::buffer_info vyy_buf = vyy_np.request();
        py::buffer_info vxy_buf = vxy_np.request();
        double *outputx = static_cast<double *>(outputx_buf.ptr);
        double *outputy = static_cast<double *>(outputy_buf.ptr);
        double *phi = static_cast<double *>(phi_buf.ptr);
        double *psi = static_cast<double *>(psi_buf.ptr);
        double *rhsx = static_cast<double *>(rhsx_buf.ptr);
        double *rhsy = static_cast<double *>(rhsy_buf.ptr);

        // calculate_gradient_vxx(vxx_, psi, dx, n);
        // calculate_gradient_vyy(vyy_, psi, dx, n);
        // calculate_gradient_vxy(vxy_, psi, dx, n);

        // step 1: for each point we will find T(x)
        for(int i=0;i<m;++i){
            for(int j=0;j<m;++j){
                int ind = i*m+j;
                int jm = static_cast<int>(fmax(0,j-1));
                int im = static_cast<int>(fmax(0,i-1));
                int jp = static_cast<int>(fmin(m-1,j+1));
                int ip = static_cast<int>(fmin(m-1,i+1));
                double Sy1=(phi[i*m+jp]-phi[i*m+jm])/(2.0*dy_);
                double Sy2=(phi[ip*m+j]-phi[im*m+j])/(2.0*dy_);

                double vxx_val = - interpolate_function_X(Sy1,Sy2,dx,n,vxx_);
                double vyy_val = - interpolate_function_X(Sy1,Sy2,dx,n,vyy_);
                double vxy_val = - interpolate_function_X(Sy1,Sy2,dx,n,vxy_);
                
                // g^{\ib \jb} \partial_{\jb}(\phi - b)
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
    void compute_inverse_g2(
            py::array_t<double, py::array::c_style | py::array::forcecast> outputx_np, 
            py::array_t<double, py::array::c_style | py::array::forcecast> outputy_np, 
            py::array_t<double, py::array::c_style | py::array::forcecast> phi_np, 
            py::array_t<double, py::array::c_style | py::array::forcecast> psi_np, 
            py::array_t<double, py::array::c_style | py::array::forcecast> rhs_np){
                
        py::buffer_info outputx_buf = outputx_np.request();
        py::buffer_info outputy_buf = outputy_np.request();
        py::buffer_info phi_buf = phi_np.request();
        py::buffer_info rhs_buf = rhs_np.request();
        
        double *outputx = static_cast<double *>(outputx_buf.ptr);
        double *outputy = static_cast<double *>(outputy_buf.ptr);
        double *phi = static_cast<double *>(phi_buf.ptr);
        double *rhs = static_cast<double *>(rhs_buf.ptr);

        // step 1: for each point we will find T(x)
        for(int i=0;i<n_;++i){
            for(int j=0;j<n_;++j){
                int ind = i*n_+j;
                auto xi_sol = stencils_[0];
                double min_val = INT_MAX;
                for(auto xi : stencils_){
                    int jp = j + xi[0];
                    int ip = i + xi[1];
                    // check if interior
                    if(jp >= 0 && jp <= n_-1 && ip >= 0 && ip <= n_-1){
                        double val = 0;
                        for(auto eta : stencils_){
                            int jp = j + eta[0];
                            int ip = i + eta[1];
                            if(jp >= 0 && jp <= n_-1 && ip >= 0 && ip <= n_-1){
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

    double compute_theta(std::vector<int>& eta, const double *rhs, int i, int j){
        int jp = j + eta[0];
        int ip = i + eta[1];
        // check if interior
        return (rhs[ip*n_+jp] - rhs[i*n_+j])/(dy_ * sqrt(eta[0]*eta[0] + eta[1]*eta[1]));
    }

    double compute_g(std::vector<int>& xi, std::vector<int>& eta, const double *phi, int i, int j){
        std::vector<int> xi_eta = {xi[0]+eta[0], xi[1]+eta[1]};
        return 0.5 * (compute_delta(xi_eta,phi,i,j) - compute_delta(xi,phi,i,j) - compute_delta(eta,phi,i,j));
    }

    double compute_delta(std::vector<int>& xi, const double *phi, int i, int j){
        double y1 = (j+0.5) * dy_;
        double y2 = (i+0.5) * dy_;

        int ind = i*n_+j;
        int jp = static_cast<int>(fmin(n_-1,j+1));
        int ip = static_cast<int>(fmin(n_-1,i+1));

        double Sy1=(phi[i*n_+jp]-phi[ind])/dy_;
        double Sy2=(phi[ip*n_+j]-phi[ind])/dy_;

        int j_new = j + xi[0];
        int i_new = i + xi[1];
        
        double y1_prime = (j_new + 0.5) * dy_;
        double y2_prime = (i_new + 0.5) * dy_;

        ind = i_new*n_+j_new;
        jp = static_cast<int>(fmin(n_-1,j_new+1));
        ip = static_cast<int>(fmin(n_-1,i_new+1));

        double Sy1_prime=(phi[i_new*n_+jp]-phi[ind])/dy_;
        double Sy2_prime=(phi[ip*n_+j_new]-phi[ind])/dy_;

        return compute_delta_cost(y1_prime, y2_prime, Sy1_prime, Sy2_prime, y1, y2, Sy1, Sy2);
    }

    double compute_delta_cost(double x_prime1, double x_prime2, double y_prime1, double y_prime2, double x1, double x2, double y1, double y2) const{
        double xyprime = - (x1*y_prime1 + x2*y_prime2);
        double xprimey = - (x_prime1*y1 + x_prime2*y2);
        double xy      = - (x1*y1 + x2*y2);
        double xprimeyprime = - (x_prime1*y_prime1 + x_prime2*y_prime2);
        return xyprime + xprimey - xy - xprimeyprime;
    }

    // pushforward(nu, psi, mu)
    void pushforward(
                py::array_t<double, py::array::c_style | py::array::forcecast> rho_np, 
                py::array_t<double, py::array::c_style | py::array::forcecast> psi_np, 
                py::array_t<double, py::array::c_style | py::array::forcecast> mu_np
                ){

        py::buffer_info psi_buf  = psi_np.request();
        py::buffer_info mu_buf   = mu_np.request();
        py::buffer_info rho_buf   = rho_np.request();

        double *psi = static_cast<double *> (psi_buf.ptr);
        double *mu  = static_cast<double *> (mu_buf.ptr);
        double *rho = static_cast<double *> (rho_buf.ptr);

        calc_pushforward_map(psi, dx_, n_); // this will update x1Map and x2Map
        sampling_pushforward(rho, mu, dx_, dy_, n_, m_);
    }

    void calc_pushforward_map(double *dual, double dx, int n){
        for(int i=1;i<n;i++){
            for(int j=1;j<n;j++){
                int im = fmax(0, i-1);
                int jm = fmax(0, j-1);
                
                x1Map_[i*(n+1)+j] = - (dual[i*n+j+1]  -dual[i*n+jm])/(2.0*dx) + (j+0.5)*dx;
                x2Map_[i*(n+1)+j] = - (dual[(i+1)*n+j]-dual[im*n+j])/(2.0*dx) + (i+0.5)*dx;
            }
        }
        
    }
    int sgn(double x){
        
        int truth=(x>0)-(x<0);
        return truth;
        
    }
    double interpolate_function(double *function, double x, const double y, const double dx, const int n){
        
        int xIndex=fmin(fmax(x/dx-.5 ,0),n-1);
        int yIndex=fmin(fmax(y/dx-.5 ,0),n-1);
        
        double xfrac=x/dx-xIndex-.5;
        double yfrac=y/dx-yIndex-.5;
        
        int xOther=xIndex+sgn(xfrac);
        int yOther=yIndex+sgn(yfrac);
        
        xOther=fmax(fmin(xOther, n-1),0);
        yOther=fmax(fmin(yOther, n-1),0);
        
        double v1=(1-fabs(xfrac))*(1-fabs(yfrac))*function[yIndex*n+xIndex];
        double v2=fabs(xfrac)*(1-fabs(yfrac))*function[yIndex*n+xOther];
        double v3=(1-fabs(xfrac))*fabs(yfrac)*function[yOther*n+xIndex];
        double v4=fabs(xfrac)*fabs(yfrac)*function[yOther*n+xOther];
        
        double v=v1+v2+v3+v4;
        
        return v;
        
    }
    void sampling_pushforward(double *rho, const double *mu, const double dx, const double dy, const int n, const int m){
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                
                double mass=mu[i*n+j];
                
                if(mass>0){
                    
                    double xStretch0=fabs(x1Map_[i*(n+1)+j+1]-x1Map_[i*(n+1)+j]);
                    double xStretch1=fabs(x1Map_[(i+1)*(n+1)+j+1]-x1Map_[(i+1)*(n+1)+j]);
                    
                    double yStretch0=fabs(x2Map_[(i+1)*(n+1)+j]-x2Map_[i*(n+1)+j]);
                    double yStretch1=fabs(x2Map_[(i+1)*(n+1)+j+1]-x2Map_[i*(n+1)+j+1]);
                    
                    double xStretch=fmax(xStretch0, xStretch1);
                    double yStretch=fmax(yStretch0, yStretch1);
                    
                    int xSamples=fmax(xStretch/dx,1);
                    int ySamples=fmax(yStretch/dx,1);

                    double factor=1/(xSamples*ySamples*1.0);
                    
                    for(int l=0;l<ySamples;l++){
                        for(int k=0;k<xSamples;k++){
                            
                            double a=(k+.5)/(xSamples*1.0);
                            double b=(l+.5)/(ySamples*1.0);
                            
                            double xPoint=(1-b)*(1-a)*x1Map_[i*(n+1)+j]+(1-b)*a*x1Map_[i*(n+1)+j+1]+b*(1-a)*x1Map_[(i+1)*(n+1)+j]+a*b*x1Map_[(i+1)*(n+1)+(j+1)];
                            double yPoint=(1-b)*(1-a)*x2Map_[i*(n+1)+j]+(1-b)*a*x2Map_[i*(n+1)+j+1]+b*(1-a)*x2Map_[(i+1)*(n+1)+j]+a*b*x2Map_[(i+1)*(n+1)+(j+1)];
                            
                            double X=xPoint/dy-.5;
                            double Y=yPoint/dy-.5;
                            
                            int xIndex=X;
                            int yIndex=Y;
                            
                            double xFrac=X-xIndex;
                            double yFrac=Y-yIndex;
                            
                            int xOther=xIndex+1;
                            int yOther=yIndex+1;
                            
                            xIndex=fmin(fmax(xIndex,0),m-1);
                            xOther=fmin(fmax(xOther,0),m-1);
                            
                            yIndex=fmin(fmax(yIndex,0),m-1);
                            yOther=fmin(fmax(yOther,0),m-1);
                            
                            
                            rho[yIndex*m+xIndex]+=(1-xFrac)*(1-yFrac)*mass*factor;
                            rho[yOther*m+xIndex]+=(1-xFrac)*yFrac*mass*factor;
                            rho[yIndex*m+xOther]+=xFrac*(1-yFrac)*mass*factor;
                            rho[yOther*m+xOther]+=xFrac*yFrac*mass*factor;
                            
                        }
                    }   
                }
            }
        }
        
        double sum       = 0;
        double totalMass = 0;
        for(int i=0,N=m*m;i<N;i++){
            sum += rho[i];
        }
        sum *= dy * dy;
        for(int i=0,N=n*n;i<N;i++){
            totalMass +=  mu[i];
        }
        totalMass *= dx * dx;
        for(int i=0,N=m*m;i<N;i++){
            rho[i]*=totalMass/sum;
        }
    }
};
    

void setup_indices(int& ip, int& im, int& jp, int& jm, int i, int j, int n){
    ip = fmin(i+1,n-1);
    im = fmax(i-1,0);
    jp = fmin(j+1,n-1);
    jm = fmax(j-1,0);
}
/**
 * input f_np, a_np
 * output f_np
 * (nu: torch.tensor, psi: torch.tensor, phi: torch.tensor, cost: torch.tensor, epsilon: float, dx: float, dy: float)
 */
void compute_first_variation_cpp(
        py::array_t<double, py::array::c_style | py::array::forcecast> out_np, 
        py::array_t<double, py::array::c_style | py::array::forcecast> psi_eps_np, 
        py::array_t<double, py::array::c_style | py::array::forcecast> phi_np, 
        py::array_t<double, py::array::c_style | py::array::forcecast> nu_np, 
        py::array_t<double, py::array::c_style | py::array::forcecast> cost_np, 
        py::array_t<double, py::array::c_style | py::array::forcecast> b_np, 
        double dx, double dy, double yMax, int n, int m){
    
    py::buffer_info out_buf   = out_np.request();
    py::buffer_info psi_eps_buf   = psi_eps_np.request();
    py::buffer_info phi_buf  = phi_np.request();
    // py::buffer_info nu_buf  = nu_np.request();
    // py::buffer_info cost_buf = cost_np.request();
    // py::buffer_info b_buf    = b_np.request();

    double *out   = static_cast<double *>(out_buf.ptr);
    double *psi   = static_cast<double *>(psi_eps_buf.ptr);
    double *phi   = static_cast<double *>(phi_buf.ptr);
    // double *nu    = static_cast<double *>(nu_buf.ptr);
    // double *cost  = static_cast<double *>(cost_buf.ptr);
    // double *b     = static_cast<double *>(b_buf.ptr);
    
    // int N = n*n;
    int M = m*m;
    
    // I will just assume neumann although neumann is not really the correct thing to use
    // compute the hessian of phi - b at each y
    std::vector<double> phi_b_11(M);
    std::vector<double> phi_b_12(M);
    std::vector<double> phi_b_22(M);

    std::vector<double> psi_hessian_11(M);
    std::vector<double> psi_hessian_12(M);
    std::vector<double> psi_hessian_22(M);

    // compute \nabla^2 (\phi-b) and \nabla^2 \psi(\nabla \phi)
    int ip,im,jp,jm;
    for(int i=0;i<m;++i){
        for(int j=0;j<m;++j){
            int ind = i*m+j;
            setup_indices(ip,im,jp,jm,i,j,m);
            double x = (j+0.5)/m;
            double y = (i+0.5)/m;
            phi_b_11[ind] = ((phi[i*m+jp]-2*phi[i*m+j]+phi[i*m+jm]) - x)/(dy*dy);
            phi_b_22[ind] = ((phi[ip*m+j]-2*phi[i*m+j]+phi[im*m+j]) - y)/(dy*dy);
            phi_b_12[ind] = ((  phi[ip*m+jp]-phi[ip*m+jm]-phi[im*m+jp]+phi[im*m+jm]))/(4.0*dy*dy);
            
        }
    }

    for(int i=0;i<m;++i){
        for(int j=0;j<m;++j){
            int ind = i*m+j;
            // find S_psi(y) = \nabla \phi(y)
            setup_indices(ip,im,jp,jm,i,j,m);
            double new_x = (phi[i*m+jp]-phi[i*m+jm])/(2.0*dy);
            double new_y = (phi[ip*m+j]-phi[im*m+j])/(2.0*dy);
            int new_j = new_x/dx - 0.5;
            int new_i = new_y/dx - 0.5;
            new_j = fmin(n-1, fmax(0, new_j));
            new_i = fmin(n-1, fmax(0, new_i));

            // py::print("new_j:",new_j, new_i);
            setup_indices(ip,im,jp,jm,new_i,new_j,n);
            psi_hessian_11[ind] = ((psi[new_i*n+jp]-2*psi[new_i*n+new_j]+psi[new_i*n+jm]))/(dx*dx);
            psi_hessian_22[ind] = ((psi[ip*n+new_j]-2*psi[new_i*n+new_j]+psi[im*n+new_j]))/(dx*dx);
            psi_hessian_12[ind] = ((psi[ip*n+jp]-psi[ip*n+jm]-psi[im*n+jp]+psi[im*n+jm]))/(4.0*dx*dx);
            
        }
    }


    // compute the first variation of J
    for(int i=0;i<m;++i){
        for(int j=0;j<m;++j){
            int ind = i*m+j;
            double val = 1
                    + (- phi_b_11[ind]*psi_hessian_11[ind])
                    + 2*(- phi_b_12[ind]*psi_hessian_12[ind])
                    + (- phi_b_22[ind]*psi_hessian_22[ind]);
            out[ind] = val;
        }
    }
    
}

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---


PYBIND11_MODULE(screening, m) {
    // optional module docstring
    m.doc() = "C++ wrapper for screening code";

    m.def("compute_dx", &compute_dx, "compute gradient x");
    m.def("compute_dy", &compute_dy, "compute gradient y");
    
    m.def("c_transform_cpp", &c_transform_cpp, "c transform from phi -> psi");
    m.def("c_transform_forward_cpp", &c_transform_forward_cpp, "c transform from psi -> phi");

    m.def("approx_push_cpp", &approx_push_cpp, "approximate pushforward");
    m.def("pushforward_entropic_cpp", &pushforward_entropic_cpp, "entropic pushforward");
    
    m.def("c_transform_epsilon_cpp", &c_transform_epsilon_cpp, "entropic c-transform");
    m.def("compute_nu_and_rho_cpp", &compute_nu_and_rho_cpp, "compute nu and rho");

    m.def("compute_Sy_cpp", &compute_Sy_cpp, "computing S(y) from for each y");
    m.def("compute_Gy_cpp", &compute_Gy_cpp, "Compute a matrix for G(y)");

    m.def("compute_first_variation_cpp", &compute_first_variation_cpp, "compute the first variation of J");

    py::class_<HelperClass>(m, "HelperClass")
        .def(py::init<
                double, double, int, int, double, double>()) // py::array_t<double, py::array::c_style | py::array::forcecast> phi_np, const double dx, const double dy
        .def("compute_inverse_g", &HelperClass::compute_inverse_g)
        .def("compute_inverse_g2", &HelperClass::compute_inverse_g2)
        .def("pushforward", &HelperClass::pushforward);
}
