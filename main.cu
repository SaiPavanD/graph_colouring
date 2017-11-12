#include "kernels.h"

#include <cuda.h>
#include <thrust/transform.h>
#include <thrust/count.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <ctime>


int main(int argc, char *argv[])
{
    int default_alg = MIS_ALG;

    if (argc == 2) {
        std::string a = std::string(argv[1]);
        if (a.compare(6, 3, "mis") == 0)
            default_alg = MIS_ALG;
        else if (a.compare(6, 3, "jpl") == 0)
            default_alg = JPL_ALG;
        else if (a.compare(6, 3, "ldf") == 0)
            default_alg = LDF_ALG;
    }
    unsigned int n_nodes, n_values;
    /*unsigned n_edges;*/
    std::cin >> n_nodes >> n_values;
    /*n_values = 2 * n_edges;*/

    /*thrust::host_vector<unsigned int> t_Ax(n_values), t_Ay(n_values), t_Ao(n_nodes+1);*/
    /*thrust::fill(t_Ao.begin(), t_Ao.end(), 0);*/
    /*unsigned int x_cord, y_cord;*/
    /*for (int ti=0; ti<n_values; ti++) {*/
        /*std::cin >> x_cord >> y_cord; */
        /*t_Ax[2*ti] = x_cord;*/
        /*t_Ay[2*ti] = y_cord;*/
        /*t_Ax[2*ti+1] = y_cord;*/
        /*t_Ay[2*ti+1] = x_cord;*/
        /*t_Ao[x_cord+1]++;*/
        /*t_Ao[y_cord+1]++;*/
    /*}*/

    /*thrust::inclusive_scan(t_Ao.begin(), t_Ao.end(), t_Ao.begin());*/

    /*for (int ind = 0; ind < n_nodes+1; ind++) {*/
        /*std::cout << t_Ao[ind] << " " ;*/
    /*}*/
    /*std::cout << "AO" << std::endl;*/

    thrust::host_vector<unsigned int> h_Ao(n_nodes+1), h_Ac(n_values);
    for (int i=0; i<n_nodes+1; i++) {
        std::cin >> h_Ao[i] ;
    }
    for (int i=0; i<n_values; i++) {
        std::cin >> h_Ac[i] ;
    }

    /*thrust::sort_by_key(t_Ax.begin(), t_Ax.end(), t_Ay.begin());*/
    thrust::device_vector<unsigned int> d_Ao = h_Ao, d_Ac = h_Ac;
    thrust::device_vector<int> d_colors(n_nodes);

    /*thrust::fill(d_colors.begin(), d_colors.end(), -1); */

    unsigned int *r_Ao = thrust::raw_pointer_cast(d_Ao.data());
    unsigned int *r_Ac = thrust::raw_pointer_cast(d_Ac.data());
    int *r_c = thrust::raw_pointer_cast(d_colors.data());

    bool *result;
    cudaHostAlloc(&result, sizeof(bool), 0);
    *result = true;

    switch(default_alg) {
        
        case MIS_ALG:
            std::cout << "Running MIS_ALG on " << n_nodes << " nodes" << std::endl;
            mis_coloring(n_nodes, r_Ao, r_Ac, r_c);
            break;
        case JPL_ALG:
            std::cout << "Running JPL_ALG on " << n_nodes << " nodes" << std::endl;
            jpl_coloring(n_nodes, r_Ao, r_Ac, r_c);
            break;
        case LDF_ALG:
            std::cout << "Running LDF_ALG on " << n_nodes << " nodes" << std::endl;
            ldf_coloring(n_nodes, r_Ao, r_Ac, r_c);
            break;
        default:
            std::cerr << "Invalid alg" << std::endl;
            return 0;
    }
    /*cudaDeviceSynchronize();*/
    check_correctness(n_nodes, r_Ao, r_Ac, r_c, result);

    cudaDeviceSynchronize();
    if(*result)
      std::cout << "Check successful " << std::endl;
    else
      std::cout << "Check failed " << std::endl;

    /*for(int i=0; i<n_nodes; i++)*/
        /*std::cout << d_colors[i] << " ";*/
    /*std::cout << std::endl;*/

}
