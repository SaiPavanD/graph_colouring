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

#define CUDA_MAX_BLOCKS 1024

int main(int argc, char *argv[])
{
    unsigned int n_nodes, n_values;
    std::cin >> n_nodes >> n_values;

    thrust::host_vector<unsigned int> t_Ax(n_values), t_Ay(n_values), t_Ao(n_nodes+1);
    thrust::fill(t_Ao.begin(), t_Ao.end(), 0);
    unsigned int x_cord, y_cord;
    for (int ti=0; ti<n_values; ti++) {
        std::cin >> x_cord >> y_cord;
        t_Ax[ti] = x_cord;
        t_Ay[ti] = y_cord;
        t_Ao[x_cord+1]++;
    }

    std::cout << "Input reading done!" << std::endl;

    thrust::inclusive_scan(t_Ao.begin(), t_Ao.end(), t_Ao.begin());

    thrust::sort_by_key(t_Ay.begin(), t_Ay.end(), t_Ax.begin());
    thrust::sort_by_key(t_Ax.begin(), t_Ax.end(), t_Ay.begin());

    std::cout << n_nodes << " " << n_values << std::endl;
    for (unsigned int i=0; i<n_nodes+1; i++) {
        std::cout <<  d_Ao[i] << " ";
    }
    printf("\n");

    for (unsigned int i=0; i<n_values; i++) {
        std::cout << d_Ay[i] << " ";
    }
    printf("\n");
    return 0;
}
