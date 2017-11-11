#include <cuda.h>
#include <thrust/transform.h>
#include <thrust/count.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <ctime>

#define CUDA_MAX_BLOCKS 1024

__global__ void check_correctness (unsigned int num_nodes, unsigned int *offset_arr,
                                 unsigned int *cols_arr,  int *color_assignment, bool *result)

{
  unsigned int tid = threadIdx.x+blockIdx.x*blockDim.x;
  unsigned int num_threads = blockDim.x*gridDim.x;
  for (unsigned int i = tid; i < num_nodes; i += num_threads)
  {
    // Iterate over neighbours
    for (unsigned int j = offset_arr[i]; j < offset_arr[i+1]; j++) {
      // Get neighbour vertex id
      int k = cols_arr[j];
      // Check neighbors color
      if (color_assignment[i] == color_assignment[k]) {
        printf("Node coloring error at %d,%d with color %d\n", i, k, color_assignment[i]);
        *result = false;
      }
    }
  }
}


int main(int argc, char *argv[])
{
    unsigned int n_nodes, n_values;
    std::cin >> n_nodes >> n_values;
    n_nodes -= 1;

    thrust::host_vector<unsigned int> h_Ao(n_nodes+1), h_Ac(n_values);
    for (int i=0; i<n_nodes+1; i++) {
        std::cin >> h_Ao[i] ;
    }
    for (int i=0; i<n_values; i++) {
        std::cin >> h_Ac[i] ;
    }
    thrust::device_vector<unsigned int> d_Ao = h_Ao, d_Ac = h_Ac;
    thrust::device_vector<int> d_colors(n_nodes);

    /*thrust::fill(d_colors.begin(), d_colors.end(), -1); */

    unsigned int *r_Ao = thrust::raw_pointer_cast(d_Ao.data());
    unsigned int *r_Ac = thrust::raw_pointer_cast(d_Ac.data());
    int *r_c = thrust::raw_pointer_cast(d_colors.data());

    bool *result;
    cudaHostAlloc(&result, sizeof(bool), 0);
    *result = true;

    mis_coloring(n_nodes, r_Ao, r_Ac, r_c);
    int num_threads = 256;
    int num_blocks = min(n_nodes/num_threads + 1,CUDA_MAX_BLOCKS);
    check_correctness<<<num_blocks, num_threads>>>(n_nodes, r_Ao, r_Ac, r_c, result);
    cudaDeviceSynchronize();

    if(*result)
      std::cout << "Check successful " << std::endl;
    else
      std::cout << "Check failed " << std::endl;

    /*for(int i=0; i<n_nodes; i++)*/
        /*std::cout << d_colors[i] << " ";*/
    /*std::cout << std::endl;*/

}
