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

__global__ void jpl_coloring_kernel(unsigned int num_nodes, unsigned int color, unsigned int *offset_arr,
                                 unsigned int *cols_arr,  unsigned int *random_wts, int *color_assignment)

{
  unsigned int tid = threadIdx.x+blockIdx.x*blockDim.x;
  unsigned int num_threads = blockDim.x*gridDim.x;
  for (unsigned int i = tid; i < num_nodes; i += num_threads)
  {
    // Initialize to true
    bool is_leader = true;

    // Ignore if already colored
    if ((color_assignment[i] != -1)) continue;

    bool *n_colors = new bool[color+1];
    memset(n_colors, 0, sizeof(n_colors));
    // Iterate over neighbours
    for (unsigned int j = offset_arr[i]; j < offset_arr[i+1]; j++) {
      // Get neighbour vertex id
      int k = cols_arr[j];
      // Get neighbors color
      int kc = color_assignment[k];
      n_colors[kc] = true;
      // Check if neighbors is already assigned the current color
      if (((kc != -1) && (kc != color)) || (i == k)) continue;
      // Get the neighbour random weight
      int kr = random_wts[k];
      // Local maximum condition
      if (random_wts[i] <= kr) {
          is_leader = false;
      }
    }

    // Assign least possible color if the current vertex is the leader
    if (is_leader) {
        for (int ci=0; ci <= color; ci++)
           if (!n_colors[ci]){
               color_assignment[i] = ci;
               break;
           }
    }
    free(n_colors);
  }
}

__host__ void jpl_coloring(unsigned int num_nodes, unsigned int *offset_arr,
                                 unsigned int *cols_arr, int *color_assignment)
{
    std::srand(std::time(0));

    //generate rand perm
    thrust::device_vector<unsigned int> d_randoms(num_nodes);
    thrust::sequence(d_randoms.begin(), d_randoms.end());
    std::random_shuffle(d_randoms.begin(), d_randoms.end());

    unsigned int *r_randoms = thrust::raw_pointer_cast(d_randoms.data());
    // init colors to -1
    thrust::fill(thrust::device, color_assignment, color_assignment + num_nodes, -1);

    // Find all maximal independent sets and assign colors to them
    int num_threads = 256;
    int num_blocks = min(num_nodes/num_threads + 1,CUDA_MAX_BLOCKS);
    int c = 0;
    for(c = 0; c < num_nodes; c++) {
        jpl_coloring_kernel<<<num_blocks, num_threads>>>(num_nodes, c, offset_arr, cols_arr, r_randoms, color_assignment);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("Error: %s\n", cudaGetErrorString(err));
        cudaDeviceSynchronize();
        // Exit if all nodes are colored
        int nodes_left = (int)thrust::count(thrust::device, color_assignment, color_assignment + num_nodes, -1);
        if (nodes_left == 0)
            break;
    }
    printf("Nodes : %d, colors : %d\n", num_nodes, c);
}

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

    jpl_coloring(n_nodes, r_Ao, r_Ac, r_c);
    int num_threads = 256;
    int num_blocks = min(n_nodes/num_threads + 1,CUDA_MAX_BLOCKS);
    check_correctness<<<num_blocks, num_threads>>>(n_nodes, r_Ao, r_Ac, r_c, result);

    cudaDeviceSynchronize();
    if(result)
      std::cout << "Check successful " << std::endl;
    else
      std::cout << "Check failed " << std::endl;

    /*for(int i=0; i<n_nodes; i++)*/
        /*std::cout << d_colors[i] << " ";*/
    /*std::cout << std::endl;*/

}
