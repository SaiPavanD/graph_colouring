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

#define CUDA_MAX_BLOCKS 32*1024
#define CUDA_MAX_THREADS 1024

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
      unsigned int k =cols_arr[j];
      // Check neighbors color
      if (i == k) continue;
      if (color_assignment[i] == color_assignment[k]) {
        printf("Node coloring error at %d,%d with color %d\n", i, k, color_assignment[i]);
        *result = false;
      }
    }
  }
}

__global__ void mis_coloring_kernel(unsigned int num_nodes, unsigned int color, unsigned int *offset_arr,
                                 unsigned int *cols_arr,  unsigned int *random_wts, int *color_assignment, unsigned int* r_iflags)

{
  unsigned int tid = threadIdx.x+blockIdx.x*blockDim.x;
  unsigned int num_threads = blockDim.x*gridDim.x;
  /*}*/
  for (unsigned int i = tid; i < num_nodes; i += num_threads)
  {
    // Initialize to true
    bool is_leader = true;

    // Ignore if already colored
    if ((color_assignment[i] != -1)) continue;

    // Iterate over neighbours
    for (int mis_i = 0; mis_i < 10; mis_i++) {
        if (r_iflags[i] != 0)
        {
          if(r_iflags[i] == 1)  is_leader = false;
          break;
        }
        for (unsigned int j = offset_arr[i]; j < offset_arr[i+1]; j++) {
            // Get neighbour vertex id
            unsigned int k =cols_arr[j];
            // Get neighbors color
            int kc = color_assignment[k];
            // Skip if neighbor is already colored(removed from graph)
            if (((kc != -1) && (kc != color)) || (i == k)) continue;
            // Get the neighbour random weight
            unsigned int kr = random_wts[k];
            // Local maximum condition
            /*if (r_iflags[k] != 2) continue;*/
            if ((unsigned int)random_wts[i] <= kr) {
                is_leader = false;
            }
        }
        if (is_leader) {
           r_iflags[i] = 2;
           /*printf("l%d->%d ",color, i);*/
          for (unsigned int jj = offset_arr[i]; jj < offset_arr[i+1]; jj++) {
            r_iflags[jj] = 1;
          }
          /*printf("\n");*/
          break;
        }
    }
    // Assign least possible color if the current vertex is the leader
    if (is_leader) {
        color_assignment[i] = color;
    }
  }
}

__host__ void mis_coloring(unsigned int num_nodes, unsigned int *offset_arr,
                                 unsigned int *cols_arr, int *color_assignment)
{
    std::srand(std::time(0));

    //generate rand perm
    thrust::device_vector<unsigned int> d_randoms(num_nodes);
    thrust::device_vector<unsigned int> d_iflags(num_nodes);

    thrust::sequence(d_randoms.begin(), d_randoms.end());
    /*for (int lo=0; lo<num_nodes; lo++) {*/
        /*[>printf("%d ", d_randoms[lo]);<]*/
        /*std::cout << d_randoms[lo] << " ";*/
    /*}*/
    /*printf("Rwts\n");*/
    unsigned int *r_randoms = thrust::raw_pointer_cast(d_randoms.data());
    unsigned int *r_iflags  = thrust::raw_pointer_cast(d_iflags.data());
    // init colors to -1
    thrust::fill(thrust::device, color_assignment, color_assignment + num_nodes, -1);

    // Find all maximal independent sets and assign colors to them
    int num_threads = CUDA_MAX_THREADS;
    int num_blocks = min(num_nodes/num_threads + 1,CUDA_MAX_BLOCKS);
    int c = 0;
    for(c = 0; c < num_nodes; c++) {
        std::random_shuffle(d_randoms.begin(), d_randoms.end());
        /*if ((c == 1 || c == 0)) {*/
            /*for (int lo=0; lo<num_nodes; lo++) {*/
                /*std::cout << d_randoms[lo] << " ";*/
            /*}*/
            /*printf("Rwts\n");*/
        /*}*/
        thrust::fill(thrust::device, r_iflags, r_iflags + num_nodes, 0);
        mis_coloring_kernel<<<num_blocks, num_threads>>>(num_nodes, c, offset_arr, cols_arr, r_randoms, color_assignment, r_iflags);
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
      unsigned int k =cols_arr[j];
      // Get neighbors color
      int kc = color_assignment[k];
      if(kc != -1)  n_colors[kc] = true;
      // Skip if neighbor is already colored(removed from graph)
      if (((kc != -1) && (kc != color)) || (i == k)) continue;
      // Get the neighbour random weight
      unsigned int kr = random_wts[k];
      // Local maximum condition
      if ((unsigned int)random_wts[i] <= kr) {
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
    int num_threads = CUDA_MAX_THREADS;
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

__global__ void ldf_coloring_kernel(unsigned int num_nodes, unsigned int color, unsigned int *offset_arr,
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
    // Get the random weights and degrees of the current vertex
    unsigned int id = offset_arr[i+1] - offset_arr[i];
    unsigned int ir = random_wts[i];
    // Iterate over neighbours
    for (unsigned int j = offset_arr[i]; j < offset_arr[i+1]; j++) {
      // Get neighbour vertex id
      unsigned int k =cols_arr[j];
      // Get neighbors color
      int kc = color_assignment[k];
      if(kc != -1)  n_colors[kc] = true;
      // Skip if neighbor is already colored(removed from graph)
      if (((kc != -1) && (kc != color)) || (i == k)) continue;
      // Get the random weights and degrees of the neighbors
      unsigned int kd = offset_arr[k+1] - offset_arr[k];
      unsigned int kr = random_wts[k];
      // Local maximum condition
      if ((id<kd) || ((id==kd)&&(ir<=kr))) {
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

__host__ void ldf_coloring(unsigned int num_nodes, unsigned int *offset_arr,
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
    int num_threads = CUDA_MAX_THREADS;
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
