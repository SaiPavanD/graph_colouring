#include <cuda.h>
#include <thrust/transform.h>
#include <thrust/count.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <algorithm>
#include <iostream>

// Ao matrix row offsets
// Av non-zero values
// Ac column indices


__global__ void color_jpl_kernel(int n, int c, int *Ao, 
                                 int *Ac,  
                                 int *randoms, int *colors)
{   
  for (int i = threadIdx.x+blockIdx.x*blockDim.x; 
       i < n; 
       i += blockDim.x*gridDim.x) 
  {   

    /*if (threadIdx.x == 0 && blockIdx.x == 0) {*/
        /*for (int i =0; i< n; i++)*/
            /*printf("-%d ", Ao[i]);*/
        /*printf("Ao\n");*/
        /*for (int i =0; i< Ao[n]; i++)*/
            /*printf("-%d ", Ac[i]);*/
        /*printf("Ac\n");*/
    /*}*/
    bool f=true; // true iff you have max random

    // ignore nodes colored earlier
    /*printf("COL %d\n", colors[i]);*/
    if ((colors[i] != -1)) continue; 

    int ir = randoms[i];

    // look at neighbors to check their random number
    /*printf("YUI %d %d\n" , Ao[i], Ao[i+1]);*/
    for (int k = Ao[i]; k < Ao[i+1]; k++) {        
      // ignore nodes colored earlier (and yourself)
      int j = Ac[k];
      int jc = colors[j];
      printf("Inside for loop %d %d %d %d\n", jc, c, i ,j);
      if (((jc != -1) && (jc != c)) || (i == j)) continue; 
      printf("Inside after for loop %d %d %d %d\n", jc, c, i ,j);
      int jr = randoms[j];
      if (ir <= jr) {
          f=false;        
          printf("%d %d %d %d\n", ir, jr, k, i);
      }
      /*else*/
          /*printf("ER%d %d %d %d\n", ir, jr, k, i);*/
    }

    // assign color if you have the maximum random number
    if (f) colors[i] = c;
  }
}

#define CUDA_MAX_BLOCKS 1024

/*__global__ void color_jpl(int n, thrust::device_vector<int> Ao, thrust::device_vector<int> Ac, thrust::device_vector<int> randoms, thrust::device_vector<int> colors) */
/*{*/
    /*thrust::host_vector<int> h_randoms(n); // allocate and init random array */
    /*//generate rand perm*/
    /*thrust::copy(h_randoms.begin(), h_randoms.end(), randoms.begin());*/
    /*[>thrust::fill(colors, colors+n, -1); // init colors to -1<]*/
    /*for(int c=0; c < n; c++) {*/
        /*int nt = 256;*/
        /*int nb = min((n + nt - 1)/nt,CUDA_MAX_BLOCKS);*/
        /*color_jpl_kernel<<<nb,nt>>>(n, c, Ao, Ac, randoms, colors);*/
        /*int left = (int)thrust::count(colors.begin(), colors.end(), -1);*/
        /*if (left == 0) break;*/
    /*}*/
/*}*/

int main(int argc, char *argv[])
{
    int n_nodes, n_values;
    std::cin >> n_nodes >> n_values;
    n_nodes -= 1;
    thrust::host_vector<int> h_Ao(n_nodes+1), h_Av(n_values), h_Ac(n_values);
    /*for (int i=0; i<n_values; i++) {*/
        /*cin >> h_Av[i] ;*/
    /*} */
    for (int i=0; i<n_nodes+1; i++) {
        std::cin >> h_Ao[i] ;
    } 
    for (int i=0; i<n_values; i++) {
        std::cin >> h_Ac[i] ;
    } 
    thrust::host_vector<int> h_colors(n_nodes);
    for(int i=0; i<n_nodes; i++)
        h_colors[i] = -1;
    thrust::device_vector<int> d_colors = h_colors;
    /*thrust::copy(h_colors.begin(), h_colors.end(), d_colors.begin());*/
    /*std::cout << "HI" << std::endl;*/
    /*std::cout << n_nodes << " " <<   n_values <<  std::endl;*/

    thrust::host_vector<int> h_randoms(n_nodes);
    thrust::sequence(h_randoms.begin(), h_randoms.end());
    std::random_shuffle(h_randoms.begin(), h_randoms.end());
    /*for (int i=0; i<n_nodes; i++)*/
        /*std::cout << ": " << h_randoms[i] ;*/
    std::cout << std::endl;
    /*//FIll h_randoms()*/
    
    thrust::device_vector<int> d_randoms = h_randoms;
    thrust::device_vector<int> d_Ao = h_Ao, d_Ac = h_Ac; 
    /*[>color_jpl(n_nodes, d_Ao, d_Ac, d_randoms, d_colors); <]*/
    int *r_Ao = thrust::raw_pointer_cast(d_Ao.data());
    int *r_Ac = thrust::raw_pointer_cast(d_Ac.data());
    int *r_r = thrust::raw_pointer_cast(d_randoms.data());
    int *r_c = thrust::raw_pointer_cast(d_colors.data());
    for(int c=0; c < n_nodes; c++) {
        std::cout << "color: " << c << std::endl;
        int nt = 256;
        int nb = min((n_nodes + nt - 1)/nt,CUDA_MAX_BLOCKS);
        std::cout << "nb:nt" << nb << "-" << nt << std::endl;
        color_jpl_kernel<<<1, 1>>>(n_nodes, c, r_Ao, r_Ac, r_r, r_c);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("Error: %s\n", cudaGetErrorString(err));
        int left = (int)thrust::count(d_colors.begin(), d_colors.end(), -1);
        if (left == 0) break;
    }
    for(int i=0; i<n_nodes; i++)
        std::cout << d_colors[i] << " ";
    std::cout << std::endl;

}
