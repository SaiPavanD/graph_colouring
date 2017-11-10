#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

using namespace std;
using namespace thrust;

class Graph{
  private:
    unsigned int n;

  public:
    Graph(unsigned int num_vertices){
      n = num_vertices;
    }
    
}
