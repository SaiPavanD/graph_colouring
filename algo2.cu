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




