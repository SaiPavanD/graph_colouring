#define MIS_ALG 0
#define JPL_ALG 1
#define LDF_ALG 2

/* Include this file in your main.cu */

#ifndef KERNEL_H
#define KERNEL_H

__host__ void check_correctness (unsigned int num_nodes, unsigned int *offset_arr,
                                 unsigned int *cols_arr,  int *color_assignment, bool *result);

__host__ void mis_coloring(unsigned int num_nodes, unsigned int *offset_arr,
                                unsigned int *cols_arr, int *color_assignment);

__host__ void jpl_coloring(unsigned int num_nodes, unsigned int *offset_arr,
                                 unsigned int *cols_arr, int *color_assignment);

__host__ void ldf_coloring(unsigned int num_nodes, unsigned int *offset_arr,
                                unsigned int *cols_arr, int *color_assignment);

#endif
