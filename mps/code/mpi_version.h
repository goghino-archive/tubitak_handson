/*
 *
 * Copyright (c) 2014 Juraj Kardos
 *
 * This software is provided 'as-is', without any express or implied warranty.
 * In no event will the authors be held liable for any damages arising 
 * from the use of this software.
 * Permission is granted to anyone to use this software for any purpose, 
 * including commercial applications, and to alter it and redistribute it freely,
 * without any restrictons.
 */

#define POPULATION_SIZE (1024) /* must be multiple of 64 == BLOCK */
#define INDIVIDUAL_LEN 6       /* order of polynom +1 for c0*/
#define N_POINTS 100

// Forward declarations
extern "C" {
    // Reads input file with noisy points. Points will be approximated by 
    // polynomial function using GA.
    float *readData(const char *name, const int POINTS_CNT);

    // GA algorithm running on GPU
    void computeGA(float *points, int deviceID,
       float *solution, float *bestFitness_o, int *genNumber_o, double *time_o);

    // Gets last error and prints message when error is present
    void check_cuda_error(const char *message);

    // Finished MPI and aborts
    void my_abort(int err);

    // returns index of minimal value in the input array
    int findMinimum(float *array, int arrayLen);
}
