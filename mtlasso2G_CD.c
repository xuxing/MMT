/* 
 * mtlasso2G_CD.c
 * Perform coordinate descent loop for the two-graph guided multi-task Lasso.
 * Author: Xing Xu (xing@ttic.edu)
 * Last update: 4-23-2012
 */
#include <mex.h>
#include <math.h>
#include <omp.h>


#define CHUNKSIZE 100
#define EPOWER 4


/*
 * Global variables, values are assigned in mexFunction
 */
double *X; /* Input observations, size n by J */
double *Y; /* Input labels, size n by K */
int J;     /* Number of features or covarites */
int K;     /* Number of tasks */
int n;     /* Number of samples */
int num_edges1; /* Number of edges in task graph */
int num_edges2; /* Number of edges in feature graph */
double lambda;  /* regularization parameter for ell_1 penalty */
double gamma1;  /* regularization parameter for task graph penalty */
double gamma2;  /* regularization parameter for feature graph penalty */


/*
 * Standard sign function, get the sign of a number
 * Input - num, a scalar
 * Output - the sign of num, should be 1,-1 or 0
 */
int sign(double num)
{
    if(num > 0) return 1;
    if(num < 0) return -1;
    return 0;
}


/*
 * Update B matrix using current value of D, D_1 and D_2
 * Result store in Matrix B_new
 * Input - B, current association matrix
 *         D_s, common denominator for D
 *         D1_s, common denominator for D1
 *         D2_s, common denominator for D2
 *         W1, weights of edges in task graph
 *         C1, correlations of tasks that connected in task graph
 *         E1, edges in task graph
 *         W2, C2, E1, counter part of feature graph
 * Output - B_new, updated association matrix
 */
void updateB(double *B, double D_s, double D1_s, double D2_s, double *W1, double *C1, int *E1, 
            double *W2, double *C2, int *E2, double *B_new)
{
    int j1, v, e, i, m, l, f, g, sice;
    double tmp, epsilon;
    double *B_up, *B_down, *R;
    
    /* B_new equals to B_up divide B_down */
    B_up = (double *)malloc(J * K * sizeof(double));
    B_down = B_new;
    epsilon = 1 / pow(n, EPOWER); /* A small number for algorithm's stability */

    /* Parallel update matrix B_up */
#pragma omp parallel for schedule(dynamic, CHUNKSIZE) private(i, j1, v, tmp, R)
    for (v = 0; v < K; v ++)
    {
        R = (double *)malloc(n * sizeof(double));
        for (i = 0; i < n; i++)
        {
            tmp = 0;
            for (j1 = 0; j1 < J; j1++)
            {
                tmp += X[j1*n+i] * B[v*J+j1];
            }
            R[i] = tmp;
        }
        for (j1 = 0; j1 < J; j1 ++)
        {
            tmp = 0;
            for (i = 0; i < n; i++)
            {
                tmp += X[j1*n+i] * (Y[v*n+i] + X[j1*n+i]*B[v*J+j1] - R[i]);
            }
            B_up[v*J+j1] = tmp;
        }
        free(R);
    }

    /* Parallel update matrix B_down */
#pragma omp parallel for schedule(dynamic, CHUNKSIZE) private(i, j1, v, tmp)
    for (j1 = 0; j1 < J; j1++)
    {
        tmp = 0;
        for (i = 0; i < n; i++)
        {
            tmp += pow(X[j1*n+i], 2);
        }
        for (v = 0; v < K; v++)
        {
            B_down[v*J + j1] = tmp + lambda * D_s / (fabs(B[v*J+j1]) + epsilon);
        }
    }

    /* Information from the first graph */
#pragma omp parallel for schedule(dynamic, CHUNKSIZE) private(e, sice, j1, v, tmp)
    for (e = 0; e < num_edges1; e++)
    {
        sice = sign(C1[e]);
        m = E1[2*e];
        v = E1[2*e+1];
        for (j1 = 0; j1 < J; j1++)
        {
            tmp = gamma1 * W1[e] * D1_s / (fabs(B[m*J+j1] - sice*B[v*J+j1]) + epsilon);
            B_down[v*J+j1] += tmp;
            B_up[v*J+j1] += tmp * B[m*J+j1] * sice;
        }

        v = E1[2*e];
        l = E1[2*e+1];
        for (j1 = 0; j1 < J; j1++)
        {
            tmp = gamma1 * W1[e] * D1_s / (fabs(B[v*J+j1] - sice*B[l*J+j1]) + epsilon);
            B_down[v*J+j1] += tmp;
            B_up[v*J+j1] += tmp * B[l*J+j1] * sice;
        }
    }

    /* Information from the second graph */
#pragma omp parallel for schedule(dynamic, CHUNKSIZE) private(e, sice, j1, v, tmp)
    for (e = 0; e < num_edges2; e++)
    {
        sice = sign(C2[e]);
        f = E2[2*e];
        j1 = E2[2*e+1];
        for (v = 0; v < K; v++)
        {
            tmp = gamma2 * W2[e] * D2_s / (fabs(B[v*J+f] - sice*B[v*J+j1]) + epsilon);
            B_down[v*J+j1] += tmp;
            B_up[v*J+j1] += tmp * B[v*J+f] * sice;
        }

        j1 = E2[2*e];
        g = E2[2*e+1];
        for (v = 0; v < K; v++)
        {
            tmp = gamma2 * W2[e] * D2_s / (fabs(B[v*J+j1] - sice*B[v*J+g]) + epsilon);
            B_down[v*J+j1] += tmp;
            B_up[v*J+j1] += tmp * B[v*J+g] * sice;
        }
    }
    
    /* Finally, update B_new from B_up and B_down */
    for (v = 0; v < K; v++)
    {
        for (j1 = 0; j1 < J; j1++)
        {
            B_new[v*J+j1] = B_up[v*J+j1] / B_down[v*J+j1];
        }
    }

    free(B_up);
}


/*
 * Get the common denominator for auxiliary variables D
 * Input - B, association matrix
 * Output - a scalar, the common denominator of D
 */
double getDSum(double * B)
{
    int j1, v;
    double s = 0;
        
    for(v = 0; v < K; v++)
    {
        for(j1 = 0; j1 < J; j1++)
        {
            s += fabs(B[v * J + j1]);
        }
    }
    
    return s + J * K * 1 / pow(n, EPOWER);
}


/*
 * Get the common denominator for auxiliary variables D1
 * Input - B, association matrix
 *         W1, weights of edges in task graph
 *         C1, correlations of tasks that connected in task graph
 *         E1, edges in task graph
 * Output - a scalar, the common denominator of D1
 */
double getD1Sum(double *B, double *W1, double *C1, int *E1)
{
    int j1, e;
    double s = 0;
    
    for(e=0; e<num_edges1; e++)
    {
        for(j1=0; j1<J; j1++)
        {
            s += fabs( W1[e] *  ( B[E1[2*e]*J+j1] - sign(C1[e]) * B[E1[2*e+1]*J+j1] ) );
        }
    }
    
    return s + num_edges1 * J * 1 / pow(n, EPOWER);
}


/*
 * Get the common denominator for auxiliary variables D2
 * Input - B, association matrix
 *         W2, weights of edges in feature graph
 *         C2, correlations of features that connected in feature graph
 *         E2, edges in feature graph
 * Output - a scalar, the common denominator of D2
 */
double getD2Sum(double *B, double *W2, double *C2, int *E2)
{
    int v, e;
    double s = 0;
    
    for(e=0; e<num_edges2; e++)
    {
        for(v=0; v<K; v++)
        {
            s += fabs( W2[e] * ( B[v*J+E2[2*e]] - sign(C2[e]) * B[v*J+E2[2*e+1]] ) );
        }
    }

    return s + num_edges2 * K * 1 / pow(n, EPOWER);
}


/*
 * Entrance of this c file, equals to the function:
 * B_new = mtlasso2G_CD(B, W1, C1, E1_d, W2, C2, E2_d, X, Y, lambda, gamma1, gamma2, tol, max_it)
 * where the function inputs are stored according to pointer array prhs,
 * outputs are in pointer array plhs, and nrhs, nlhs represent the size
 * of each array.
 * Input - B, or prhs[0], current association matrix
 *         W1, weights of edges in task graph
 *         C1, correlations of tasks that connected in task graph
 *         E1_d, edges in task graph, stored as double, transform to integer later
 *         W2, C2, E1_d, counter part of feature graph
 *         X, input observations, size n by J
 *         Y, input labels matrix, size n by K
 *         lambda, regularization parameter for ell_1 penalty
 *         gamma1, regularization parameter for task graph fused penalty
 *         gamma2, regularization parameter for feature graph fused penalty
 *         tol, maximum allowed difference between two iterations, convergence criterion
 *         max_it, maximum allowed number of iterations
 * Output - B_new, updated association matrix
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    double *B, *W1, *C1, *W2, *C2, *B_new, *B_prime, *E1_d, *E2_d;
    int *E1, *E2;
    double diff, tol, D1_s, D2_s, D_s;
    int e, flag_it = 0, ind = 0, max_it;
    
    /* From right(input) hand side */
    B = mxGetPr(prhs[0]);
    W1 = mxGetPr(prhs[1]);
    C1 = mxGetPr(prhs[2]);
    E1_d = mxGetPr(prhs[3]);
    W2 = mxGetPr(prhs[4]);
    C2 = mxGetPr(prhs[5]);
    E2_d = mxGetPr(prhs[6]);
    X = mxGetPr(prhs[7]);
    Y = mxGetPr(prhs[8]);
    lambda = mxGetScalar(prhs[9]);
    gamma1 = mxGetScalar(prhs[10]);
    gamma2 = mxGetScalar(prhs[11]);
    tol = mxGetScalar(prhs[12]);
    max_it = mxGetScalar(prhs[13]);
    
    diff = tol + 1;
    
    J = mxGetM(prhs[0]);
    K = mxGetN(prhs[0]);
    n = mxGetM(prhs[7]);
    num_edges1 = mxGetN(prhs[3]);
    num_edges2 = mxGetN(prhs[6]);
    
    E1 = (int *)malloc(2 * num_edges1 * sizeof(int));
    E2 = (int *)malloc(2 * num_edges2 * sizeof(int));
    B_prime = (double *)malloc(J * K * sizeof(double));
    plhs[0] = mxCreateDoubleMatrix(J, K, mxREAL);
    B_new = mxGetPr(plhs[0]);

    /* Mex does not allow to reuse the memory of inputs */
    for (ind = 0; ind < J * K; ind ++)
    {
        *(B_prime + ind) = *(B + ind);
    }

    /* Change E1 and E2 from double matrix to integer matrix */
    for (e = 0; e < 2 * num_edges1; e++)
    {
        E1[e] = (int)(E1_d[e] + 0.1);
    }
    for (e = 0; e < 2 * num_edges2; e++)
    {
        E2[e] = (int)(E2_d[e] + 0.1);
    }
    
    /* End if converge or reach maximum iterations allowed */
    while(diff > tol && flag_it < max_it)
    {
        /* First update all normalizers */
#pragma omp parallel
        {
    #pragma omp sections
            {
        #pragma omp section
                D1_s = getD1Sum(B_prime, W1, C1, E1);
        #pragma omp section
                D2_s = getD2Sum(B_prime, W2, C2, E2);
        #pragma omp section
                D_s = getDSum(B_prime);
            }
        }

        /* Then update matrix B */
        updateB(B_prime, D_s, D1_s, D2_s, W1, C1, E1, W2, C2, E2, B_new);

        /* Check if converge */
        diff = 0;
        for(ind=0; ind < J*K; ind++)
        {
            diff += fabs(B_new[ind] - B_prime[ind]);
            *(B_prime + ind) = *(B_new + ind);
        }
        
        flag_it++;
    }
    
    /* Free malloced spaces here */
    free(B_prime);
    free(E1);
    free(E2);
}
