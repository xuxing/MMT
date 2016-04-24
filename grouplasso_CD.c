/* 
 * grouplasso_CD.c
 * Perform coordinate descent algorithm for standard multi-task Lasso, or group lasso
 * Author: Xing Xu @ TTIC
 * Last update: Sep-16-2011
 */
#include "mex.h"
#include "math.h"


/*
 * Global variables, values are assigned in mexFunction
 */
double *X; /* Input observations, size n by J */
double *Y; /* Input labels, size n by K */
int J;     /* Number of features or covarites */
int K;     /* Number of tasks */
int n;     /* Number of samples */
double lambda; /* regularization parameter for group lasso penalty */


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
 * Update matrix B using current D
 * Input - B, current association matrix
 *         D, current auxiliary variables
 * Output - B_new, updated association matrix
 */
void updateB(double *B, double *D, double *B_new)
{
    int i, j1, v, j2;
    double residual, b_up, b_down;
    for(j1 = 0; j1 < J; j1++)
    {
        for(v = 0; v < K; v++)
        {
            b_up = 0;  b_down = 0;
            for(i = 0; i < n; i++)
            {
				residual = 0;
                for(j2 = 0; j2 < J; j2++)
                {
					if (j2 != j1)
					{
						residual += X[j2*n+i] * B[v*J+j2];
					}
                }
                residual = Y[v*n+i] - residual;
                b_up += X[j1*n+i] * residual;
            }
            
            for(i = 0; i < n; i++)
            {
                b_down += pow(X[j1*n+i], 2);
            }
			b_down += lambda / D[j1];
            
            *(B_new + v*J + j1) = b_up / b_down;
        }
    }
}


/*
 * Update auxiliary variables using current B
 * Input - B, current association matrix
 * Output - D, updated auxiliary variables
 */
void updateD(double *D, double *B)
{
    int j1, v, ind;
    double s = 0, *D_tmp = NULL;
    
    D_tmp = (double *)malloc(J * sizeof(double));
    
    for(j1 = 0; j1 < J; j1++)
    {
        D_tmp[j1] = 0;
        for(v = 0; v < K; v++)
        {
            ind = v * J + j1;
            D_tmp[j1] += pow(B[ind], 2);
        }
		D_tmp[j1] = sqrt(D_tmp[j1]) + 1 / pow(n, 2);
		s += D_tmp[j1];
    }
    for(j1 = 0; j1 < J; j1++)
    {
        D[j1] = D_tmp[j1] / s;
    }
    free(D_tmp);
}


/*
 * Entrance of this c file, equals to the function:
 * [B_new D_new] = grouplasso_CD(B, X, Y, D, lambda)
 * where the function inputs are stored according to pointer array prhs,
 * outputs are in pointer array plhs, and nrhs, nlhs represent the size
 * of each array.
 * Input - B, or prhs[0], current association matrix
 *         X, observations, size n by J
 *         Y, label matrix, size n by K
 *         D, current auxiliary variables
 *         lambda, regularization parameter for group lasso penalty
 * Output - B_new, updated associatoin matrix after this CD Loop
 *          D_new, updated auxiliary variables after this CD Loop
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    double *B, *D;
    double *B_new, *D_new;

    /* Get input */
    B = mxGetPr(prhs[0]);
    X = mxGetPr(prhs[1]);
    Y = mxGetPr(prhs[2]);
    D = mxGetPr(prhs[3]);
    lambda = mxGetScalar(prhs[4]);
    
    J = mxGetM(prhs[0]);
    K = mxGetN(prhs[0]);
    n = mxGetM(prhs[1]);
    
    /* Set output */
    plhs[0] = mxCreateDoubleMatrix(J, K, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(J, 1, mxREAL);
    B_new = mxGetPr(plhs[0]);
    D_new = mxGetPr(plhs[1]);
    
    /* Update results to output */
    updateB(B, D, B_new);
    updateD(D_new, B_new);
}
