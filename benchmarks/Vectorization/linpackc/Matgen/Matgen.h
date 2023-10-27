#include <stdio.h>
#include <math.h>

void matgen_float_gcc(float a[],int lda, int n,float b[],float* norma);
void matgen_double_gcc(double a[],int lda, int n,double b[],double* norma);
void matgen_float_clang(float a[],int lda, int n,float b[],float* norma);
void matgen_double_clang(double a[],int lda, int n,double b[],double* norma);