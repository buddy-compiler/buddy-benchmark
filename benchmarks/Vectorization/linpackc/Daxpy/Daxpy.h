#include <cstdio>
#include <cmath>

void daxpy_ROLL_float_gcc(int n,float da, float dx[],int incx, float dy[], int incy);
void daxpy_ROLL_double_gcc(int n,double da, double dx[],int incx,double dy[], int incy);
void daxpy_UNROLL_float_gcc(int n,float da, float dx[],int incx,float dy[], int incy);
void daxpy_UNROLL_double_gcc(int n,double da, double dx[],int incx,double dy[], int incy);
void daxpy_ROLL_float_clang(int n,float da, float dx[],int incx, float dy[], int incy);
void daxpy_ROLL_double_clang(int n,double da, double dx[],int incx,double dy[], int incy);
void daxpy_UNROLL_float_clang(int n,float da, float dx[],int incx,float dy[], int incy);
void daxpy_UNROLL_double_clang(int n,double da, double dx[],int incx,double dy[], int incy);


