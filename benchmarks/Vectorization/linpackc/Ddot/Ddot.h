#include <stdio.h>
#include <math.h>

float ddot_ROLL_float_gcc(int n,float dx[],int incx,float dy[], int incy);
float ddot_UNROLL_float_gcc(int n,float dx[],int incx,float dy[], int incy);
double ddot_ROLL_double_gcc(int n,double dx[],int incx,double dy[], int incy);
double ddot_UNROLL_double_gcc(int n,double dx[],int incx,double dy[], int incy);

float ddot_ROLL_float_clang(int n,float dx[],int incx,float dy[], int incy);
float ddot_UNROLL_float_clang(int n,float dx[],int incx,float dy[], int incy);
double ddot_ROLL_double_clang(int n,double dx[],int incx,double dy[], int incy);
double ddot_UNROLL_double_clang(int n,double dx[],int incx,double dy[], int incy);