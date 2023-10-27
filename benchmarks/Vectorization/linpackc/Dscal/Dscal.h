#include <stdio.h>
#include <math.h>

void dscal_ROLL_float_gcc(int n,float da,float dx[], int incx);
void dscal_UNROLL_float_gcc(int n,float da,float dx[], int incx);
void dscal_ROLL_double_gcc(int n,double da,double dx[], int incx);
void dscal_UNROLL_double_gcc(int n,double da,double dx[], int incx);

void dscal_ROLL_float_clang(int n,float da,float dx[], int incx);
void dscal_UNROLL_float_clang(int n,float da,float dx[], int incx);
void dscal_ROLL_double_clang(int n,double da,double dx[], int incx);
void dscal_UNROLL_double_clang(int n,double da,double dx[], int incx);