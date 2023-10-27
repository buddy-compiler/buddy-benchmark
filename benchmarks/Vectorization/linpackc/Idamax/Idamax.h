#include <stdio.h>
#include <math.h>

int idamax_float_gcc(int n,float dx[],int incx);
int idamax_double_gcc(int n,double dx[],int incx);
int idamax_float_clang(int n,float dx[],int incx);
int idamax_double_clang(int n,double dx[],int incx);