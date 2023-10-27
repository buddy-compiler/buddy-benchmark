#include <stdio.h>
#include <math.h>
void dmxpy_float_gcc (int n1, float y[], int n2, int ldm, float x[], float m[]);
void dmxpy_double_gcc (int n1, double y[], int n2, int ldm, double x[], double m[]);
void dmxpy_float_clang (int n1, float y[], int n2, int ldm, float x[], float m[]);
void dmxpy_double_clang (int n1, double y[], int n2, int ldm, double x[], double m[]);