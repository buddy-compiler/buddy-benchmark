void SAXPY(const int n, const float *x, float *y,
                               const float a) {
  for (int i = 0; i < n; ++i) {
    y[i] = a * x[i] + y[i];
  }
}
