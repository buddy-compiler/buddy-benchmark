#include <arm_neon.h>

void SAXPY(const int n, const float *x, float *y, const float a) {
  int i = 0;
  const float32x4_t a_vec = vdupq_n_f32(a);

  for (; i <= n - 4; i += 4) {
    float32x4_t x_vec = vld1q_f32(x + i);
    float32x4_t y_vec = vld1q_f32(y + i);

    float32x4_t result = vmlaq_f32(y_vec, x_vec, a_vec);

    vst1q_f32(y + i, result);
  }

  for (; i < n; i++) {
    y[i] = a * x[i] + y[i];
  }
}
