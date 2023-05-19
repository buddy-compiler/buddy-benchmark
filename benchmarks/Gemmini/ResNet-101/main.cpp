#include <iostream>
#include <math.h>
#include <buddy/Core/Container.h>
#include <cat.h>
#include <labels.h>

extern "C" {
  void _mlir_ciface_resnet101(MemRef<float, 2>* output, MemRef<float, 4>* input);
}

void softmax(float *input, size_t size) {
  assert(0 <= size <= sizeof(input) / sizeof(float));
  int i;
  float m, sum, constant;
  m = -INFINITY;
  for (i = 0; i < size; ++i) {
    if (m < input[i]) {
      m = input[i];
    }
  }

  sum = 0.0;
  for (i = 0; i < size; ++i) {
    sum += exp(input[i] - m);
  }

  constant = m + log(sum);
  for (i = 0; i < size; ++i) {
    input[i] = exp(input[i] - constant);
  }
}

int main() {
  intptr_t sizesInput[4] = {1, 299, 299, 3};
  intptr_t sizesOutput[2] = {1, 1001};
  MemRef<float, 4> input(catImg, sizesInput);
  MemRef<float, 2> output(sizesOutput);
  std::cout << output.getSize() << std::endl;
  _mlir_ciface_resnet101(&output, &input);
  float* out = output.getData();
  softmax(out, 1001);
  float maxVal = 0;
  int maxIdx = 0;
  for (int i = 0; i < 1001; ++i) {
    if (out[i] > maxVal) {
      maxVal = out[i];
      maxIdx = i;
    }
  }
  std::cout << "Classification Index: " << maxVal << std::endl;
  std::cout << "Classification: " << maxIdx << std::endl;
  printf("Probability: %s \n", labels[maxIdx]);
  return 0;
}
