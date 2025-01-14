#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <vector>

extern "C" {
void _mlir_ciface_gesummv(int, double, double, MemRef<double, 2> *,
                          MemRef<double, 2> *, MemRef<double, 1> *,
                          MemRef<double, 1> *, MemRef<double, 1> *);
void _mlir_ciface_gesummv_init_array(int, MemRef<double, 1> *,
                                     MemRef<double, 1> *, MemRef<double, 2> *,
                                     MemRef<double, 2> *, MemRef<double, 1> *);
}

const std::vector<std::pair<std::string, std::vector<size_t>>> sizes = {
    {"mini", {30}},    {"small", {90}},        {"medium", {250}},
    {"large", {1300}}, {"extralarge", {2800}},
};

static void runPolybench(benchmark::State &state,
                         const std::vector<size_t> &size) {
  const size_t N = size[0];

  MemRef<double, 1> alpha({1}, 0);
  MemRef<double, 1> beta({1}, 0);

  MemRef<double, 2> inputA({N, N}, 0);
  MemRef<double, 2> inputB({N, N}, 0);
  MemRef<double, 1> inputTmp({N}, 0);
  MemRef<double, 1> inputX({N}, 0);
  MemRef<double, 1> inputY({N}, 0);

  for (auto _ : state) {
    state.PauseTiming();
    _mlir_ciface_gesummv_init_array(N, &alpha, &beta, &inputA, &inputB,
                                    &inputX);
    state.ResumeTiming();
    _mlir_ciface_gesummv(N, alpha.getData()[0], beta.getData()[0], &inputA,
                         &inputB, &inputTmp, &inputX, &inputY);
  }
}

static void printArray(int n, double *y) {
  int i;
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) {
      printf("\n");
    }
    printf("%0.2lf ", y[i]);
  }
  printf("\n");
}

void registerMLIRPolybenchGesummv() {
  for (const auto &sizePair : sizes) {
    std::string benchmarkName = "gesummv-" + sizePair.first;
    benchmark::RegisterBenchmark(benchmarkName.c_str(),
                                 [sizePair](benchmark::State &state) {
                                   runPolybench(state, sizePair.second);
                                 })
        ->Unit(benchmark::kMillisecond);
  }
}

void generateResultMLIRPolybenchGesummv(size_t size_id) {
  const std::string benchmarkName = "gesummv-" + sizes[size_id].first;
  const std::vector<size_t> &size = sizes[size_id].second;

  const size_t N = size[0];

  MemRef<double, 1> alpha({1}, 0);
  MemRef<double, 1> beta({1}, 0);

  MemRef<double, 2> inputA({N, N}, 0);
  MemRef<double, 2> inputB({N, N}, 0);
  MemRef<double, 1> inputTmp({N}, 0);
  MemRef<double, 1> inputX({N}, 0);
  MemRef<double, 1> inputY({N}, 0);

  _mlir_ciface_gesummv_init_array(N, &alpha, &beta, &inputA, &inputB, &inputX);

  _mlir_ciface_gesummv(N, alpha.getData()[0], beta.getData()[0], &inputA,
                       &inputB, &inputTmp, &inputX, &inputY);

  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "Result for " << benchmarkName << ":\n";
  printArray(N, inputY.getData());
  std::cout << "--------------------------------------------------------"
            << std::endl;
}
