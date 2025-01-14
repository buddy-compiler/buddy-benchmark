#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <cstddef>
#include <vector>

extern "C" {
void _mlir_ciface_trmm(int, int, double, MemRef<double, 2> *,
                       MemRef<double, 2> *);
void _mlir_ciface_trmm_init_array(int, int, MemRef<double, 1> *,
                                  MemRef<double, 2> *, MemRef<double, 2> *);
}

const std::vector<std::pair<std::string, std::vector<size_t>>> sizes = {
    {"mini", {20, 30}},           {"small", {60, 80}},
    {"medium", {200, 240}},       {"large", {1000, 1200}},
    {"extralarge", {2000, 2600}},
};

static void runPolybench(benchmark::State &state,
                         const std::vector<size_t> &size) {
  const size_t M = size[0];
  const size_t N = size[1];

  MemRef<double, 1> alpha({1}, 0);
  MemRef<double, 2> inputA({M, M}, 0);
  MemRef<double, 2> inputB({M, N}, 0);

  for (auto _ : state) {
    state.PauseTiming();
    _mlir_ciface_trmm_init_array(M, N, &alpha, &inputA, &inputB);
    state.ResumeTiming();
    _mlir_ciface_trmm(M, N, alpha.getData()[0], &inputA, &inputB);
  }
}

static void printArray(int m, int n, double *b) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if ((i * m + j) % 20 == 0) {
        printf("\n");
      }
      printf("%0.2lf ", b[i * n + j]);
    }
  }
  printf("\n");
}

void registerMLIRPolybenchTrmm() {
  for (const auto &sizePair : sizes) {
    std::string benchmarkName = "trmm-" + sizePair.first;
    benchmark::RegisterBenchmark(benchmarkName.c_str(),
                                 [sizePair](benchmark::State &state) {
                                   runPolybench(state, sizePair.second);
                                 })
        ->Unit(benchmark::kMillisecond);
  }
}

void generateResultMLIRPolybenchTrmm(size_t size_id) {
  const std::string benchmarkName = "trmm-" + sizes[size_id].first;
  const std::vector<size_t> &size = sizes[size_id].second;

  const size_t M = size[0];
  const size_t N = size[1];

  MemRef<double, 1> alpha({1}, 0);
  MemRef<double, 2> inputA({M, M}, 0);
  MemRef<double, 2> inputB({M, N}, 0);

  _mlir_ciface_trmm_init_array(M, N, &alpha, &inputA, &inputB);
  _mlir_ciface_trmm(M, N, alpha.getData()[0], &inputA, &inputB);

  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "Result for " << benchmarkName << ":\n";
  printArray(M, N, inputB.getData());
  std::cout << "--------------------------------------------------------"
            << std::endl;
}
