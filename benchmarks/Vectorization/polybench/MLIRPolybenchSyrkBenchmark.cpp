#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <cstddef>
#include <vector>

extern "C" {
void _mlir_ciface_syrk(int, int, double, double, MemRef<double, 2> *,
                       MemRef<double, 2> *);
void _mlir_ciface_syrk_init_array(int, int, MemRef<double, 1> *,
                                  MemRef<double, 1> *, MemRef<double, 2> *,
                                  MemRef<double, 2> *);
}

const std::vector<std::pair<std::string, std::vector<size_t>>> sizes = {
    {"mini", {20, 30}},           {"small", {60, 80}},
    {"medium", {200, 240}},       
    // {"large", {1000, 1200}},
    // {"extralarge", {2000, 2600}},
};

static void runPolybench(benchmark::State &state,
                         const std::vector<size_t> &size) {
  const size_t M = size[0];
  const size_t N = size[1];

  MemRef<double, 1> alpha({1}, 0);
  MemRef<double, 1> beta({1}, 0);
  MemRef<double, 2> inputC({N, N}, 0);
  MemRef<double, 2> inputA({N, M}, 0);

  for (auto _ : state) {
    state.PauseTiming();
    _mlir_ciface_syrk_init_array(N, M, &alpha, &beta, &inputC, &inputA);
    state.ResumeTiming();
    _mlir_ciface_syrk(N, M, alpha.getData()[0], beta.getData()[0], &inputC,
                      &inputA);
  }
}

static void printArray(int n, double *C) {
  int i, j;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) {
        printf("\n");
      }
      printf("%0.2lf ", C[i * n + j]);
    }
  }
  printf("\n");
}

void registerMLIRPolybenchSyrk() {
  for (const auto &sizePair : sizes) {
    std::string benchmarkName = "syrk-" + sizePair.first;
    benchmark::RegisterBenchmark(benchmarkName.c_str(),
                                 [sizePair](benchmark::State &state) {
                                   runPolybench(state, sizePair.second);
                                 })
        ->Unit(benchmark::kMillisecond);
  }
}

void generateResultMLIRPolybenchSyrk(size_t size_id) {
  const std::string benchmarkName = "syrk-" + sizes[size_id].first;
  const std::vector<size_t> &size = sizes[size_id].second;

  const size_t M = size[0];
  const size_t N = size[1];

  MemRef<double, 1> alpha({1}, 0);
  MemRef<double, 1> beta({1}, 0);
  MemRef<double, 2> inputC({N, N}, 0);
  MemRef<double, 2> inputA({N, M}, 0);

  _mlir_ciface_syrk_init_array(N, M, &alpha, &beta, &inputC, &inputA);
  _mlir_ciface_syrk(N, M, alpha.getData()[0], beta.getData()[0], &inputC,
                    &inputA);

  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "Result for " << benchmarkName << ":\n";
  printArray(N, inputC.getData());
  std::cout << "--------------------------------------------------------"
            << std::endl;
}
