#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <vector>

extern "C" {
void _mlir_ciface_gemm(int, int, int, double, double, MemRef<double, 2> *,
                       MemRef<double, 2> *, MemRef<double, 2> *);
void _mlir_ciface_gemm_init_array(int, int, int, MemRef<double, 1> *,
                                  MemRef<double, 1> *, MemRef<double, 2> *,
                                  MemRef<double, 2> *, MemRef<double, 2> *);
}

const std::vector<std::pair<std::string, std::vector<size_t>>> sizes = {
    {"mini", {20, 25, 30}},
    {"small", {60, 70, 80}},
    {"medium", {200, 220, 240}},
    {"large", {1000, 1100, 1200}},
    {"extralarge", {2000, 2300, 2600}},
};

static void runPolybench(benchmark::State &state,
                         const std::vector<size_t> &size) {
  const size_t NI = size[0];
  const size_t NJ = size[1];
  const size_t NK = size[2];

  MemRef<double, 1> alpha({1}, 0);
  MemRef<double, 1> beta({1}, 0);

  MemRef<double, 2> inputC({NI, NJ}, 0);
  MemRef<double, 2> inputA({NI, NK}, 0);
  MemRef<double, 2> inputB({NK, NJ}, 0);

  for (auto _ : state) {
    state.PauseTiming();
    _mlir_ciface_gemm_init_array(NI, NJ, NK, &alpha, &beta, &inputC, &inputA,
                                 &inputB);
    state.ResumeTiming();
    _mlir_ciface_gemm(NI, NJ, NK, alpha.getData()[0], beta.getData()[0],
                      &inputC, &inputA, &inputB);
  }
}

static void printArray(int ni, int nj, double *c) {
  int i, j;
  for (i = 0; i < ni; i++) {
    for (j = 0; j < nj; j++) {
      if ((i * ni + j) % 20 == 0) {
        printf("\n");
      }
      printf("%0.2lf ", c[i * nj + j]);
    }
  }
  printf("\n");
}

void registerMLIRPolybenchGemm(const std::set<std::string> &disabledSizes) {
  for (const auto &sizePair : sizes) {
    if (disabledSizes.count(sizePair.first)) {
      continue;
    }
    std::string benchmarkName = "gemm-" + sizePair.first;
    benchmark::RegisterBenchmark(benchmarkName.c_str(),
                                 [sizePair](benchmark::State &state) {
                                   runPolybench(state, sizePair.second);
                                 })
        ->Unit(benchmark::kMillisecond);
  }
}

void generateResultMLIRPolybenchGemm(size_t size_id) {
  const std::string benchmarkName = "gemm-" + sizes[size_id].first;
  const std::vector<size_t> &size = sizes[size_id].second;

  const size_t NI = size[0];
  const size_t NJ = size[1];
  const size_t NK = size[2];

  MemRef<double, 1> alpha({1}, 0);
  MemRef<double, 1> beta({1}, 0);

  MemRef<double, 2> inputC({NI, NJ}, 0);
  MemRef<double, 2> inputA({NI, NK}, 0);
  MemRef<double, 2> inputB({NK, NJ}, 0);

  _mlir_ciface_gemm_init_array(NI, NJ, NK, &alpha, &beta, &inputC, &inputA,
                               &inputB);
  _mlir_ciface_gemm(NI, NJ, NK, alpha.getData()[0], beta.getData()[0], &inputC,
                    &inputA, &inputB);

  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "Result for " << benchmarkName << ":\n";
  printArray(NI, NJ, inputC.getData());
  std::cout << "--------------------------------------------------------"
            << std::endl;
}
