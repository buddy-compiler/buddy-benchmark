#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <vector>

extern "C" {
void _mlir_ciface_cholesky(int, MemRef<double, 2> *);
void _mlir_ciface_cholesky_init_array(int, MemRef<double, 2> *);
}

const std::vector<std::pair<std::string, std::vector<size_t>>> sizes = {
    {"mini", {40}}, {"small", {120}}, {"medium", {400}},
    // {"large", {2000}}, {"extralarge", {4000}},
};

static void runPolybench(benchmark::State &state,
                         const std::vector<size_t> &size) {
  const size_t N = size[0];

  MemRef<double, 2> inputA({N, N}, 0);

  for (auto _ : state) {
    state.PauseTiming();
    _mlir_ciface_cholesky_init_array(N, &inputA);
    state.ResumeTiming();
    _mlir_ciface_cholesky(N, &inputA);
  }
}

static void printArray(int n, double *A) {
  int i, j;
  for (i = 0; i < n; i++) {
    for (j = 0; j <= i; j++) {
      if ((i * n + j) % 20 == 0) {
        printf("\n");
      }
      printf("%0.2lf ", A[i * n + j]);
    }
  }
  printf("\n");
}

void registerMLIRPolybenchCholesky() {
  for (const auto &sizePair : sizes) {
    std::string benchmarkName = "cholesky-" + sizePair.first;
    benchmark::RegisterBenchmark(benchmarkName.c_str(),
                                 [sizePair](benchmark::State &state) {
                                   runPolybench(state, sizePair.second);
                                 })
        ->Unit(benchmark::kMillisecond);
  }
}

void generateResultMLIRPolybenchCholesky(size_t size_id) {
  const std::string benchmarkName = "cholesky-" + sizes[size_id].first;
  const std::vector<size_t> &size = sizes[size_id].second;

  const size_t N = size[0];

  MemRef<double, 2> inputA({N, N}, 0);

  _mlir_ciface_cholesky_init_array(N, &inputA);

  _mlir_ciface_cholesky(N, &inputA);

  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "Result for " << benchmarkName << ":\n";
  printArray(N, inputA.getData());
  std::cout << "--------------------------------------------------------"
            << std::endl;
}
