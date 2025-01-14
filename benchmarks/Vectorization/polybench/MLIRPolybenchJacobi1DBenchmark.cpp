#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <cstdio>
#include <vector>

extern "C" {
void _mlir_ciface_jacobi_1d(int, int, MemRef<double, 1> *, MemRef<double, 1> *);
void _mlir_ciface_jacobi_1d_init_array(int, MemRef<double, 1> *,
                                       MemRef<double, 1> *);
}

const std::vector<std::pair<std::string, std::vector<size_t>>> sizes = {
    {"mini", {20, 30}},           {"small", {40, 120}},
    {"medium", {100, 400}},       {"large", {500, 2000}},
    {"extralarge", {1000, 4000}},
};

static void runPolybench(benchmark::State &state,
                         const std::vector<size_t> &size) {
  const size_t TSTEPS = size[0];
  const size_t N = size[1];

  MemRef<double, 1> inputA({N}, 0);
  MemRef<double, 1> inputB({N}, 0);

  for (auto _ : state) {
    state.PauseTiming();
    _mlir_ciface_jacobi_1d_init_array(N, &inputA, &inputB);
    state.ResumeTiming();
    _mlir_ciface_jacobi_1d(TSTEPS, N, &inputA, &inputB);
  }
}

static void printArray(int n, double *A) {
  for (int i = 0; i < n; i++) {
    if (i % 20 == 0) {
      printf("\n");
    }
    printf("%0.2lf ", A[i]);
  }
  printf("\n");
}

void registerMLIRPolybenchJacobi1D() {
  for (const auto &sizePair : sizes) {
    std::string benchmarkName = "jacobi-1d-" + sizePair.first;
    benchmark::RegisterBenchmark(benchmarkName.c_str(),
                                 [sizePair](benchmark::State &state) {
                                   runPolybench(state, sizePair.second);
                                 })
        ->Unit(benchmark::kMillisecond);
  }
}

void generateResultMLIRPolybenchJacobi1D(size_t size_id) {
  const std::string benchmarkName = "jacobi-1d-" + sizes[size_id].first;
  const std::vector<size_t> &size = sizes[size_id].second;

  const size_t TSTEPS = size[0];
  const size_t N = size[1];

  MemRef<double, 1> inputA({N}, 0);
  MemRef<double, 1> inputB({N}, 0);

  _mlir_ciface_jacobi_1d_init_array(N, &inputA, &inputB);
  _mlir_ciface_jacobi_1d(TSTEPS, N, &inputA, &inputB);

  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "Result for " << benchmarkName << ":\n";
  printArray(N, inputA.getData());
  std::cout << "--------------------------------------------------------"
            << std::endl;
}
