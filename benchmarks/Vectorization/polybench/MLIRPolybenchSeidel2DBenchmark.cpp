#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <cstddef>
#include <vector>

extern "C" {
void _mlir_ciface_seidel_2d(int, int, MemRef<double, 2> *);
void _mlir_ciface_seidel_2d_init_array(int, MemRef<double, 2> *);
}

const std::vector<std::pair<std::string, std::vector<size_t>>> sizes = {
    {"mini", {20, 40}},           {"small", {40, 120}},
    {"medium", {100, 400}},       
    // {"large", {500, 2000}},
    // {"extralarge", {1000, 4000}},
};

static void runPolybench(benchmark::State &state,
                         const std::vector<size_t> &size) {
  const size_t TSTEPS = size[0];
  const size_t N = size[1];

  MemRef<double, 2> inputA({N, N}, 0);

  for (auto _ : state) {
    state.PauseTiming();
    _mlir_ciface_seidel_2d_init_array(N, &inputA);
    state.ResumeTiming();
    _mlir_ciface_seidel_2d(TSTEPS, N, &inputA);
  }
}

static void printArray(int n, double *A) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) {
        printf("\n");
      }
      printf("%0.2lf ", A[i * n + j]);
    }
  }
  printf("\n");
}

void registerMLIRPolybenchSeidel2D() {
  for (const auto &sizePair : sizes) {
    std::string benchmarkName = "seidel-2d-" + sizePair.first;
    benchmark::RegisterBenchmark(benchmarkName.c_str(),
                                 [sizePair](benchmark::State &state) {
                                   runPolybench(state, sizePair.second);
                                 })
        ->Unit(benchmark::kMillisecond);
  }
}

void generateResultMLIRPolybenchSeidel2D(size_t size_id) {
  const std::string benchmarkName = "seidel-2d-" + sizes[size_id].first;
  const std::vector<size_t> &size = sizes[size_id].second;

  const size_t TSTEPS = size[0];
  const size_t N = size[1];

  MemRef<double, 2> inputA({N, N}, 0);

  _mlir_ciface_seidel_2d_init_array(N, &inputA);
  _mlir_ciface_seidel_2d(TSTEPS, N, &inputA);

  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "Result for " << benchmarkName << ":\n";
  printArray(N, inputA.getData());
  std::cout << "--------------------------------------------------------"
            << std::endl;
}
