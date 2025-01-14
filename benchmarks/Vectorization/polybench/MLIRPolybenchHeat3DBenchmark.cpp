#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <cstdio>
#include <vector>

extern "C" {
void _mlir_ciface_heat_3d(int, int, MemRef<double, 3> *, MemRef<double, 3> *);
void _mlir_ciface_heat_3d_init_array(int, MemRef<double, 3> *,
                                     MemRef<double, 3> *);
}

const std::vector<std::pair<std::string, std::vector<size_t>>> sizes = {
    {"mini", {20, 10}},    {"small", {40, 20}},         {"medium", {100, 40}},
    {"large", {500, 120}}, {"extralarge", {1000, 200}},
};

static void runPolybench(benchmark::State &state,
                         const std::vector<size_t> &size) {
  const size_t TSTEPS = size[0];
  const size_t N = size[1];

  MemRef<double, 3> inputA({N, N, N}, 0);
  MemRef<double, 3> inputB({N, N, N}, 0);

  for (auto _ : state) {
    state.PauseTiming();
    _mlir_ciface_heat_3d_init_array(N, &inputA, &inputB);
    state.ResumeTiming();
    _mlir_ciface_heat_3d(TSTEPS, N, &inputA, &inputB);
  }
}

static void printArray(int n, double *A) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        if ((i * n * n + j * n + k) % 20 == 0) {
          printf("\n");
        }
        printf("%0.2lf ", A[i * n * n + j * n + k]);
      }
    }
  }
  printf("\n");
}

void registerMLIRPolybenchHeat3D(const std::set<std::string> &disabledSizes) {
  for (const auto &sizePair : sizes) {
    if (disabledSizes.count(sizePair.first)) {
      continue;
    }
    std::string benchmarkName = "heat-3d-" + sizePair.first;
    benchmark::RegisterBenchmark(benchmarkName.c_str(),
                                 [sizePair](benchmark::State &state) {
                                   runPolybench(state, sizePair.second);
                                 })
        ->Unit(benchmark::kMillisecond);
  }
}

void generateResultMLIRPolybenchHeat3D(size_t size_id) {
  const std::string benchmarkName = "heat-3d-" + sizes[size_id].first;
  const std::vector<size_t> &size = sizes[size_id].second;

  const size_t TSTEPS = size[0];
  const size_t N = size[1];

  MemRef<double, 3> inputA({N, N, N}, 0);
  MemRef<double, 3> inputB({N, N, N}, 0);

  _mlir_ciface_heat_3d_init_array(N, &inputA, &inputB);
  _mlir_ciface_heat_3d(TSTEPS, N, &inputA, &inputB);

  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "Result for " << benchmarkName << ":\n";
  printArray(N, inputA.getData());
  std::cout << "--------------------------------------------------------"
            << std::endl;
}
