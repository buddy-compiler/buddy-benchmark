#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <cstddef>
#include <vector>

extern "C" {
void _mlir_ciface_trisolv(int, MemRef<double, 2> *, MemRef<double, 1> *,
                          MemRef<double, 1> *);
void _mlir_ciface_trisolv_init_array(int, MemRef<double, 2> *,
                                     MemRef<double, 1> *, MemRef<double, 1> *);
}

const std::vector<std::pair<std::string, std::vector<size_t>>> sizes = {
    {"mini", {40}},    {"small", {120}},       {"medium", {400}},
    {"large", {2000}}, {"extralarge", {4000}},
};

static void runPolybench(benchmark::State &state,
                         const std::vector<size_t> &size) {
  const size_t N = size[0];

  MemRef<double, 2> inputL({N, N}, 0);
  MemRef<double, 1> inputX({N}, 0);
  MemRef<double, 1> inputB({N}, 0);

  for (auto _ : state) {
    state.PauseTiming();
    _mlir_ciface_trisolv_init_array(N, &inputL, &inputX, &inputB);
    state.ResumeTiming();
    _mlir_ciface_trisolv(N, &inputL, &inputX, &inputB);
  }
}

static void printArray(int n, double *x) {
  int i;
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) {
      printf("\n");
    }
    printf("%0.2lf ", x[i]);
  }
  printf("\n");
}

void registerMLIRPolybenchTrisolv(const std::set<std::string> &disabledSizes) {
  for (const auto &sizePair : sizes) {
    if (disabledSizes.count(sizePair.first)) {
      continue;
    }
    std::string benchmarkName = "trisolv-" + sizePair.first;
    benchmark::RegisterBenchmark(benchmarkName.c_str(),
                                 [sizePair](benchmark::State &state) {
                                   runPolybench(state, sizePair.second);
                                 })
        ->Unit(benchmark::kMillisecond);
  }
}

void generateResultMLIRPolybenchTrisolv(size_t size_id) {
  const std::string benchmarkName = "trisolv-" + sizes[size_id].first;
  const std::vector<size_t> &size = sizes[size_id].second;

  const size_t N = size[0];

  MemRef<double, 2> inputL({N, N}, 0);
  MemRef<double, 1> inputX({N}, 0);
  MemRef<double, 1> inputB({N}, 0);

  _mlir_ciface_trisolv_init_array(N, &inputL, &inputX, &inputB);

  _mlir_ciface_trisolv(N, &inputL, &inputX, &inputB);

  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "Result for " << benchmarkName << ":\n";
  printArray(N, inputX.getData());
  std::cout << "--------------------------------------------------------"
            << std::endl;
}
