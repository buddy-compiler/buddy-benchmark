#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <vector>

extern "C" {
void _mlir_ciface_atax(int, int, MemRef<double, 2> *, MemRef<double, 1> *,
                       MemRef<double, 1> *, MemRef<double, 1> *);
void _mlir_ciface_atax_init_array(int, int, MemRef<double, 2> *,
                                  MemRef<double, 1> *);
}

const std::vector<std::pair<std::string, std::vector<size_t>>> sizes = {
    {"mini", {38, 42}},           {"small", {116, 124}},
    {"medium", {390, 410}},       {"large", {1900, 2100}},
    {"extralarge", {1800, 2200}},
};

static void runPolybench(benchmark::State &state,
                         const std::vector<size_t> &size) {
  const size_t M = size[0];
  const size_t N = size[1];

  MemRef<double, 2> inputA({M, N}, 0);
  MemRef<double, 1> inputX({N}, 0);
  MemRef<double, 1> inputY({N}, 0);
  MemRef<double, 1> inputTMP({M}, 0);

  for (auto _ : state) {
    state.PauseTiming();
    _mlir_ciface_atax_init_array(M, N, &inputA, &inputX);
    state.ResumeTiming();
    _mlir_ciface_atax(M, N, &inputA, &inputX, &inputY, &inputTMP);
  }
}

static void printArray(int n, double *y) {
  for (int i = 0; i < n; i++) {
    if (i % 20 == 0) {
      printf("\n");
    }
    printf("%0.2lf ", y[i]);
  }
  printf("\n");
}

void registerMLIRPolybenchAtax(const std::set<std::string> &disabledSizes) {
  for (const auto &sizePair : sizes) {
    if (disabledSizes.count(sizePair.first)) {
      continue;
    }
    std::string benchmarkName = "atax-" + sizePair.first;
    benchmark::RegisterBenchmark(benchmarkName.c_str(),
                                 [sizePair](benchmark::State &state) {
                                   runPolybench(state, sizePair.second);
                                 })
        ->Unit(benchmark::kMillisecond);
  }
}

void generateResultMLIRPolybenchAtax(size_t size_id) {
  const std::string benchmarkName = "atax-" + sizes[size_id].first;
  const std::vector<size_t> &size = sizes[size_id].second;

  const size_t M = size[0];
  const size_t N = size[1];

  MemRef<double, 2> inputA({M, N}, 0);
  MemRef<double, 1> inputX({N}, 0);
  MemRef<double, 1> inputY({N}, 0);
  MemRef<double, 1> inputTMP({M}, 0);

  _mlir_ciface_atax_init_array(M, N, &inputA, &inputX);

  _mlir_ciface_atax(M, N, &inputA, &inputX, &inputY, &inputTMP);

  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "Result for " << benchmarkName << ":\n";
  printArray(N, inputY.getData());
  std::cout << "--------------------------------------------------------"
            << std::endl;
}
