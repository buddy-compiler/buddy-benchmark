#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <vector>

extern "C" {
void _mlir_ciface_durbin(int, MemRef<double, 1> *, MemRef<double, 1> *);
void _mlir_ciface_durbin_init_array(int, MemRef<double, 1> *);
}

const std::vector<std::pair<std::string, std::vector<size_t>>> sizes = {
    {"mini", {40}},    {"small", {120}},       {"medium", {400}},
    {"large", {2000}}, {"extralarge", {4000}},
};

static void runPolybench(benchmark::State &state,
                         const std::vector<size_t> &size) {
  const size_t N = size[0];

  MemRef<double, 1> inputR({N}, 0);
  MemRef<double, 1> inputY({N}, 0);

  for (auto _ : state) {
    state.PauseTiming();
    _mlir_ciface_durbin_init_array(N, &inputR);
    state.ResumeTiming();
    _mlir_ciface_durbin(N, &inputR, &inputY);
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

void registerMLIRPolybenchDurbin(const std::set<std::string> &disabledSizes) {
  for (const auto &sizePair : sizes) {
    if (disabledSizes.count(sizePair.first)) {
      continue;
    }
    std::string benchmarkName = "durbin-" + sizePair.first;
    benchmark::RegisterBenchmark(benchmarkName.c_str(),
                                 [sizePair](benchmark::State &state) {
                                   runPolybench(state, sizePair.second);
                                 })
        ->Unit(benchmark::kMillisecond);
  }
}

void generateResultMLIRPolybenchDurbin(size_t size_id) {
  const std::string benchmarkName = "durbin-" + sizes[size_id].first;
  const std::vector<size_t> &size = sizes[size_id].second;

  const size_t N = size[0];

  MemRef<double, 1> inputR({N}, 0);
  MemRef<double, 1> inputY({N}, 0);

  _mlir_ciface_durbin_init_array(N, &inputR);

  _mlir_ciface_durbin(N, &inputR, &inputY);

  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "Result for " << benchmarkName << ":\n";
  printArray(N, inputY.getData());
  std::cout << "--------------------------------------------------------"
            << std::endl;
}
