#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <vector>

extern "C" {
void _mlir_ciface_adi(int, int, MemRef<double, 2> *, MemRef<double, 2> *,
                      MemRef<double, 2> *, MemRef<double, 2> *);
void _mlir_ciface_adi_init_array(int, MemRef<double, 2> *);
}

const std::vector<std::pair<std::string, std::vector<size_t>>> sizes = {
    {"mini", {20, 20}},           {"small", {40, 60}},
    {"medium", {100, 200}},       {"large", {500, 1000}},
    {"extralarge", {1000, 2000}},
};

static void runPolybench(benchmark::State &state,
                         const std::vector<size_t> &size) {
  const size_t TSTEPS = size[0];
  const size_t N = size[1];

  MemRef<double, 2> inputU({N, N}, 0);
  MemRef<double, 2> inputV({N, N}, 0);
  MemRef<double, 2> inputP({N, N}, 0);
  MemRef<double, 2> inputQ({N, N}, 0);

  for (auto _ : state) {
    state.PauseTiming();
    _mlir_ciface_adi_init_array(N, &inputU);
    state.ResumeTiming();
    _mlir_ciface_adi(TSTEPS, N, &inputU, &inputV, &inputP, &inputQ);
  }
}

static void printArray(int n, double *u) {
  int i, j;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) {
        printf("\n");
      }
      printf("%0.2lf ", u[i * n + j]);
    }
  }
  printf("\n");
}

void registerMLIRPolybenchAdi(const std::set<std::string> &disabledSizes) {
  for (const auto &sizePair : sizes) {
    if (disabledSizes.count(sizePair.first)) {
      continue;
    }
    std::string benchmarkName = "adi-" + sizePair.first;
    benchmark::RegisterBenchmark(benchmarkName.c_str(),
                                 [sizePair](benchmark::State &state) {
                                   runPolybench(state, sizePair.second);
                                 })
        ->Unit(benchmark::kMillisecond);
  }
}

void generateResultMLIRPolybenchAdi(size_t size_id) {
  const std::string benchmarkName = "adi-" + sizes[size_id].first;
  const std::vector<size_t> &size = sizes[size_id].second;

  const size_t TSTEPS = size[0];
  const size_t N = size[1];

  MemRef<double, 2> inputU({N, N}, 0);
  MemRef<double, 2> inputV({N, N}, 0);
  MemRef<double, 2> inputP({N, N}, 0);
  MemRef<double, 2> inputQ({N, N}, 0);

  _mlir_ciface_adi_init_array(N, &inputU);

  _mlir_ciface_adi(TSTEPS, N, &inputU, &inputV, &inputP, &inputQ);

  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "Result for " << benchmarkName << ":\n";
  printArray(N, inputU.getData());
  std::cout << "--------------------------------------------------------"
            << std::endl;
}
