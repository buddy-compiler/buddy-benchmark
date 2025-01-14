#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <vector>

extern "C" {
void _mlir_ciface_mvt(int, MemRef<double, 1> *, MemRef<double, 1> *,
                      MemRef<double, 1> *, MemRef<double, 1> *,
                      MemRef<double, 2> *);
void _mlir_ciface_mvt_init_array(int, MemRef<double, 1> *, MemRef<double, 1> *,
                                 MemRef<double, 1> *, MemRef<double, 1> *,
                                 MemRef<double, 2> *);
}

const std::vector<std::pair<std::string, std::vector<size_t>>> sizes = {
    {"mini", {40}},    {"small", {120}},       {"medium", {400}},
    {"large", {2000}}, {"extralarge", {4000}},
};

static void runPolybench(benchmark::State &state,
                         const std::vector<size_t> &size) {
  const size_t N = size[0];

  MemRef<double, 1> inputX1({N}, 0);
  MemRef<double, 1> inputX2({N}, 0);
  MemRef<double, 1> inputY1({N}, 0);
  MemRef<double, 1> inputY2({N}, 0);
  MemRef<double, 2> inputA({N, N}, 0);

  for (auto _ : state) {
    state.PauseTiming();
    _mlir_ciface_mvt_init_array(N, &inputX1, &inputX2, &inputY1, &inputY2,
                                &inputA);
    state.ResumeTiming();
    _mlir_ciface_mvt(N, &inputX1, &inputX2, &inputY1, &inputY2, &inputA);
  }
}

static void printArray(int n, double *x1, double *x2) {
  printf("begin dump: x1");
  for (int i = 0; i < n; i++) {
    if (i % 20 == 0) {
      printf("\n");
    }
    printf("%0.2lf ", x1[i]);
  }
  printf("\nend   dump: x1\n");

  printf("begin dump: x2");
  for (int i = 0; i < n; i++) {
    if (i % 20 == 0) {
      printf("\n");
    }
    printf("%0.2lf ", x2[i]);
  }
  printf("\nend   dump: x2\n");
}

void registerMLIRPolybenchMvt(const std::set<std::string> &disabledSizes) {
  for (const auto &sizePair : sizes) {
    if (disabledSizes.count(sizePair.first)) {
      continue;
    }
    std::string benchmarkName = "mvt-" + sizePair.first;
    benchmark::RegisterBenchmark(benchmarkName.c_str(),
                                 [sizePair](benchmark::State &state) {
                                   runPolybench(state, sizePair.second);
                                 })
        ->Unit(benchmark::kMillisecond);
  }
}

void generateResultMLIRPolybenchMvt(size_t size_id) {
  const std::string benchmarkName = "mvt-" + sizes[size_id].first;
  const std::vector<size_t> &size = sizes[size_id].second;

  const size_t N = size[0];

  MemRef<double, 1> inputX1({N}, 0);
  MemRef<double, 1> inputX2({N}, 0);
  MemRef<double, 1> inputY1({N}, 0);
  MemRef<double, 1> inputY2({N}, 0);
  MemRef<double, 2> inputA({N, N}, 0);

  _mlir_ciface_mvt_init_array(N, &inputX1, &inputX2, &inputY1, &inputY2,
                              &inputA);
  _mlir_ciface_mvt(N, &inputX1, &inputX2, &inputY1, &inputY2, &inputA);

  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "Result for " << benchmarkName << ":\n";
  printArray(N, inputX1.getData(), inputX2.getData());
  std::cout << "--------------------------------------------------------"
            << std::endl;
}
