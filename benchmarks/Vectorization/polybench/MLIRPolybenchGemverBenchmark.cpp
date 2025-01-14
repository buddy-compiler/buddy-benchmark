#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <vector>

extern "C" {
void _mlir_ciface_gemver(int, double, double, MemRef<double, 2> *,
                         MemRef<double, 1> *, MemRef<double, 1> *,
                         MemRef<double, 1> *, MemRef<double, 1> *,
                         MemRef<double, 1> *, MemRef<double, 1> *,
                         MemRef<double, 1> *, MemRef<double, 1> *);
void _mlir_ciface_gemver_init_array(int, MemRef<double, 1> *,
                                    MemRef<double, 1> *, MemRef<double, 2> *,
                                    MemRef<double, 1> *, MemRef<double, 1> *,
                                    MemRef<double, 1> *, MemRef<double, 1> *,
                                    MemRef<double, 1> *, MemRef<double, 1> *,
                                    MemRef<double, 1> *, MemRef<double, 1> *);
}

const std::vector<std::pair<std::string, std::vector<size_t>>> sizes = {
    {"mini", {40}},    {"small", {120}},       {"medium", {400}},
    {"large", {2000}}, {"extralarge", {4000}},
};

static void runPolybench(benchmark::State &state,
                         const std::vector<size_t> &size) {
  const size_t N = size[0];

  MemRef<double, 1> alpha({1}, 0);
  MemRef<double, 1> beta({1}, 0);

  MemRef<double, 2> inputA({N, N}, 0);
  MemRef<double, 1> inputU1({N}, 0);
  MemRef<double, 1> inputV1({N}, 0);
  MemRef<double, 1> inputU2({N}, 0);
  MemRef<double, 1> inputV2({N}, 0);
  MemRef<double, 1> inputW({N}, 0);
  MemRef<double, 1> inputX({N}, 0);
  MemRef<double, 1> inputY({N}, 0);
  MemRef<double, 1> inputZ({N}, 0);

  for (auto _ : state) {
    state.PauseTiming();
    _mlir_ciface_gemver_init_array(N, &alpha, &beta, &inputA, &inputU1,
                                   &inputV1, &inputU2, &inputV2, &inputW,
                                   &inputX, &inputY, &inputZ);
    state.ResumeTiming();
    _mlir_ciface_gemver(N, alpha.getData()[0], beta.getData()[0], &inputA,
                        &inputU1, &inputV1, &inputU2, &inputV2, &inputW,
                        &inputX, &inputY, &inputZ);
  }
}

static void printArray(int n, double *w) {
  int i;
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) {
      printf("\n");
    }
    printf("%0.2lf ", w[i]);
  }
  printf("\n");
}

void registerMLIRPolybenchGemver() {
  for (const auto &sizePair : sizes) {
    std::string benchmarkName = "gemver-" + sizePair.first;
    benchmark::RegisterBenchmark(benchmarkName.c_str(),
                                 [sizePair](benchmark::State &state) {
                                   runPolybench(state, sizePair.second);
                                 })
        ->Unit(benchmark::kMillisecond);
  }
}

void generateResultMLIRPolybenchGemver(size_t size_id) {
  const std::string benchmarkName = "gemver-" + sizes[size_id].first;
  const std::vector<size_t> &size = sizes[size_id].second;
  const size_t N = size[0];

  MemRef<double, 1> alpha({1}, 0);
  MemRef<double, 1> beta({1}, 0);

  MemRef<double, 2> inputA({N, N}, 0);
  MemRef<double, 1> inputU1({N}, 0);
  MemRef<double, 1> inputV1({N}, 0);
  MemRef<double, 1> inputU2({N}, 0);
  MemRef<double, 1> inputV2({N}, 0);
  MemRef<double, 1> inputW({N}, 0);
  MemRef<double, 1> inputX({N}, 0);
  MemRef<double, 1> inputY({N}, 0);
  MemRef<double, 1> inputZ({N}, 0);

  _mlir_ciface_gemver_init_array(N, &alpha, &beta, &inputA, &inputU1, &inputV1,
                                 &inputU2, &inputV2, &inputW, &inputX, &inputY,
                                 &inputZ);

  _mlir_ciface_gemver(N, alpha.getData()[0], beta.getData()[0], &inputA,
                      &inputU1, &inputV1, &inputU2, &inputV2, &inputW, &inputX,
                      &inputY, &inputZ);

  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "Result for " << benchmarkName << ":\n";
  printArray(N, inputW.getData());
  std::cout << "--------------------------------------------------------"
            << std::endl;
}
