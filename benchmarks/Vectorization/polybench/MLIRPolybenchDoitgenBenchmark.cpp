#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <vector>

extern "C" {
void _mlir_ciface_doitgen(int, int, int, MemRef<double, 3> *,
                          MemRef<double, 2> *, MemRef<double, 1> *);
void _mlir_ciface_doitgen_init_array(int, int, int, MemRef<double, 3> *,
                                     MemRef<double, 2> *);
}

const std::vector<std::pair<std::string, std::vector<size_t>>> sizes = {
    {"mini", {8, 10, 12}}, {"small", {20, 25, 30}}, {"medium", {40, 50, 60}},
    // {"large", {140, 150, 160}},
    // {"extralarge", {220, 250, 270}},
};

static void runPolybench(benchmark::State &state,
                         const std::vector<size_t> &size) {
  const size_t NQ = size[0];
  const size_t NR = size[1];
  const size_t NP = size[2];

  MemRef<double, 3> inputA({NR, NQ, NP}, 0);
  MemRef<double, 2> inputC4({NP, NP}, 0);
  MemRef<double, 1> inputSum({NP}, 0);

  for (auto _ : state) {
    state.PauseTiming();
    _mlir_ciface_doitgen_init_array(NR, NQ, NP, &inputA, &inputC4);
    state.ResumeTiming();
    _mlir_ciface_doitgen(NR, NQ, NP, &inputA, &inputC4, &inputSum);
  }
}

static void printArray(int nr, int nq, int np, double *A) {
  int i, j, k;
  for (i = 0; i < nr; i++) {
    for (j = 0; j < nq; j++) {
      for (k = 0; k < np; k++) {
        if ((i * nq * np + j * np + k) % 20 == 0) {
          printf("\n");
        }
        printf("%0.2lf ", A[i * nq * np + j * np + k]);
      }
    }
  }
  printf("\n");
}

void registerMLIRPolybenchDoitgen() {
  for (const auto &sizePair : sizes) {
    std::string benchmarkName = "doitgen-" + sizePair.first;
    benchmark::RegisterBenchmark(benchmarkName.c_str(),
                                 [sizePair](benchmark::State &state) {
                                   runPolybench(state, sizePair.second);
                                 })
        ->Unit(benchmark::kMillisecond);
  }
}

void generateResultMLIRPolybenchDoitgen(size_t size_id) {
  const std::string benchmarkName = "doitgen-" + sizes[size_id].first;
  const std::vector<size_t> &size = sizes[size_id].second;

  const size_t NQ = size[0];
  const size_t NR = size[1];
  const size_t NP = size[2];

  MemRef<double, 3> inputA({NR, NQ, NP}, 0);
  MemRef<double, 2> inputC4({NP, NP}, 0);
  MemRef<double, 1> inputSum({NP}, 0);

  _mlir_ciface_doitgen_init_array(NR, NQ, NP, &inputA, &inputC4);

  _mlir_ciface_doitgen(NR, NQ, NP, &inputA, &inputC4, &inputSum);

  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "Result for " << benchmarkName << ":\n";
  printArray(NR, NQ, NP, inputA.getData());
  std::cout << "--------------------------------------------------------"
            << std::endl;
}
