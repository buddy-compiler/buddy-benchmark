#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <cstdio>
#include <vector>

extern "C" {
void _mlir_ciface_gramschmidt(int, int, MemRef<double, 2> *,
                              MemRef<double, 2> *, MemRef<double, 2> *);
void _mlir_ciface_gramschmidt_init_array(int, int, MemRef<double, 2> *,
                                         MemRef<double, 2> *,
                                         MemRef<double, 2> *);
}

const std::vector<std::pair<std::string, std::vector<size_t>>> sizes = {
    {"mini", {20, 30}}, {"small", {60, 80}}, {"medium", {200, 240}},
    // {"large", {1000, 1200}},
    // {"extralarge", {2000, 2600}},
};

static void runPolybench(benchmark::State &state,
                         const std::vector<size_t> &size) {
  const size_t M = size[0];
  const size_t N = size[1];

  MemRef<double, 2> inputA({M, N}, 0);
  MemRef<double, 2> inputR({N, N}, 0);
  MemRef<double, 2> inputQ({M, N}, 0);

  for (auto _ : state) {
    state.PauseTiming();
    _mlir_ciface_gramschmidt_init_array(M, N, &inputA, &inputR, &inputQ);
    state.ResumeTiming();
    _mlir_ciface_gramschmidt(M, N, &inputA, &inputR, &inputQ);
  }
}

static void printArray(int m, int n, double *A, double *R, double *Q) {
  printf("begin dump: R");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) {
        printf("\n");
      }
      printf("%0.2lf ", R[i * n + j]);
    }
  }
  printf("\nend   dump: R\n");

  printf("begin dump: Q");
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) {
        printf("\n");
      }
      printf("%0.2lf ", Q[i * n + j]);
    }
  }
  printf("\nend   dump: Q\n");
}

void registerMLIRPolybenchGramschmidt() {
  for (const auto &sizePair : sizes) {
    std::string benchmarkName = "gramschmidt-" + sizePair.first;
    benchmark::RegisterBenchmark(benchmarkName.c_str(),
                                 [sizePair](benchmark::State &state) {
                                   runPolybench(state, sizePair.second);
                                 })
        ->Unit(benchmark::kMillisecond);
  }
}

void generateResultMLIRPolybenchGramschmidt(size_t size_id) {
  const std::string benchmarkName = "gramschmidt-" + sizes[size_id].first;
  const std::vector<size_t> &size = sizes[size_id].second;

  const size_t M = size[0];
  const size_t N = size[1];

  MemRef<double, 2> inputA({M, N}, 0);
  MemRef<double, 2> inputR({N, N}, 0);
  MemRef<double, 2> inputQ({M, N}, 0);

  _mlir_ciface_gramschmidt_init_array(M, N, &inputA, &inputR, &inputQ);

  _mlir_ciface_gramschmidt(M, N, &inputA, &inputR, &inputQ);

  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "Result for " << benchmarkName << ":\n";
  printArray(M, N, inputA.getData(), inputR.getData(), inputQ.getData());
  std::cout << "--------------------------------------------------------"
            << std::endl;
}
