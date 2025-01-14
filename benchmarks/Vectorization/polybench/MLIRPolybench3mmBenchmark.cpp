#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <vector>

extern "C" {
void _mlir_ciface_kernel_3mm(int, int, int, int, int, MemRef<double, 2> *,
                             MemRef<double, 2> *, MemRef<double, 2> *,
                             MemRef<double, 2> *, MemRef<double, 2> *,
                             MemRef<double, 2> *, MemRef<double, 2> *);
void _mlir_ciface_kernel_3mm_init_array(int, int, int, int, int,
                                        MemRef<double, 2> *,
                                        MemRef<double, 2> *,
                                        MemRef<double, 2> *,
                                        MemRef<double, 2> *);
}

const std::vector<std::pair<std::string, std::vector<size_t>>> sizes = {
    {"mini", {16, 18, 20, 22, 24}},
    {"small", {40, 50, 60, 70, 80}},
    {"medium", {180, 190, 200, 210, 220}},
    // {"large", {800, 900, 1000, 1100, 1200}},
    // {"extralarge", {1600, 1800, 2000, 2200, 2400}},
};

static void runPolybench(benchmark::State &state,
                         const std::vector<size_t> &size) {
  const size_t NI = size[0];
  const size_t NJ = size[1];
  const size_t NK = size[2];
  const size_t NL = size[3];
  const size_t NM = size[4];

  MemRef<double, 2> inputE({NI, NJ}, 0);
  MemRef<double, 2> inputA({NI, NK}, 0);
  MemRef<double, 2> inputB({NK, NJ}, 0);
  MemRef<double, 2> inputF({NJ, NL}, 0);
  MemRef<double, 2> inputC({NJ, NM}, 0);
  MemRef<double, 2> inputD({NM, NL}, 0);
  MemRef<double, 2> inputG({NI, NL}, 0);

  for (auto _ : state) {
    state.PauseTiming();
    _mlir_ciface_kernel_3mm_init_array(NI, NJ, NK, NL, NM, &inputA, &inputB,
                                       &inputC, &inputD);
    state.ResumeTiming();
    _mlir_ciface_kernel_3mm(NI, NJ, NK, NL, NM, &inputE, &inputA, &inputB,
                            &inputF, &inputC, &inputD, &inputG);
  }
}

static void printArray(int ni, int nl, double *G) {
  int i, j;
  for (i = 0; i < ni; i++) {
    for (j = 0; j < nl; j++) {
      if ((i * ni + j) % 20 == 0) {
        printf("\n");
      }
      printf("%0.2lf ", G[i * nl + j]);
    }
  }
  printf("\n");
}

void registerMLIRPolybench3mm() {
  for (const auto &sizePair : sizes) {
    std::string benchmarkName = "3mm-" + sizePair.first;
    benchmark::RegisterBenchmark(benchmarkName.c_str(),
                                 [sizePair](benchmark::State &state) {
                                   runPolybench(state, sizePair.second);
                                 })
        ->Unit(benchmark::kMillisecond);
  }
}

void generateResultMLIRPolybench3mm(size_t size_id) {
  const std::string benchmarkName = "3mm-" + sizes[size_id].first;
  const std::vector<size_t> &size = sizes[size_id].second;

  const size_t NI = size[0];
  const size_t NJ = size[1];
  const size_t NK = size[2];
  const size_t NL = size[3];
  const size_t NM = size[4];

  MemRef<double, 2> inputE({NI, NJ}, 0);
  MemRef<double, 2> inputA({NI, NK}, 0);
  MemRef<double, 2> inputB({NK, NJ}, 0);
  MemRef<double, 2> inputF({NJ, NL}, 0);
  MemRef<double, 2> inputC({NJ, NM}, 0);
  MemRef<double, 2> inputD({NM, NL}, 0);
  MemRef<double, 2> inputG({NI, NL}, 0);

  _mlir_ciface_kernel_3mm_init_array(NI, NJ, NK, NL, NM, &inputA, &inputB,
                                     &inputC, &inputD);

  _mlir_ciface_kernel_3mm(NI, NJ, NK, NL, NM, &inputE, &inputA, &inputB,
                          &inputF, &inputC, &inputD, &inputG);

  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "Result for " << benchmarkName << ":\n";
  printArray(NI, NL, inputG.getData());
  std::cout << "--------------------------------------------------------"
            << std::endl;
}
