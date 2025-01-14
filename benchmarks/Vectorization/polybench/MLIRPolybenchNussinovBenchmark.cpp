#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <vector>

extern "C" {
void _mlir_ciface_nussinov(int, MemRef<char, 1> *, MemRef<int, 2> *);
void _mlir_ciface_nussinov_init_array(int, MemRef<char, 1> *, MemRef<int, 2> *);
}

const std::vector<std::pair<std::string, std::vector<size_t>>> sizes = {
    {"mini", {60}},    {"small", {180}},       {"medium", {500}},
    {"large", {2500}}, {"extralarge", {5500}},
};

static void runPolybench(benchmark::State &state,
                         const std::vector<size_t> &size) {
  const size_t N = size[0];

  MemRef<char, 1> inputSeq({N}, 0);
  MemRef<int, 2> inputTable({N, N}, 0);

  for (auto _ : state) {
    state.PauseTiming();
    _mlir_ciface_nussinov_init_array(N, &inputSeq, &inputTable);
    state.ResumeTiming();
    _mlir_ciface_nussinov(N, &inputSeq, &inputTable);
  }
}

static void printArray(int n, int *table) {
  int i, j;
  int t = 0;
  for (i = 0; i < n; i++) {
    for (j = i; j < n; j++) {
      if (t % 20 == 0) {
        printf("\n");
      }
      printf("%d ", table[i * n + j]);
      t++;
    }
  }
  printf("\n");
}

void registerMLIRPolybenchNussinov() {
  for (const auto &sizePair : sizes) {
    std::string benchmarkName = "nussinov-" + sizePair.first;
    benchmark::RegisterBenchmark(benchmarkName.c_str(),
                                 [sizePair](benchmark::State &state) {
                                   runPolybench(state, sizePair.second);
                                 })
        ->Unit(benchmark::kMillisecond);
  }
}

void generateResultMLIRPolybenchNussinov(size_t size_id) {
  const std::string benchmarkName = "nussinov-" + sizes[size_id].first;
  const std::vector<size_t> &size = sizes[size_id].second;

  const size_t N = size[0];

  MemRef<char, 1> inputSeq({N}, 0);
  MemRef<int, 2> inputTable({N, N}, 0);

  _mlir_ciface_nussinov_init_array(N, &inputSeq, &inputTable);
  _mlir_ciface_nussinov(N, &inputSeq, &inputTable);

  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "Result for " << benchmarkName << ":\n";
  printArray(N, inputTable.getData());
  std::cout << "--------------------------------------------------------"
            << std::endl;
}
