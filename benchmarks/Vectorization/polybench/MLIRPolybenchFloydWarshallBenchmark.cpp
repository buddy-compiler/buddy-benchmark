#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <vector>

extern "C" {
void _mlir_ciface_floyd_warshall(int, MemRef<int, 2> *);
void _mlir_ciface_floyd_warshall_init_array(int, MemRef<int, 2> *);
}

const std::vector<std::pair<std::string, std::vector<size_t>>> sizes = {
    {"mini", {60}},    {"small", {180}},       {"medium", {500}},
    {"large", {2800}}, {"extralarge", {5600}},
};

static void runPolybench(benchmark::State &state,
                         const std::vector<size_t> &size) {
  const size_t N = size[0];

  MemRef<int, 2> inputPath({N, N}, 0);

  for (auto _ : state) {
    state.PauseTiming();
    _mlir_ciface_floyd_warshall_init_array(N, &inputPath);
    state.ResumeTiming();
    _mlir_ciface_floyd_warshall(N, &inputPath);
  }
}

static void printArray(int n, int *path) {
  int i, j;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) {
        printf("\n");
      }
      printf("%d ", path[i * n + j]);
    }
  }
  printf("\n");
}

void registerMLIRPolybenchFloydWarshall(
    const std::set<std::string> &disabledSizes) {
  for (const auto &sizePair : sizes) {
    if (disabledSizes.count(sizePair.first)) {
      continue;
    }
    std::string benchmarkName = "floyd-warshall-" + sizePair.first;
    benchmark::RegisterBenchmark(benchmarkName.c_str(),
                                 [sizePair](benchmark::State &state) {
                                   runPolybench(state, sizePair.second);
                                 })
        ->Unit(benchmark::kMillisecond);
  }
}

void generateResultMLIRPolybenchFloydWarshall(size_t size_id) {
  const std::string benchmarkName = "floyd-warshall-" + sizes[size_id].first;
  const std::vector<size_t> &size = sizes[size_id].second;

  const size_t N = size[0];

  MemRef<int, 2> inputPath({N, N}, 0);

  _mlir_ciface_floyd_warshall_init_array(N, &inputPath);
  _mlir_ciface_floyd_warshall(N, &inputPath);

  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "Result for " << benchmarkName << ":\n";
  printArray(N, inputPath.getData());
  std::cout << "--------------------------------------------------------"
            << std::endl;
}
