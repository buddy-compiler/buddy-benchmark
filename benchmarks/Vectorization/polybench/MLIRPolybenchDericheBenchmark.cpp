#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <vector>

extern "C" {
void _mlir_ciface_deriche(int, int, float, MemRef<float, 2> *,
                          MemRef<float, 2> *, MemRef<float, 2> *,
                          MemRef<float, 2> *);
void _mlir_ciface_deriche_init_array(int, int, MemRef<float, 1> *,
                                     MemRef<float, 2> *, MemRef<float, 2> *);
}

const std::vector<std::pair<std::string, std::vector<size_t>>> sizes = {
    {"mini", {64, 64}},           {"small", {192, 128}},
    {"medium", {720, 480}},       {"large", {4096, 2160}},
    {"extralarge", {7680, 4320}},
};

static void runPolybench(benchmark::State &state,
                         const std::vector<size_t> &size) {
  const size_t W = size[0];
  const size_t H = size[1];

  MemRef<float, 1> alpha({1}, 0);
  MemRef<float, 2> inputImgIn({W, H}, 0);
  MemRef<float, 2> inputImgOut({W, H}, 0);
  MemRef<float, 2> inputY1({W, H}, 0);
  MemRef<float, 2> inputY2({W, H}, 0);

  for (auto _ : state) {
    state.PauseTiming();
    _mlir_ciface_deriche_init_array(W, H, &alpha, &inputImgIn, &inputImgOut);
    state.ResumeTiming();
    _mlir_ciface_deriche(W, H, alpha.getData()[0], &inputImgIn, &inputImgOut,
                         &inputY1, &inputY2);
  }
}

static void printArray(int w, int h, float *imgOut) {
  for (int i = 0; i < w; i++) {
    for (int j = 0; j < h; j++) {
      if ((i * h + j) % 20 == 0) {
        printf("\n");
      }
      printf("%0.2f ", imgOut[i * h + j]);
    }
  }
  printf("\n");
}

void registerMLIRPolybenchDeriche(const std::set<std::string> &disabledSizes) {
  for (const auto &sizePair : sizes) {
    if (disabledSizes.count(sizePair.first)) {
      continue;
    }
    std::string benchmarkName = "deriche-" + sizePair.first;
    benchmark::RegisterBenchmark(benchmarkName.c_str(),
                                 [sizePair](benchmark::State &state) {
                                   runPolybench(state, sizePair.second);
                                 })
        ->Unit(benchmark::kMillisecond);
  }
}

void generateResultMLIRPolybenchDeriche(size_t size_id) {
  const std::string benchmarkName = "deriche-" + sizes[size_id].first;
  const std::vector<size_t> &size = sizes[size_id].second;

  const size_t W = size[0];
  const size_t H = size[1];

  MemRef<float, 1> alpha({1}, 0);
  MemRef<float, 2> inputImgIn({W, H}, 0);
  MemRef<float, 2> inputImgOut({W, H}, 0);
  MemRef<float, 2> inputY1({W, H}, 0);
  MemRef<float, 2> inputY2({W, H}, 0);

  _mlir_ciface_deriche_init_array(W, H, &alpha, &inputImgIn, &inputImgOut);

  _mlir_ciface_deriche(W, H, alpha.getData()[0], &inputImgIn, &inputImgOut,
                       &inputY1, &inputY2);

  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "Result for " << benchmarkName << ":\n";
  printArray(W, H, inputImgOut.getData());
  std::cout << "--------------------------------------------------------"
            << std::endl;
}
