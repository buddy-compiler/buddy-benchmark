#include "perf_test.h"

extern int cvtBGRtoHSV(int start, int end, const unsigned char * src, size_t src_step,
                       unsigned char * dst, size_t dst_step, int width, int scn,
                       bool swapBlue, bool isFullRange, bool isHSV);

static void BM_CvtBGRtoHSV(benchmark::State& state) {
  const int width = state.range(0);
  const int height = state.range(1);
  const int scn = state.range(2);

  const size_t src_step = width * scn;
  const size_t dst_step = width * 3;

  auto src = sequence_vector<unsigned char>(height * src_step);
  std::vector<unsigned char> dst(height * dst_step);

  for (auto _ : state) {
    int result = cvtBGRtoHSV(0, height, src.data(), src_step,
                              dst.data(), dst_step,
                              width, scn,
                              false, false, true);

    benchmark::DoNotOptimize(result);
    benchmark::DoNotOptimize(dst.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * width * height);
}

BENCHMARK(BM_CvtBGRtoHSV)
    ->Name("CvtBGRtoHSV_BGR3_128x128")
    ->Args({128, 128, 3})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1000)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoHSV)
    ->Name("CvtBGRtoHSV_BGR3_256x256")
    ->Args({256, 256, 3})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(500)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoHSV)
    ->Name("CvtBGRtoHSV_BGR3_512x512")
    ->Args({512, 512, 3})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(100)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoHSV)
    ->Name("CvtBGRtoHSV_BGR3_1024x1024")
    ->Args({1024, 1024, 3})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(50)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoHSV)
    ->Name("CvtBGRtoHSV_BGR3_1920x1080")
    ->Args({1920, 1080, 3})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(20)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoHSV)
    ->Name("CvtBGRtoHSV_BGRX4_128x128")
    ->Args({128, 128, 4})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1000)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoHSV)
    ->Name("CvtBGRtoHSV_BGRX4_256x256")
    ->Args({256, 256, 4})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(500)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoHSV)
    ->Name("CvtBGRtoHSV_BGRX4_512x512")
    ->Args({512, 512, 4})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(100)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoHSV)
    ->Name("CvtBGRtoHSV_BGRX4_1024x1024")
    ->Args({1024, 1024, 4})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(50)
    ->Repetitions(3);
