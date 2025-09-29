#include "perf_test.h"

extern int cvtBGRtoBGR5x5(const unsigned char* src_data, size_t src_step,
                          unsigned char* dst_data, size_t dst_step,
                          int width, int height,
                          int scn, bool swapBlue, int greenBits);

static void BM_CvtBGRtoBGR5x5(benchmark::State& state) {
  const int width = state.range(0);
  const int height = state.range(1);
  const int scn = state.range(2);
  const bool swapBlue = state.range(3);
  const int greenBits = state.range(4);

  const size_t src_step = width * scn;
  const size_t dst_step = width * 2;

  auto src = sequence_vector<unsigned char>(height * src_step);
  std::vector<unsigned char> dst(height * dst_step);

  for (auto _ : state) {
    int result = cvtBGRtoBGR5x5(src.data(), src_step,
                                dst.data(), dst_step,
                                width, height,
                                scn, swapBlue, greenBits);

    benchmark::DoNotOptimize(result);
    benchmark::DoNotOptimize(dst.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * width * height);
}

BENCHMARK(BM_CvtBGRtoBGR5x5)
    ->Name("CvtBGRtoBGR5x5_BGR3_to_BGR565_128x128")
    ->Args({128, 128, 3, 0, 6})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1000)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR5x5)
    ->Name("CvtBGRtoBGR5x5_BGR3_to_BGR565_256x256")
    ->Args({256, 256, 3, 0, 6})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(500)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR5x5)
    ->Name("CvtBGRtoBGR5x5_BGR3_to_BGR565_512x512")
    ->Args({512, 512, 3, 0, 6})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(100)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR5x5)
    ->Name("CvtBGRtoBGR5x5_BGR3_to_BGR565_1024x1024")
    ->Args({1024, 1024, 3, 0, 6})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(50)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR5x5)
    ->Name("CvtBGRtoBGR5x5_BGR3_to_BGR565_1920x1080")
    ->Args({1920, 1080, 3, 0, 6})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(20)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR5x5)
    ->Name("CvtBGRtoBGR5x5_BGR3_to_BGR555_128x128")
    ->Args({128, 128, 3, 0, 5})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1000)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR5x5)
    ->Name("CvtBGRtoBGR5x5_BGR3_to_BGR555_256x256")
    ->Args({256, 256, 3, 0, 5})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(500)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR5x5)
    ->Name("CvtBGRtoBGR5x5_BGR3_to_BGR555_512x512")
    ->Args({512, 512, 3, 0, 5})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(100)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR5x5)
    ->Name("CvtBGRtoBGR5x5_BGR3_to_BGR555_1024x1024")
    ->Args({1024, 1024, 3, 0, 5})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(50)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR5x5)
    ->Name("CvtBGRtoBGR5x5_BGRX4_to_BGR565_128x128")
    ->Args({128, 128, 4, 0, 6})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1000)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR5x5)
    ->Name("CvtBGRtoBGR5x5_BGRX4_to_BGR565_256x256")
    ->Args({256, 256, 4, 0, 6})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(500)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR5x5)
    ->Name("CvtBGRtoBGR5x5_BGRX4_to_BGR565_512x512")
    ->Args({512, 512, 4, 0, 6})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(100)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR5x5)
    ->Name("CvtBGRtoBGR5x5_BGRX4_to_BGR565_1024x1024")
    ->Args({1024, 1024, 4, 0, 6})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(50)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR5x5)
    ->Name("CvtBGRtoBGR5x5_BGRX4_to_BGR555_128x128")
    ->Args({128, 128, 4, 0, 5})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1000)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR5x5)
    ->Name("CvtBGRtoBGR5x5_BGRX4_to_BGR555_256x256")
    ->Args({256, 256, 4, 0, 5})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(500)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR5x5)
    ->Name("CvtBGRtoBGR5x5_BGRX4_to_BGR555_512x512")
    ->Args({512, 512, 4, 0, 5})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(100)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR5x5)
    ->Name("CvtBGRtoBGR5x5_BGR3_to_BGR565_SwapBlue_512x512")
    ->Args({512, 512, 3, 1, 6})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(100)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR5x5)
    ->Name("CvtBGRtoBGR5x5_BGRX4_to_BGR555_SwapBlue_512x512")
    ->Args({512, 512, 4, 1, 5})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(100)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR5x5)
    ->Name("CvtBGRtoBGR5x5_BGR3_to_BGR565_64x64")
    ->Args({64, 64, 3, 0, 6})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(2000)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR5x5)
    ->Name("CvtBGRtoBGR5x5_BGR3_to_BGR565_1280x720")
    ->Args({1280, 720, 3, 0, 6})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(30)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR5x5)
    ->Name("CvtBGRtoBGR5x5_BGR3_to_BGR565_3840x2160")
    ->Args({3840, 2160, 3, 0, 6})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(5)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR5x5)
    ->Name("CvtBGRtoBGR5x5_BGR3_to_BGR565_8x512")
    ->Args({8, 512, 3, 0, 6})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(200)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR5x5)
    ->Name("CvtBGRtoBGR5x5_BGR3_to_BGR565_15x512")
    ->Args({15, 512, 3, 0, 6})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(200)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR5x5)
    ->Name("CvtBGRtoBGR5x5_BGR3_to_BGR565_16x512")
    ->Args({16, 512, 3, 0, 6})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(200)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR5x5)
    ->Name("CvtBGRtoBGR5x5_BGR3_to_BGR565_17x512")
    ->Args({17, 512, 3, 0, 6})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(200)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR5x5)
    ->Name("CvtBGRtoBGR5x5_BGR3_to_BGR555_PERF_1024x1024")
    ->Args({1024, 1024, 3, 0, 5})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100)
    ->Repetitions(5);

BENCHMARK(BM_CvtBGRtoBGR5x5)
    ->Name("CvtBGRtoBGR5x5_BGR3_to_BGR565_PERF_1024x1024")
    ->Args({1024, 1024, 3, 0, 6})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100)
    ->Repetitions(5);

BENCHMARK(BM_CvtBGRtoBGR5x5)
    ->Name("CvtBGRtoBGR5x5_BGR3_to_BGR565_PERF_512x512")
    ->Args({512, 512, 3, 0, 6})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(200)
    ->Repetitions(5);

BENCHMARK(BM_CvtBGRtoBGR5x5)
    ->Name("CvtBGRtoBGR5x5_BGRX4_to_BGR565_PERF_512x512")
    ->Args({512, 512, 4, 0, 6})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(200)
    ->Repetitions(5);

BENCHMARK(BM_CvtBGRtoBGR5x5)
    ->Name("CvtBGRtoBGR5x5_BGR3_to_BGR565_NoSwap_512x512")
    ->Args({512, 512, 3, 0, 6})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(200)
    ->Repetitions(5);

BENCHMARK(BM_CvtBGRtoBGR5x5)
    ->Name("CvtBGRtoBGR5x5_BGR3_to_BGR565_WithSwap_512x512")
    ->Args({512, 512, 3, 1, 6})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(200)
    ->Repetitions(5);
