#include "perf_test.h"

extern void cvtBGRtoYUV(const unsigned char* src_data, size_t src_step,
                        unsigned char* dst_data, size_t dst_step,
                        int width, int height,
                        int depth, int scn);

static void BM_CvtBGRtoYUV(benchmark::State& state) {
  const int width = state.range(0);
  const int height = state.range(1);
  const int scn = state.range(2);

  const size_t src_step = width * scn;
  const size_t dst_step = width * 3;

  auto src = sequence_vector<unsigned char>(height * src_step);
  std::vector<unsigned char> dst(height * dst_step);

  for (auto _ : state) {
    cvtBGRtoYUV(src.data(), src_step,
                dst.data(), dst_step,
                width, height,
                0, scn);

    benchmark::DoNotOptimize(dst.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * width * height);
}

BENCHMARK(BM_CvtBGRtoYUV)
    ->Name("CvtBGRtoYUV_BGR3_128x128")
    ->Args({128, 128, 3})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1000)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoYUV)
    ->Name("CvtBGRtoYUV_BGR3_256x256")
    ->Args({256, 256, 3})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(500)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoYUV)
    ->Name("CvtBGRtoYUV_BGR3_512x512")
    ->Args({512, 512, 3})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(100)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoYUV)
    ->Name("CvtBGRtoYUV_BGR3_1024x1024")
    ->Args({1024, 1024, 3})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(50)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoYUV)
    ->Name("CvtBGRtoYUV_BGR3_1920x1080")
    ->Args({1920, 1080, 3})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(20)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoYUV)
    ->Name("CvtBGRtoYUV_BGRX4_128x128")
    ->Args({128, 128, 4})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1000)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoYUV)
    ->Name("CvtBGRtoYUV_BGRX4_256x256")
    ->Args({256, 256, 4})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(500)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoYUV)
    ->Name("CvtBGRtoYUV_BGRX4_512x512")
    ->Args({512, 512, 4})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(100)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoYUV)
    ->Name("CvtBGRtoYUV_BGRX4_1024x1024")
    ->Args({1024, 1024, 4})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(50)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoYUV)
    ->Name("CvtBGRtoYUV_BGRX4_1920x1080")
    ->Args({1920, 1080, 4})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(20)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoYUV)
    ->Name("CvtBGRtoYUV_BGR3_64x64")
    ->Args({64, 64, 3})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(2000)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoYUV)
    ->Name("CvtBGRtoYUV_BGR3_32x32")
    ->Args({32, 32, 3})
    ->Unit(benchmark::kNanosecond)
    ->Iterations(5000)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoYUV)
    ->Name("CvtBGRtoYUV_BGR3_640x480")
    ->Args({640, 480, 3})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(200)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoYUV)
    ->Name("CvtBGRtoYUV_BGR3_320x240")
    ->Args({320, 240, 3})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(500)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoYUV)
    ->Name("CvtBGRtoYUV_BGR3_2048x256")
    ->Args({2048, 256, 3})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(200)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoYUV)
    ->Name("CvtBGRtoYUV_BGR3_256x2048")
    ->Args({256, 2048, 3})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(200)
    ->Repetitions(3);
