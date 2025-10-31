#include "perf_test.h"

extern int cvtBGRtoBGR(const unsigned char* src_data, size_t src_step,
                       unsigned char* dst_data, size_t dst_step,
                       int width, int height,
                       int depth, int scn, int dcn);

static void BM_CvtBGRtoBGR(benchmark::State& state) {
  const int width = state.range(0);
  const int height = state.range(1);
  const int scn = state.range(2);
  const int dcn = state.range(3);

  const size_t src_step = width * scn;
  const size_t dst_step = width * dcn;

  auto src = sequence_vector<unsigned char>(height * src_step);
  std::vector<unsigned char> dst(height * dst_step);

  for (auto _ : state) {
    int result = cvtBGRtoBGR(src.data(), src_step,
                             dst.data(), dst_step,
                             width, height,
                             0, scn, dcn);

    benchmark::DoNotOptimize(result);
    benchmark::DoNotOptimize(dst.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * width * height);
}

BENCHMARK(BM_CvtBGRtoBGR)
    ->Name("CvtBGRtoBGR_BGR3toBGR4_128x128")
    ->Args({128, 128, 3, 4})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1000)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR)
    ->Name("CvtBGRtoBGR_BGR3toBGR4_256x256")
    ->Args({256, 256, 3, 4})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(500)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR)
    ->Name("CvtBGRtoBGR_BGR3toBGR4_512x512")
    ->Args({512, 512, 3, 4})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(100)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR)
    ->Name("CvtBGRtoBGR_BGR3toBGR4_1024x1024")
    ->Args({1024, 1024, 3, 4})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(50)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR)
    ->Name("CvtBGRtoBGR_BGR3toBGR4_1920x1080")
    ->Args({1920, 1080, 3, 4})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(20)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR)
    ->Name("CvtBGRtoBGR_BGR4toBGR3_128x128")
    ->Args({128, 128, 4, 3})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1000)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR)
    ->Name("CvtBGRtoBGR_BGR4toBGR3_256x256")
    ->Args({256, 256, 4, 3})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(500)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR)
    ->Name("CvtBGRtoBGR_BGR4toBGR3_512x512")
    ->Args({512, 512, 4, 3})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(100)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR)
    ->Name("CvtBGRtoBGR_BGR4toBGR3_1024x1024")
    ->Args({1024, 1024, 4, 3})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(50)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR)
    ->Name("CvtBGRtoBGR_BGR4toBGR3_1920x1080")
    ->Args({1920, 1080, 4, 3})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(20)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR)
    ->Name("CvtBGRtoBGR_BGR3toBGR3_128x128")
    ->Args({128, 128, 3, 3})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1000)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR)
    ->Name("CvtBGRtoBGR_BGR3toBGR3_256x256")
    ->Args({256, 256, 3, 3})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(500)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR)
    ->Name("CvtBGRtoBGR_BGR3toBGR3_512x512")
    ->Args({512, 512, 3, 3})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(100)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR)
    ->Name("CvtBGRtoBGR_BGR3toBGR3_1024x1024")
    ->Args({1024, 1024, 3, 3})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(50)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR)
    ->Name("CvtBGRtoBGR_BGR3toBGR3_1920x1080")
    ->Args({1920, 1080, 3, 3})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(20)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR)
    ->Name("CvtBGRtoBGR_BGR4toBGR4_128x128")
    ->Args({128, 128, 4, 4})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1000)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR)
    ->Name("CvtBGRtoBGR_BGR4toBGR4_256x256")
    ->Args({256, 256, 4, 4})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(500)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR)
    ->Name("CvtBGRtoBGR_BGR4toBGR4_512x512")
    ->Args({512, 512, 4, 4})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(100)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR)
    ->Name("CvtBGRtoBGR_BGR4toBGR4_1024x1024")
    ->Args({1024, 1024, 4, 4})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(50)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR)
    ->Name("CvtBGRtoBGR_BGR4toBGR4_1920x1080")
    ->Args({1920, 1080, 4, 4})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(20)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR)
    ->Name("CvtBGRtoBGR_BGR3toBGR4_Edge_7x7")
    ->Args({7, 7, 3, 4})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(5000)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR)
    ->Name("CvtBGRtoBGR_BGR3toBGR4_Edge_15x15")
    ->Args({15, 15, 3, 4})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(2000)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR)
    ->Name("CvtBGRtoBGR_BGR3toBGR4_Edge_17x17")
    ->Args({17, 17, 3, 4})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(2000)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR)
    ->Name("CvtBGRtoBGR_BGR4toBGR3_Edge_31x31")
    ->Args({31, 31, 4, 3})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1000)
    ->Repetitions(3);

BENCHMARK(BM_CvtBGRtoBGR)
    ->Name("CvtBGRtoBGR_BGR4toBGR3_Edge_33x33")
    ->Args({33, 33, 4, 3})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1000)
    ->Repetitions(3);
