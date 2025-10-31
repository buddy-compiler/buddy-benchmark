#include "perf_test.h"

extern void split8u(const unsigned char* src, unsigned char** dst, int len, int cn);

static void BM_Split(benchmark::State& state) {
  const int len = state.range(0);
  const int cn = state.range(1);

  auto src = sequence_vector<unsigned char>(len * cn);

  std::vector<unsigned char*> dst_ptrs(cn);
  std::vector<std::vector<unsigned char>> dst_arrays(cn);

  for (int i = 0; i < cn; ++i) {
    dst_arrays[i].resize(len);
    dst_ptrs[i] = dst_arrays[i].data();
  }

  for (auto _ : state) {
    split8u(src.data(), dst_ptrs.data(), len, cn);

    for (int i = 0; i < cn; ++i) {
      benchmark::DoNotOptimize(dst_ptrs[i]);
    }
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * len * cn);
}

BENCHMARK(BM_Split)
    ->Name("Split_C1")
    ->Args({1024, 1})
    ->Args({4096, 1})
    ->Args({16384, 1})
    ->Args({65536, 1})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1000)
    ->Repetitions(3);

BENCHMARK(BM_Split)
    ->Name("Split_C2")
    ->Args({1024, 2})
    ->Args({4096, 2})
    ->Args({16384, 2})
    ->Args({65536, 2})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1000)
    ->Repetitions(3);

BENCHMARK(BM_Split)
    ->Name("Split_C3")
    ->Args({1024, 3})
    ->Args({4096, 3})
    ->Args({16384, 3})
    ->Args({65536, 3})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1000)
    ->Repetitions(3);

BENCHMARK(BM_Split)
    ->Name("Split_C4")
    ->Args({1024, 4})
    ->Args({4096, 4})
    ->Args({16384, 4})
    ->Args({65536, 4})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1000)
    ->Repetitions(3);

BENCHMARK(BM_Split)
    ->Name("Split_C8")
    ->Args({1024, 8})
    ->Args({4096, 8})
    ->Args({16384, 8})
    ->Args({65536, 8})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(500)
    ->Repetitions(3);
