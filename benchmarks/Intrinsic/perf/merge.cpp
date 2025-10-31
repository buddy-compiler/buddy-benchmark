#include "perf_test.h"

extern void merge8u(const unsigned char** src, unsigned char* dst, int len, int cn);

static void BM_Merge(benchmark::State& state) {
  const int len = state.range(0);
  const int cn = state.range(1);

  std::vector<std::vector<unsigned char>> src_arrays(cn);
  std::vector<const unsigned char*> src_ptrs(cn);

  for (int i = 0; i < cn; ++i) {
    src_arrays[i] = sequence_vector<unsigned char>(len);
    src_ptrs[i] = src_arrays[i].data();
  }

  auto dst = sequence_vector<unsigned char>(len * cn);

  for (auto _ : state) {
    merge8u(src_ptrs.data(), dst.data(), len, cn);

    benchmark::DoNotOptimize(dst.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * len * cn);
}

BENCHMARK(BM_Merge)
    ->Name("Merge_C1")
    ->Args({1024, 1})
    ->Args({4096, 1})
    ->Args({16384, 1})
    ->Args({65536, 1})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1000)
    ->Repetitions(3);

BENCHMARK(BM_Merge)
    ->Name("Merge_C2")
    ->Args({1024, 2})
    ->Args({4096, 2})
    ->Args({16384, 2})
    ->Args({65536, 2})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1000)
    ->Repetitions(3);

BENCHMARK(BM_Merge)
    ->Name("Merge_C3")
    ->Args({1024, 3})
    ->Args({4096, 3})
    ->Args({16384, 3})
    ->Args({65536, 3})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1000)
    ->Repetitions(3);

BENCHMARK(BM_Merge)
    ->Name("Merge_C4")
    ->Args({1024, 4})
    ->Args({4096, 4})
    ->Args({16384, 4})
    ->Args({65536, 4})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1000)
    ->Repetitions(3);

BENCHMARK(BM_Merge)
    ->Name("Merge_C8")
    ->Args({1024, 8})
    ->Args({4096, 8})
    ->Args({16384, 8})
    ->Args({65536, 8})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(500)
    ->Repetitions(3);
