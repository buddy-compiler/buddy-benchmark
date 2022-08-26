buddy-opt opt_gemm.mlir  \
    -convert-vector-to-llvm \
    -convert-memref-to-llvm \
    --lower-affine \
    -convert-scf-to-cf \
    -convert-linalg-to-llvm \
    -llvm-request-c-wrappers \
    -convert-func-to-llvm \
    -reconcile-unrealized-casts   \
    | \
buddy-translate --mlir-to-llvmir \
| \
llc -mtriple='x86_64-unknown-linux-gnu' \
-mattr='avx512f' --filetype=obj -O2 -o mlir-gemm.o
llvm-objdump -d mlir-gemm.o > dump.ll
rm mlir-gemm.o
cat dump.ll | less
