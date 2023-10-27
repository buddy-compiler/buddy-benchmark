#-------------------------------------------------------------------------------
# MLIR SCF Dialect linpackc Operation + Upstream Lowering Passes
#-------------------------------------------------------------------------------
function(build_mlir_linpackcdaxpy type)
  add_custom_command(OUTPUT mlir-linpackcdaxpy${type}.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/linpackc/Daxpy/MLIRLinpackCDaxpy.mlir |
          sed 's/TYPE_PLACEHOLDER/${type}/g' |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt
            -convert-scf-to-cf
            -expand-strided-metadata
            -convert-arith-to-llvm
            -llvm-request-c-wrappers
            --convert-math-to-llvm
            -finalize-memref-to-llvm
            -convert-func-to-llvm 
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
            -o ${BUDDY_BINARY_DIR}/../benchmarks/Vectorization/linpackc/Daxpy/mlir-linpackcdaxpy${type}.o
  )
  add_library(MLIRLinpackDaxpy${type} STATIC mlir-linpackcdaxpy${type}.o)
  set_target_properties(MLIRLinpackDaxpy${type} PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

build_mlir_linpackcdaxpy(f32)
build_mlir_linpackcdaxpy(f64)

# Daxpy gcc
function(build_linpackcdaxpy_gcc type compiler)
add_custom_command(OUTPUT Daxpy${type}_gcc.o
COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/linpackc/Daxpy/Daxpy.cpp |
        sed -e 's/TYPE_PLACEHOLDER/${type}/g' -e 's/COMPILER_PLACEHOLDER/${compiler}/g' |
        g++ -x c++ -c  -ftree-vectorize -march=native -o Daxpy${type}_gcc.o -
        
)
add_library(Daxpy${type}_gcc STATIC Daxpy${type}_gcc.o)
set_target_properties(Daxpy${type}_gcc PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

build_linpackcdaxpy_gcc(float gcc)
build_linpackcdaxpy_gcc(double gcc)

#Daxpy clang 
function(build_linpackcdaxpy_clang type compiler)
add_custom_command(OUTPUT Daxpy${type}_clang.o
COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/linpackc/Daxpy/Daxpy.cpp |
        sed -e 's/TYPE_PLACEHOLDER/${type}/g' -e 's/COMPILER_PLACEHOLDER/${compiler}/g' |
        ${LLVM_MLIR_BINARY_DIR}/clang++ -x c++ -c  -fvectorize -march=native -o Daxpy${type}_clang.o -    
)
add_library(Daxpy${type}_clang STATIC Daxpy${type}_clang.o)
set_target_properties(Daxpy${type}_clang PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

build_linpackcdaxpy_clang(float clang)
build_linpackcdaxpy_clang(double clang)