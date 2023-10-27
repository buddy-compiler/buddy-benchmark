#-------------------------------------------------------------------------------
# MLIR SCF Dialect linpackc Operation + Upstream Lowering Passes
#-------------------------------------------------------------------------------
# dscal
function(build_mlir_linpackcdscal type)
  add_custom_command(OUTPUT mlir-linpackcdscal${type}.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/linpackc/Dscal/MLIRLinpackCDscal.mlir |
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
            -o ${BUDDY_BINARY_DIR}/../benchmarks/Vectorization/linpackc/Dscal/mlir-linpackcdscal${type}.o
  )
  add_library(MLIRLinpackDscal${type} STATIC mlir-linpackcdscal${type}.o)
  set_target_properties(MLIRLinpackDscal${type} PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

build_mlir_linpackcdscal(f32)
build_mlir_linpackcdscal(f64)

# Dscal gcc
function(build_linpackcdscal_gcc type compiler)
add_custom_command(OUTPUT Dscal${type}_gcc.o
COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/linpackc/Dscal/Dscal.cpp |
        sed -e 's/TYPE_PLACEHOLDER/${type}/g' -e 's/COMPILER_PLACEHOLDER/${compiler}/g' |
        g++ -x c++ -c  -ftree-vectorize -march=native -o Dscal${type}_gcc.o -
        
)
add_library(Dscal${type}_gcc STATIC Dscal${type}_gcc.o)
set_target_properties(Dscal${type}_gcc PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

build_linpackcdscal_gcc(float gcc)
build_linpackcdscal_gcc(double gcc)

#Dscal clang 
function(build_linpackcdscal_clang type compiler)
add_custom_command(OUTPUT Dscal${type}_clang.o
COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/linpackc/Dscal/Dscal.cpp |
        sed -e 's/TYPE_PLACEHOLDER/${type}/g' -e 's/COMPILER_PLACEHOLDER/${compiler}/g' |
        ${LLVM_MLIR_BINARY_DIR}/clang++ -x c++ -c  -fvectorize -march=native -o Dscal${type}_clang.o -    
)
add_library(Dscal${type}_clang STATIC Dscal${type}_clang.o)
set_target_properties(Dscal${type}_clang PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

build_linpackcdscal_clang(float clang)
build_linpackcdscal_clang(double clang)
