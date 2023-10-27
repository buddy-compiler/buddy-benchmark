#-------------------------------------------------------------------------------
# MLIR SCF Dialect linpackc Operation + Upstream Lowering Passes
#-------------------------------------------------------------------------------
add_custom_command(OUTPUT mlir-linpackcepslonf32.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
          ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/linpackc/Epslon/MLIRLinpackCEpslonF32.mlir 
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
            -o ${BUDDY_BINARY_DIR}/../benchmarks/Vectorization/linpackc/Epslon/mlir-linpackcepslonf32.o
)
add_library(MLIRLinpackEpslonf32 STATIC mlir-linpackcepslonf32.o)
set_target_properties(MLIRLinpackEpslonf32 PROPERTIES LINKER_LANGUAGE CXX)

#-------------------------------------------------------------------------------
# MLIR SCF Dialect linpackc Operation + Upstream Lowering Passes
#-------------------------------------------------------------------------------
add_custom_command(OUTPUT mlir-linpackcepslonf64.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
          ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/linpackc/Epslon/MLIRLinpackCEpslonF64.mlir 
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
            -o ${BUDDY_BINARY_DIR}/../benchmarks/Vectorization/linpackc/Epslon/mlir-linpackcepslonf64.o
)
add_library(MLIRLinpackEpslonf64 STATIC mlir-linpackcepslonf64.o)
set_target_properties(MLIRLinpackEpslonf64 PROPERTIES LINKER_LANGUAGE CXX)

# Epslon gcc
function(build_linpackcepslon_gcc type compiler)
add_custom_command(OUTPUT Epslon${type}_gcc.o
COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/linpackc/Epslon/Epslon.cpp |
        sed -e 's/TYPE_PLACEHOLDER/${type}/g' -e 's/COMPILER_PLACEHOLDER/${compiler}/g' |
        g++ -x c++ -c  -ftree-vectorize -march=native -o Epslon${type}_gcc.o -
        
)
add_library(Epslon${type}_gcc STATIC Epslon${type}_gcc.o)
set_target_properties(Epslon${type}_gcc PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

build_linpackcepslon_gcc(float gcc)
build_linpackcepslon_gcc(double gcc)

#Epslon clang 
function(build_linpackcepslon_clang type compiler)
add_custom_command(OUTPUT Epslon${type}_clang.o
COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/linpackc/Epslon/Epslon.cpp |
        sed -e 's/TYPE_PLACEHOLDER/${type}/g' -e 's/COMPILER_PLACEHOLDER/${compiler}/g' |
        ${LLVM_MLIR_BINARY_DIR}/clang++ -x c++ -c  -fvectorize -march=native -o Epslon${type}_clang.o -    
)
add_library(Epslon${type}_clang STATIC Epslon${type}_clang.o)
set_target_properties(Epslon${type}_clang PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

build_linpackcepslon_clang(float clang)
build_linpackcepslon_clang(double clang)
