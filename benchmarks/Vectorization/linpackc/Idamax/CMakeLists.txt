#-------------------------------------------------------------------------------
# MLIR SCF Dialect linpackc Operation + Upstream Lowering Passes
#-------------------------------------------------------------------------------
add_custom_command(OUTPUT mlir-linpackcidamaxf32.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
          ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/linpackc/Idamax/MLIRLinpackCIdamaxF32.mlir 
            -convert-scf-to-cf
            -expand-strided-metadata
            -convert-arith-to-llvm
            -llvm-request-c-wrappers
            -finalize-memref-to-llvm
            -convert-func-to-llvm 
            --convert-math-to-llvm
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
            -o ${BUDDY_BINARY_DIR}/../benchmarks/Vectorization/linpackc/Idamax/mlir-linpackcidamaxf32.o
)
add_library(MLIRLinpackIdamaxf32 STATIC mlir-linpackcidamaxf32.o)
set_target_properties(MLIRLinpackIdamaxf32 PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT mlir-linpackcidamaxf64.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
          ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/linpackc/Idamax/MLIRLinpackCIdamaxF64.mlir 
            -convert-scf-to-cf
            -expand-strided-metadata
            -convert-arith-to-llvm
            -llvm-request-c-wrappers
            -finalize-memref-to-llvm
            -convert-func-to-llvm 
            --convert-math-to-llvm
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
            -o ${BUDDY_BINARY_DIR}/../benchmarks/Vectorization/linpackc/Idamax/mlir-linpackcidamaxf64.o
)
add_library(MLIRLinpackIdamaxf64 STATIC mlir-linpackcidamaxf64.o)
set_target_properties(MLIRLinpackIdamaxf64 PROPERTIES LINKER_LANGUAGE CXX)

# Idamax gcc
function(build_linpackcidamax_gcc type compiler)
add_custom_command(OUTPUT Idamax${type}_gcc.o
COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/linpackc/Idamax/Idamax.cpp |
        sed -e 's/TYPE_PLACEHOLDER/${type}/g' -e 's/COMPILER_PLACEHOLDER/${compiler}/g' |
        g++ -x c++ -c  -ftree-vectorize -march=native -o Idamax${type}_gcc.o -
        
)
add_library(Idamax${type}_gcc STATIC Idamax${type}_gcc.o)
set_target_properties(Idamax${type}_gcc PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

build_linpackcidamax_gcc(float gcc)
build_linpackcidamax_gcc(double gcc)

#Idamax clang 
function(build_linpackcidamax_clang type compiler)
add_custom_command(OUTPUT Idamax${type}_clang.o
COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/linpackc/Idamax/Idamax.cpp |
        sed -e 's/TYPE_PLACEHOLDER/${type}/g' -e 's/COMPILER_PLACEHOLDER/${compiler}/g' |
        ${LLVM_MLIR_BINARY_DIR}/clang++ -x c++ -c  -fvectorize -march=native -o Idamax${type}_clang.o -    
)
add_library(Idamax${type}_clang STATIC Idamax${type}_clang.o)
set_target_properties(Idamax${type}_clang PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

build_linpackcidamax_clang(float clang)
build_linpackcidamax_clang(double clang)
