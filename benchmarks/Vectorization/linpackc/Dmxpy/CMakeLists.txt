#-------------------------------------------------------------------------------
# MLIR SCF Dialect linpackc Operation + Upstream Lowering Passes
#-------------------------------------------------------------------------------
# dmxpy
function(build_mlir_linpackcdmxpy type)
  add_custom_command(OUTPUT mlir-linpackcdmxpy${type}.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/linpackc/Dmxpy/MLIRLinpackCDmxpy.mlir |
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
            -o ${BUDDY_BINARY_DIR}/../benchmarks/Vectorization/linpackc/Dmxpy/mlir-linpackcdmxpy${type}.o
  )
  add_library(MLIRLinpackDmxpy${type} STATIC mlir-linpackcdmxpy${type}.o)
  set_target_properties(MLIRLinpackDmxpy${type} PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

build_mlir_linpackcdmxpy(f32)
build_mlir_linpackcdmxpy(f64)

# Dmxpy gcc
function(build_linpackcdmxpy_gcc type compiler)
add_custom_command(OUTPUT Dmxpy${type}_gcc.o
COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/linpackc/Dmxpy/Dmxpy.cpp |
        sed -e 's/TYPE_PLACEHOLDER/${type}/g' -e 's/COMPILER_PLACEHOLDER/${compiler}/g' |
        g++ -x c++ -c  -ftree-vectorize -march=native -o Dmxpy${type}_gcc.o -
        
)
add_library(Dmxpy${type}_gcc STATIC Dmxpy${type}_gcc.o)
set_target_properties(Dmxpy${type}_gcc PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

build_linpackcdmxpy_gcc(float gcc)
build_linpackcdmxpy_gcc(double gcc)

#Dmxpy clang 
function(build_linpackcdmxpy_clang type compiler)
add_custom_command(OUTPUT Dmxpy${type}_clang.o
COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/linpackc/Dmxpy/Dmxpy.cpp |
        sed -e 's/TYPE_PLACEHOLDER/${type}/g' -e 's/COMPILER_PLACEHOLDER/${compiler}/g' |
        ${LLVM_MLIR_BINARY_DIR}/clang++ -x c++ -c  -fvectorize -march=native -o Dmxpy${type}_clang.o -    
)
add_library(Dmxpy${type}_clang STATIC Dmxpy${type}_clang.o)
set_target_properties(Dmxpy${type}_clang PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

build_linpackcdmxpy_clang(float clang)
build_linpackcdmxpy_clang(double clang)
