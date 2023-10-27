#-------------------------------------------------------------------------------
# MLIR SCF Dialect linpackc Operation + Upstream Lowering Passes
#-------------------------------------------------------------------------------
# matgen
function(build_mlir_linpackcmatgen type)
  add_custom_command(OUTPUT mlir-linpackcmatgen${type}.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/linpackc/Matgen/MLIRLinpackCMatgen.mlir |
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
            -o ${BUDDY_BINARY_DIR}/../benchmarks/Vectorization/linpackc/Matgen/mlir-linpackcmatgen${type}.o
  )
  add_library(MLIRLinpackMatgen${type} STATIC mlir-linpackcmatgen${type}.o)
  set_target_properties(MLIRLinpackMatgen${type} PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

build_mlir_linpackcmatgen(f32)
build_mlir_linpackcmatgen(f64)

# Matgen gcc
function(build_linpackcmatgen_gcc type compiler)
add_custom_command(OUTPUT Matgen${type}_gcc.o
COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/linpackc/Matgen/Matgen.cpp |
        sed -e 's/TYPE_PLACEHOLDER/${type}/g' -e 's/COMPILER_PLACEHOLDER/${compiler}/g' |
        g++ -x c++ -c  -ftree-vectorize -march=native -o Matgen${type}_gcc.o -
        
)
add_library(Matgen${type}_gcc STATIC Matgen${type}_gcc.o)
set_target_properties(Matgen${type}_gcc PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

build_linpackcmatgen_gcc(float gcc)
build_linpackcmatgen_gcc(double gcc)

#Matgen clang 
function(build_linpackcmatgen_clang type compiler)
add_custom_command(OUTPUT Matgen${type}_clang.o
COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/linpackc/Matgen/Matgen.cpp |
        sed -e 's/TYPE_PLACEHOLDER/${type}/g' -e 's/COMPILER_PLACEHOLDER/${compiler}/g' |
        ${LLVM_MLIR_BINARY_DIR}/clang++ -x c++ -c  -fvectorize -march=native -o Matgen${type}_clang.o -    
)
add_library(Matgen${type}_clang STATIC Matgen${type}_clang.o)
set_target_properties(Matgen${type}_clang PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

build_linpackcmatgen_clang(float clang)
build_linpackcmatgen_clang(double clang)
