#-------------------------------------------------------------------------------
# MLIR SCF Dialect linpackc Operation + Upstream Lowering Passes
#-------------------------------------------------------------------------------
# ddot
function(build_mlir_linpackcddot type)
  add_custom_command(OUTPUT mlir-linpackcddot${type}.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/linpackc/Ddot/MLIRLinpackCDdot.mlir |
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
            -o ${BUDDY_BINARY_DIR}/../benchmarks/Vectorization/linpackc/Ddot/mlir-linpackcddot${type}.o
  )
  add_library(MLIRLinpackDdot${type} STATIC mlir-linpackcddot${type}.o)
  set_target_properties(MLIRLinpackDdot${type} PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

build_mlir_linpackcddot(f32)
build_mlir_linpackcddot(f64)

# Ddot gcc
function(build_linpackcddot_gcc type compiler)
add_custom_command(OUTPUT Ddot${type}_gcc.o
COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/linpackc/Ddot/Ddot.cpp |
        sed -e 's/TYPE_PLACEHOLDER/${type}/g' -e 's/COMPILER_PLACEHOLDER/${compiler}/g' |
        g++ -x c++ -c  -ftree-vectorize -march=native -o Ddot${type}_gcc.o -
        
)
add_library(Ddot${type}_gcc STATIC Ddot${type}_gcc.o)
set_target_properties(Ddot${type}_gcc PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

build_linpackcddot_gcc(float gcc)
build_linpackcddot_gcc(double gcc)

#Ddot clang 
function(build_linpackcddot_clang type compiler)
add_custom_command(OUTPUT Ddot${type}_clang.o
COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/linpackc/Ddot/Ddot.cpp |
        sed -e 's/TYPE_PLACEHOLDER/${type}/g' -e 's/COMPILER_PLACEHOLDER/${compiler}/g' |
        ${LLVM_MLIR_BINARY_DIR}/clang++ -x c++ -c  -fvectorize -march=native -o Ddot${type}_clang.o -    
)
add_library(Ddot${type}_clang STATIC Ddot${type}_clang.o)
set_target_properties(Ddot${type}_clang PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

build_linpackcddot_clang(float clang)
build_linpackcddot_clang(double clang)
