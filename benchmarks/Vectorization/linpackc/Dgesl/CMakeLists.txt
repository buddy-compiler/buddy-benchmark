#-------------------------------------------------------------------------------
# MLIR SCF Dialect linpackc Operation + Upstream Lowering Passes
#-------------------------------------------------------------------------------
# dgesl
link_directories(CMAKE_ARCHIVE_OUTPUT_DIRECTORY)

add_custom_command(OUTPUT mlir-linpackcdgeslf32.o
  COMMAND 
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
            ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/linpackc/Dgesl/mlir-dgeslf32.ll
            -o ${BUDDY_BINARY_DIR}/../benchmarks/Vectorization/linpackc/Dgesl/mlir-linpackcdgeslf32.o
)
add_library(MLIRLinpackDgeslf32 STATIC mlir-linpackcdgeslf32.o)
target_link_libraries(  MLIRLinpackDgeslf32 
                        MLIRLinpackDaxpyf32
                        MLIRLinpackDdotf32
)
set_target_properties(MLIRLinpackDgeslf32 PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT mlir-linpackcdgeslf64.o
  COMMAND 
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
            ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/linpackc/Dgesl/mlir-dgeslf64.ll
            -o ${BUDDY_BINARY_DIR}/../benchmarks/Vectorization/linpackc/Dgesl/mlir-linpackcdgeslf64.o
)
add_library(MLIRLinpackDgeslf64 STATIC mlir-linpackcdgeslf64.o)
target_link_libraries(  MLIRLinpackDgeslf64 
                        MLIRLinpackDaxpyf64
                        MLIRLinpackDdotf64
)
set_target_properties(MLIRLinpackDgeslf64 PROPERTIES LINKER_LANGUAGE CXX)

# Dgesl gcc
function(build_linpackcdgesl_gcc type compiler)
add_custom_command(OUTPUT Dgesl${type}_gcc.o
COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/linpackc/Dgesl/Dgesl.cpp |
        sed -e 's/TYPE_PLACEHOLDER/${type}/g' -e 's/COMPILER_PLACEHOLDER/${compiler}/g' |
        g++ -x c++ -c  -ftree-vectorize -march=native -static
        -I${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/linpackc
        -o Dgesl${type}_gcc.o -
        
)
add_library(Dgesl${type}_gcc STATIC Dgesl${type}_gcc.o)
set_target_properties(Dgesl${type}_gcc PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

build_linpackcdgesl_gcc(float gcc)
build_linpackcdgesl_gcc(double gcc)

#Dgesl clang 
function(build_linpackcdgesl_clang type compiler)
add_custom_command(OUTPUT Dgesl${type}_clang.o
COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/linpackc/Dgesl/Dgesl.cpp |
        sed -e 's/TYPE_PLACEHOLDER/${type}/g' -e 's/COMPILER_PLACEHOLDER/${compiler}/g' |
        ${LLVM_MLIR_BINARY_DIR}/clang++ -x c++ -c  -fvectorize -march=native 
        -I${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/linpackc
        -o Dgesl${type}_clang.o -    
)
add_library(Dgesl${type}_clang STATIC Dgesl${type}_clang.o)
set_target_properties(Dgesl${type}_clang PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

build_linpackcdgesl_clang(float clang)
build_linpackcdgesl_clang(double clang)




