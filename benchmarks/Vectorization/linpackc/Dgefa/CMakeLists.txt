#-------------------------------------------------------------------------------
# MLIR SCF Dialect linpackc Operation + Upstream Lowering Passes
#-------------------------------------------------------------------------------
# dgefa
add_custom_command(OUTPUT mlir-linpackcdgefaf32.o
  COMMAND 
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
            ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/linpackc/Dgefa/mlir-dgefaf32.ll
            -o ${BUDDY_BINARY_DIR}/../benchmarks/Vectorization/linpackc/Dgefa/mlir-linpackcdgefaf32.o
)
add_library(MLIRLinpackDgefaf32 STATIC mlir-linpackcdgefaf32.o)
target_link_libraries(  MLIRLinpackDgefaf32 
                        MLIRLinpackDaxpyf32
                        MLIRLinpackIdamaxf32
                        MLIRLinpackDscalf32
)
set_target_properties(MLIRLinpackDgefaf32 PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT mlir-linpackcdgefaf64.o
  COMMAND 
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
            ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/linpackc/Dgefa/mlir-dgefaf64.ll
            -o ${BUDDY_BINARY_DIR}/../benchmarks/Vectorization/linpackc/Dgefa/mlir-linpackcdgefaf64.o
)
add_library(MLIRLinpackDgefaf64 STATIC mlir-linpackcdgefaf64.o)
target_link_libraries(  MLIRLinpackDgefaf64 
                        MLIRLinpackDaxpyf64
                        MLIRLinpackIdamaxf64
                        MLIRLinpackDscalf64
)
set_target_properties(MLIRLinpackDgefaf64 PROPERTIES LINKER_LANGUAGE CXX)


# Dgefa gcc
function(build_linpackcdgefa_gcc type compiler)
add_custom_command(OUTPUT Dgefa${type}_gcc.o
COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/linpackc/Dgefa/Dgefa.cpp |
        sed -e 's/TYPE_PLACEHOLDER/${type}/g' -e 's/COMPILER_PLACEHOLDER/${compiler}/g' |
        g++ -x c++ -c -ftree-vectorize -march=native  
        -I${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/linpackc
        -o Dgefa${type}_gcc.o -
        
)
add_library(Dgefa${type}_gcc STATIC Dgefa${type}_gcc.o)
set_target_properties(Dgefa${type}_gcc PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

build_linpackcdgefa_gcc(float gcc)
build_linpackcdgefa_gcc(double gcc)

#Dgefa clang 
function(build_linpackcdgefa_clang type compiler)
add_custom_command(OUTPUT Dgefa${type}_clang.o
COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/linpackc/Dgefa/Dgefa.cpp |
        sed -e 's/TYPE_PLACEHOLDER/${type}/g' -e 's/COMPILER_PLACEHOLDER/${compiler}/g' |
        ${LLVM_MLIR_BINARY_DIR}/clang++ -x c++ -c  -fvectorize -march=native 
        -I${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/linpackc
        -o Dgefa${type}_clang.o -    
)
add_library(Dgefa${type}_clang STATIC Dgefa${type}_clang.o)
set_target_properties(Dgefa${type}_clang PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

build_linpackcdgefa_clang(float clang)
build_linpackcdgefa_clang(double clang)




