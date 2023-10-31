add_custom_command(OUTPUT MiniLM_1.o
COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
        ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/operator/Linalg_generic/MLIRMiniLM_1.mlir
          -convert-linalg-to-loops
          -convert-scf-to-cf
          -convert-linalg-to-llvm
          -llvm-request-c-wrappers
          -finalize-memref-to-llvm
          -convert-func-to-llvm
          -reconcile-unrealized-casts | 
        ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
        ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
          -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
          -o ${BUDDY_BINARY_DIR}/../benchmarks/Vectorization/operator/Linalg_generic/MiniLM_1.o
)
add_library(MiniLM_1 STATIC MiniLM_1.o)
set_target_properties(MiniLM_1 PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT MiniLM_2.o
COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
      ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/operator/Linalg_generic/MLIRMiniLM_2.mlir
        -convert-linalg-to-loops
        -convert-scf-to-cf
        -convert-linalg-to-llvm
        -llvm-request-c-wrappers
        -finalize-memref-to-llvm
        -convert-func-to-llvm
        -reconcile-unrealized-casts | 
      ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
      ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
        -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
        -o ${BUDDY_BINARY_DIR}/../benchmarks/Vectorization/operator/Linalg_generic/MiniLM_2.o
)
add_library(MiniLM_2 STATIC MiniLM_2.o)
set_target_properties(MiniLM_2 PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT MiniLM_3.o
COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
      ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/operator/Linalg_generic/MLIRMiniLM_3.mlir
        -convert-linalg-to-loops
        -convert-scf-to-cf
        -convert-linalg-to-llvm
        -llvm-request-c-wrappers
        -finalize-memref-to-llvm
        -convert-func-to-llvm
        -reconcile-unrealized-casts | 
      ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
      ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
        -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
        -o ${BUDDY_BINARY_DIR}/../benchmarks/Vectorization/operator/Linalg_generic/MiniLM_3.o
)
add_library(MiniLM_3 STATIC MiniLM_3.o)
set_target_properties(MiniLM_3 PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT MiniLM_4.o
COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
      ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/operator/Linalg_generic/MLIRMiniLM_4.mlir
        -convert-linalg-to-loops
        -convert-scf-to-cf
        -convert-linalg-to-llvm
        -llvm-request-c-wrappers
        -finalize-memref-to-llvm
        -convert-func-to-llvm
        -reconcile-unrealized-casts | 
      ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
      ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
        -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
        -o ${BUDDY_BINARY_DIR}/../benchmarks/Vectorization/operator/Linalg_generic/MiniLM_4.o
)
add_library(MiniLM_4 STATIC MiniLM_4.o)
set_target_properties(MiniLM_4 PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT MiniLM_5.o
COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
      ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/operator/Linalg_generic/MLIRMiniLM_5.mlir
        -convert-linalg-to-loops
        -convert-scf-to-cf
        -convert-linalg-to-llvm
        -llvm-request-c-wrappers
        -finalize-memref-to-llvm
        -convert-func-to-llvm
        -reconcile-unrealized-casts | 
      ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
      ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
        -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
        -o ${BUDDY_BINARY_DIR}/../benchmarks/Vectorization/operator/Linalg_generic/MiniLM_5.o
)
add_library(MiniLM_5 STATIC MiniLM_5.o)
set_target_properties(MiniLM_5 PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT MiniLM_6.o
COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
      ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/operator/Linalg_generic/MLIRMiniLM_6.mlir
        -convert-linalg-to-loops
        -convert-scf-to-cf
        -convert-linalg-to-llvm
        -llvm-request-c-wrappers
        -finalize-memref-to-llvm
        -convert-func-to-llvm
        -reconcile-unrealized-casts | 
      ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
      ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
        -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
        -o ${BUDDY_BINARY_DIR}/../benchmarks/Vectorization/operator/Linalg_generic/MiniLM_6.o
)
add_library(MiniLM_6 STATIC MiniLM_6.o)
set_target_properties(MiniLM_6 PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT MiniLM_7.o
COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
      ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/operator/Linalg_generic/MLIRMiniLM_7.mlir
        -convert-linalg-to-loops
        -convert-scf-to-cf
        -convert-linalg-to-llvm
        -llvm-request-c-wrappers
        -finalize-memref-to-llvm
        -convert-func-to-llvm
        -reconcile-unrealized-casts | 
      ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
      ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
        -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
        -o ${BUDDY_BINARY_DIR}/../benchmarks/Vectorization/operator/Linalg_generic/MiniLM_7.o
)
add_library(MiniLM_7 STATIC MiniLM_7.o)
set_target_properties(MiniLM_7 PROPERTIES LINKER_LANGUAGE CXX)

