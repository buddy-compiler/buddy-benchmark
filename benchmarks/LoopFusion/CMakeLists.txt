add_custom_command(
    OUTPUT mlir-loop.o
    COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
    ${BUDDY_SOURCE_DIR}/benchmarks/LoopFusion/log.mlir
        -affine-loop-fusion
		-lower-affine 
		-func-bufferize 
		-arith-bufferize 
		-tensor-bufferize 
		-buffer-deallocation 
		-finalizing-bufferize 
		-convert-vector-to-scf 
		-expand-strided-metadata 
		-convert-vector-to-llvm 
		-memref-expand 
		-arith-expand 
		-convert-arith-to-llvm 
		-finalize-memref-to-llvm 
		-convert-scf-to-cf 
		-llvm-request-c-wrappers 
		-convert-openmp-to-llvm
		-convert-arith-to-llvm 
		-convert-math-to-llvm 
		-convert-math-to-libm  
		-convert-func-to-llvm 
		-reconcile-unrealized-casts | 
        ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
        ${LLVM_MLIR_BINARY_DIR}/llc  -mtriple=${TARGET_TRIPLE}  -mattr=${BUDDY_OPT_ATTR}
		--filetype=obj 
          -o ${BUDDY_BINARY_DIR}/../benchmarks/LoopFusion/mlir-loop.o
)
add_library(MLIRLoop STATIC mlir-loop.o)
set_target_properties(MLIRLoop PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(
    OUTPUT mlir-fusionloop.o
    COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
    ${BUDDY_SOURCE_DIR}/benchmarks/LoopFusion/finish-fusion.mlir
        -affine-loop-fusion
		-lower-affine 
		-func-bufferize 
		-arith-bufferize 
		-tensor-bufferize 
		-buffer-deallocation 
		-finalizing-bufferize 
		-convert-vector-to-scf 
		-expand-strided-metadata 
		-convert-vector-to-llvm 
		-memref-expand 
		-arith-expand 
		-convert-arith-to-llvm 
		-finalize-memref-to-llvm 
		-convert-scf-to-cf 
		-llvm-request-c-wrappers 
		-convert-openmp-to-llvm
		-convert-arith-to-llvm 
		-convert-math-to-llvm 
		-convert-math-to-libm  
		-convert-func-to-llvm 
		-reconcile-unrealized-casts |
        ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
        ${LLVM_MLIR_BINARY_DIR}/llc  -mtriple=${TARGET_TRIPLE} -mattr=${BUDDY_OPT_ATTR}
		--filetype=obj 
            -o ${BUDDY_BINARY_DIR}/../benchmarks/LoopFusion/mlir-fusionloop.o
)
add_library(MLIRLoopFusion STATIC mlir-fusionloop.o)
set_target_properties(MLIRLoopFusion PROPERTIES LINKER_LANGUAGE CXX)

add_executable(loopfusion-test-benchmark
    main.cpp
    MLIRlogBenchmark.cpp
    MLIRfinish-fusionBenchmark.cpp
)

target_link_libraries(loopfusion-test-benchmark
    GoogleBenchmark
    MLIRLoop
    MLIRLoopFusion
)