add_custom_command(
    OUTPUT attention-nooptimize.o
    COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
    ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Layers/Attention/log.mlir
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
          -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Layers/Attention/attention-nooptimize.o
)
add_library(attention-noop STATIC attention-nooptimize.o)
set_target_properties(attention-noop PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(
    OUTPUT attention-loopfuse.o
    COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
    ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Layers/Attention/finish-fusion.mlir
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
            -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Layers/Attention/attention-loopfuse.o
)
add_library(attention-opfuse STATIC attention-loopfuse.o)
set_target_properties(attention-opfuse PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(
    OUTPUT attention-fusevec.o
    COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
    ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Layers/Attention/fusionvec.mlir
        -affine-loop-fusion
		-affine-parallelize 
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
            -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Layers/Attention/attention-fusevec.o
)
add_library(attention-fusevec STATIC attention-fusevec.o)
set_target_properties(attention-fusevec PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(
    OUTPUT attention-nofusepartvec.o
    COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
    ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Layers/Attention/nofusion-partvec.mlir
        -affine-loop-fusion
		-affine-parallelize 
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
            -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Layers/Attention/attention-nofusepartvec.o
)
add_library(attention-nofusepartvec STATIC attention-nofusepartvec.o)
set_target_properties(attention-nofusepartvec PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(
    OUTPUT attention-nofusevec.o
    COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
    ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Layers/Attention/nofusion-vec.mlir
        -affine-loop-fusion
		-affine-parallelize 
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
            -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Layers/Attention/attention-nofusevec.o
)
add_library(attention-nofusevec STATIC attention-nofusevec.o)
set_target_properties(attention-nofusevec PROPERTIES LINKER_LANGUAGE CXX)

add_executable(attention-test-benchmark
    main.cpp
    MLIRlogBenchmark.cpp
    MLIRfinish-fusionBenchmark.cpp
	MLIRfusion-vecBenchmark.cpp
	MLIRnofuse-partvecBenchmark.cpp
	MLIRnofuse-vecBenchmark.cpp
)

target_link_libraries(attention-test-benchmark
    GoogleBenchmark
    attention-noop
    attention-opfuse
	attention-fusevec
	attention-nofusepartvec
	attention-nofusevec
)