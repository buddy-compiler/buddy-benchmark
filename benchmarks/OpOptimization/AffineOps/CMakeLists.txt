function(add_e2e_custom_command output_file mlir_file sed_command mlir_opt_flags mlir_translate_flags)
  if(CMAKE_TOOLCHAIN_FILE STREQUAL ${BUDDY_SOURCE_DIR}/cmake/riscv-toolchain.cmake)
    # Generate RISC-V object file.
    set(target_triple "riscv64")
    set(target_attr "+m,+d,+v")
    set(target_attr_additional "--target-abi=lp64d" "--relocation-model=pic" "-riscv-v-vector-bits-min=128")
  else()
    # Generate x86 object file.
    set(target_triple "${BUDDY_OPT_TRIPLE}")
    set(target_attr "${BUDDY_OPT_ATTR}")
  endif()

  set(mlir_opt_commands "${LLVM_MLIR_BINARY_DIR}/mlir-opt")
  foreach(flag IN LISTS mlir_opt_flags)
    list(APPEND mlir_opt_commands "${flag}")
  endforeach()

  add_custom_command(OUTPUT ${output_file}
    COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/OpOptimization/AffineOps/${mlir_file} |
    sed '${sed_command}' | ${mlir_opt_commands} |
            ${LLVM_MLIR_BINARY_DIR}/mlir-translate ${mlir_translate_flags} |
            ${LLVM_MLIR_BINARY_DIR}/llc -O0 -mtriple=${target_triple} -mattr=${target_attr}
              ${target_attr_additional} --filetype=obj
              -o ${BUDDY_BINARY_DIR}/../benchmarks/OpOptimization/AffineOps/${output_file}
  )
endfunction()

function(build_function_vector func_name step)
  string(TOLOWER ${func_name} func_name_lower_case)
  add_e2e_custom_command(
    "${func_name}${step}.o" 
    "${func_name}.mlir" 
    "s/${func_name_lower_case}/${func_name_lower_case}_vector_${step}/g"
    "-affine-super-vectorize=\"virtual-vector-size=${step};vectorize-reductions=true\";-convert-vector-to-scf;-lower-affine;-convert-scf-to-cf;-convert-vector-to-llvm;-finalize-memref-to-llvm;-llvm-request-c-wrappers;-convert-func-to-llvm;-reconcile-unrealized-casts"
    "--mlir-to-llvmir"
  )
  add_library(${func_name}Vector${step} STATIC "${func_name}${step}.o")
  target_link_libraries(${func_name}Vector${step} -static)
  set_target_properties(${func_name}Vector${step} PROPERTIES LINKER_LANGUAGE CXX) 
endfunction()

function(build_function_all func_name)
  add_e2e_custom_command(
    "${func_name}.o" 
    "${func_name}.mlir" 
    ""
    "-lower-affine;-convert-scf-to-cf;-convert-vector-to-llvm;-finalize-memref-to-llvm;-llvm-request-c-wrappers;-convert-func-to-llvm;-reconcile-unrealized-casts"
    "--mlir-to-llvmir"
  )
  add_library(${func_name} STATIC "${func_name}.o")
  target_link_libraries(${func_name} -static)
  set_target_properties(${func_name} PROPERTIES LINKER_LANGUAGE CXX) 
  
  build_function_vector(${func_name} 8)
  build_function_vector(${func_name} 16)
  build_function_vector(${func_name} 32)
  build_function_vector(${func_name} 64)
  build_function_vector(${func_name} 128)
endfunction()

build_function_all("Add2D") 
build_function_all("Reduction")
build_function_all("Matmul2D")

add_executable(affine-ops-benchmark Main.cpp AffineOpsBenchmark.cpp)
target_link_libraries(affine-ops-benchmark -static
                      GoogleBenchmark
                      Add2D
                      Add2DVector8
                      Add2DVector16
                      Add2DVector32
                      Add2DVector64
                      Add2DVector128
                      Reduction
                      ReductionVector8
                      ReductionVector16
                      ReductionVector32
                      ReductionVector64
                      ReductionVector128
                      Matmul2D
                      Matmul2DVector8
                      Matmul2DVector16
                      Matmul2DVector32
                      Matmul2DVector64
                      Matmul2DVector128
                      )

