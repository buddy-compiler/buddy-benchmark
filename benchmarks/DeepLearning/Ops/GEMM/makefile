#!/bin/bash
gemm-affine:
	@buddy-opt ./gemm.mlir -convert-linalg-to-affine-loops -o opt_gemm.mlir

gemm-affine-opt:
	@buddy-opt ./gemm.mlir -convert-linalg-to-affine-loops --affine-loop-tile=tile-sizes=64,128,8 --loop-order-change=new-order-list=1,2,0,5,3,4 -o opt_gemm.mlir
	
clean:
	rm opt_gemm.mlir
