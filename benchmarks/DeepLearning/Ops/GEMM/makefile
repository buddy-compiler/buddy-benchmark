#!/bin/bash
gemm-affine:
	@buddy-opt ./gemm.mlir -convert-linalg-to-affine-loops -o opt_gemm.mlir

gemm-affine-opt:
	@buddy-opt ./gemm.mlir -convert-linalg-to-affine-loops --affine-loop-tile=tile-sizes=64,8,256 --loop-order-change=new-order-list=2,0,1,5,4,3 -o opt_gemm.mlir
	
clean:
	rm opt_gemm.mlir
