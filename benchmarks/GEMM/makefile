#!/bin/bash
base-line:
	@buddy-opt ./gemm.mlir -convert-linalg-to-affine-loops -o opt_gemm.mlir

v-base-line:
	@buddy-opt ./gemm.mlir -convert-linalg-to-affine-loops --affine-super-vectorize=virtual-vector-size=4 -o opt_gemm.mlir

tile:
	@buddy-opt ./gemm.mlir -convert-linalg-to-affine-loops --affine-loop-tile=tile-sizes=64,8,256 --loop-order-change=new-order-list=2,0,1,5,4,3 -o opt_gemm.mlir

tile-reorder:
	@buddy-opt ./gemm.mlir -convert-linalg-to-affine-loops --affine-super-tilling=stratege=110,3,-1,16,-1,480,-1 --loop-order-change=new-order-list=2,0,1,3,6,5,4 -canonicalize -o opt_gemm.mlir
	
tile-reorder-pack:
	@buddy-opt ./gemm.mlir -convert-linalg-to-affine-loops --affine-super-tilling=stratege=110,3,-1,16,-1,480,-1 --loop-order-change=new-order-list=2,0,1,3,6,5,4 -canonicalize -gemm-data-copy-generate  -o opt_gemm.mlir

tile-reorder-unroll:
	@buddy-opt ./gemm.mlir -convert-linalg-to-affine-loops --affine-super-tilling=stratege=110,3,-1,16,-1,480,-1 --loop-order-change=new-order-list=2,0,1,3,6,5,4 -canonicalize --affine-loop-unroll=unroll-factor=4  -o opt_gemm.mlir

tile-reorder-pack-unroll:
	@buddy-opt ./gemm.mlir -convert-linalg-to-affine-loops --affine-super-tilling=stratege=110,3,-1,16,-1,480,-1 --loop-order-change=new-order-list=2,0,1,3,6,5,4 -canonicalize -gemm-data-copy-generate --affine-loop-unroll=unroll-factor=3  -o opt_gemm.mlir

v-tile-reorder:
	@buddy-opt ./gemm.mlir -convert-linalg-to-affine-loops --convert-linalg-to-affine-loops --affine-super-vectorize=virtual-vector-size=4 --canonicalize --affine-loop-normalize --affine-simplify-structures --gemm-loop-perfect --canonicalize  --affine-super-tilling=stratege=110,3,-1,16,-1,480,-1 --loop-order-change=new-order-list=2,0,1,3,6,5,4 -canonicalize -o opt_gemm.mlir
	
v-tile-reorder-unroll:
	@buddy-opt ./gemm.mlir -convert-linalg-to-affine-loops --convert-linalg-to-affine-loops --affine-super-vectorize=virtual-vector-size=4 --canonicalize --affine-loop-normalize --affine-simplify-structures --gemm-loop-perfect --canonicalize  --affine-super-tilling=stratege=110,3,-1,16,-1,480,-1 --loop-order-change=new-order-list=2,0,1,3,6,5,4 -canonicalize --affine-loop-unroll=unroll-factor=3 -o opt_gemm.mlir

clean:
	rm opt_gemm.mlir
