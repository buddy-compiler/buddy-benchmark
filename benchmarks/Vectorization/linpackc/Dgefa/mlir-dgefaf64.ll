; //===- mlir-dgefaf32.ll -------------------------------------------------===//
; //
; // Licensed under the Apache License, Version 2.0 (the "License");
; // you may not use this file except in compliance with the License.
; // You may obtain a copy of the License at
; //
; //     http://www.apache.org/licenses/LICENSE-2.0
; //
; // Unless required by applicable law or agreed to in writing, software
; // distributed under the License is distributed on an "AS IS" BASIS,
; // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
; // See the License for the specific language governing permissions and
; // limitations under the License.
; //
; //===--------------------------------------------------------------------===//
; //
; // This file provides the LLVM IR of linpackc dgefa function.
; //
; //===--------------------------------------------------------------------===//
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

declare i32 @mlir_linpackcidamaxf64(i32, ptr, ptr, i64, i64, i64, i32)

declare void @mlir_linpackcdscalrollf64(i32, double, ptr, ptr, i64, i64, i64, i32)

declare void @mlir_linpackcdaxpyrollf64(i32, double, ptr, ptr, i64, i64, i64, i32, ptr, ptr, i64, i64, i64, i32)

declare void @mlir_linpackcdscalunrollf64(i32, double, ptr, ptr, i64, i64, i64, i32)

declare void @mlir_linpackcdaxpyunrollf64(i32, double, ptr, ptr, i64, i64, i64, i32, ptr, ptr, i64, i64, i64, i32)

declare i32 @_mlir_ciface_mlir_linpackcidamaxf64(i32, ptr, i32)

declare void @_mlir_ciface_mlir_linpackcdscalrollf64(i32, double, ptr, i32)

declare void @_mlir_ciface_mlir_linpackcdaxpyrollf64(i32, double, ptr, i32, ptr, i32)

declare void @_mlir_ciface_mlir_linpackcdscalunrollf64(i32, double, ptr, i32)

declare void @_mlir_ciface_mlir_linpackcdaxpyunrollf64(i32, double, ptr, i32, ptr, i32)

define double @get_val_dgefa_f64(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7) {
  %9 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %0, 0
  %10 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %9, ptr %1, 1
  %11 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, i64 %2, 2
  %12 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, i64 %3, 3, 0
  %13 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, i64 %4, 4, 0
  %14 = mul i64 %5, %6
  %15 = add i64 %14, %7
  %16 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, 1
  %17 = getelementptr double, ptr %16, i64 %15
  %18 = load double, ptr %17, align 8
  ret double %18
}

define double @_mlir_ciface_get_val_dgefa_f64(ptr %0, i64 %1, i64 %2, i64 %3) {
  %5 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %0, align 8
  %6 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, 0
  %7 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, 1
  %8 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, 2
  %9 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, 3, 0
  %10 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, 4, 0
  %11 = call double @get_val_dgefa_f64(ptr %6, ptr %7, i64 %8, i64 %9, i64 %10, i64 %1, i64 %2, i64 %3)
  ret double %11
}

define void @set_val_dgefa_f64(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, double %8) {
  %10 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %0, 0
  %11 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, ptr %1, 1
  %12 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, i64 %2, 2
  %13 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, i64 %3, 3, 0
  %14 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, i64 %4, 4, 0
  %15 = mul i64 %5, %6
  %16 = add i64 %15, %7
  %17 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %14, 1
  %18 = getelementptr double, ptr %17, i64 %16
  store double %8, ptr %18, align 8
  ret void
}

define void @_mlir_ciface_set_val_dgefa_f64(ptr %0, i64 %1, i64 %2, i64 %3, double %4) {
  %6 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %0, align 8
  %7 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 0
  %8 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 1
  %9 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 2
  %10 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 3, 0
  %11 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 4, 0
  call void @set_val_dgefa_f64(ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, i64 %1, i64 %2, i64 %3, double %4)
  ret void
}

define i64 @get_offset_dgefa_f64(i64 %0, i64 %1, i64 %2) {
  %4 = mul i64 %0, %1
  %5 = add i64 %4, %2
  ret i64 %5
}

define i64 @_mlir_ciface_get_offset_dgefa_f64(i64 %0, i64 %1, i64 %2) {
  %4 = call i64 @get_offset_dgefa_f64(i64 %0, i64 %1, i64 %2)
  ret i64 %4
}

define void @mlir_linpackcdgefarollf64(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i32 %5, i32 %6, ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, ptr %12, ptr %13, i64 %14) {
  %16 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %0, 0
  %17 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %16, ptr %1, 1
  %18 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %17, i64 %2, 2
  %19 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %18, i64 %3, 3, 0
  %20 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, i64 %4, 4, 0
  %21 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %7, 0
  %22 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %21, ptr %8, 1
  %23 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %22, i64 %9, 2
  %24 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, i64 %10, 3, 0
  %25 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %24, i64 %11, 4, 0
  %26 = insertvalue { ptr, ptr, i64 } undef, ptr %12, 0
  %27 = insertvalue { ptr, ptr, i64 } %26, ptr %13, 1
  %28 = insertvalue { ptr, ptr, i64 } %27, i64 %14, 2
  %29 = extractvalue { ptr, ptr, i64 } %28, 1
  store i32 0, ptr %29, align 4
  %30 = sub i32 %6, 1
  %31 = sext i32 %6 to i64
  %32 = sext i32 %30 to i64
  %33 = sext i32 %5 to i64
  %34 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 3
  %35 = alloca [1 x i64], i64 1, align 8
  store [1 x i64] %34, ptr %35, align 4
  %36 = getelementptr [1 x i64], ptr %35, i32 0, i32 0
  %37 = load i64, ptr %36, align 4
  %38 = icmp sge i32 %30, 0
  br i1 %38, label %39, label %139

39:                                               ; preds = %15
  br label %40

40:                                               ; preds = %136, %39
  %41 = phi i64 [ %137, %136 ], [ 0, %39 ]
  %42 = icmp slt i64 %41, %32
  br i1 %42, label %43, label %138

43:                                               ; preds = %40
  %44 = add i64 %41, 1
  %45 = trunc i64 %41 to i32
  %46 = add i32 %45, 1
  %47 = sub i32 %6, %45
  %48 = call i64 @get_offset_dgefa_f64(i64 %33, i64 %41, i64 %41)
  %49 = sub i64 %37, %48
  %50 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 0
  %51 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 1
  %52 = insertvalue { ptr, ptr, i64 } undef, ptr %50, 0
  %53 = insertvalue { ptr, ptr, i64 } %52, ptr %51, 1
  %54 = insertvalue { ptr, ptr, i64 } %53, i64 0, 2
  %55 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 2
  %56 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 3, 0
  %57 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 4, 0
  %58 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %50, 0
  %59 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %58, ptr %51, 1
  %60 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %59, i64 %48, 2
  %61 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %60, i64 %49, 3, 0
  %62 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %61, i64 1, 4, 0
  %63 = call i32 @mlir_linpackcidamaxf64(i32 %47, ptr %50, ptr %51, i64 %48, i64 %49, i64 1, i32 1)
  %64 = add i32 %63, %45
  %65 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %25, 1
  %66 = getelementptr i32, ptr %65, i64 %41
  store i32 %64, ptr %66, align 4
  %67 = sext i32 %64 to i64
  %68 = call double @get_val_dgefa_f64(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %33, i64 %41, i64 %41)
  %69 = call double @get_val_dgefa_f64(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %33, i64 %41, i64 %67)
  %70 = fcmp une double %69, 0.000000e+00
  br i1 %70, label %71, label %134

71:                                               ; preds = %43
  %72 = icmp ne i32 %64, %45
  br i1 %72, label %73, label %74

73:                                               ; preds = %71
  call void @set_val_dgefa_f64(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %33, i64 %41, i64 %67, double %68)
  call void @set_val_dgefa_f64(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %33, i64 %41, i64 %41, double %69)
  br label %74

74:                                               ; preds = %73, %71
  %75 = call double @get_val_dgefa_f64(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %33, i64 %41, i64 %41)
  %76 = fdiv double -1.000000e+00, %75
  %77 = sub i32 %6, %46
  %78 = call double @get_val_dgefa_f64(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %33, i64 %41, i64 %44)
  %79 = call i64 @get_offset_dgefa_f64(i64 %33, i64 %41, i64 %44)
  %80 = sub i64 %37, %79
  %81 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 0
  %82 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 1
  %83 = insertvalue { ptr, ptr, i64 } undef, ptr %81, 0
  %84 = insertvalue { ptr, ptr, i64 } %83, ptr %82, 1
  %85 = insertvalue { ptr, ptr, i64 } %84, i64 0, 2
  %86 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 2
  %87 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 3, 0
  %88 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 4, 0
  %89 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %81, 0
  %90 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %89, ptr %82, 1
  %91 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %90, i64 %79, 2
  %92 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %91, i64 %80, 3, 0
  %93 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %92, i64 1, 4, 0
  call void @mlir_linpackcdscalrollf64(i32 %77, double %76, ptr %81, ptr %82, i64 %79, i64 %80, i64 1, i32 1)
  br label %94

94:                                               ; preds = %101, %74
  %95 = phi i64 [ %132, %101 ], [ %44, %74 ]
  %96 = icmp slt i64 %95, %31
  br i1 %96, label %97, label %133

97:                                               ; preds = %94
  %98 = call double @get_val_dgefa_f64(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %33, i64 %95, i64 %67)
  br i1 %72, label %99, label %101

99:                                               ; preds = %97
  %100 = call double @get_val_dgefa_f64(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %33, i64 %95, i64 %41)
  call void @set_val_dgefa_f64(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %33, i64 %95, i64 %67, double %100)
  call void @set_val_dgefa_f64(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %33, i64 %95, i64 %41, double %98)
  br label %101

101:                                              ; preds = %99, %97
  %102 = call i64 @get_offset_dgefa_f64(i64 %33, i64 %41, i64 %44)
  %103 = sub i64 %37, %102
  %104 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 0
  %105 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 1
  %106 = insertvalue { ptr, ptr, i64 } undef, ptr %104, 0
  %107 = insertvalue { ptr, ptr, i64 } %106, ptr %105, 1
  %108 = insertvalue { ptr, ptr, i64 } %107, i64 0, 2
  %109 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 2
  %110 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 3, 0
  %111 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 4, 0
  %112 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %104, 0
  %113 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %112, ptr %105, 1
  %114 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %113, i64 %102, 2
  %115 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %114, i64 %103, 3, 0
  %116 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %115, i64 1, 4, 0
  %117 = call i64 @get_offset_dgefa_f64(i64 %33, i64 %95, i64 %44)
  %118 = sub i64 %37, %117
  %119 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 0
  %120 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 1
  %121 = insertvalue { ptr, ptr, i64 } undef, ptr %119, 0
  %122 = insertvalue { ptr, ptr, i64 } %121, ptr %120, 1
  %123 = insertvalue { ptr, ptr, i64 } %122, i64 0, 2
  %124 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 2
  %125 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 3, 0
  %126 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 4, 0
  %127 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %119, 0
  %128 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %127, ptr %120, 1
  %129 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %128, i64 %117, 2
  %130 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %129, i64 %118, 3, 0
  %131 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %130, i64 1, 4, 0
  call void @mlir_linpackcdaxpyrollf64(i32 %77, double %98, ptr %104, ptr %105, i64 %102, i64 %103, i64 1, i32 1, ptr %119, ptr %120, i64 %117, i64 %118, i64 1, i32 1)
  %132 = add i64 %95, 1
  br label %94

133:                                              ; preds = %94
  br label %136

134:                                              ; preds = %43
  %135 = extractvalue { ptr, ptr, i64 } %28, 1
  store i32 %45, ptr %135, align 4
  br label %136

136:                                              ; preds = %133, %134
  %137 = add i64 %41, 1
  br label %40

138:                                              ; preds = %40
  br label %139

139:                                              ; preds = %138, %15
  %140 = sub i64 %31, 1
  %141 = sub i32 %6, 1
  %142 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %25, 1
  %143 = getelementptr i32, ptr %142, i64 %140
  store i32 %141, ptr %143, align 4
  %144 = call double @get_val_dgefa_f64(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %33, i64 %140, i64 %140)
  %145 = fcmp ueq double %144, 0.000000e+00
  br i1 %145, label %146, label %148

146:                                              ; preds = %139
  %147 = extractvalue { ptr, ptr, i64 } %28, 1
  store i32 %141, ptr %147, align 4
  br label %148

148:                                              ; preds = %146, %139
  ret void
}

define void @_mlir_ciface_mlir_linpackcdgefarollf64(ptr %0, i32 %1, i32 %2, ptr %3, ptr %4) {
  %6 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %0, align 8
  %7 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 0
  %8 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 1
  %9 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 2
  %10 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 3, 0
  %11 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 4, 0
  %12 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %3, align 8
  %13 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, 0
  %14 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, 1
  %15 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, 2
  %16 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, 3, 0
  %17 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, 4, 0
  %18 = load { ptr, ptr, i64 }, ptr %4, align 8
  %19 = extractvalue { ptr, ptr, i64 } %18, 0
  %20 = extractvalue { ptr, ptr, i64 } %18, 1
  %21 = extractvalue { ptr, ptr, i64 } %18, 2
  call void @mlir_linpackcdgefarollf64(ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, i32 %1, i32 %2, ptr %13, ptr %14, i64 %15, i64 %16, i64 %17, ptr %19, ptr %20, i64 %21)
  ret void
}

define void @mlir_linpackcdgefaunrollf64(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i32 %5, i32 %6, ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, ptr %12, ptr %13, i64 %14) {
  %16 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %0, 0
  %17 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %16, ptr %1, 1
  %18 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %17, i64 %2, 2
  %19 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %18, i64 %3, 3, 0
  %20 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, i64 %4, 4, 0
  %21 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %7, 0
  %22 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %21, ptr %8, 1
  %23 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %22, i64 %9, 2
  %24 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, i64 %10, 3, 0
  %25 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %24, i64 %11, 4, 0
  %26 = insertvalue { ptr, ptr, i64 } undef, ptr %12, 0
  %27 = insertvalue { ptr, ptr, i64 } %26, ptr %13, 1
  %28 = insertvalue { ptr, ptr, i64 } %27, i64 %14, 2
  %29 = extractvalue { ptr, ptr, i64 } %28, 1
  store i32 0, ptr %29, align 4
  %30 = sub i32 %6, 1
  %31 = sext i32 %6 to i64
  %32 = sext i32 %30 to i64
  %33 = sext i32 %5 to i64
  %34 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 3
  %35 = alloca [1 x i64], i64 1, align 8
  store [1 x i64] %34, ptr %35, align 4
  %36 = getelementptr [1 x i64], ptr %35, i32 0, i32 0
  %37 = load i64, ptr %36, align 4
  %38 = icmp sge i32 %30, 0
  br i1 %38, label %39, label %139

39:                                               ; preds = %15
  br label %40

40:                                               ; preds = %136, %39
  %41 = phi i64 [ %137, %136 ], [ 0, %39 ]
  %42 = icmp slt i64 %41, %32
  br i1 %42, label %43, label %138

43:                                               ; preds = %40
  %44 = add i64 %41, 1
  %45 = trunc i64 %41 to i32
  %46 = add i32 %45, 1
  %47 = sub i32 %6, %45
  %48 = call i64 @get_offset_dgefa_f64(i64 %33, i64 %41, i64 %41)
  %49 = sub i64 %37, %48
  %50 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 0
  %51 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 1
  %52 = insertvalue { ptr, ptr, i64 } undef, ptr %50, 0
  %53 = insertvalue { ptr, ptr, i64 } %52, ptr %51, 1
  %54 = insertvalue { ptr, ptr, i64 } %53, i64 0, 2
  %55 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 2
  %56 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 3, 0
  %57 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 4, 0
  %58 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %50, 0
  %59 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %58, ptr %51, 1
  %60 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %59, i64 %48, 2
  %61 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %60, i64 %49, 3, 0
  %62 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %61, i64 1, 4, 0
  %63 = call i32 @mlir_linpackcidamaxf64(i32 %47, ptr %50, ptr %51, i64 %48, i64 %49, i64 1, i32 1)
  %64 = add i32 %63, %45
  %65 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %25, 1
  %66 = getelementptr i32, ptr %65, i64 %41
  store i32 %64, ptr %66, align 4
  %67 = sext i32 %64 to i64
  %68 = call double @get_val_dgefa_f64(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %33, i64 %41, i64 %41)
  %69 = call double @get_val_dgefa_f64(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %33, i64 %41, i64 %67)
  %70 = fcmp une double %69, 0.000000e+00
  br i1 %70, label %71, label %134

71:                                               ; preds = %43
  %72 = icmp ne i32 %64, %45
  br i1 %72, label %73, label %74

73:                                               ; preds = %71
  call void @set_val_dgefa_f64(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %33, i64 %41, i64 %67, double %68)
  call void @set_val_dgefa_f64(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %33, i64 %41, i64 %41, double %69)
  br label %74

74:                                               ; preds = %73, %71
  %75 = call double @get_val_dgefa_f64(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %33, i64 %41, i64 %41)
  %76 = fdiv double -1.000000e+00, %75
  %77 = sub i32 %6, %46
  %78 = call double @get_val_dgefa_f64(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %33, i64 %41, i64 %44)
  %79 = call i64 @get_offset_dgefa_f64(i64 %33, i64 %41, i64 %44)
  %80 = sub i64 %37, %79
  %81 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 0
  %82 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 1
  %83 = insertvalue { ptr, ptr, i64 } undef, ptr %81, 0
  %84 = insertvalue { ptr, ptr, i64 } %83, ptr %82, 1
  %85 = insertvalue { ptr, ptr, i64 } %84, i64 0, 2
  %86 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 2
  %87 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 3, 0
  %88 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 4, 0
  %89 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %81, 0
  %90 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %89, ptr %82, 1
  %91 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %90, i64 %79, 2
  %92 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %91, i64 %80, 3, 0
  %93 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %92, i64 1, 4, 0
  call void @mlir_linpackcdscalunrollf64(i32 %77, double %76, ptr %81, ptr %82, i64 %79, i64 %80, i64 1, i32 1)
  br label %94

94:                                               ; preds = %101, %74
  %95 = phi i64 [ %132, %101 ], [ %44, %74 ]
  %96 = icmp slt i64 %95, %31
  br i1 %96, label %97, label %133

97:                                               ; preds = %94
  %98 = call double @get_val_dgefa_f64(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %33, i64 %95, i64 %67)
  br i1 %72, label %99, label %101

99:                                               ; preds = %97
  %100 = call double @get_val_dgefa_f64(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %33, i64 %95, i64 %41)
  call void @set_val_dgefa_f64(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %33, i64 %95, i64 %67, double %100)
  call void @set_val_dgefa_f64(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %33, i64 %95, i64 %41, double %98)
  br label %101

101:                                              ; preds = %99, %97
  %102 = call i64 @get_offset_dgefa_f64(i64 %33, i64 %41, i64 %44)
  %103 = sub i64 %37, %102
  %104 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 0
  %105 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 1
  %106 = insertvalue { ptr, ptr, i64 } undef, ptr %104, 0
  %107 = insertvalue { ptr, ptr, i64 } %106, ptr %105, 1
  %108 = insertvalue { ptr, ptr, i64 } %107, i64 0, 2
  %109 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 2
  %110 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 3, 0
  %111 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 4, 0
  %112 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %104, 0
  %113 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %112, ptr %105, 1
  %114 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %113, i64 %102, 2
  %115 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %114, i64 %103, 3, 0
  %116 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %115, i64 1, 4, 0
  %117 = call i64 @get_offset_dgefa_f64(i64 %33, i64 %95, i64 %44)
  %118 = sub i64 %37, %117
  %119 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 0
  %120 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 1
  %121 = insertvalue { ptr, ptr, i64 } undef, ptr %119, 0
  %122 = insertvalue { ptr, ptr, i64 } %121, ptr %120, 1
  %123 = insertvalue { ptr, ptr, i64 } %122, i64 0, 2
  %124 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 2
  %125 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 3, 0
  %126 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 4, 0
  %127 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %119, 0
  %128 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %127, ptr %120, 1
  %129 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %128, i64 %117, 2
  %130 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %129, i64 %118, 3, 0
  %131 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %130, i64 1, 4, 0
  call void @mlir_linpackcdaxpyunrollf64(i32 %77, double %98, ptr %104, ptr %105, i64 %102, i64 %103, i64 1, i32 1, ptr %119, ptr %120, i64 %117, i64 %118, i64 1, i32 1)
  %132 = add i64 %95, 1
  br label %94

133:                                              ; preds = %94
  br label %136

134:                                              ; preds = %43
  %135 = extractvalue { ptr, ptr, i64 } %28, 1
  store i32 %45, ptr %135, align 4
  br label %136

136:                                              ; preds = %133, %134
  %137 = add i64 %41, 1
  br label %40

138:                                              ; preds = %40
  br label %139

139:                                              ; preds = %138, %15
  %140 = sub i64 %31, 1
  %141 = sub i32 %6, 1
  %142 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %25, 1
  %143 = getelementptr i32, ptr %142, i64 %140
  store i32 %141, ptr %143, align 4
  %144 = call double @get_val_dgefa_f64(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %33, i64 %140, i64 %140)
  %145 = fcmp ueq double %144, 0.000000e+00
  br i1 %145, label %146, label %148

146:                                              ; preds = %139
  %147 = extractvalue { ptr, ptr, i64 } %28, 1
  store i32 %141, ptr %147, align 4
  br label %148

148:                                              ; preds = %146, %139
  ret void
}

define void @_mlir_ciface_mlir_linpackcdgefaunrollf64(ptr %0, i32 %1, i32 %2, ptr %3, ptr %4) {
  %6 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %0, align 8
  %7 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 0
  %8 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 1
  %9 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 2
  %10 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 3, 0
  %11 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 4, 0
  %12 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %3, align 8
  %13 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, 0
  %14 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, 1
  %15 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, 2
  %16 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, 3, 0
  %17 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, 4, 0
  %18 = load { ptr, ptr, i64 }, ptr %4, align 8
  %19 = extractvalue { ptr, ptr, i64 } %18, 0
  %20 = extractvalue { ptr, ptr, i64 } %18, 1
  %21 = extractvalue { ptr, ptr, i64 } %18, 2
  call void @mlir_linpackcdgefaunrollf64(ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, i32 %1, i32 %2, ptr %13, ptr %14, i64 %15, i64 %16, i64 %17, ptr %19, ptr %20, i64 %21)
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
