; //===- mlir-dgeslf32.ll -------------------------------------------------===//
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
; // This file provides the LLVM IR of linpackc dgesl function.
; //
; //===--------------------------------------------------------------------===//
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

declare void @mlir_linpackcdaxpyrollf32(i32, float, ptr, ptr, i64, i64, i64, i32, ptr, ptr, i64, i64, i64, i32)

declare void @mlir_linpackcdaxpyunrollf32(i32, float, ptr, ptr, i64, i64, i64, i32, ptr, ptr, i64, i64, i64, i32)

declare float @mlir_linpackcddotrollf32(i32, ptr, ptr, i64, i64, i64, i32, ptr, ptr, i64, i64, i64, i32)

declare float @mlir_linpackcddotunrollf32(i32, ptr, ptr, i64, i64, i64, i32, ptr, ptr, i64, i64, i64, i32)

declare void @_mlir_ciface_mlir_linpackcdaxpyrollf32(i32, float, ptr, i32, ptr, i32)

declare float @_mlir_ciface_mlir_linpackcddotrollf32(i32, ptr, i32, ptr, i32)

declare void @_mlir_ciface_mlir_linpackcdaxpyunrollf32(i32, float, ptr, i32, ptr, i32)

declare float @_mlir_ciface_mlir_linpackcddotunrollf32(i32, ptr, i32, ptr, i32)

define float @get_val_dgesl_f32(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7) {
  %9 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %0, 0
  %10 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %9, ptr %1, 1
  %11 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, i64 %2, 2
  %12 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, i64 %3, 3, 0
  %13 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, i64 %4, 4, 0
  %14 = mul i64 %5, %6
  %15 = add i64 %14, %7
  %16 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, 1
  %17 = getelementptr float, ptr %16, i64 %15
  %18 = load float, ptr %17, align 4
  ret float %18
}

define float @_mlir_ciface_get_val_dgesl_f32(ptr %0, i64 %1, i64 %2, i64 %3) {
  %5 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %0, align 8
  %6 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, 0
  %7 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, 1
  %8 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, 2
  %9 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, 3, 0
  %10 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, 4, 0
  %11 = call float @get_val_dgesl_f32(ptr %6, ptr %7, i64 %8, i64 %9, i64 %10, i64 %1, i64 %2, i64 %3)
  ret float %11
}

define void @set_val_dgesl_f32(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, float %8) {
  %10 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %0, 0
  %11 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, ptr %1, 1
  %12 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, i64 %2, 2
  %13 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, i64 %3, 3, 0
  %14 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, i64 %4, 4, 0
  %15 = mul i64 %5, %6
  %16 = add i64 %15, %7
  %17 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %14, 1
  %18 = getelementptr float, ptr %17, i64 %16
  store float %8, ptr %18, align 4
  ret void
}

define void @_mlir_ciface_set_val_dgesl_f32(ptr %0, i64 %1, i64 %2, i64 %3, float %4) {
  %6 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %0, align 8
  %7 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 0
  %8 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 1
  %9 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 2
  %10 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 3, 0
  %11 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 4, 0
  call void @set_val_dgesl_f32(ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, i64 %1, i64 %2, i64 %3, float %4)
  ret void
}

define i64 @get_offset_dgesl_f32(i64 %0, i64 %1, i64 %2) {
  %4 = mul i64 %0, %1
  %5 = add i64 %4, %2
  ret i64 %5
}

define i64 @_mlir_ciface_get_offset_dgesl_f32(i64 %0, i64 %1, i64 %2) {
  %4 = call i64 @get_offset_dgesl_f32(i64 %0, i64 %1, i64 %2)
  ret i64 %4
}

define void @mlir_linpackcdgeslrollf32(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i32 %5, i32 %6, ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, ptr %12, ptr %13, i64 %14, i64 %15, i64 %16, i32 %17) {
  %19 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %0, 0
  %20 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, ptr %1, 1
  %21 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, i64 %2, 2
  %22 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %21, i64 %3, 3, 0
  %23 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %22, i64 %4, 4, 0
  %24 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %7, 0
  %25 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %24, ptr %8, 1
  %26 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %25, i64 %9, 2
  %27 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %26, i64 %10, 3, 0
  %28 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %27, i64 %11, 4, 0
  %29 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %12, 0
  %30 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %29, ptr %13, 1
  %31 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %30, i64 %14, 2
  %32 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %31, i64 %15, 3, 0
  %33 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %32, i64 %16, 4, 0
  %34 = sub i32 %6, 1
  %35 = sext i32 %6 to i64
  %36 = sext i32 %34 to i64
  %37 = sext i32 %5 to i64
  %38 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 3
  %39 = alloca [1 x i64], i64 1, align 8
  store [1 x i64] %38, ptr %39, align 4
  %40 = getelementptr [1 x i64], ptr %39, i32 0, i32 0
  %41 = load i64, ptr %40, align 4
  %42 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 3
  %43 = alloca [1 x i64], i64 1, align 8
  store [1 x i64] %42, ptr %43, align 4
  %44 = getelementptr [1 x i64], ptr %43, i32 0, i32 0
  %45 = load i64, ptr %44, align 4
  %46 = icmp eq i32 %17, 0
  br i1 %46, label %47, label %138

47:                                               ; preds = %18
  %48 = icmp sge i32 %34, 1
  br i1 %48, label %49, label %105

49:                                               ; preds = %47
  br label %50

50:                                               ; preds = %72, %49
  %51 = phi i64 [ %103, %72 ], [ 0, %49 ]
  %52 = icmp slt i64 %51, %36
  br i1 %52, label %53, label %104

53:                                               ; preds = %50
  %54 = add i64 %51, 1
  %55 = trunc i64 %54 to i32
  %56 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %28, 1
  %57 = getelementptr i32, ptr %56, i64 %51
  %58 = load i32, ptr %57, align 4
  %59 = sext i32 %58 to i64
  %60 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 1
  %61 = getelementptr float, ptr %60, i64 %59
  %62 = load float, ptr %61, align 4
  %63 = icmp ne i64 %59, %51
  br i1 %63, label %64, label %72

64:                                               ; preds = %53
  %65 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 1
  %66 = getelementptr float, ptr %65, i64 %51
  %67 = load float, ptr %66, align 4
  %68 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 1
  %69 = getelementptr float, ptr %68, i64 %59
  store float %67, ptr %69, align 4
  %70 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 1
  %71 = getelementptr float, ptr %70, i64 %51
  store float %62, ptr %71, align 4
  br label %72

72:                                               ; preds = %64, %53
  %73 = sub i32 %6, %55
  %74 = call i64 @get_offset_dgesl_f32(i64 %37, i64 %51, i64 %54)
  %75 = sub i64 %41, %74
  %76 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 0
  %77 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 1
  %78 = insertvalue { ptr, ptr, i64 } undef, ptr %76, 0
  %79 = insertvalue { ptr, ptr, i64 } %78, ptr %77, 1
  %80 = insertvalue { ptr, ptr, i64 } %79, i64 0, 2
  %81 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 2
  %82 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 3, 0
  %83 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 4, 0
  %84 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %76, 0
  %85 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %84, ptr %77, 1
  %86 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %85, i64 %74, 2
  %87 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %86, i64 %75, 3, 0
  %88 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %87, i64 1, 4, 0
  %89 = sub i64 %45, %54
  %90 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 0
  %91 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 1
  %92 = insertvalue { ptr, ptr, i64 } undef, ptr %90, 0
  %93 = insertvalue { ptr, ptr, i64 } %92, ptr %91, 1
  %94 = insertvalue { ptr, ptr, i64 } %93, i64 0, 2
  %95 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 2
  %96 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 3, 0
  %97 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 4, 0
  %98 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %90, 0
  %99 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %98, ptr %91, 1
  %100 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %99, i64 %54, 2
  %101 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %100, i64 %89, 3, 0
  %102 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %101, i64 1, 4, 0
  call void @mlir_linpackcdaxpyrollf32(i32 %73, float %62, ptr %76, ptr %77, i64 %74, i64 %75, i64 1, i32 1, ptr %90, ptr %91, i64 %54, i64 %89, i64 1, i32 1)
  %103 = add i64 %51, 1
  br label %50

104:                                              ; preds = %50
  br label %105

105:                                              ; preds = %104, %47
  br label %106

106:                                              ; preds = %109, %105
  %107 = phi i64 [ %136, %109 ], [ 0, %105 ]
  %108 = icmp slt i64 %107, %35
  br i1 %108, label %109, label %137

109:                                              ; preds = %106
  %110 = add i64 %107, 1
  %111 = sub i64 %35, %110
  %112 = trunc i64 %111 to i32
  %113 = call float @get_val_dgesl_f32(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %37, i64 %111, i64 %111)
  %114 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 1
  %115 = getelementptr float, ptr %114, i64 %111
  %116 = load float, ptr %115, align 4
  %117 = fdiv float %116, %113
  %118 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 1
  %119 = getelementptr float, ptr %118, i64 %111
  store float %117, ptr %119, align 4
  %120 = fneg float %117
  %121 = call i64 @get_offset_dgesl_f32(i64 %37, i64 %111, i64 0)
  %122 = sub i64 %41, %121
  %123 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 0
  %124 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 1
  %125 = insertvalue { ptr, ptr, i64 } undef, ptr %123, 0
  %126 = insertvalue { ptr, ptr, i64 } %125, ptr %124, 1
  %127 = insertvalue { ptr, ptr, i64 } %126, i64 0, 2
  %128 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 2
  %129 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 3, 0
  %130 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 4, 0
  %131 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %123, 0
  %132 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %131, ptr %124, 1
  %133 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %132, i64 %121, 2
  %134 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %133, i64 %122, 3, 0
  %135 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %134, i64 1, 4, 0
  call void @mlir_linpackcdaxpyrollf32(i32 %112, float %120, ptr %123, ptr %124, i64 %121, i64 %122, i64 1, i32 1, ptr %12, ptr %13, i64 %14, i64 %15, i64 %16, i32 1)
  %136 = add i64 %107, 1
  br label %106

137:                                              ; preds = %106, %222, %161
  br label %223

138:                                              ; preds = %18
  br label %139

139:                                              ; preds = %142, %138
  %140 = phi i64 [ %160, %142 ], [ 0, %138 ]
  %141 = icmp slt i64 %140, %35
  br i1 %141, label %142, label %161

142:                                              ; preds = %139
  %143 = trunc i64 %140 to i32
  %144 = call i64 @get_offset_dgesl_f32(i64 %37, i64 %140, i64 0)
  %145 = sub i64 %41, %144
  %146 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 0
  %147 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 1
  %148 = insertvalue { ptr, ptr, i64 } undef, ptr %146, 0
  %149 = insertvalue { ptr, ptr, i64 } %148, ptr %147, 1
  %150 = insertvalue { ptr, ptr, i64 } %149, i64 0, 2
  %151 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 2
  %152 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 3, 0
  %153 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 4, 0
  %154 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %146, 0
  %155 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %154, ptr %147, 1
  %156 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %155, i64 %144, 2
  %157 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %156, i64 %145, 3, 0
  %158 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %157, i64 1, 4, 0
  %159 = call float @mlir_linpackcddotrollf32(i32 %143, ptr %146, ptr %147, i64 %144, i64 %145, i64 1, i32 1, ptr %12, ptr %13, i64 %14, i64 %15, i64 %16, i32 1)
  %160 = add i64 %140, 1
  br label %139

161:                                              ; preds = %139
  %162 = icmp sge i32 %34, 1
  br i1 %162, label %163, label %137

163:                                              ; preds = %161
  br label %164

164:                                              ; preds = %220, %163
  %165 = phi i64 [ %221, %220 ], [ 1, %163 ]
  %166 = icmp slt i64 %165, %36
  br i1 %166, label %167, label %222

167:                                              ; preds = %164
  %168 = add i64 %165, 1
  %169 = sub i64 %35, %168
  %170 = add i64 %169, 1
  %171 = sub i64 %35, %170
  %172 = trunc i64 %171 to i32
  %173 = trunc i64 %169 to i32
  %174 = call i64 @get_offset_dgesl_f32(i64 %37, i64 %169, i64 %170)
  %175 = sub i64 %41, %174
  %176 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 0
  %177 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 1
  %178 = insertvalue { ptr, ptr, i64 } undef, ptr %176, 0
  %179 = insertvalue { ptr, ptr, i64 } %178, ptr %177, 1
  %180 = insertvalue { ptr, ptr, i64 } %179, i64 0, 2
  %181 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 2
  %182 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 3, 0
  %183 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 4, 0
  %184 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %176, 0
  %185 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %184, ptr %177, 1
  %186 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %185, i64 %174, 2
  %187 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %186, i64 %175, 3, 0
  %188 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %187, i64 1, 4, 0
  %189 = sub i64 %45, %170
  %190 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 0
  %191 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 1
  %192 = insertvalue { ptr, ptr, i64 } undef, ptr %190, 0
  %193 = insertvalue { ptr, ptr, i64 } %192, ptr %191, 1
  %194 = insertvalue { ptr, ptr, i64 } %193, i64 0, 2
  %195 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 2
  %196 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 3, 0
  %197 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 4, 0
  %198 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %190, 0
  %199 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %198, ptr %191, 1
  %200 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %199, i64 %170, 2
  %201 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %200, i64 %189, 3, 0
  %202 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %201, i64 1, 4, 0
  %203 = call float @mlir_linpackcddotrollf32(i32 %172, ptr %176, ptr %177, i64 %174, i64 %175, i64 1, i32 1, ptr %190, ptr %191, i64 %170, i64 %189, i64 1, i32 1)
  %204 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %28, 1
  %205 = getelementptr i32, ptr %204, i64 %169
  %206 = load i32, ptr %205, align 4
  %207 = sext i32 %206 to i64
  %208 = icmp ne i32 %206, %173
  br i1 %208, label %209, label %220

209:                                              ; preds = %167
  %210 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 1
  %211 = getelementptr float, ptr %210, i64 %207
  %212 = load float, ptr %211, align 4
  %213 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 1
  %214 = getelementptr float, ptr %213, i64 %169
  %215 = load float, ptr %214, align 4
  %216 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 1
  %217 = getelementptr float, ptr %216, i64 %207
  store float %215, ptr %217, align 4
  %218 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 1
  %219 = getelementptr float, ptr %218, i64 %169
  store float %212, ptr %219, align 4
  br label %220

220:                                              ; preds = %209, %167
  %221 = add i64 %165, 1
  br label %164

222:                                              ; preds = %164
  br label %137

223:                                              ; preds = %137
  ret void
}

define void @_mlir_ciface_mlir_linpackcdgeslrollf32(ptr %0, i32 %1, i32 %2, ptr %3, ptr %4, i32 %5) {
  %7 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %0, align 8
  %8 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 0
  %9 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %10 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 2
  %11 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 3, 0
  %12 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 4, 0
  %13 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %3, align 8
  %14 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, 0
  %15 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, 1
  %16 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, 2
  %17 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, 3, 0
  %18 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, 4, 0
  %19 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %4, align 8
  %20 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, 0
  %21 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, 1
  %22 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, 2
  %23 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, 3, 0
  %24 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, 4, 0
  call void @mlir_linpackcdgeslrollf32(ptr %8, ptr %9, i64 %10, i64 %11, i64 %12, i32 %1, i32 %2, ptr %14, ptr %15, i64 %16, i64 %17, i64 %18, ptr %20, ptr %21, i64 %22, i64 %23, i64 %24, i32 %5)
  ret void
}

define void @mlir_linpackcdgeslunrollf32(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i32 %5, i32 %6, ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, ptr %12, ptr %13, i64 %14, i64 %15, i64 %16, i32 %17) {
  %19 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %0, 0
  %20 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, ptr %1, 1
  %21 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, i64 %2, 2
  %22 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %21, i64 %3, 3, 0
  %23 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %22, i64 %4, 4, 0
  %24 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %7, 0
  %25 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %24, ptr %8, 1
  %26 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %25, i64 %9, 2
  %27 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %26, i64 %10, 3, 0
  %28 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %27, i64 %11, 4, 0
  %29 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %12, 0
  %30 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %29, ptr %13, 1
  %31 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %30, i64 %14, 2
  %32 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %31, i64 %15, 3, 0
  %33 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %32, i64 %16, 4, 0
  %34 = sub i32 %6, 1
  %35 = sext i32 %6 to i64
  %36 = sext i32 %34 to i64
  %37 = sext i32 %5 to i64
  %38 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 3
  %39 = alloca [1 x i64], i64 1, align 8
  store [1 x i64] %38, ptr %39, align 4
  %40 = getelementptr [1 x i64], ptr %39, i32 0, i32 0
  %41 = load i64, ptr %40, align 4
  %42 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 3
  %43 = alloca [1 x i64], i64 1, align 8
  store [1 x i64] %42, ptr %43, align 4
  %44 = getelementptr [1 x i64], ptr %43, i32 0, i32 0
  %45 = load i64, ptr %44, align 4
  %46 = icmp eq i32 %17, 0
  br i1 %46, label %47, label %138

47:                                               ; preds = %18
  %48 = icmp sge i32 %34, 1
  br i1 %48, label %49, label %105

49:                                               ; preds = %47
  br label %50

50:                                               ; preds = %72, %49
  %51 = phi i64 [ %103, %72 ], [ 0, %49 ]
  %52 = icmp slt i64 %51, %36
  br i1 %52, label %53, label %104

53:                                               ; preds = %50
  %54 = add i64 %51, 1
  %55 = trunc i64 %54 to i32
  %56 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %28, 1
  %57 = getelementptr i32, ptr %56, i64 %51
  %58 = load i32, ptr %57, align 4
  %59 = sext i32 %58 to i64
  %60 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 1
  %61 = getelementptr float, ptr %60, i64 %59
  %62 = load float, ptr %61, align 4
  %63 = icmp ne i64 %59, %51
  br i1 %63, label %64, label %72

64:                                               ; preds = %53
  %65 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 1
  %66 = getelementptr float, ptr %65, i64 %51
  %67 = load float, ptr %66, align 4
  %68 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 1
  %69 = getelementptr float, ptr %68, i64 %59
  store float %67, ptr %69, align 4
  %70 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 1
  %71 = getelementptr float, ptr %70, i64 %51
  store float %62, ptr %71, align 4
  br label %72

72:                                               ; preds = %64, %53
  %73 = sub i32 %6, %55
  %74 = call i64 @get_offset_dgesl_f32(i64 %37, i64 %51, i64 %54)
  %75 = sub i64 %41, %74
  %76 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 0
  %77 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 1
  %78 = insertvalue { ptr, ptr, i64 } undef, ptr %76, 0
  %79 = insertvalue { ptr, ptr, i64 } %78, ptr %77, 1
  %80 = insertvalue { ptr, ptr, i64 } %79, i64 0, 2
  %81 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 2
  %82 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 3, 0
  %83 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 4, 0
  %84 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %76, 0
  %85 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %84, ptr %77, 1
  %86 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %85, i64 %74, 2
  %87 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %86, i64 %75, 3, 0
  %88 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %87, i64 1, 4, 0
  %89 = sub i64 %45, %54
  %90 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 0
  %91 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 1
  %92 = insertvalue { ptr, ptr, i64 } undef, ptr %90, 0
  %93 = insertvalue { ptr, ptr, i64 } %92, ptr %91, 1
  %94 = insertvalue { ptr, ptr, i64 } %93, i64 0, 2
  %95 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 2
  %96 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 3, 0
  %97 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 4, 0
  %98 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %90, 0
  %99 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %98, ptr %91, 1
  %100 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %99, i64 %54, 2
  %101 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %100, i64 %89, 3, 0
  %102 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %101, i64 1, 4, 0
  call void @mlir_linpackcdaxpyunrollf32(i32 %73, float %62, ptr %76, ptr %77, i64 %74, i64 %75, i64 1, i32 1, ptr %90, ptr %91, i64 %54, i64 %89, i64 1, i32 1)
  %103 = add i64 %51, 1
  br label %50

104:                                              ; preds = %50
  br label %105

105:                                              ; preds = %104, %47
  br label %106

106:                                              ; preds = %109, %105
  %107 = phi i64 [ %136, %109 ], [ 0, %105 ]
  %108 = icmp slt i64 %107, %35
  br i1 %108, label %109, label %137

109:                                              ; preds = %106
  %110 = add i64 %107, 1
  %111 = sub i64 %35, %110
  %112 = trunc i64 %111 to i32
  %113 = call float @get_val_dgesl_f32(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %37, i64 %111, i64 %111)
  %114 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 1
  %115 = getelementptr float, ptr %114, i64 %111
  %116 = load float, ptr %115, align 4
  %117 = fdiv float %116, %113
  %118 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 1
  %119 = getelementptr float, ptr %118, i64 %111
  store float %117, ptr %119, align 4
  %120 = fneg float %117
  %121 = call i64 @get_offset_dgesl_f32(i64 %37, i64 %111, i64 0)
  %122 = sub i64 %41, %121
  %123 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 0
  %124 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 1
  %125 = insertvalue { ptr, ptr, i64 } undef, ptr %123, 0
  %126 = insertvalue { ptr, ptr, i64 } %125, ptr %124, 1
  %127 = insertvalue { ptr, ptr, i64 } %126, i64 0, 2
  %128 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 2
  %129 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 3, 0
  %130 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 4, 0
  %131 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %123, 0
  %132 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %131, ptr %124, 1
  %133 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %132, i64 %121, 2
  %134 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %133, i64 %122, 3, 0
  %135 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %134, i64 1, 4, 0
  call void @mlir_linpackcdaxpyunrollf32(i32 %112, float %120, ptr %123, ptr %124, i64 %121, i64 %122, i64 1, i32 1, ptr %12, ptr %13, i64 %14, i64 %15, i64 %16, i32 1)
  %136 = add i64 %107, 1
  br label %106

137:                                              ; preds = %106, %222, %161
  br label %223

138:                                              ; preds = %18
  br label %139

139:                                              ; preds = %142, %138
  %140 = phi i64 [ %160, %142 ], [ 0, %138 ]
  %141 = icmp slt i64 %140, %35
  br i1 %141, label %142, label %161

142:                                              ; preds = %139
  %143 = trunc i64 %140 to i32
  %144 = call i64 @get_offset_dgesl_f32(i64 %37, i64 %140, i64 0)
  %145 = sub i64 %41, %144
  %146 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 0
  %147 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 1
  %148 = insertvalue { ptr, ptr, i64 } undef, ptr %146, 0
  %149 = insertvalue { ptr, ptr, i64 } %148, ptr %147, 1
  %150 = insertvalue { ptr, ptr, i64 } %149, i64 0, 2
  %151 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 2
  %152 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 3, 0
  %153 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 4, 0
  %154 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %146, 0
  %155 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %154, ptr %147, 1
  %156 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %155, i64 %144, 2
  %157 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %156, i64 %145, 3, 0
  %158 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %157, i64 1, 4, 0
  %159 = call float @mlir_linpackcddotunrollf32(i32 %143, ptr %146, ptr %147, i64 %144, i64 %145, i64 1, i32 1, ptr %12, ptr %13, i64 %14, i64 %15, i64 %16, i32 1)
  %160 = add i64 %140, 1
  br label %139

161:                                              ; preds = %139
  %162 = icmp sge i32 %34, 1
  br i1 %162, label %163, label %137

163:                                              ; preds = %161
  br label %164

164:                                              ; preds = %220, %163
  %165 = phi i64 [ %221, %220 ], [ 1, %163 ]
  %166 = icmp slt i64 %165, %36
  br i1 %166, label %167, label %222

167:                                              ; preds = %164
  %168 = add i64 %165, 1
  %169 = sub i64 %35, %168
  %170 = add i64 %169, 1
  %171 = sub i64 %35, %170
  %172 = trunc i64 %171 to i32
  %173 = trunc i64 %169 to i32
  %174 = call i64 @get_offset_dgesl_f32(i64 %37, i64 %169, i64 %170)
  %175 = sub i64 %41, %174
  %176 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 0
  %177 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 1
  %178 = insertvalue { ptr, ptr, i64 } undef, ptr %176, 0
  %179 = insertvalue { ptr, ptr, i64 } %178, ptr %177, 1
  %180 = insertvalue { ptr, ptr, i64 } %179, i64 0, 2
  %181 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 2
  %182 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 3, 0
  %183 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 4, 0
  %184 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %176, 0
  %185 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %184, ptr %177, 1
  %186 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %185, i64 %174, 2
  %187 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %186, i64 %175, 3, 0
  %188 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %187, i64 1, 4, 0
  %189 = sub i64 %45, %170
  %190 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 0
  %191 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 1
  %192 = insertvalue { ptr, ptr, i64 } undef, ptr %190, 0
  %193 = insertvalue { ptr, ptr, i64 } %192, ptr %191, 1
  %194 = insertvalue { ptr, ptr, i64 } %193, i64 0, 2
  %195 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 2
  %196 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 3, 0
  %197 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 4, 0
  %198 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %190, 0
  %199 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %198, ptr %191, 1
  %200 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %199, i64 %170, 2
  %201 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %200, i64 %189, 3, 0
  %202 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %201, i64 1, 4, 0
  %203 = call float @mlir_linpackcddotunrollf32(i32 %172, ptr %176, ptr %177, i64 %174, i64 %175, i64 1, i32 1, ptr %190, ptr %191, i64 %170, i64 %189, i64 1, i32 1)
  %204 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %28, 1
  %205 = getelementptr i32, ptr %204, i64 %169
  %206 = load i32, ptr %205, align 4
  %207 = sext i32 %206 to i64
  %208 = icmp ne i32 %206, %173
  br i1 %208, label %209, label %220

209:                                              ; preds = %167
  %210 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 1
  %211 = getelementptr float, ptr %210, i64 %207
  %212 = load float, ptr %211, align 4
  %213 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 1
  %214 = getelementptr float, ptr %213, i64 %169
  %215 = load float, ptr %214, align 4
  %216 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 1
  %217 = getelementptr float, ptr %216, i64 %207
  store float %215, ptr %217, align 4
  %218 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 1
  %219 = getelementptr float, ptr %218, i64 %169
  store float %212, ptr %219, align 4
  br label %220

220:                                              ; preds = %209, %167
  %221 = add i64 %165, 1
  br label %164

222:                                              ; preds = %164
  br label %137

223:                                              ; preds = %137
  ret void
}

define void @_mlir_ciface_mlir_linpackcdgeslunrollf32(ptr %0, i32 %1, i32 %2, ptr %3, ptr %4, i32 %5) {
  %7 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %0, align 8
  %8 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 0
  %9 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %10 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 2
  %11 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 3, 0
  %12 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 4, 0
  %13 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %3, align 8
  %14 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, 0
  %15 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, 1
  %16 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, 2
  %17 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, 3, 0
  %18 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, 4, 0
  %19 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %4, align 8
  %20 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, 0
  %21 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, 1
  %22 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, 2
  %23 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, 3, 0
  %24 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, 4, 0
  call void @mlir_linpackcdgeslunrollf32(ptr %8, ptr %9, i64 %10, i64 %11, i64 %12, i32 %1, i32 %2, ptr %14, ptr %15, i64 %16, i64 %17, i64 %18, ptr %20, ptr %21, i64 %22, i64 %23, i64 %24, i32 %5)
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
