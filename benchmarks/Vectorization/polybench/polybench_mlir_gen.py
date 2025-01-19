# ===- polybench_mlir_gen.py ---------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# This file generates the MLIR source code and binaries of Polybench benchmarks
# using Polygeist and clang.
#
# ===---------------------------------------------------------------------------

import subprocess
import os
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Generate MLIR and Binary for Polybench benchmarks using Polygeist and clang"
    )
    parser.add_argument(
        "--output-dir", type=str, default="output", help="Output directory"
    )
    parser.add_argument(
        "--polygeist-build-dir",
        type=str,
        required=True,
        help="Polygeist build directory",
    )
    parser.add_argument(
        "--polybench-dir", type=str, required=True, help="Polybench directory"
    )
    parser.add_argument(
        "--generate-mlir", action="store_true", help="Generate MLIR files"
    )
    parser.add_argument(
        "--generate-binary", action="store_true", help="Generate binaries"
    )
    parser.add_argument(
        "--binary-compiler",
        type=str,
        help="The compiler to generate binaries",
        default="clang",
    )
    parser.add_argument(
        "--generate-std-output",
        action="store_true",
        help="Generate standard output for the given dataset size",
    )
    parser.add_argument(
        "--std-output-file",
        type=str,
        default="std_output.txt",
    )
    parser.add_argument(
        "--std-output-dataset-size",
        type=str,
        default="small",
    )

    args = parser.parse_args()

    # using polygeist to generate MLIR
    template = (
        "{polygeist_build_dir}/bin/cgeist "
        "{polybench_dir}/{benchmark_path}/{benchmark_id}.c "
        "{polybench_dir}/utilities/polybench.c "
        "-resource-dir={polygeist_build_dir}/../llvm-project/build/lib/clang/18 "
        "-DPOLYBENCH_NO_FLUSH_CACHE -D{dataset_size_macro} "
        "-I{polybench_dir}/utilities "
        "-O0 -S -o {output_dir}/{benchmark_id}-{dataset_size}.mlir"
    )

    # using clang to generate binary, with DUMP_ARRAYS enabled
    if args.binary_compiler == "clang":
        template_bin = (
            "clang {polybench_dir}/{benchmark_path}/{benchmark_id}.c "
            "{polybench_dir}/utilities/polybench.c "
            "-DPOLYBENCH_NO_FLUSH_CACHE -D{dataset_size_macro} "
            "-I{polybench_dir}/utilities "
            "-DPOLYBENCH_DUMP_ARRAYS "
            "-O0 -lm -o {output_dir}/{benchmark_id}-{dataset_size}"
        )
    elif args.binary_compiler == "cgeist":
        template_bin = (
            "{polygeist_build_dir}/bin/cgeist "
            "{polybench_dir}/{benchmark_path}/{benchmark_id}.c "
            "{polybench_dir}/utilities/polybench.c "
            "-resource-dir={polygeist_build_dir}/../llvm-project/build/lib/clang/18 "
            "-DPOLYBENCH_NO_FLUSH_CACHE -D{dataset_size_macro} "
            "-I{polybench_dir}/utilities "
            "-DPOLYBENCH_DUMP_ARRAYS "
            "-O0 -lm -o {output_dir}/{benchmark_id}-{dataset_size}"
        )
    else:
        print("Unsupported compiler")
        return

    benchmarks = {
        "2mm": "linear-algebra/kernels/2mm",
        "3mm": "linear-algebra/kernels/3mm",
        "adi": "stencils/adi",
        "atax": "linear-algebra/kernels/atax",
        "bicg": "linear-algebra/kernels/bicg",
        "cholesky": "linear-algebra/solvers/cholesky",
        "correlation": "datamining/correlation",
        "covariance": "datamining/covariance",
        "deriche": "medley/deriche",
        "doitgen": "linear-algebra/kernels/doitgen",
        "durbin": "linear-algebra/solvers/durbin",
        "fdtd-2d": "stencils/fdtd-2d",
        "floyd-warshall": "medley/floyd-warshall",
        "gemm": "linear-algebra/blas/gemm",
        "gemver": "linear-algebra/blas/gemver",
        "gesummv": "linear-algebra/blas/gesummv",
        "gramschmidt": "linear-algebra/solvers/gramschmidt",
        "heat-3d": "stencils/heat-3d",
        "jacobi-1d": "stencils/jacobi-1d",
        "jacobi-2d": "stencils/jacobi-2d",
        "lu": "linear-algebra/solvers/lu",
        "ludcmp": "linear-algebra/solvers/ludcmp",
        "mvt": "linear-algebra/kernels/mvt",
        "nussinov": "medley/nussinov",
        "seidel-2d": "stencils/seidel-2d",
        "symm": "linear-algebra/blas/symm",
        "syr2k": "linear-algebra/blas/syr2k",
        "syrk": "linear-algebra/blas/syrk",
        "trisolv": "linear-algebra/solvers/trisolv",
        "trmm": "linear-algebra/blas/trmm",
    }

    dataset_sizes = {
        "mini": "MINI_DATASET",
        "small": "SMALL_DATASET",
        "medium": "MEDIUM_DATASET",
        "large": "LARGE_DATASET",
        "extralarge": "EXTRALARGE_DATASET",
    }

    output_dir = os.path.abspath(args.output_dir)

    if os.path.exists(output_dir):
        if args.generate_mlir or args.generate_binary:
            os.system(f"rm -rf {output_dir}")
            os.makedirs(output_dir)
    else:
        os.makedirs(output_dir)

    for benchmark_id, benchmark_path in benchmarks.items():
        for dataset_size, dataset_size_macro in dataset_sizes.items():
            if args.generate_mlir:
                command = template.format(
                    polygeist_build_dir=args.polygeist_build_dir,
                    polybench_dir=args.polybench_dir,
                    benchmark_path=benchmark_path,
                    dataset_size=dataset_size,
                    dataset_size_macro=dataset_size_macro,
                    output_dir=output_dir,
                    benchmark_id=benchmark_id,
                )
                print("[ Running ]", command)
                result = subprocess.run(command, shell=True)
                if result.returncode != 0:
                    print("[ Failed  ]", f"{benchmark_id}-{dataset_size}.mlir")
                    break

                print("[Generated]", f"{benchmark_id}-{dataset_size}.mlir")

            if args.generate_binary:
                if args.binary_compiler == "clang":
                    command = template_bin.format(
                        polybench_dir=args.polybench_dir,
                        benchmark_path=benchmark_path,
                        dataset_size=dataset_size,
                        dataset_size_macro=dataset_size_macro,
                        output_dir=output_dir,
                        benchmark_id=benchmark_id,
                    )
                else:
                    command = template_bin.format(
                        polygeist_build_dir=args.polygeist_build_dir,
                        polybench_dir=args.polybench_dir,
                        benchmark_path=benchmark_path,
                        dataset_size=dataset_size,
                        dataset_size_macro=dataset_size_macro,
                        output_dir=output_dir,
                        benchmark_id=benchmark_id,
                    )

                print("[ Running ]", command)
                result = subprocess.run(command, shell=True)
                if result.returncode != 0:
                    print("[ Failed  ]", f"binary {benchmark_id}-{dataset_size}")
                    break

                print("[Generated]", f"binary {benchmark_id}-{dataset_size}")

    if args.generate_std_output:
        # Run the generated binaries and redirect stderr to append to the file
        output_file_path = os.path.join(output_dir, args.std_output_file)
        if os.path.exists(output_file_path):
            os.remove(output_file_path)

        for benchmark_id in sorted(benchmarks.keys()):
            dataset_size = args.std_output_dataset_size
            command = os.path.join(output_dir, f"{benchmark_id}-{dataset_size}")

            with open(output_file_path, "a") as f:
                print(f"{'-' * 48}", file=f)
                print(f"Result for {benchmark_id}-{dataset_size}:", file=f)

            command = f"cd {output_dir} && {command} 2>> {args.std_output_file}"
            print("[ Running ]", command)

            result = subprocess.run(command, shell=True)
            if result.returncode != 0:
                print("[ Failed  ]", f"running {benchmark_id}-{dataset_size}")
                break

            with open(output_file_path, "a") as f:
                print(f"{'-' * 48}", file=f)

    print("Done")


if __name__ == "__main__":
    main()
