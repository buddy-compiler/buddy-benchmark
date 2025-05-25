#!/usr/bin/env bash
set -e

# ➊ one container per run, killed automatically on exit
CID=$(docker run -d --name buddy-mlir-ci-test \
        --privileged \
        -v "${GITHUB_WORKSPACE}:/home/buddy-complier-workspace" \
        liuqun1006/buddycompiler-base:python sleep infinity)

trap "docker rm -f ${CID}" EXIT

# ➋ execute the whole build-and-test sequence inside
docker exec "${CID}" bash -lc '
  set -e
  cd /home/buddy-complier-workspace/buddy-mlir
  ./test.sh build-llvm
  ./test.sh build-buddy
  ./test.sh run

  cd /home/buddy-complier-workspace/buddy-benchmark/test
  ./test_script_vectorizationprocessing.sh
'

# ➌ bring the logs back to the host (under ./test_result)
docker cp "${CID}":/home/buddy-complier-workspace/buddy-benchmark/test_result ./test_result
