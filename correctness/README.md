# Correctness Checking Framework

Please configure following parameters for building correctness checking framework.

| CMake Options  | Default Value            |
| -------------- |--------------------------|
| `-DBUILD_CORRECTNESS`  | OFF                      |
| `-DAUDIO_PROCESSING_BENCHMARKS`  | OFF                      |

## Audio processing correctness
Currently supported correctness checking:

| Item  | Compare to |
|-------|------------|
| `FIR` | KFR        |