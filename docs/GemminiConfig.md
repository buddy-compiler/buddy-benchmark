# Gemmini Configuration for Benchmark Cases

Gemmini is Berkeley's spatial array generator, and this project provides a full-stack DNN hardware exploration and evaluation platform.
We implemented an MLIR-based compiler for the Gemmini backend and added benchmark cases.
Before running these cases, you may need to use the specific Gemmini configuration (please see the official docs and tutorials).
We also provide some simple steps here.

## Using Default Float Point Configuration

As for this configuration, we assume you have already built all the components in the Gemmini README file. In the generated file
`chipyard/generators/gemmini/configs/GemminiCustomConfigs.scala`,
define the custom configuration with the default float point configuration.

```scala
object GemminiCustomConfigs {
  // Default configurations
  val defaultConfig = GemminiConfigs.defaultConfig
  val defaultFpConfig = GemminiFPConfigs.defaultFPConfig

  ... ...

  // Specify which of your custom configs you want to build here
  // val customConfig = baselineInferenceConfig
  val customConfig = defaultFpConfig
}
```

And then, enter the Chipyard environment and rebuild the Spike simulator.

```
$ cd chipyard
$ source env.sh
$ cd generators/gemmini
$ ./scripts/build-spike.sh
```
