#-------------------------------------------------------------------------------
# Generate Non-Schedule Version Conv Layer Static Library
#-------------------------------------------------------------------------------

add_halide_generator(conv_layer_nonschedule.generator SOURCES conv_layer_generator.cpp)
add_halide_library(conv_layer_nonschedule FROM conv_layer_nonschedule.generator)

#-------------------------------------------------------------------------------
# Generate Auto-Schedule Version Conv Layer Static Library
#-------------------------------------------------------------------------------

add_halide_generator(conv_layer_autoschedule.generator SOURCES conv_layer_generator-autoschedule.cpp)
add_halide_library(conv_layer_autoschedule FROM conv_layer_autoschedule.generator
                   AUTOSCHEDULER Halide::Mullapudi2016)

#-------------------------------------------------------------------------------
# Generate Manually-Schedule Version Conv Layer Static Library
#-------------------------------------------------------------------------------

add_halide_generator(conv_layer_manually.generator SOURCES conv_layer_generator-manually.cpp)
add_halide_library(conv_layer_manuallyschedule FROM conv_layer_manually.generator)

#-------------------------------------------------------------------------------
# Halide ConvLayer Benchmark Target
#-------------------------------------------------------------------------------

add_executable(halide-convlayer-benchmark
               Main.cpp 
               HalideConvLayerBenchmark.cpp)

add_custom_command(TARGET halide-convlayer-benchmark POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E rm ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/*.generator
                COMMENT "Cleaning up all generator files")

target_link_libraries(halide-convlayer-benchmark 
                       GoogleBenchmark
                       Halide::ImageIO
                       conv_layer_nonschedule
                       conv_layer_manuallyschedule
                       conv_layer_autoschedule)