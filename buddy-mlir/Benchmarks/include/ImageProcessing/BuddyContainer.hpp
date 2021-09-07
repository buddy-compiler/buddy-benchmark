#ifndef BUDDY_CONTAINER
#define BUDDY_CONTAINER

#include <stdint.h>
#include <stdlib.h>

// Define Memref Descriptor.
typedef struct MemRef_descriptor_ *MemRef_descriptor;
typedef struct MemRef_descriptor_ {
  float *allocated;
  float *aligned;
  intptr_t offset;
  intptr_t sizes[2];
  intptr_t strides[2];
} Memref;

// Constructor
MemRef_descriptor MemRef_Descriptor(float *allocated, float *aligned,
                                    intptr_t offset, intptr_t sizes[2],
                                    intptr_t strides[2]) {
  MemRef_descriptor n = (MemRef_descriptor)malloc(sizeof(*n));
  n->allocated = allocated;
  n->aligned = aligned;
  n->offset = offset;
  for (int i = 0; i < 2; i++)
    n->sizes[i] = sizes[i];
  for (int j = 0; j < 2; j++)
    n->strides[j] = strides[j];

  return n;
}

// Declare the conv2d C interface.
extern "C" {
void _mlir_ciface_conv_2d(MemRef_descriptor input, MemRef_descriptor kernel,
                          MemRef_descriptor output);
}

#endif
