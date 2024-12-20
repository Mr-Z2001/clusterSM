#ifndef DEFS_H
#define DEFS_H

#include <cstdint>

#define GRID_DIM 1024u
#define BLOCK_DIM 512u
#define WARP_SIZE 32u
#define WARP_PER_BLOCK 16u

#define MAX_VQ 32
#define MAX_EQ 64u
#define MAX_VLQ 16u

#define MAX_CLUSTERS 150
#define MAX_LAYERS 100

#define ENC_SIZE 32

#define MAX_RES 500000000

// #define MAX_CUCKOO_LOOP 64u
// #define BUCKET_DIM 8u
// #define CUCKOO_SCALE_PER_TABLE 2u
// #define NUM_TABLE 2u

// #define MAX_RES_MEM_SPACE 6000000000ul

using vtype = uint32_t;
using etype = uint32_t;
using vltype = uint32_t;
using numtype = uint32_t;
using offtype = uint32_t;
using degtype = uint32_t;
// using eltype = uint32_t;

inline unsigned int calc_grid_dim(int N, int block_size)
{
  if (N == 0)
    ++N;
  return (N - 1) / block_size + 1;
}

#endif