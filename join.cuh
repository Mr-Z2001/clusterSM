#ifndef JOIN_H
#define JOIN_H

#include "globals.cuh"
#include "cpuGraph.h"
#include "gpuGraph.h"
#include "structure.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// __global__ void
// firstJoinCGKernel(
//     etype e, vtype u, vtype u_nbr,
//     numtype num_res,
//     vtype *candidate_vs_1_, vtype *candidate_vs_2_,
//     vtype *d_res_table, size_t d_res_table_pitch);

// void firstJoinCG(
//     cpuGraph *hq,
//     etype e,
//     cpuRelation *cpu_relations_, gpuRelation *gpu_relations_,
//     numtype *num_res,
//     vtype *d_res_table, size_t d_res_table_pitch);

// __global__ void
// joinOneEdgeCountCGKernel(
//     vtype *keys_,
//     numtype num_candidates,
//     vtype *candidate_vs_1_, vtype *candidate_vs_2_,
//     vtype *d_res_table, size_t d_res_table_pitch, numtype num_res_old,
//     numtype *num_res_new);

// void joinOneEdgeCountCG(
//     etype e, vtype u, vtype u_nbr,
//     numtype num_res_old,
//     cpuRelation *cpu_relations_, gpuRelation *gpu_relations_,
//     vtype *d_res_table, size_t d_res_table_pitch,
//     vtype *d_temp_res_table, size_t d_temp_res_table_pitch,
//     numtype *num_res);

// __global__ void joinOneEdgeWriteCGKernel(
//     vtype *keys_,
//     numtype num_candidates,
//     vtype *candidate_vs_1_, vtype *candidate_vs_2_,
//     vtype *d_res_table, size_t d_res_table_pitch, numtype num_res_old,
//     vtype *d_temp_res_table, size_t d_temp_res_table_pitch,
//     numtype *num_res_new);

// void joinOneEdgeWriteCG(
//     etype e, vtype u, vtype u_nbr,
//     numtype num_res_old,
//     cpuRelation *cpu_relations_, gpuRelation *gpu_relations_,
//     vtype *d_res_table, size_t d_res_table_pitch,
//     vtype *d_temp_res_table, size_t d_temp_res_table_pitch,
//     numtype *num_res);

// __global__ void
// selectPartialMatchingsCGKernel(
//     vtype *keys_,
//     numtype num_candidates,
//     vtype *candidate_vs_1_, vtype *candidate_vs_2_,
//     vtype *d_res_table, size_t d_res_table_pitch,
//     vtype *d_temp_res_table, size_t d_temp_res_table_pitch,
//     numtype num_res_old,
//     numtype *num_res_new);

// void selectPartialMatchingsCG(
//     etype e, vtype u, vtype u_nbr,
//     numtype num_res_old,
//     cpuRelation *cpu_relations_, gpuRelation *gpu_relations_,
//     vtype *d_res_table, size_t d_res_table_pitch,
//     vtype *d_temp_res_table, size_t d_temp_res_table_pitch,
//     numtype *num_res);

// void joinCG(
//     cpuGraph *hq, cpuGraph *hg,
//     gpuGraph *dq,
//     etype *order,
//     cpuRelation *cpu_relations_, gpuRelation *gpu_relations_,
//     numtype *num_res, vtype *h_res_table,
//     vtype *d_res_table, size_t d_res_table_pitch,
//     vtype *d_temp_res_table, size_t d_temp_res_table_pitch);

__global__ void
selectPartialMatchingsKernel(
    offtype *offsets_, vtype *nbrs_,
    vtype u, vtype u_matched,
    vtype *d_res_table_old_, numtype num_res_old,
    vtype *d_res_table_, numtype *num_res_new,

    uint32_t *d_encodings_, numtype num_blocks,
    uint32_t enc_pos_u, uint32_t enc_pos_u_matched);

__global__ void collectMappedVs(
    vtype u_matched,
    bool *d_flag_,
    vtype *d_res_table_old_, numtype num_res_old);

__global__ void
joinOneEdgeKernel(
    // structure
    offtype *offsets_, vtype *nbrs_,

    vtype u, vtype u_matched,
    vtype *d_res_table_old_, numtype num_res_old,
    vtype *d_res_table_, numtype *num_res_new,
    vtype *candidate_v_buffer_,

    uint32_t *v_candidate_us_, // bitmap reverse

    uint32_t *encodings_, numtype num_blocks,
    int enc_pos,

    int *num_candidates_in_buffer,
    bool *flag_);

void join(
    cpuGraph *hq, cpuGraph *hg,
    gpuGraph *dq, gpuGraph *dg,
    etype *order,
    vtype *h_res_table_, numtype &num_res,

    encodingMeta *enc_meta,
    uint32_t *h_encodings_, uint32_t *d_encodings_,

    vtype *h_u_candidate_vs_, numtype *h_num_u_candidate_vs_,
    vtype *d_u_candidate_vs_, numtype *d_num_u_candidate_vs_,
    vtype *h_v_candidate_us_, numtype *h_num_v_candidate_us_,
    vtype *d_v_candidate_us_, numtype *d_num_v_candidate_us_,

    cpuCluster *cpu_clusters_, gpuCluster *gpu_clusters_);

#endif