#ifndef FILTER_H
#define FILTER_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "gpuGraph.h"
#include "cpuGraph.h"
#include "globals.cuh"
#include "structure.cuh"

__global__ void
oneRoundFilterCG(
		vltype *d_q_vLabels_, degtype *d_q_degrees_,
		vltype *d_v_labels_, degtype *d_v_degrees_,
		uint32_t *d_bitmap, size_t bitmap_pitch);

void getVertexCover( // DP on spanning tree.
		cpuGraph *hq,
		// return
		vtype *vertex_cover_, numtype *vertex_cover_size);

void clustering(
		cpuGraph *hq,
		cpuCluster *&cpu_clusters_,
		encodingMeta *enc_meta);

__global__ void
NLCFilter(
		offtype *d_offsets_, vtype *d_nbrs_, vltype *d_v_labels_, degtype *d_v_degrees_,
		uint32_t *d_query_NLC,
		uint32_t *d_bitmap, size_t bitmap_pitch,
		numtype *d_v_candidate_size_, vtype *d_v_candidate_us_);

void oneRoundFilterBidirection(
		cpuGraph *hq, cpuGraph *hg,
		gpuGraph *dq, gpuGraph *dg,

		vtype *d_u_candidate_vs_, numtype *d_num_u_candidate_vs_,
		vtype *d_v_candidate_us_, numtype *d_num_v_candidate_us_);

__global__ void
oneRoundFilterBidirectionKernel(
		// structure info
		vltype *query_vLabels_, degtype *query_out_degrees_,
		offtype *d_offsets_, vtype *d_nbrs_, vltype *d_v_labels_, degtype *d_v_degrees_,

		vtype *d_u_candidate_vs_, numtype *d_num_u_candidate_vs_,
		vtype *d_v_candidate_us_, numtype *d_num_v_candidate_us_,

		numtype *d_query_nlc_table_);

void oneRoundFilterReverse(
		cpuGraph *hq, cpuGraph *hg,
		gpuGraph *dq, gpuGraph *dg,
		uint32_t *d_bitmap_, size_t bitmap_pitch,
		vtype *d_v_candidate_us_, numtype *d_num_v_candidates_);

__global__ void
compactCandidatesKernel(
		vtype *d_u_candidate_vs_, numtype *d_num_u_candidate_vs_,
		vtype *d_u_candidate_vs_new_, numtype *d_num_u_candidate_vs_new_);

void encode(
		gpuGraph *dg,
		cpuCluster *cpu_clusters_, gpuCluster *gpu_clusters_,
		uint32_t *h_encodings_, uint32_t *d_encodings_, encodingMeta *encoding_meta,
		vtype *d_u_candidate_vs_, numtype *d_num_u_candidate_vs_,
		vtype *d_v_candidate_us_, numtype *d_num_v_candidate_us_);

__global__ void encodeKernel(
		// graph info
		offtype *d_offsets_, vtype *d_nbrs_,
		// candidate vertices
		vtype core_u, uint32_t cluster_index,
		vtype *d_u_candidate_vs_, numtype *d_num_u_candidate_vs_,
		vtype *d_v_candidate_us_,
		// uint32_t *d_bitmap_reverse_, numtype d_bitmap_reverse_width,

		// encoding info
		uint32_t *encodings_,
		numtype *enc_num_query_us_, numtype enc_num_blocks,
		vtype *enc_query_us_compact_, offtype *enc_cluster_offsets_);

__global__ void
mergeKernel(
		int left_pos, int right_pos,
		// graph info
		offtype *d_offsets_, vtype *d_nbrs_,
		// candidate vertices
		vtype core_u, uint32_t layer_index, uint32_t cluster_index,
		vtype *d_u_candidate_vs_, numtype *d_num_u_candidate_vs_,
		vtype *d_v_candidate_us_, numtype *d_num_v_candidate_us_,
		// uint32_t *d_bitmap_reverse_, numtype d_bitmap_reverse_width,

		// encoding info
		uint32_t *encodings_,

		numtype enc_num_clusters, numtype *enc_num_query_us_,
		numtype enc_num_total_us, numtype enc_num_blocks,
		vtype *enc_query_us_compact_, offtype *enc_cluster_offsets_,

		numtype enc_num_layers, numtype *enc_num_clusters_per_layer_,

		numtype enc_merge_count,
		numtype *enc_merged_cluster_left_, numtype *enc_merged_cluster_right_,
		vtype *enc_merged_cluster_vertex_, numtype *enc_merged_cluster_layer_);

__global__ void
combineMultipleClustersKernel(
		offtype *d_offsets_, vtype *nbrs_,
		vtype core_u, bool combine_type,
		int big_cluster, int *small_clusters_arr_, int num_small_clusters,
		uint32_t *d_encodings_,
		numtype *num_query_us_,
		numtype num_blocks,
		vtype *query_us_compact_, offtype *cluster_offsets_,
		vtype *d_u_candidate_vs_, numtype *d_num_u_candidate_vs_);

__global__ void
collectCandidatesKernel(
		vtype *d_u_candidate_vs_, vtype *d_num_u_candidate_vs_,
		vtype *d_v_candidate_us_, vtype *d_num_v_candidate_us_,
		uint32_t *d_encodings_, int *d_pos_array_, vtype *d_query_us_compact_,
		int num_blocks);

void clusterFilter(
		cpuGraph *hq_backup, gpuGraph *dq_backup,
		cpuGraph *hq, cpuGraph *hg,
		gpuGraph *dq, gpuGraph *dg,

		// cluster related
		cpuCluster *&cpu_clusters_, gpuCluster *&gpu_clusters_,
		uint32_t *&h_encodings_, uint32_t *&d_encodings_,
		encodingMeta *encoding_meta,

		// return
		vtype *&h_u_candidate_vs_, numtype *&h_num_u_candidate_vs_,
		vtype *&d_u_candidate_vs_, numtype *&d_num_u_candidate_vs_,
		vtype *&h_v_candidate_us_, numtype *&h_num_v_candidate_us_,
		vtype *&d_v_candidate_us_, numtype *&d_num_v_candidate_us_);

#endif