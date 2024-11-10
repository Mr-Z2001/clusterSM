#ifndef STRUCTURE_H
#define STRUCTURE_H

#include "globals.cuh"
#include "cpuGraph.h"
#include "gpuGraph.h"

#include <set>

struct cpuRelation;

struct gpuRelation // n*2 table.
{
public:
	gpuRelation();
	~gpuRelation();

	void copy_from_cpu(const cpuRelation &cpu_relation, bool copy_keys = false);
	void copy_to_cpu(cpuRelation &cpu_relation, bool copy_keys = false);

	vtype *keys_;							// size: 2
	numtype *num_candidates_; // size: 1
	vtype *candidate_vs_[2];	// size of `candidate_vs_[i]`: num_candidates_[i]
};

struct cpuRelation
{
public:
	cpuRelation();
	~cpuRelation();

	void copy_from_gpu(const gpuRelation &gpu_relation, bool copy_keys = false);
	void copy_to_gpu(gpuRelation &gpu_relation, bool copy_keys = false);

	vtype *keys_;							// size: 2
	numtype *num_candidates_; // size: 1
	vtype *candidate_vs_[2];	// size of `candidate_vs_[i]`: num_candidates_[i]
};

struct cpuCluster
{
	numtype num_query_us; // size: 1
	vtype *query_us_;			// `num_query_us` query vertices. 0-th vertex is the root, it has `num_query_us-1` neighbors.

	cpuCluster();
	~cpuCluster();

	cpuCluster &operator=(const cpuCluster &rhs);
};

struct gpuCluster
{
	numtype num_query_us; // size: 1
	vtype *query_us_;			// `num_query_us` query vertices. 0-th vertex is the root, it has `num_query_us-1` neighbors.

	gpuCluster();
	~gpuCluster();
};

struct encodingMeta
{
	numtype num_clusters;
	numtype *num_query_us_;		 // size: num_clusters
	numtype num_total_us;			 // num_total_us = sum(num_query_us_)
	numtype num_blocks;				 // num_blocks = ceil(sum(num_query_us_) / 32);
	vtype *query_us_compact_;	 // size: sum(num_query_us_)
	offtype *cluster_offsets_; // size: num_clusters
	bool *is_a_valid_cluster_;

	// layer info
	numtype num_layers;
	numtype *num_clusters_per_layer_;
	offtype *layer_offsets_;

	// merge
	numtype merge_count;						// how many times did the merge happen.
	numtype *merged_cluster_left_;	// cluster id
	numtype *merged_cluster_right_; // cluster id
	vtype *merged_cluster_vertex_;	// connection vertex
	numtype *merged_cluster_layer_; // which layer are those clusters in.

	// combine
	numtype combine_cnt;
	numtype *combine_cluster_out_;
	int *combine_checkpoints_;
	std::set<int> *combine_clusters_other_;
	int *combine_type_; // 0 means new, 1 means use old.

	encodingMeta();
	~encodingMeta();

	void init(cpuCluster *cpu_clusters_);
	void print();
};

#endif // STRUCTURE_H