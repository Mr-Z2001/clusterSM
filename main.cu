#include <iostream>
#include <algorithm>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "globals.cuh"
#include "io.cuh"
#include "order.h"
#include "join.cuh"
#include "decycle.h"
#include "cuda_helpers.h"

#include "structure.cuh"
#include "filter.cuh"

#include "CLI11.hpp"

using std::cout;
using std::endl;

int main(int argc, char **argv)
{
	CLI::App app{"App description"};

	std::string query_path, data_path;
	// method = "BFS-DFS";
	// bool filtering_3rd = true, adaptive_ordering = true, load_balancing = true;
	// uint32_t filtering_order_start_v = UINT32_MAX;
	uint32_t gpu_num = 0u;
	// uint32_t threshold;
	app.add_option("-q", query_path, "query graph path")->required();
	app.add_option("-d", data_path, "data graph path")->required();
	// app.add_option("-m", method, "enumeration method");
	// app.add_option("--f3", filtering_3rd,
	//  "enable the third filtering step or not");
	// app.add_option("--f3start", filtering_order_start_v,
	//  "start vertex of the third filtering step");
	// app.add_option("--ao", adaptive_ordering, "enable adaptive ordering or not");
	// app.add_option("--lb", load_balancing, "enable load balancing or not");
	// app.add_option("-t", threshold, "threadshold <= 4")->required();
	app.add_option("--gpu", gpu_num, "gpu number");

	CLI11_PARSE(app, argc, argv);

	int device = gpu_num;
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
	cout << "Device " << device << ": " << prop.name << endl;
	cudaSetDevice(device);
	GPU_NUM = gpu_num;

#ifndef NDEBUG
	std::cout << "query: " << query_path << std::endl;
#endif

	cpuGraph hq_backup;
	gpuGraph dq_backup;
	cpuGraph hq, hg;
	gpuGraph dq, dg;
	// gpuGraph dg;
	hg.isQuery = false;
	readGraphToCPU2(&hq_backup, query_path.c_str());
	readGraphToCPU2(&hq, query_path.c_str());
	readGraphToCPU2(&hg, data_path.c_str());
	copyMeta(&hq_backup, &hg);

	// chage hq into spanning tree.
	decycle(&hq);
#ifndef NDEBUG
	std::cout << "decycle done" << std::endl;
	hq.Print();
#endif

	allocateMemGPU(&dq_backup, &hq_backup);
	allocateMemGPU(&dq, &hq);
	allocateMemGPU(&dg, &hg);

#ifndef NDEBUG
	std::cout << "allocate done" << std::endl;
#endif
	copyGraphToGPU(&dq_backup, &hq_backup);
	copyGraphToGPU(&dq, &hq);
	copyGraphToGPU(&dg, &hg);
#ifndef NDEBUG
	std::cout << "copy done" << std::endl;
#endif

	vtype *h_u_candidate_vs_ = nullptr, *d_u_candidate_vs_ = nullptr;
	numtype *h_num_u_candidate_vs_ = new numtype[NUM_VQ];
	memset(h_num_u_candidate_vs_, 0, sizeof(numtype) * NUM_VQ);
	numtype *d_num_u_candidate_vs_;
	cuchk(cudaMalloc((void **)&d_num_u_candidate_vs_, NUM_VQ * sizeof(numtype)));
	cuchk(cudaMemset((void **)d_num_u_candidate_vs_, 0, sizeof(numtype) * NUM_VQ));

	cuchk(cudaMalloc((void **)&d_u_candidate_vs_, sizeof(vtype) * NUM_VQ * MAX_L_FREQ));
	cuchk(cudaMemset(d_u_candidate_vs_, -1, sizeof(vtype) * NUM_VQ * MAX_L_FREQ));

	vtype *h_v_candidate_us_ = nullptr, *d_v_candidate_us_ = nullptr;
	cuchk(cudaMallocHost((void **)&h_v_candidate_us_, NUM_VD * NUM_VQ * sizeof(vtype)));
	cuchk(cudaMalloc((void **)&d_v_candidate_us_, NUM_VD * NUM_VQ * sizeof(vtype)));

	numtype *h_num_v_candidate_us_ = new numtype[NUM_VD];
	memset(h_num_v_candidate_us_, 0, sizeof(numtype) * NUM_VD);
	numtype *d_num_v_candidate_us_ = nullptr;
	cuchk(cudaMalloc((void **)&d_num_v_candidate_us_, NUM_VD * sizeof(numtype)));
	cuchk(cudaMemset(d_num_v_candidate_us_, 0, NUM_VD * sizeof(numtype)));

	// cluster structures.
	cpuCluster *cpu_clusters_ = nullptr;
	gpuCluster *gpu_clusters_ = nullptr;
	numtype num_clusters = 0;
	encodingMeta enc_meta;

	uint32_t *h_encodings_ = nullptr;
	uint32_t *d_encodings_ = nullptr;

#ifndef NDEBUG
	std::cout << "filter start" << std::endl;
#endif

	clusterFilter(&hq_backup, &dq_backup, &hq, &hg, &dq, &dg,
								// cluster related
								cpu_clusters_, gpu_clusters_, &num_clusters,
								h_encodings_, d_encodings_, &enc_meta,
								// return
								h_u_candidate_vs_, h_num_u_candidate_vs_,
								d_u_candidate_vs_, d_num_u_candidate_vs_);
	// filterCG(&hq, &hg, &dq, h_u_candidate_vs_, h_num_u_candidate_vs_, d_u_candidate_vs_);

#ifndef NDEBUG
	std::cout << std::dec << std::endl;
	std::cout << "filter done" << std::endl;
	enc_meta.print();
#endif

	vtype root = enc_meta.query_us_compact_[enc_meta.cluster_offsets_[enc_meta.num_clusters - 1]];
	etype *order = new vtype[NUM_EQ];
	getBFSEdgeOrder(&hq_backup, order, root);

#ifndef NDEBUG
	std::cout << "order done" << std::endl;
	for (int i = 0; i < NUM_EQ; ++i)
		std::cout << order[i] << " " << hq_backup.evv[order[i]].first << " " << hq_backup.evv[order[i]].second << std::endl;
	std::cout << std::endl;
#endif

	numtype MAX_RES = 500000;

	vtype *h_res_table = new vtype[MAX_RES * NUM_VQ]; // around 4.5 MB.
	numtype num_res = 0;
	join(
			&hq_backup, &hg, &dq_backup, &dg,
			order,
			h_res_table, num_res,

			&enc_meta,
			h_encodings_, d_encodings_,

			h_u_candidate_vs_, h_num_u_candidate_vs_,
			d_u_candidate_vs_, d_num_u_candidate_vs_,
			h_v_candidate_us_, h_num_v_candidate_us_,
			d_v_candidate_us_, d_num_v_candidate_us_,

			cpu_clusters_, gpu_clusters_);

	std::cout << "num_res: " << num_res << std::endl;

	if (h_v_candidate_us_)
		cuchk(cudaFreeHost(h_v_candidate_us_));
	if (h_num_v_candidate_us_ != nullptr)
		delete[] h_num_v_candidate_us_;
	if (h_num_u_candidate_vs_ != nullptr)
		delete[] h_num_u_candidate_vs_;
	// if (cpu_clusters_ != nullptr)
	// delete[] cpu_clusters_;
	// if (gpu_clusters_ != nullptr)
	// delete[] gpu_clusters_;
	if (h_encodings_ != nullptr)
		delete[] h_encodings_;
	// if (h_u_candidate_vs_ != nullptr)
	// delete[] h_u_candidate_vs_;
	if (order != nullptr)
		delete[] order;
	if (h_res_table != nullptr)
		delete[] h_res_table;

	// cpuRelation *cpu_relations_ = new cpuRelation[NUM_EQ * 2];
	// gpuRelation *gpu_relations_ = new gpuRelation[NUM_EQ * 2];

	// #ifndef NDEBUG
	// 	std::cout << "construct edge candidate start" << std::endl;
	// #endif
	// 	// construct edge candidate hashed tries.
	// 	constructEdgeCandidates(gpu_relations_, cpu_relations_,
	// 													&hq, &hg, &dq,
	// 													h_u_candidate_vs_, h_num_u_candidate_vs_, d_u_candidate_vs_);

	// #ifndef NDEBUG
	// 	std::cout << "construct edge candidates done" << std::endl;
	// #endif

	// #ifndef NDEBUG
	// 	std::cout << "start ordering" << std::endl;
	// #endif
	// 	// ordering
	// 	etype *order = new etype[NUM_EQ];
	// 	getBFSEdgeOrder(&hq, order, cpu_relations_);
	// #ifndef NDEBUG
	// 	std::cout << "ordering done" << std::endl;
	// 	for (int i = 0; i < NUM_EQ; ++i)
	// 	{
	// 		std::cout << order[i] << " " << hq.evv[order[i]].first << " " << hq.evv[order[i]].second << std::endl;
	// 	}
	// #endif

	// 	numtype num_res = 0;
	// 	numtype max_res = (uint32_t)1e5;
	// 	vtype *res_table = new vtype[max_res * NUM_VQ];

	// 	vtype *d_res_table = nullptr;
	// 	size_t d_res_table_pitch;
	// 	cuchk(cudaMallocPitch((void **)&d_res_table, &d_res_table_pitch, NUM_VQ * sizeof(vtype), max_res));
	// 	vtype *temp_res_table = nullptr;
	// 	size_t temp_res_table_pitch;
	// 	cuchk(cudaMallocPitch((void **)&temp_res_table, &temp_res_table_pitch, NUM_VQ * sizeof(vtype), max_res));

	// #ifndef NDEBUG
	// 	std::cout << "join start" << std::endl;
	// #endif
	// 	// join
	// 	joinCG(&hq, &hg, &dq,
	// 				 order,
	// 				 cpu_relations_, gpu_relations_,
	// 				 &num_res, res_table,
	// 				 d_res_table, d_res_table_pitch,
	// 				 temp_res_table, temp_res_table_pitch);
	// #ifndef NDEBUG
	// 	std::cout << "join done" << std::endl;
	// #endif

	// 	std::cout << num_res << std::endl;

	return 0;
}