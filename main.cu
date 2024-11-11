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
	uint32_t gpu_num = 0u;
	app.add_option("-q", query_path, "query graph path")->required();
	app.add_option("-d", data_path, "data graph path")->required();
	app.add_option("--gpu", gpu_num, "gpu number");

	CLI11_PARSE(app, argc, argv);

	int device = gpu_num;
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
#ifndef NDEBUG
	cout << "Device " << device << ": " << prop.name << endl;
#endif
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
	std::cout << "MAX_L_FREQ: " << MAX_L_FREQ << std::endl;
	std::cout << "NUM_VLQ: " << NUM_VLQ << std::endl;
#endif

	vtype *h_u_candidate_vs_ = nullptr;
	vtype *d_u_candidate_vs_ = nullptr;
	vtype *h_v_candidate_us_ = nullptr; // the same as `bitmap_reverse`
	vtype *d_v_candidate_us_ = nullptr; // the same sa `d_bitmap_reverse`

	numtype *h_num_u_candidate_vs_ = nullptr;
	numtype *d_num_u_candidate_vs_ = nullptr;
	numtype *h_num_v_candidate_us_ = nullptr;
	numtype *d_num_v_candidate_us_ = nullptr;

	h_u_candidate_vs_ = new vtype[NUM_VQ * MAX_L_FREQ];
	memset(h_u_candidate_vs_, -1, sizeof(vtype) * NUM_VQ * MAX_L_FREQ);
	cuchk(cudaMalloc((void **)&d_u_candidate_vs_, sizeof(vtype) * NUM_VQ * MAX_L_FREQ));
	cuchk(cudaMemset(d_u_candidate_vs_, -1, sizeof(vtype) * NUM_VQ * MAX_L_FREQ));
	h_v_candidate_us_ = new vtype[NUM_VD];
	memset(h_v_candidate_us_, -1, sizeof(vtype) * NUM_VD);
	cuchk(cudaMalloc((void **)&d_v_candidate_us_, sizeof(vtype) * NUM_VD));
	cuchk(cudaMemset(d_v_candidate_us_, -1, sizeof(vtype) * NUM_VD));

	h_num_u_candidate_vs_ = new numtype[NUM_VQ];
	memset(h_num_u_candidate_vs_, 0, sizeof(numtype) * NUM_VQ);
	cuchk(cudaMalloc((void **)&d_num_u_candidate_vs_, sizeof(numtype) * NUM_VQ));
	cuchk(cudaMemset((void **)d_num_u_candidate_vs_, 0, sizeof(numtype) * NUM_VQ));
	h_num_v_candidate_us_ = new numtype[NUM_VD];
	memset(h_num_v_candidate_us_, 0, sizeof(numtype) * NUM_VD);
	cuchk(cudaMalloc((void **)&d_num_v_candidate_us_, NUM_VD * sizeof(numtype)));
	cuchk(cudaMemset(d_num_v_candidate_us_, 0, NUM_VD * sizeof(numtype)));

	// cluster structures.
	cpuCluster *cpu_clusters_ = nullptr;
	gpuCluster *gpu_clusters_ = nullptr;
	encodingMeta enc_meta;

	uint32_t *h_encodings_ = nullptr;
	uint32_t *d_encodings_ = nullptr;

#ifndef NDEBUG
	std::cout << "filter start" << std::endl;
#endif

	clusterFilter(&hq_backup, &dq_backup, &hq, &hg, &dq, &dg,
								// cluster related
								cpu_clusters_, gpu_clusters_,
								h_encodings_, d_encodings_, &enc_meta,
								// return
								h_u_candidate_vs_, h_num_u_candidate_vs_,
								d_u_candidate_vs_, d_num_u_candidate_vs_,
								h_v_candidate_us_, h_num_v_candidate_us_,
								d_v_candidate_us_, d_num_v_candidate_us_);
	// filterCG(&hq, &hg, &dq, h_u_candidate_vs_, h_num_u_candidate_vs_, d_u_candidate_vs_);

#ifndef NDEBUG
	std::cout << std::dec << std::endl;
	std::cout << "filter done" << std::endl;
	enc_meta.print();

	for (int i = 0; i < h_num_u_candidate_vs_[3]; ++i)
		std::cout << h_u_candidate_vs_[3 * MAX_L_FREQ + i] << " ";
	std::cout << std::endl;
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

	return 0;
}