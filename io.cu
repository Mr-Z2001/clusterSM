#include "io.cuh"
#include "globals.cuh"
#include "cuda_helpers.h"

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <unordered_set>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <unordered_map>

using std::cerr;
using std::cout;
using std::endl;

// no `eid` in the graph file.
// for an undirected edge, appear only once in the file. So we should store each edge twice.
void readGraphToCPU2(
		cpuGraph *graph,
		const char *filename)
{
	std::ifstream ifs(filename);
	if (!ifs.is_open())
	{
		cerr << "Error: Unable to open file " << filename << endl;
		exit(1);
	}
	char type;
	numtype num_v, num_e;
	ifs >> type >> num_v >> num_e;
	if (type != 't')
	{
		cerr << "Error: Expected 't' at the beginning \n"
				 << "While found " << type << endl;
		exit(1);
	}

	graph->num_v = num_v;
	graph->num_e = num_e;
	graph->largest_l = 0;

	graph->vertexIDs_ = new vtype[graph->num_v];
	graph->offsets_ = new offtype[graph->num_v + 1];
	graph->neighbors_ = new vtype[graph->num_e << 1];
	graph->edgeIDs_ = new etype[graph->num_e << 1];

	graph->outdeg_ = new degtype[graph->num_v];
	graph->vLabels_ = new vltype[graph->num_v];
	memset(graph->outdeg_, 0, sizeof(degtype) * graph->num_v);

	offtype *offs = new offtype[graph->num_v];
	memset(offs, 0, sizeof(offtype) * (graph->num_v));
	std::unordered_map<vltype, numtype> labelCount;
	graph->offsets_[0] = 0;
	etype eid_global = 0;

	while (ifs >> type)
	{
		if (type == 'v')
		{
			vtype vid;
			vltype vLabel;
			degtype deg;
			ifs >> vid >> vLabel >> deg;
			graph->vertexIDs_[vid] = vid;
			graph->vLabels_[vid] = vLabel;
			graph->outdeg_[vid] = deg;
			labelCount[vLabel]++;
			graph->offsets_[vid + 1] = graph->offsets_[vid] + deg;
		}
		else
		{
			vtype src, dst;
			// etype eid = eid_global++;
			ifs >> src >> dst;
			graph->vve.insert({std::pair(src, dst), eid_global});
			graph->evv.insert(std::make_pair(eid_global, std::make_pair(src, dst)));
			graph->vve.insert({std::pair(dst, src), eid_global + 1});
			graph->evv.insert(std::make_pair(eid_global + 1, std::make_pair(dst, src)));
			offtype off = graph->offsets_[src] + offs[src];
			graph->neighbors_[off] = dst;
			graph->edgeIDs_[off] = eid_global;
			offs[src]++;
			off = graph->offsets_[dst] + offs[dst];
			graph->neighbors_[off] = src;
			graph->edgeIDs_[off] = eid_global + 1;
			offs[dst]++;
			eid_global += 2;
		}
	}
	delete[] offs;
	for (auto l_c : labelCount)
	{
		graph->maxLabelFreq = std::max(graph->maxLabelFreq, l_c.second);
		graph->largest_l = std::max(graph->largest_l, l_c.first);
	}
	for (int i = 0; i < graph->num_v; i++)
		graph->maxDegree = std::max(graph->maxDegree, graph->outdeg_[i]);

	graph->keep = new bool[graph->num_e * 2];
	memset(graph->keep, false, sizeof(bool) * graph->num_e * 2);
}

// Up to 64 edges for query graph.
// If query graph is a complete graph, up to 11 vertices.

// `eid` in the file.
// for undirected edges appear twice in the file.
void readGraphToCPU(
		cpuGraph *graph,
		const char *filename)
{
	std::ifstream
			ifs(filename);
	if (!ifs.is_open())
	{
		cerr
				<< "Error: Unable to open file "
				<< filename
				<< endl;
		exit(1);
	}

	char type;
	numtype num_v, num_e;
	ifs >> type >> num_v >> num_e;
	if (type != 't')
	{
		cerr
				<< "Error: Expected 't' at the beginning \n"
				<< "While found " << type
				<< endl;
		exit(1);
	}

	// init.
	graph->num_v = num_v;
	graph->num_e = num_e;
	graph->largest_l = 0;

	graph->vertexIDs_ = new vtype[graph->num_v];
	graph->offsets_ = new offtype[graph->num_v + 1];
	graph->neighbors_ = new vtype[graph->num_e << 1];
	graph->edgeIDs_ = new etype[graph->num_e << 1];

	graph->outdeg_ = new degtype[graph->num_v];
	graph->vLabels_ = new vltype[graph->num_v];
	memset(graph->outdeg_, 0, sizeof(degtype) * graph->num_v);

	// v_id starts from 0 and is continuous
	vtype v_id;
	vltype vLabel;
	std::unordered_map<vltype, numtype> labelCount;
	for (vtype i = 0; i < graph->num_v; ++i)
	{
		ifs >> type >> v_id >> vLabel;
		if (type != 'v')
		{
			std::cerr << "Error: Invalid vertex label" << std::endl;
			exit(1);
		}
		graph->vertexIDs_[i] = v_id;
		graph->vLabels_[v_id] = vLabel;
		labelCount[vLabel]++;
	}
	for (auto l_c : labelCount)
	{
		graph->largest_l = std::max(graph->largest_l, l_c.first);
		graph->maxLabelFreq = std::max(graph->maxLabelFreq, l_c.second);
	}

	// edgeID grows from 0.

	offtype off = 0; // point to both `neighbors_` and `edgeIDs_`
	vtype lastSrc = 0, src, dst;
	etype eid;
	graph->offsets_[0] = 0;
	// cout << (graph->num_e << 1) << endl;
	for (etype i = 0; i < (graph->num_e << 1); ++i)
	{
		ifs >> type >> src >> dst >> eid;
		if (type != 'e')
		{
			std::cerr << "Error: Expected 'e' at the beginning. \n"
								<< "while found " << type
								<< std::endl;
			exit(1);
		}
		graph->vve.insert({std::pair(src, dst), eid});

		if (src == dst)
			continue;
		graph->neighbors_[off] = dst;
		graph->edgeIDs_[off] = eid;
		graph->outdeg_[src]++;
		// // graph->eLabels[off] = label;

		if (src != lastSrc)
		{
			graph->offsets_[src] = off;
			lastSrc = src;
		}

		++off;
	}
	graph->offsets_[graph->num_v] = off;
	// cout << "edge done" << endl; // pass
	for (offtype u_off = 0; u_off < graph->num_v; ++u_off)
		graph->maxDegree = std::max(graph->maxDegree, graph->outdeg_[u_off]);

	// cout << "read to cpu all done" << endl; // pass
}

void allocateMemGPU(
		gpuGraph *gpuGraph,
		const cpuGraph *cpuGraph)
{
	uint32_t num = cpuGraph->num_v;
	if (cpuGraph->isQuery)
		num = NUM_VQ;
	cuchk(cudaMalloc((void **)&gpuGraph->degree_, sizeof(degtype) * num));
	cuchk(cudaMalloc((void **)&gpuGraph->vLabels_, sizeof(vltype) * num));

	cuchk(cudaMalloc((void **)&gpuGraph->offsets_, sizeof(offtype) * (num + 1)));
	cuchk(cudaMalloc((void **)&gpuGraph->neighbors_, sizeof(vtype) * cpuGraph->num_e * 2));
	cuchk(cudaMalloc((void **)&gpuGraph->edgeIDs_, sizeof(etype) * cpuGraph->num_e * 2));
}

void copyGraphToGPU(
		gpuGraph *gpuGraph,
		const cpuGraph *cpuGraph)
{
	uint32_t num = cpuGraph->num_v;
	if (cpuGraph->isQuery)
		num = NUM_VQ;
	cuchk(cudaMemcpy(gpuGraph->degree_, cpuGraph->outdeg_, sizeof(uint32_t) * num, cudaMemcpyHostToDevice));
	cuchk(cudaMemcpy(gpuGraph->vLabels_, cpuGraph->vLabels_, sizeof(vltype) * num, cudaMemcpyHostToDevice));
	// cuchk(cudaMemcpy(gpuGraph->eLabels, cpuGraph->eLabels, sizeof(eltype) * cpuGraph->num_e * 2, cudaMemcpyHostToDevice));

	cuchk(cudaMemcpy(gpuGraph->offsets_, cpuGraph->offsets_, sizeof(uint32_t) * (num + 1), cudaMemcpyHostToDevice));
	cuchk(cudaMemcpy(gpuGraph->neighbors_, cpuGraph->neighbors_, sizeof(vtype) * cpuGraph->num_e * 2, cudaMemcpyHostToDevice));
	cuchk(cudaMemcpy(gpuGraph->edgeIDs_, cpuGraph->edgeIDs_, sizeof(etype) * cpuGraph->num_e * 2, cudaMemcpyHostToDevice));
}

void copyGraphToCPU(
		gpuGraph *gpuGraph,
		cpuGraph *cpuGraph)
{
	uint32_t num = NUM_VQ;
	cuchk(cudaMemcpy(cpuGraph->outdeg_, gpuGraph->degree_, sizeof(uint32_t) * num, cudaMemcpyDeviceToHost));
	cuchk(cudaMemcpy(cpuGraph->vLabels_, gpuGraph->vLabels_, sizeof(vltype) * num, cudaMemcpyDeviceToHost));
	// cuchk(cudaMemcpy(cpuGraph->eLabels, gpuGraph->eLabels, sizeof(eltype) * cpuGraph->num_e * 2, cudaMemcpyDeviceToHost));

	cuchk(cudaMemcpy(cpuGraph->offsets_, gpuGraph->offsets_, sizeof(uint32_t) * (num + 1), cudaMemcpyDeviceToHost));
	cuchk(cudaMemcpy(cpuGraph->neighbors_, gpuGraph->neighbors_, sizeof(vtype) * cpuGraph->num_e * 2, cudaMemcpyDeviceToHost));
	cuchk(cudaMemcpy(cpuGraph->edgeIDs_, gpuGraph->edgeIDs_, sizeof(etype) * cpuGraph->num_e * 2, cudaMemcpyDeviceToHost));
}

void copyMeta(cpuGraph *query, cpuGraph *data)
{
	NUM_VQ = query->num_v;
	NUM_EQ = query->num_e;
	NUM_VLQ = query->largest_l + 1;

	// make sure that label corresponding to max_l_freq appears in the query graph.
	// labels that not in the query are invalid, useless.
	std::unordered_set<vltype> valid_vLabels;
	for (int i = 0; i < query->num_v; ++i)
		valid_vLabels.insert(query->vLabels_[i]);
	std::map<vltype, numtype> labelMap;
	for (int i = 0; i < data->num_v; ++i)
		if (valid_vLabels.find(data->vLabels_[i]) != valid_vLabels.end())
			labelMap[data->vLabels_[i]]++;
	numtype maxFreq = 0;
	for (auto l_c : labelMap)
		maxFreq = std::max(maxFreq, l_c.second);

	MAX_L_FREQ = maxFreq;
	MAX_DEGREE = data->maxDegree;

	NUM_VD = data->num_v;
	NUM_ED = data->num_e;
	// NUM_VLD = data->largest_l;
	// NUM_ELD = data->elCount;

	cuchk(cudaMemcpyToSymbol(C_NUM_VQ, &NUM_VQ, sizeof(uint32_t)));
	cuchk(cudaMemcpyToSymbol(C_NUM_EQ, &NUM_EQ, sizeof(uint32_t)));
	cuchk(cudaMemcpyToSymbol(C_NUM_VLQ, &NUM_VLQ, sizeof(uint32_t)));
	// cuchk(cudaMemcpyToSymbol(C_NUM_ELQ, &NUM_ELQ, sizeof(uint32_t)));
	cuchk(cudaMemcpyToSymbol(C_MAX_L_FREQ, &MAX_L_FREQ, sizeof(uint32_t)));
	cuchk(cudaMemcpyToSymbol(C_MAX_DEGREE, &MAX_DEGREE, sizeof(uint32_t)));

	cuchk(cudaMemcpyToSymbol(C_NUM_VD, &NUM_VD, sizeof(uint32_t)));
	cuchk(cudaMemcpyToSymbol(C_NUM_ED, &NUM_ED, sizeof(uint32_t)));
	// cudaMemcpyToSymbol(&C_NUM_VLD, &NUM_VLD, sizeof(uint32_t));
	// cudaMemcpyToSymbol(&C_NUM_ELD, &NUM_ELD, sizeof(uint32_t));
}