#ifndef IO_H
#define IO_H

#include "cpuGraph.h"
#include "gpuGraph.h"

void readGraphToCPU(
    cpuGraph *graph,
    const char *filename);

void readGraphToCPU2(
    cpuGraph *graph,
    const char *filename);

void copyGraphToGPU(
    gpuGraph *gpuGraph,
    const cpuGraph *cpuGraph);

void allocateMemGPU(
    gpuGraph *gpuGraph,
    const cpuGraph *cpuGraph);

void copyGraphToCPU(
    gpuGraph *gpuGraph,
    cpuGraph *cpuGraph);

void copyMeta(cpuGraph *query, cpuGraph *data);
#endif