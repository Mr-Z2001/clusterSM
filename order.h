#ifndef ORDER_H
#define ORDER_H

#include "cpuGraph.h"
#include "structure.cuh"

void getBFSorder(
    cpuGraph *g,
    vtype *order,
    vtype start_v = 0);

void getCFLorder(
    cpuGraph *g,
    vtype *order);

void getBFSEdgeOrder(
    cpuGraph *g,
    etype *order,
    // cpuRelation *cpu_relations_,
    vtype start_v = 0);

#endif // ORDER_H