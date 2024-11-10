#include "cpuGraph.h"
#include "globals.cuh"

#include <iostream>

cpuGraph::cpuGraph() : vve()
{
  num_v = 0;
  num_e = 0;
  largest_l = 0;
  // elCount = 0;
  maxDegree = 0;

  outdeg_ = nullptr;
  vLabels_ = nullptr;
  maxLabelFreq = 0;
  // eLabels = nullptr;

  vertexIDs_ = nullptr;
  offsets_ = nullptr;
  neighbors_ = nullptr;
  edgeIDs_ = nullptr;
  isQuery = true;
  keep = nullptr;
}

cpuGraph::~cpuGraph()
{
  if (outdeg_ != nullptr)
    delete[] outdeg_;
  if (vLabels_ != nullptr)
    delete[] vLabels_;
  if (vertexIDs_ != nullptr)
    delete[] vertexIDs_;
  if (offsets_ != nullptr)
    delete[] offsets_;
  if (neighbors_ != nullptr)
    delete[] neighbors_;
  if (edgeIDs_ != nullptr)
    delete[] edgeIDs_;
  if (keep != nullptr)
    delete[] keep;
}

uint32_t cpuGraph::get_u_off(vtype u)
{
  uint32_t u_off = UINT32_MAX;
  for (uint32_t i = 0; i < num_v; i++)
  {
    if (vertexIDs_[i] == u)
    {
      u_off = i;
      break;
    }
  }

  if (u_off == UINT32_MAX)
  {
    std::cerr << "ERROR! in get_u_off(): u_off = uint32max" << std::endl;
    exit(EXIT_FAILURE);
  }

  return u_off;
}

void cpuGraph::Print()
{
  std::cout << "============================\n";
  std::cout << "num_v: " << num_v << std::endl;
  std::cout << "num_e: " << num_e << std::endl;
  std::cout << "largest_l: " << largest_l << std::endl;
  std::cout << "maxDegree: " << maxDegree << std::endl;

  std::cout << "outdeg_: \n";
  for (int i = 0; i < NUM_VQ; ++i)
    std::cout << outdeg_[i] << " \n"[i == NUM_VQ - 1];
  std::cout << "vLabels: \n";
  for (int i = 0; i < NUM_VQ; ++i)
    std::cout << vLabels_[i] << " \n"[i == NUM_VQ - 1];
  std::cout << "maxLabelFreq: " << maxLabelFreq << std::endl;

  std::cout << "vertexIDs_: \n";
  for (int i = 0; i < num_v; ++i)
    std::cout << vertexIDs_[i] << " \n"[i == num_v - 1];
  std::cout << "offsets_: \n";
  for (int i = 0; i < NUM_VQ + 1; ++i)
    std::cout << offsets_[i] << " \n"[i == NUM_VQ];
  std::cout << "neighbors_: \n";
  for (int i = 0; i < num_e * 2; ++i)
    std::cout << neighbors_[i] << " \n"[i == num_e * 2 - 1];
  std::cout << "edgeIDs_: \n";
  for (int i = 0; i < num_e * 2; ++i)
    std::cout << edgeIDs_[i] << " \n"[i == num_e * 2 - 1];
  std::cout << "============================" << std::endl;
}