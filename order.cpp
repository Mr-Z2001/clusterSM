#include "order.h"
#include "globals.cuh"
#include "structure.cuh"

#include <cstring>
#include <queue>
#include <iostream>
#include <set>

using namespace std;

#define NDEBUG

void getBFSorder(
    cpuGraph *g,
    vtype *order,
    vtype start_v)
{
#ifndef NDEBUG
  std::cout << "offsets_:" << std::endl;
  for (int i = 0; i < g->num_v; ++i)
    std::cout << g->vertexIDs_[i] << " ";
  std::cout << std::endl;
  for (int i = 0; i < NUM_VQ + 1; ++i)
    std::cout << g->offsets_[i] << " ";
  std::cout << std::endl;
#endif

  uint32_t num_v = g->num_v;

  bool vis[MAX_VQ];
  memset(vis, false, sizeof(bool) * MAX_VQ);
  int order_off = 0;

  // go bfs
  queue<vtype> q;
  vtype start_u = start_v;
  q.push(start_u);
  while (!q.empty())
  {
    vtype u = q.front();
    q.pop();
    vis[u] = true;

    order[order_off++] = u;

#ifndef NDEBUG
    std::cout << u << std::endl;
#endif

    vtype u_nxt = NUM_VQ;
    uint32_t u_off = g->get_u_off(u);
    if (u_off != num_v - 1)
      u_nxt = g->vertexIDs_[u_off + 1];

#ifndef NDEBUG
    std::cout << "u=" << u << ", u_nxt=" << u_nxt << std::endl;
    std::cout << "off_st=" << g->offsets_[u] << ", off_end=" << g->offsets_[u_nxt] << std::endl;
#endif
    for (uint32_t off = g->offsets_[u]; off < g->offsets_[u_nxt]; ++off)
    {
#ifndef NDEBUG
      std::cout << "off=" << off << std::endl;
#endif
      vtype u_nbr = g->neighbors_[off];
#ifndef NDEBUG
      std::cout << "u_nbr = " << u_nbr << std::endl;
#endif
      if (!vis[u_nbr] && u_nbr != UINT32_MAX)
        q.push(u_nbr);
    }
  }

#ifndef NDEBUG
  std::cout << "In getBFSorder(): " << std::endl;
  for (int i = 0; i < num_v; ++i)
    std::cout << order[i] << " ";
  std::cout << std::endl;
#endif
}

struct cmp
{
  bool operator()(std::pair<degtype, vtype> a, std::pair<degtype, vtype> b)
  {
    if (a.first == b.first)
      return a.second < b.second;
    return a.first > b.first;
  }
};

// designed for undirected graph
// if directed, need to modify some details. --> change of degree.
void getCFLorder(
    cpuGraph *q,
    vtype *order)
{
  bool *vis = new bool[NUM_VQ];
  memset(vis, false, sizeof(vis));
  degtype *out_deg_copy = new degtype[NUM_VQ];
  memcpy(out_deg_copy, q->outdeg_, sizeof(degtype) * NUM_VQ);

  offtype order_off = 0;

  bool is_leaf_left = true;
  while (is_leaf_left)
  {
    is_leaf_left = false;
    for (vtype u = 0; u < NUM_VQ; ++u)
    {
      if (vis[u])
        continue;
      if (out_deg_copy[u] == 1)
      {
        order[order_off++] = u;
        vis[u] = true;
        is_leaf_left = true;
        // decrease the out degree of the neighbor
        for (offtype off = q->offsets_[u]; off < q->offsets_[u + 1]; ++off)
        {
          vtype u_nbr = q->neighbors_[off];
          if (!vis[u_nbr])
            --out_deg_copy[u_nbr];
        }
      }
    }
  }

  std::priority_queue<std::pair<degtype, vtype>, std::vector<std::pair<degtype, vtype>>, cmp> pq;
  for (vtype u = 0; u < NUM_VQ; ++u)
  {
    if (!vis[u])
      pq.push(std::make_pair(out_deg_copy[u], u));
  }

  while (!pq.empty())
  {
    vtype u = pq.top().second;
    pq.pop();
    order[order_off++] = u;
  }

  reverse(order, order + NUM_VQ);

  delete[] vis;
  delete[] out_deg_copy;
}

// one direction of an undirected edge is enough.
void getBFSEdgeOrder(
    cpuGraph *g,
    etype *order,
    // cpuRelation *cpu_relations_,
    vtype start_v)
{
  uint32_t num_v = g->num_v;

  vtype start_v_nbr = g->neighbors_[g->offsets_[start_v]];

  // int min_num_relations = cpu_relations_[0].num_candidates_[0];
  // int min_relation_index = 0;

  // for (etype e = 0; e < g->num_e * 2; ++e)
  // {
  //   if (cpu_relations_[e].num_candidates_[0] < min_num_relations)
  //   {
  //     min_num_relations = cpu_relations_[e].num_candidates_[0];
  //     min_relation_index = e;
  //   }
  // }

  // start_v = cpu_relations_[min_relation_index].keys_[0];
  // start_v_nbr = cpu_relations_[min_relation_index].keys_[1];

  order[0] = g->vve[{start_v, start_v_nbr}];

  bool *vis = new bool[g->num_v];
  memset(vis, false, sizeof(bool) * g->num_v);
  int order_off = 1;

  // go bfs
  std::set<std::pair<vtype, vtype>> visited_edges;
  visited_edges.insert({start_v, start_v_nbr});

  queue<vtype> q;
  q.push(start_v);
  while (!q.empty())
  {
    vtype u = q.front();
    q.pop();
    vis[u] = true;

    for (uint32_t off = g->offsets_[u]; off < g->offsets_[u + 1]; ++off)
    {
      vtype u_nbr = g->neighbors_[off];
      if (!vis[u_nbr])
        q.push(u_nbr);
      if (visited_edges.find(std::make_pair(u_nbr, u)) == visited_edges.end() &&
          visited_edges.find(std::make_pair(u, u_nbr)) == visited_edges.end())
      {
        order[order_off++] = g->edgeIDs_[off];
        visited_edges.insert(std::make_pair(u, u_nbr));
      }
    }
  }
}