#include "filter.cuh"
#include "cuda_helpers.h"
#include "structure.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

#include <set>

#include "order.h"
#include "memManag.cuh"

#define row_size ((NUM_VD - 1) / 32 + 1)

/**
 * TODO: need a lot of optimization.
 */

__global__ void
oneRoundFilterCG(
    vltype *d_q_vLabels_, degtype *d_q_degrees_,
    vltype *d_v_labels_, degtype *d_v_degrees_,
    uint32_t *d_bitmap, size_t bitmap_pitch)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int idx = tid + bid * blockDim.x;
  int wid = tid / warpSize;
  int lid = tid % warpSize;
  int wid_g = idx / warpSize;

  __shared__ vltype s_q_vLabels[MAX_VQ];
  __shared__ degtype s_q_degrees[MAX_VQ];
  __shared__ uint32_t s_bitmap[MAX_VQ][BLOCK_DIM / 32]; // 512/32=16 elements one row per block. each element 32 vertices.

  if (tid < C_NUM_VQ)
  {
    s_q_vLabels[tid] = d_q_vLabels_[tid];
    s_q_degrees[tid] = d_q_degrees_[tid];
  }
  if (lid == 31)
    for (vtype u = 0; u < C_NUM_VQ; ++u)
      s_bitmap[u][wid] = 0;
  __syncthreads();

  vtype v = idx;
  vltype vlabel = UINT32_MAX;
  degtype deg = 0;
  if (v >= C_NUM_VD)
    v = UINT32_MAX;
  if (v != UINT32_MAX)
  {
    vlabel = d_v_labels_[v];
    deg = d_v_degrees_[v];
  }
  __syncthreads();

  if (v < C_NUM_VD)
    for (vtype u = 0; u < C_NUM_VQ; ++u)
    {
      if (s_q_vLabels[u] == vlabel &&
          s_q_degrees[u] <= deg)
        atomicOr(&s_bitmap[u][wid], 1 << lid);
    }
  __syncwarp();

  // Write shared memory to global memory in batch
  if (lid == 0 && v < C_NUM_VD)
  {
    for (vtype u = 0; u < C_NUM_VQ; ++u)
    {
      d_bitmap[u * bitmap_pitch / sizeof(uint32_t) + wid_g] = s_bitmap[u][wid];
    }
  }
  __syncwarp();
}

void getVertexCoverHeuristic(
    cpuGraph *hq,
    // return
    vtype *vertex_cover_, numtype *num_vertex_covers)
{
}

void getVertexCover(
    cpuGraph *hq,
    // return
    vtype *vertex_cover_, numtype *num_vertex_covers)
{
  std::set<vtype> vertex_cover;

  bool all_edges_visited = false;
  bool *vis_e = new bool[hq->num_e * 2];
  bool *vis_v = new bool[hq->num_v];
  memset(vis_e, false, sizeof(bool) * hq->num_e * 2);
  memset(vis_v, false, sizeof(bool) * hq->num_v);
  etype e = 0;
  while (!all_edges_visited)
  {
    if (e >= hq->num_e * 2)
    {
      all_edges_visited = true;
      break;
    }
    if (!hq->keep[e] || vis_e[e])
    {
      e += 2;
      continue;
    }
    vtype u = hq->evv[e].first;
    vtype v = hq->evv[e].second;
    vis_e[e] = true;
    if ((e & 1) == 0)
    {
      if (hq->outdeg_[u] == 1 || hq->outdeg_[v] == 1)
      {
        if (hq->outdeg_[u] == 1)
          vertex_cover.insert(v);
        else
          vertex_cover.insert(u);
        vis_v[u] = vis_v[v] = true;
      }
      // else
      // {
      //   vertex_cover.insert(u);
      //   vertex_cover.insert(v);
      // }
      else
      {
        bool flag_u_nbrs_all_visited = true;
        for (offtype u_nbr_off = hq->offsets_[u]; u_nbr_off < hq->offsets_[u + 1]; ++u_nbr_off)
        {
          vtype u_nbr = hq->neighbors_[u_nbr_off];
          if (!vis_v[u_nbr])
          {
            if (u_nbr == v)
              continue;
            flag_u_nbrs_all_visited = false;
            break;
          }
        }
        if (!flag_u_nbrs_all_visited)
        {
          vertex_cover.insert(u);
          vis_v[u] = true;
          bool flag_v_nbrs_all_visited = true;
          for (offtype v_nbr_off = hq->offsets_[v]; v_nbr_off < hq->offsets_[v + 1]; ++v_nbr_off)
          {
            vtype v_nbr = hq->neighbors_[v_nbr_off];
            if (!vis_v[v_nbr])
            {
              flag_v_nbrs_all_visited = false;
              break;
            }
          }
          if (!flag_v_nbrs_all_visited)
            vertex_cover.insert(v);
        }
        else // if (flag_u_nbrs_all_visited)
          vertex_cover.insert(v);
      }
      if (vertex_cover.find(u) != vertex_cover.end())
        for (offtype u_off = hq->offsets_[u]; u_off < hq->offsets_[u + 1]; ++u_off)
        {
          vtype u_nbr = hq->neighbors_[u_off];
          etype e_nbr = hq->vve[{u, u_nbr}];
          // if (!hq->keep[e_nbr])
          //   continue;
          vis_e[e_nbr] = true;
          etype e_nbr_reverse = hq->vve[{u_nbr, u}];
          // if (!hq->keep[e_nbr_reverse])
          //   continue;
          vis_e[e_nbr_reverse] = true;
        }
      if (vertex_cover.find(v) != vertex_cover.end())
        for (offtype v_off = hq->offsets_[v]; v_off < hq->offsets_[v + 1]; ++v_off)
        {
          vtype v_nbr = hq->neighbors_[v_off];
          etype e_nbr = hq->vve[{v, v_nbr}];
          // if (!hq->keep[e_nbr])
          //   continue;
          vis_e[e_nbr] = true;
          etype e_nbr_reverse = hq->vve[{v_nbr, v}];
          // if (!hq->keep[e_nbr_reverse])
          //   continue;
          vis_e[e_nbr_reverse] = true;
        }
    }
    e += 2;
  }
  delete[] vis_e;
  num_vertex_covers[0] = vertex_cover.size();
  int i = 0;
  for (auto v : vertex_cover)
    vertex_cover_[i++] = v;
}

// TODO: change it to an array of `T*`(i.e. T** for an 1-d array), save `new` and `delete` operations.
// make sure all type `T` support operator `=`
template <typename T>
inline void extendArray(T *&arr_old, numtype old_length, numtype addition_length = 1)
{
  T *arr_new = new T[old_length + addition_length];
  for (int i = 0; i < old_length; i++)
    arr_new[i] = arr_old[i];
  if (arr_old != nullptr)
    delete[] arr_old;
  arr_old = arr_new;
}

void clustering(
    cpuGraph *hq,
    cpuCluster *&cpu_clusters_, numtype *num_clusters,
    encodingMeta *enc_meta)
{
  vtype *vertex_cover_ = new vtype[hq->num_v];
  memset(vertex_cover_, 0, sizeof(vtype) * hq->num_v);
  numtype num_vertex_covers = 0;
  getVertexCover(hq, vertex_cover_, &num_vertex_covers);
#ifndef NDEBUG
  std::cout << "number of vertex covers: " << num_vertex_covers << std::endl;
  for (int i = 0; i < num_vertex_covers; i++)
    std::cout << vertex_cover_[i] << " ";
  std::cout << std::endl;
#endif

  // construct clusters
  cpu_clusters_ = new cpuCluster[num_vertex_covers];
  enc_meta->is_a_valid_cluster_ = new bool[num_vertex_covers];
  memset(enc_meta->is_a_valid_cluster_, true, sizeof(bool) * num_vertex_covers);
  *num_clusters = num_vertex_covers;
  for (int i = 0; i < *num_clusters; ++i)
  {
    cpu_clusters_[i].num_query_us[0] = hq->outdeg_[vertex_cover_[i]] + 1;
    cpu_clusters_[i].query_us_ = new vtype[cpu_clusters_[i].num_query_us[0]];
    cpu_clusters_[i].query_us_[0] = vertex_cover_[i];
    memcpy(&cpu_clusters_[i].query_us_[1],
           hq->neighbors_ + hq->offsets_[vertex_cover_[i]],
           sizeof(vtype) * (cpu_clusters_[i].num_query_us[0] - 1));
  }

  // join clusters
  vtype connection_vertex;
  numtype num_new_clusters = 0;
  numtype num_actual_new_clusters = 2; // assigned arbitrarily, larger than 1.
  int layer_index = 0;
  int *num_clusters_per_layer_ = new int[100]; // I don't know actually how many layers would be, but definitely less than 100.
  memset(num_clusters_per_layer_, 0, sizeof(int) * 100);
  num_clusters_per_layer_[layer_index] = *num_clusters;
  int i = 0;
  while (num_actual_new_clusters > 1)
  {
    num_actual_new_clusters = 0;
    num_new_clusters = 0;
    // join clusters in layer (i-1) to form new clusters in layer i
    for (; i < *num_clusters; ++i) // get cluster left
    {
      cpuCluster &cluster_i = cpu_clusters_[i];
      if (enc_meta->is_a_valid_cluster_[i] == false)
        continue;

      for (int j = i + 1; j < *num_clusters; ++j) // get cluster right
      {
        // join cluster_i and cluster_j
        cpuCluster &cluster_j = cpu_clusters_[j];
        if (enc_meta->is_a_valid_cluster_[j] == false)
          continue;

        for (uint32_t i_ptr = 1; i_ptr < cpu_clusters_[i].num_query_us[0]; ++i_ptr)
        {
          vtype u_i = cpu_clusters_[i].query_us_[i_ptr];
          for (uint32_t j_ptr = 0; j_ptr < cpu_clusters_[j].num_query_us[0]; ++j_ptr)
          {
            vtype u_j = cpu_clusters_[j].query_us_[j_ptr];

            if (u_i == u_j)
            {
              // if (cpu_clusters_[i].num_query_us[0] == 2 && cpu_clusters_[j].num_query_us[0] == 2 &&
              //     cpu_clusters_[i].query_us_[0] == cpu_clusters_[j].query_us_[1])
              // {
              //   std::cout << "continue" << std::endl;
              //   continue;
              // }

              connection_vertex = u_i;
              extendArray(cpu_clusters_, *num_clusters + num_new_clusters);
              extendArray(enc_meta->is_a_valid_cluster_, *num_clusters + num_new_clusters);
              num_new_clusters++;

              enc_meta->is_a_valid_cluster_[*num_clusters + num_new_clusters - 1] = true;
              auto &var_cluster = cpu_clusters_[*num_clusters + num_new_clusters - 1];
              if (i_ptr && j_ptr) // both are not core vertex
              {
                var_cluster.num_query_us[0] = 3;
                if (cpu_clusters_[i].query_us_[0] == cpu_clusters_[j].query_us_[0])
                  var_cluster.num_query_us[0]--;
                var_cluster.query_us_ = new vtype[var_cluster.num_query_us[0]];
                var_cluster.query_us_[0] = connection_vertex;
                var_cluster.query_us_[1] = cpu_clusters_[i].query_us_[0]; // core of i-th
                if (cpu_clusters_[i].query_us_[0] != cpu_clusters_[j].query_us_[0])
                  var_cluster.query_us_[2] = cpu_clusters_[j].query_us_[0]; // core of j-th
              }
              else if (!j_ptr) // u_j is the core vertex
              {
                var_cluster.num_query_us[0] = 2;
                var_cluster.query_us_ = new vtype[2];
                var_cluster.query_us_[0] = connection_vertex;
                var_cluster.query_us_[1] = cpu_clusters_[i].query_us_[0]; // core of i-th
              }
              else
              {
                std::cout << "unexpected case" << std::endl;
                std::cout << "i: " << i << " j: " << j << " i_ptr: " << i_ptr << " j_ptr: " << j_ptr << std::endl;
                std::cout << "u_i: " << u_i << " u_j: " << u_j << std::endl;
              }

#ifndef NDEBUG
              std::cout << "new cluster: ";
              std::cout << "num query us: " << cpu_clusters_[*num_clusters + num_new_clusters - 1].num_query_us[0] << " ";
              for (int k = 0; k < cpu_clusters_[*num_clusters + num_new_clusters - 1].num_query_us[0]; k++)
                std::cout << cpu_clusters_[*num_clusters + num_new_clusters - 1].query_us_[k] << " ";
              std::cout << std::endl;
#endif

              extendArray(enc_meta->merged_cluster_layer_, enc_meta->merge_count);
              extendArray(enc_meta->merged_cluster_left_, enc_meta->merge_count);
              extendArray(enc_meta->merged_cluster_right_, enc_meta->merge_count);
              extendArray(enc_meta->merged_cluster_vertex_, enc_meta->merge_count);

              enc_meta->merged_cluster_layer_[enc_meta->merge_count] = layer_index;
              enc_meta->merged_cluster_left_[enc_meta->merge_count] = i;
              enc_meta->merged_cluster_right_[enc_meta->merge_count] = j;
              enc_meta->merged_cluster_vertex_[enc_meta->merge_count] = connection_vertex;

              enc_meta->merge_count++;
            }
          }
        }
      }
    }
    num_actual_new_clusters = num_new_clusters;

    // combine
    for (int new_cluster_outer_ptr = *num_clusters; new_cluster_outer_ptr < *num_clusters + num_new_clusters; ++new_cluster_outer_ptr)
    {
      if (enc_meta->is_a_valid_cluster_[new_cluster_outer_ptr] == false)
        continue;
      vtype core_u_outer = cpu_clusters_[new_cluster_outer_ptr].query_us_[0];

      // scan for the inner, combine all clusters that have the same core vertex.
      std::set<int> to_combine_cluster_index;
      to_combine_cluster_index.insert(new_cluster_outer_ptr);
      std::set<vtype> core_nbrs;
      for (int i = 1; i < cpu_clusters_[new_cluster_outer_ptr].num_query_us[0]; ++i)
        core_nbrs.insert(cpu_clusters_[new_cluster_outer_ptr].query_us_[i]);
      for (int new_cluster_inner_ptr = new_cluster_outer_ptr + 1; new_cluster_inner_ptr < *num_clusters + num_new_clusters; ++new_cluster_inner_ptr)
      {
        if (enc_meta->is_a_valid_cluster_[new_cluster_inner_ptr] == false)
          continue;
        if (cpu_clusters_[new_cluster_inner_ptr].query_us_[0] == core_u_outer)
        {
          to_combine_cluster_index.insert(new_cluster_inner_ptr);
          for (int i = 1; i < cpu_clusters_[new_cluster_inner_ptr].num_query_us[0]; ++i)
            core_nbrs.insert(cpu_clusters_[new_cluster_inner_ptr].query_us_[i]);
        }
      }
      bool if_create_new = true;
      auto set_iterator = to_combine_cluster_index.begin();
      while (if_create_new &&
             (set_iterator != to_combine_cluster_index.end()))
      {
        int num_query_us = cpu_clusters_[*set_iterator].num_query_us[0];
        if_create_new = if_create_new && (num_query_us != core_nbrs.size() + 1);
        set_iterator++;
      }
      int largest_cluster_index;
      extendArray(enc_meta->combine_type_, enc_meta->combine_cnt);
      if (!if_create_new) // There exists a cluster contains all.
      {
        set_iterator--;
        largest_cluster_index = *set_iterator;
        to_combine_cluster_index.erase(largest_cluster_index);
        enc_meta->combine_type_[enc_meta->combine_cnt] = 1;
      }
      else // create new cluster. no cluster contains all others.
      {
        extendArray(cpu_clusters_, *num_clusters + num_new_clusters);
        extendArray(enc_meta->is_a_valid_cluster_, *num_clusters + num_new_clusters);
        num_new_clusters++;
        enc_meta->combine_type_[enc_meta->combine_cnt] = 0;

        enc_meta->is_a_valid_cluster_[*num_clusters + num_new_clusters - 1] = true;
        auto &var_cluster = cpu_clusters_[*num_clusters + num_new_clusters - 1];
        var_cluster.num_query_us[0] = core_nbrs.size() + 1;
        var_cluster.query_us_ = new vtype[var_cluster.num_query_us[0]];
        var_cluster.query_us_[0] = core_u_outer;
        int i = 1;
        for (auto core_nbr : core_nbrs)
          var_cluster.query_us_[i++] = core_nbr;

        largest_cluster_index = *num_clusters + num_new_clusters - 1;

#ifndef NDEBUG
        std::cout << "new cluster combine: ";
        std::cout << "num query us: " << cpu_clusters_[*num_clusters + num_new_clusters - 1].num_query_us[0] << " ";
        for (int k = 0; k < cpu_clusters_[*num_clusters + num_new_clusters - 1].num_query_us[0]; k++)
          std::cout << cpu_clusters_[*num_clusters + num_new_clusters - 1].query_us_[k] << " ";
        std::cout << std::endl;
#endif
      }

      set_iterator = to_combine_cluster_index.begin();
      extendArray(enc_meta->combine_cluster_out_, enc_meta->combine_cnt);
      extendArray(enc_meta->combine_clusters_other_, enc_meta->combine_cnt);
      enc_meta->combine_cluster_out_[enc_meta->combine_cnt] = largest_cluster_index;
      enc_meta->combine_clusters_other_[enc_meta->combine_cnt] = to_combine_cluster_index;

      num_actual_new_clusters -= to_combine_cluster_index.size();
      while (set_iterator != to_combine_cluster_index.end())
      {
        enc_meta->is_a_valid_cluster_[*set_iterator] = false;
        set_iterator++;
      }
      enc_meta->combine_cnt++;
    }

    layer_index++;
    *num_clusters += num_new_clusters;
    num_clusters_per_layer_[layer_index] = num_new_clusters;
    std::cout << "layer " << layer_index << " num_clusters: " << *num_clusters << std::endl;
  }

  // construct meta
  enc_meta->init(*num_clusters, cpu_clusters_);
  enc_meta->num_layers = layer_index;
  enc_meta->num_clusters_per_layer_ = new numtype[enc_meta->num_layers];
  memcpy(enc_meta->num_clusters_per_layer_, num_clusters_per_layer_, sizeof(numtype) * enc_meta->num_layers);

  delete[] vertex_cover_;
  delete[] num_clusters_per_layer_;
}

__global__ void
NLCFilter(
    offtype *d_offsets_, vtype *d_nbrs_, vltype *d_v_labels_, degtype *d_v_degrees_,
    uint32_t *d_query_NLC,
    uint32_t *d_bitmap, size_t bitmap_pitch,
    numtype *d_v_candidate_size_, vtype *d_v_candidate_us_)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int idx = tid + bid * blockDim.x;
  int wid = tid / warpSize;
  int lid = tid % warpSize;
  int wid_g = idx / warpSize;

  __shared__ uint32_t s_bitmap[MAX_VQ][BLOCK_DIM / 32]; // 512/32=16 elements one row per block. each element 32 vertices.
  __shared__ uint32_t s_d_NLC[WARP_PER_BLOCK][MAX_VLQ];
  __shared__ numtype s_q_NLC[MAX_VQ][MAX_VLQ];
  __shared__ uint32_t warp_pos[MAX_VQ][WARP_PER_BLOCK];

  if (tid / C_NUM_VLQ < C_NUM_VQ)
    s_q_NLC[tid / C_NUM_VLQ][tid % C_NUM_VLQ] = d_query_NLC[tid];
  if (tid / WARP_PER_BLOCK < C_NUM_VQ)
    warp_pos[tid / WARP_PER_BLOCK][tid % WARP_PER_BLOCK] = 0;
  __syncthreads();
  if (lid == 0 && idx < C_NUM_VD)
    for (vtype u = 0; u < C_NUM_VQ; ++u)
      s_bitmap[u][wid] = d_bitmap[u * bitmap_pitch / sizeof(uint32_t) + wid_g];
  __syncthreads();

  // one warp : 32 vertices.
  vtype v = wid_g * warpSize;

  while (v < C_NUM_VD)
  {
    // reuse s_d_NLC
    if (lid < MAX_VLQ)
      s_d_NLC[wid][lid] = 0;
    __syncwarp();

    offtype v_nbr_off = d_offsets_[v];
    offtype v_nbr_off_end = d_offsets_[v + 1];
    while (v_nbr_off + lid < v_nbr_off_end) // enter cond: valid nbr_offset.
    {
      vtype v_nbr = d_nbrs_[v_nbr_off + lid]; // one lane - one v_nbr
      vltype v_nbr_label = d_v_labels_[v_nbr];
      if (v_nbr_label < C_NUM_VLQ)
        atomicAdd(&s_d_NLC[wid][v_nbr_label], 1);
      v_nbr_off += warpSize; // needed when v_nbr_size > warpSize.
    }
    __syncwarp();

    // compare NLC
    for (uint32_t u = 0; u < C_NUM_VQ; ++u)
    {
      if (lid < C_NUM_VLQ)
      {
        // lid == v_label

        if ((s_bitmap[u][wid] & (1 << (v % warpSize))) == 0 ||
            s_d_NLC[wid][lid] < s_q_NLC[u][lid])
        {
          auto group = cooperative_groups::coalesced_threads();
          uint32_t leader = group.thread_rank() == 0;
          if (leader)
            s_bitmap[u][wid] &= ~(1 << (v % warpSize));
          group.sync();
        }
      }
      __syncwarp();
    }
    ++v;
  }

  // count candidate vertices.
  if (wid_g * 32 < C_NUM_VD)
  {
    // lid == u
    if (lid < C_NUM_VQ)
    {
      if (s_bitmap[lid][wid] & (1 << (v % warpSize)))
      {
        auto group = cooperative_groups::coalesced_threads();
        int rank = group.thread_rank();
        d_v_candidate_us_[v] |= (1 << lid);
        if (rank == 0)
          d_v_candidate_size_[v] = group.size();
        group.sync();
      }
    }
  }
  __syncthreads();

} // NLCFilterCluster

__global__ void
oneRoundFilterBidirectionKernel(
    // structure info
    vltype *query_vLabels_, degtype *query_out_degrees_,
    offtype *d_offsets_, vtype *d_nbrs_, vltype *d_v_labels_, degtype *d_v_degrees_,

    uint32_t *d_bitmap_, size_t bitmap_pitch,
    uint32_t *d_bitmap_reverse_, size_t bitmap_reverse_pitch,

    vtype *d_u_candidate_vs_, numtype *d_num_u_candidate_vs_,
    vtype *d_v_candidate_us_, numtype *d_num_v_candidate_us_,

    numtype *d_query_nlc_table_)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int idx = tid + bid * blockDim.x;
  int wid = tid / warpSize;
  int lid = tid % warpSize;
  int wid_g = idx / warpSize;

  __shared__ vltype s_q_vlabels[MAX_VQ];
  __shared__ degtype s_q_degs[MAX_VQ];
  __shared__ uint32_t s_bitmap[MAX_VQ][BLOCK_DIM / 32];
  __shared__ uint32_t s_bitmap_reverse[BLOCK_DIM];
  __shared__ uint32_t s_d_nlc_table[WARP_PER_BLOCK][MAX_VLQ];
  __shared__ numtype s_q_nlc_table[MAX_VQ][MAX_VLQ];
  __shared__ uint32_t warp_pos[MAX_VQ][WARP_PER_BLOCK];

  if (tid < C_NUM_VQ)
  {
    s_q_degs[tid] = query_out_degrees_[tid];
    s_q_vlabels[tid] = query_vLabels_[tid];
  }
  if (tid < C_NUM_VQ * C_NUM_VLQ)
    s_q_nlc_table[tid / C_NUM_VLQ][tid % C_NUM_VLQ] = d_query_nlc_table_[tid];
  if (lid < MAX_VQ)
    warp_pos[lid][wid] = 0;
  if (lid < MAX_VLQ)
    s_d_nlc_table[wid][lid] = 0;
  if (lid < MAX_VQ)
    s_bitmap[lid][wid] = 0;
  s_bitmap_reverse[tid] = 0;
  __syncthreads();

  vtype v = wid_g * warpSize;
  vtype v_end = min(v + 32, C_NUM_VD);
  while (v < v_end)
  {
    if (lid < C_NUM_VLQ)
      s_d_nlc_table[wid][lid] = 0;
    __syncwarp();

    // build data nlc table
    offtype v_nbr_off = d_offsets_[v];
    offtype v_nbr_off_end = d_offsets_[v + 1];
    offtype my_off = v_nbr_off + lid;
    while (my_off < v_nbr_off_end)
    {
      auto group = cooperative_groups::coalesced_threads();
      vtype v_nbr = d_nbrs_[my_off];
      vltype v_nbr_label = d_v_labels_[v_nbr];
      if (v_nbr_label < C_NUM_VLQ)
        atomicAdd(&s_d_nlc_table[wid][v_nbr_label], 1);
      group.sync();
      my_off += warpSize;
    }
    __syncwarp();

    for (vtype u = 0; u < C_NUM_VQ; ++u)
    {
      if (s_q_degs[u] <= d_v_degrees_[v] &&
          s_q_vlabels[u] == d_v_labels_[v])
      {
        // lid == vLabel
        if (lid < C_NUM_VLQ)
        {
          auto group = cooperative_groups::coalesced_threads();
          int mask = group.all(s_d_nlc_table[wid][lid] >=
                               d_query_nlc_table_[u * C_NUM_VLQ + lid]);
          if (mask && group.thread_rank() == 0)
          {
            atomicOr(&s_bitmap[u][wid], (1 << (v % 32)));
            atomicOr(&s_bitmap_reverse[v % BLOCK_DIM], (1 << (u % 32)));
          }
          group.sync();
        }
      }
      __syncwarp();
    }
    v++;
  }
  __syncwarp();

  for (vtype u = 0; u < C_NUM_VQ; ++u)
  {
    if (s_bitmap[u][wid] & (1 << lid))
    {
      auto group = cooperative_groups::coalesced_threads();
      int rank = group.thread_rank();
      if (rank == 0)
        warp_pos[u][wid] = atomicAdd(&d_num_u_candidate_vs_[u], group.size());
      group.sync();
      int my_pos = warp_pos[u][wid] + rank;
      d_u_candidate_vs_[u * C_MAX_L_FREQ + my_pos] = wid_g * warpSize + lid;
    }
    __syncwarp();
  }
  __syncthreads();
  if (idx < C_NUM_VD)
    d_num_v_candidate_us_[idx] = __popc(s_bitmap_reverse[tid]);
  __syncthreads();

  if (idx < C_NUM_VD)
    d_v_candidate_us_[idx] = s_bitmap_reverse[tid];
  __syncthreads();
}

void oneRoundFilterBidirection(
    cpuGraph *hq, cpuGraph *hg,
    gpuGraph *dq, gpuGraph *dg,
    uint32_t *d_bitmap_, size_t bitmap_pitch,
    uint32_t *d_bitmap_reverse_, size_t bitmap_reverse_pitch,

    vtype *d_u_candidate_vs_, numtype *d_num_u_candidate_vs_,
    vtype *d_v_candidate_us_, numtype *d_num_v_candidate_us_)
{
  numtype *h_query_nlc = nullptr;
  cuchk(cudaMallocHost((void **)&h_query_nlc, sizeof(numtype) * NUM_VQ * NUM_VLQ));
  memset(h_query_nlc, 0, sizeof(numtype) * NUM_VQ * NUM_VLQ);

  numtype *d_query_nlc = nullptr;
  cuchk(cudaMalloc((void **)&d_query_nlc, sizeof(numtype) * NUM_VQ * NUM_VLQ));

  for (vtype u = 0; u < NUM_VQ; ++u)
  {
    uint32_t NLC_offset = u * NUM_VLQ;
    for (offtype off = hq->offsets_[u]; off < hq->offsets_[u + 1]; ++off)
    {
      vtype nbr = hq->neighbors_[off];
      vltype vlabel = hq->vLabels_[nbr];
      h_query_nlc[NLC_offset + vlabel]++;
    }
  }

  oneRoundFilterBidirectionKernel<<<GRID_DIM, BLOCK_DIM>>>(
      dq->vLabels_, dq->degree_,
      dg->offsets_, dg->neighbors_, dg->vLabels_, dg->degree_,
      d_bitmap_, bitmap_pitch,
      d_bitmap_reverse_, bitmap_reverse_pitch,
      d_u_candidate_vs_, d_num_u_candidate_vs_,
      d_v_candidate_us_, d_num_v_candidate_us_,
      d_query_nlc);
  cuchk(cudaDeviceSynchronize());
}

void oneRoundFilterReverse(
    cpuGraph *hq, cpuGraph *hg,
    gpuGraph *dq, gpuGraph *dg,
    uint32_t *d_bitmap_, size_t bitmap_pitch,
    vtype *d_v_candidate_us_, numtype *d_num_v_candidates_)
{
  // ldf
  oneRoundFilterCG<<<GRID_DIM, BLOCK_DIM>>>(
      dq->vLabels_, dq->degree_,
      dg->vLabels_, dg->degree_,
      d_bitmap_, bitmap_pitch);
  cuchk(cudaDeviceSynchronize());

  // nlf
  numtype *h_query_nlc = nullptr;
  cuchk(cudaMallocHost((void **)&h_query_nlc, sizeof(numtype) * NUM_VQ * NUM_VLQ));
  memset(h_query_nlc, 0, sizeof(numtype) * NUM_VQ * NUM_VLQ);

  numtype *d_query_nlc = nullptr;
  cuchk(cudaMalloc((void **)&d_query_nlc, sizeof(numtype) * NUM_VQ * NUM_VLQ));

  for (vtype u = 0; u < NUM_VQ; ++u)
  {
    uint32_t NLC_offset = u * NUM_VLQ;
    for (offtype off = hq->offsets_[u]; off < hq->offsets_[u + 1]; ++off)
    {
      vtype nbr = hq->neighbors_[off];
      vltype vlabel = hq->vLabels_[nbr];
      h_query_nlc[NLC_offset + vlabel]++;
    }
  }
  cuchk(cudaMemcpy(d_query_nlc, h_query_nlc, sizeof(numtype) * NUM_VQ * NUM_VLQ, cudaMemcpyHostToDevice));
  NLCFilter<<<GRID_DIM, BLOCK_DIM>>>(
      dg->offsets_, dg->neighbors_, dg->vLabels_, dg->degree_,
      d_query_nlc,
      d_bitmap_, bitmap_pitch,
      d_num_v_candidates_, d_v_candidate_us_);
  cuchk(cudaDeviceSynchronize());
}

__global__ void
encodeKernel(
    // graph info
    offtype *d_offsets_, vtype *d_nbrs_,
    // candidate vertices
    vtype core_u, uint32_t layer_index, uint32_t cluster_index,
    vtype *d_u_candidate_vs_, numtype *d_num_u_candidate_vs_,
    vtype *d_v_candidate_us_, numtype *d_num_v_candidate_us_,

    // encoding info
    uint32_t *encodings_,

    numtype enc_num_clusters, numtype *enc_num_query_us_,
    numtype enc_num_total_us, numtype enc_num_bytes,
    vtype *enc_query_us_compact_, offtype *enc_cluster_offsets_,

    numtype enc_num_layers, numtype *enc_num_clusters_per_layer_,

    numtype enc_merge_count,
    numtype *enc_merged_cluster_left_, numtype *enc_merged_cluster_right_,
    vtype *enc_merged_cluster_vertex_, numtype *enc_merged_cluster_layer_)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int idx = tid + bid * blockDim.x;
  int wid = tid / warpSize;
  int lid = tid % warpSize;
  int wid_g = idx / warpSize;

  __shared__ vtype s_core_v[WARP_PER_BLOCK];

  if (wid_g < d_num_u_candidate_vs_[core_u])
  {
    if (lid == 0)
      s_core_v[wid] = d_u_candidate_vs_[core_u * C_MAX_L_FREQ + wid_g];
  }
  else
    return;
  __syncthreads();

  // encode core vertex
  vtype core_v = s_core_v[wid];
  uint32_t enc_pos = enc_cluster_offsets_[cluster_index];

  if (lid == 0)
    encodings_[core_v * enc_num_bytes + enc_pos / 32] |= 1 << (enc_pos % 32);
  __syncwarp();

  // encode core v's neighbors
  offtype nbr_off = d_offsets_[core_v] + lid;
  while (nbr_off < d_offsets_[core_v + 1])
  {
    vtype v_nbr = d_nbrs_[nbr_off];
    for (int i = 1; i < enc_num_query_us_[cluster_index]; ++i)
    {
      vtype tobe_map_u = enc_query_us_compact_[enc_pos + i];
      // TODO: optimize using bitmap. fast check.
      if (d_v_candidate_us_[v_nbr] & (1 << tobe_map_u))
        atomicOr(&encodings_[v_nbr * enc_num_bytes + (enc_pos + i) / 32], 1 << ((enc_pos + i) % 32));
      __syncwarp();
    }
    nbr_off += warpSize;
  }
  __syncwarp();
}

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
    numtype enc_num_total_us, numtype enc_num_bytes,
    vtype *enc_query_us_compact_, offtype *enc_cluster_offsets_,

    numtype enc_num_layers, numtype *enc_num_clusters_per_layer_,

    numtype enc_merge_count,
    numtype *enc_merged_cluster_left_, numtype *enc_merged_cluster_right_,
    vtype *enc_merged_cluster_vertex_, numtype *enc_merged_cluster_layer_)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int idx = tid + bid * blockDim.x;
  int wid = tid / warpSize;
  int lid = tid % warpSize;
  int wid_g = idx / warpSize;

  __shared__ vtype s_core_v[WARP_PER_BLOCK];
  __shared__ bool s_core_v_valid[WARP_PER_BLOCK];

  if (lid == 0)
    s_core_v_valid[wid] = false;

  if (wid_g < d_num_u_candidate_vs_[core_u])
  {
    if (lid == 0)
      s_core_v[wid] = d_u_candidate_vs_[core_u * C_MAX_L_FREQ + wid_g];
  }
  else
    return;
  __syncthreads();

  // encode core vertex
  vtype core_v = s_core_v[wid];
  uint32_t enc_pos = enc_cluster_offsets_[cluster_index];

  if (lid == 0 && core_v != UINT32_MAX)
  {
    if (encodings_[core_v * enc_num_bytes + left_pos / 32] & (1 << (left_pos % 32)) &&
        encodings_[core_v * enc_num_bytes + right_pos / 32] & (1 << (right_pos % 32)))
    {
      s_core_v_valid[wid] = true;
      encodings_[core_v * enc_num_bytes + enc_pos / 32] |= 1 << (enc_pos % 32);
    }
    else
    {
      s_core_v[wid] = UINT32_MAX;
      d_u_candidate_vs_[core_u * C_MAX_L_FREQ + wid_g] = UINT32_MAX;
    }
  }
  __syncwarp();

  if (core_v != UINT32_MAX)
  {
    offtype nbr_off = d_offsets_[core_v] + lid;
    while (nbr_off < d_offsets_[core_v + 1])
    {
      vtype v_nbr = d_nbrs_[nbr_off];
      for (int i = 1; i < enc_num_query_us_[cluster_index]; ++i)
      {
        vtype tobe_map_u = enc_query_us_compact_[enc_pos + i];
        if (d_v_candidate_us_[v_nbr] & (1 << tobe_map_u))
          atomicOr(&encodings_[v_nbr * enc_num_bytes + (enc_pos + i) / 32], 1 << ((enc_pos + i) % 32));
        // TODO: optimize using bitmap. fast check.
        __syncwarp();
      }
      nbr_off += warpSize;
    }
  }
  __syncwarp();
}

__global__ void
combineMultipleClustersKernel(
    vtype core_u, bool combine_type,
    int big_cluster, int *small_clusters_arr_, int num_small_clusters,
    uint32_t *d_encodings_,
    numtype num_clusters, numtype *num_query_us_,
    numtype num_total_us, numtype num_bytes,
    vtype *query_us_compact_, offtype *cluster_offsets_)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int idx = tid + bid * blockDim.x;
  int wid = tid / warpSize;
  int lid = tid % warpSize;
  int wid_g = idx / warpSize;

  __shared__ bool s_combine_type[1];
  __shared__ int s_big_pos[1];
  __shared__ int s_small_pos[32]; // I guess 32 is enough. Need test is for larger graphs.

  if (tid == 0)
    s_combine_type[0] = combine_type;
  if (tid == 0)
    s_big_pos[0] = cluster_offsets_[big_cluster];
  if (tid < num_small_clusters)
    s_small_pos[tid] = cluster_offsets_[small_clusters_arr_[tid]];
  __syncthreads();

  vtype v = idx;
  if (v >= C_NUM_VD)
    v = UINT32_MAX;
  __syncwarp();

  if (wid_g * warpSize >= C_NUM_VD)
    return;

  for (int i = 0; i < num_query_us_[big_cluster]; ++i)
  {
    int target_pos = s_big_pos[0] + i;
    vtype target_u = query_us_compact_[target_pos];
    int final_value = 1;
    for (int small_cluster_index = 0; small_cluster_index < num_small_clusters; ++small_cluster_index)
    {
      int small_pos = s_small_pos[small_cluster_index];
      while (small_pos < s_small_pos[small_cluster_index] + num_query_us_[small_clusters_arr_[small_cluster_index]])
      {
        if (target_u == query_us_compact_[small_pos])
        {
          if (v != UINT32_MAX)
            final_value &= (d_encodings_[v * num_bytes + small_pos / 32] & (1 << (small_pos % 32)));
          __syncwarp();
          break;
        }
        small_pos++;
      }
    }
    // no warp divergence here, all the threads take the same path.
    if (v != UINT32_MAX)
    {
      if (s_combine_type[0] == 1)
        d_encodings_[v * num_bytes + target_pos / 32] &= final_value << (target_pos % 32);
      else
        d_encodings_[v * num_bytes + target_pos / 32] |= final_value << (target_pos % 32);
    }
  }
}

void encode(
    gpuGraph *dg,
    cpuCluster *cpu_clusters_, gpuCluster *gpu_clusters_, numtype num_clusters,
    uint32_t *h_encodings_, uint32_t *d_encodings_, encodingMeta *encoding_meta,
    numtype num_layers, numtype *num_clusters_per_layer,
    vtype *d_u_candidate_vs_, numtype *d_num_u_candidate_vs_,
    vtype *d_v_candidate_us_, numtype *d_num_v_candidate_us_)
{
  // device encoding meta
  numtype *enc_d_num_query_us_;
  vtype *enc_d_query_us_compact_;
  offtype *enc_d_cluster_offsets_;
  numtype *enc_d_clusters_per_layer_;
  numtype *enc_d_merged_cluster_left_;
  numtype *enc_d_merged_cluster_right_;
  vtype *enc_d_merged_cluster_vertex_;
  numtype *enc_d_merged_cluster_layer_;

  // TODO: 合并内存分配，一次cudaMalloc()，然后用指针指一下。
  cuchk(cudaMalloc((void **)&enc_d_num_query_us_, sizeof(numtype) * encoding_meta->num_clusters));
  cuchk(cudaMemcpy(enc_d_num_query_us_, encoding_meta->num_query_us_, sizeof(numtype) * encoding_meta->num_clusters, cudaMemcpyHostToDevice));

  cuchk(cudaMalloc((void **)&enc_d_query_us_compact_, sizeof(vtype) * encoding_meta->num_total_us));
  cuchk(cudaMemcpy(enc_d_query_us_compact_, encoding_meta->query_us_compact_, sizeof(vtype) * encoding_meta->num_total_us, cudaMemcpyHostToDevice));

  cuchk(cudaMalloc((void **)&enc_d_cluster_offsets_, sizeof(offtype) * (encoding_meta->num_clusters + 1)));
  cuchk(cudaMemcpy(enc_d_cluster_offsets_, encoding_meta->cluster_offsets_, sizeof(offtype) * (encoding_meta->num_clusters + 1), cudaMemcpyHostToDevice));

  cuchk(cudaMalloc((void **)&enc_d_clusters_per_layer_, sizeof(numtype) * num_layers));
  cuchk(cudaMemcpy(enc_d_clusters_per_layer_, num_clusters_per_layer, sizeof(numtype) * num_layers, cudaMemcpyHostToDevice));

  cuchk(cudaMalloc((void **)&enc_d_merged_cluster_left_, sizeof(numtype) * encoding_meta->merge_count));
  cuchk(cudaMemcpy(enc_d_merged_cluster_left_, encoding_meta->merged_cluster_left_, sizeof(numtype) * encoding_meta->merge_count, cudaMemcpyHostToDevice));

  cuchk(cudaMalloc((void **)&enc_d_merged_cluster_right_, sizeof(numtype) * encoding_meta->merge_count));
  cuchk(cudaMemcpy(enc_d_merged_cluster_right_, encoding_meta->merged_cluster_right_, sizeof(numtype) * encoding_meta->merge_count, cudaMemcpyHostToDevice));

  cuchk(cudaMalloc((void **)&enc_d_merged_cluster_vertex_, sizeof(vtype) * encoding_meta->merge_count));
  cuchk(cudaMemcpy(enc_d_merged_cluster_vertex_, encoding_meta->merged_cluster_vertex_, sizeof(vtype) * encoding_meta->merge_count, cudaMemcpyHostToDevice));

  cuchk(cudaMalloc((void **)&enc_d_merged_cluster_layer_, sizeof(numtype) * encoding_meta->merge_count));
  cuchk(cudaMemcpy(enc_d_merged_cluster_layer_, encoding_meta->merged_cluster_layer_, sizeof(numtype) * encoding_meta->merge_count, cudaMemcpyHostToDevice));

  for (int cluster_index = 0; cluster_index < num_clusters_per_layer[0]; ++cluster_index)
  {
    vtype core_u = cpu_clusters_[cluster_index].query_us_[0];

    encodeKernel<<<GRID_DIM, BLOCK_DIM>>>(
        dg->offsets_, dg->neighbors_,
        core_u, 0, cluster_index,
        d_u_candidate_vs_, d_num_u_candidate_vs_,
        d_v_candidate_us_, d_num_v_candidate_us_,

        d_encodings_,
        encoding_meta->num_clusters, enc_d_num_query_us_,
        encoding_meta->num_total_us, encoding_meta->num_bytes,
        enc_d_query_us_compact_, enc_d_cluster_offsets_,

        encoding_meta->num_layers, enc_d_clusters_per_layer_,

        encoding_meta->merge_count,
        enc_d_merged_cluster_left_, enc_d_merged_cluster_right_,
        enc_d_merged_cluster_vertex_, enc_d_merged_cluster_layer_);
    cuchk(cudaDeviceSynchronize());
  }

#ifndef NDEBUG
  std::cout << "first layer encoding done" << std::endl;
#endif

  uint32_t layer_index = 1;
  uint32_t cluster_sum = num_clusters_per_layer[0];
  uint32_t combine_ptr = 0;

  int *d_small_clusters_;
  cuchk(cudaMalloc((void **)&d_small_clusters_, sizeof(int) * encoding_meta->num_clusters));

  for (int cluster_index = num_clusters_per_layer[0]; cluster_index < encoding_meta->num_clusters; ++cluster_index)
  {
    if (cluster_index == cluster_sum + num_clusters_per_layer[layer_index])
    {
      cluster_sum += num_clusters_per_layer[layer_index];
      layer_index++;

      // combine
      if (encoding_meta->combine_cnt != 0)
      {
        while (encoding_meta->combine_cluster_out_[combine_ptr] <= cluster_index)
        {
          int big_cluster = encoding_meta->combine_cluster_out_[combine_ptr];
          std::set<int> small_clusters = encoding_meta->combine_clusters_other_[combine_ptr];

          std::vector<int> small_clusters_vec = std::vector<int>(small_clusters.begin(), small_clusters.end());
          int *small_clusters_arr = small_clusters_vec.data();
          int num_small_clusters = small_clusters.size();

          cuchk(cudaMemcpy(d_small_clusters_, small_clusters_arr, sizeof(int) * num_small_clusters, cudaMemcpyHostToDevice));

          vtype core_u = cpu_clusters_[big_cluster].query_us_[0];

          combineMultipleClustersKernel<<<GRID_DIM, BLOCK_DIM>>>(
              core_u, encoding_meta->combine_type_[combine_ptr],
              big_cluster, d_small_clusters_, num_small_clusters,
              d_encodings_,
              encoding_meta->num_clusters, enc_d_num_query_us_,
              encoding_meta->num_total_us, encoding_meta->num_bytes,
              enc_d_query_us_compact_, enc_d_cluster_offsets_);
          cuchk(cudaDeviceSynchronize());
          ++combine_ptr;
        }
#ifndef NDEBUG
        std::cout << "combine for layer " << layer_index - 1 << " done" << std::endl;
#endif
      }
    }
    if (encoding_meta->is_a_valid_cluster_[cluster_index] == false)
      continue;
    vtype core_u = cpu_clusters_[cluster_index].query_us_[0];

    // merge clusters.
    int merge_index = cluster_index - num_clusters_per_layer[0];
    int left = encoding_meta->merged_cluster_left_[merge_index];
    int left_position = encoding_meta->cluster_offsets_[left];
    while (encoding_meta->query_us_compact_[left_position] != core_u)
      left_position++;
    int right = encoding_meta->merged_cluster_right_[merge_index];
    int right_position = encoding_meta->cluster_offsets_[right];
    while (encoding_meta->query_us_compact_[right_position] != core_u)
      right_position++;
    vtype vertex = encoding_meta->merged_cluster_vertex_[merge_index];
    int layer = encoding_meta->merged_cluster_layer_[merge_index];

    mergeKernel<<<GRID_DIM, BLOCK_DIM>>>(
        left_position, right_position,
        dg->offsets_, dg->neighbors_,
        core_u, 0, cluster_index,
        d_u_candidate_vs_, d_num_u_candidate_vs_,
        d_v_candidate_us_, d_num_v_candidate_us_,

        d_encodings_,
        encoding_meta->num_clusters, enc_d_num_query_us_,
        encoding_meta->num_total_us, encoding_meta->num_bytes,
        enc_d_query_us_compact_, enc_d_cluster_offsets_,

        encoding_meta->num_layers, enc_d_clusters_per_layer_,

        encoding_meta->merge_count,
        enc_d_merged_cluster_left_, enc_d_merged_cluster_right_,
        enc_d_merged_cluster_vertex_, enc_d_merged_cluster_layer_);
    cuchk(cudaDeviceSynchronize());
  }

  // #ifndef NDEBUG
  //   cuchk(cudaMemcpy(h_encodings_, d_encodings_, sizeof(uint32_t) * NUM_VD * encoding_meta->num_bytes, cudaMemcpyDeviceToHost));

  //   std::cout << std::hex;
  //   for (vtype v = 0; v < NUM_VD; ++v)
  //   {
  //     std::cout << "vertex " << v << ": ";
  //     for (int i = 0; i < encoding_meta->num_bytes; i++)
  //       std::cout << (int)h_encodings_[v * encoding_meta->num_bytes + i] << " ";
  //     std::cout << std::endl;
  //   }
  // #endif
}

void clusterFilter(
    cpuGraph *hq_backup, gpuGraph *dq_backup,
    cpuGraph *hq, cpuGraph *hg,
    gpuGraph *dq, gpuGraph *dg,

    // cluster related
    cpuCluster *&cpu_clusters_, gpuCluster *&gpu_clusters_,
    numtype *num_clusters,
    uint32_t *&h_encodings_, uint32_t *&d_encodings_,
    encodingMeta *encoding_meta,

    // return
    vtype *h_u_candidate_vs_, numtype *h_num_u_candidate_vs_,
    vtype *d_u_candidate_vs_, numtype *d_num_u_candidate_vs_)
{
  // clustering
  clustering(hq, cpu_clusters_, num_clusters, encoding_meta);

  h_encodings_ = new uint32_t[NUM_VD * encoding_meta->num_bytes];
  memset(h_encodings_, 0, sizeof(uint32_t) * NUM_VD * encoding_meta->num_bytes);
  cuchk(cudaMalloc((void **)&d_encodings_, sizeof(uint32_t) * NUM_VD * encoding_meta->num_bytes));
  cuchk(cudaMemset(d_encodings_, 0, sizeof(uint32_t) * NUM_VD * encoding_meta->num_bytes));

  // numtype *d_num_u_candidate_vs_;
  // cuchk(cudaMalloc((void **)&d_num_u_candidate_vs_, sizeof(numtype) * NUM_VQ));
  // cuchk(cudaMemcpy(d_num_u_candidate_vs_, h_num_u_candidate_vs_, sizeof(numtype) * NUM_VQ, cudaMemcpyHostToDevice));

  vtype *d_v_candidate_us_ = nullptr;
  numtype *d_num_v_candidate_us_ = nullptr;
  cuchk(cudaMalloc((void **)&d_v_candidate_us_, sizeof(vtype) * NUM_VD));
  cuchk(cudaMalloc((void **)&d_num_v_candidate_us_, sizeof(numtype) * NUM_VD));
  cuchk(cudaMemset(d_num_v_candidate_us_, 0, sizeof(numtype) * NUM_VD));

  uint32_t *d_bitmap_reverse = nullptr;
  size_t bitmap_reverse_pitch = 0;
  cuchk(cudaMallocPitch((void **)&d_bitmap_reverse, &bitmap_reverse_pitch, sizeof(uint32_t) * NUM_VQ, NUM_VD / 32));
  cuchk(cudaMemset2D(d_bitmap_reverse, bitmap_reverse_pitch, 0, sizeof(uint32_t) * NUM_VQ, NUM_VD / 32));

  uint32_t *d_bitmap = nullptr;
  size_t bitmap_pitch = 0;
  cuchk(cudaMallocPitch((void **)&d_bitmap, &bitmap_pitch, sizeof(uint32_t) * (NUM_VD - 1) / 32 + 1, NUM_VQ));
  cuchk(cudaMemset2D(d_bitmap, bitmap_pitch, 0, sizeof(uint32_t) * (NUM_VD - 1) / 32 + 1, NUM_VQ));

  oneRoundFilterBidirection(
      hq_backup, hg,
      dq_backup, dg,
      d_bitmap, bitmap_pitch,
      d_bitmap_reverse, bitmap_reverse_pitch,
      d_u_candidate_vs_, d_num_u_candidate_vs_,
      d_v_candidate_us_, d_num_v_candidate_us_);

  vtype *d_query_us_compact_ = nullptr;
  cuchk(cudaMalloc((void **)&d_query_us_compact_, sizeof(vtype) * encoding_meta->num_total_us));
  cuchk(cudaMemcpy(d_query_us_compact_, encoding_meta->query_us_compact_, sizeof(vtype) * encoding_meta->num_total_us, cudaMemcpyHostToDevice));

#ifndef NDEBUG
  std::cout << "one round filter done" << std::endl;
#endif

  encode(
      dg,
      cpu_clusters_, gpu_clusters_, *num_clusters,
      h_encodings_, d_encodings_, encoding_meta,
      encoding_meta->num_layers, encoding_meta->num_clusters_per_layer_,
      d_u_candidate_vs_, d_num_u_candidate_vs_,
      d_v_candidate_us_, d_num_v_candidate_us_);
#ifndef NDEBUG
  std::cout << "encode done" << std::endl;
#endif

  cuchk(cudaMemcpy(h_encodings_, d_encodings_, sizeof(uint32_t) * NUM_VD * encoding_meta->num_bytes, cudaMemcpyDeviceToHost));
  cuchk(cudaMemcpy(h_num_u_candidate_vs_, d_num_u_candidate_vs_, sizeof(numtype) * NUM_VQ, cudaMemcpyDeviceToHost));
  cuchk(cudaMemcpy(h_u_candidate_vs_, d_u_candidate_vs_, sizeof(vtype) * NUM_VQ * C_MAX_L_FREQ, cudaMemcpyDeviceToHost));

  // #ifndef NDEBUG
  //   std::cout << std::hex;
  //   for (vtype v = 0; v < NUM_VD; ++v)
  //   {
  //     std::cout << "vertex " << v << ": ";
  //     for (int i = 0; i < encoding_meta->num_bytes; i++)
  //       std::cout << (int)h_encodings_[v * encoding_meta->num_bytes + i] << " ";
  //     std::cout << std::endl;
  //   }
  // #endif

  // #ifndef NDEBUG
  //   int tot_can = 0;
  //   int tot_reduced = 0;
  //   for (vtype v = 0; v < NUM_VD; ++v)
  //   {
  //     int *flag = new int[NUM_VQ];
  //     memset(flag, -1, sizeof(int) * NUM_VQ);
  //     int reduced = 0;
  //     for (int i = encoding_meta->num_total_us - 1; ~i; --i)
  //     {
  //       vtype u = encoding_meta->query_us_compact_[i];
  //       if (h_encodings_[v * encoding_meta->num_bytes + i / 32] & (1 << (i % 32)))
  //       {
  //         if (flag[u] == 0)
  //           reduced++;
  //         flag[u] = 1;
  //       }
  //       else // no encoding
  //       {
  //         flag[u] = 0;
  //       }
  //     }
  //     int num_can = 0;
  //     for (int i = 0; i < NUM_VQ; ++i)
  //       num_can += flag[i] == 1;
  //     std::cout << "reduce rate: " << (double)reduced / (reduced + num_can) << std::endl;
  //     tot_can += num_can;
  //     tot_reduced += reduced;
  //   }
  //   double tot_reduced_rate = (double)tot_reduced / (tot_reduced + tot_can);
  //   std::cout << "total reduce rate: " << tot_reduced_rate << std::endl;
  // #endif
}
