#include "filter.cuh"
#include "cuda_helpers.h"
#include "structure.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

#include <set>
#include <stack>

#include "order.h"
#include "memManag.cuh"

#define row_size ((NUM_VD - 1) / 32 + 1)

/**
 * TODO: need a lot of optimization.
 */

void dfs_cvc(
    vtype cur_v,
    vtype cur_v_father,
    int **f,
    cpuGraph *hq)
{
  f[cur_v][0] = 0;
  f[cur_v][1] = 1;

  for (offtype v_nbr_off = hq->offsets_[cur_v]; v_nbr_off < hq->offsets_[cur_v + 1]; ++v_nbr_off)
  {
    vtype v_nbr = hq->neighbors_[v_nbr_off];
    if (v_nbr == cur_v_father)
      continue;
    dfs_cvc(v_nbr, cur_v, f, hq);
    f[cur_v][0] += f[v_nbr][1];
    f[cur_v][1] += min(f[v_nbr][0], f[v_nbr][1]);
  }
}

void getVertexCover(
    cpuGraph *hq,
    // return
    vtype *vertex_cover_, numtype *num_vertex_covers)
{
  int **f = new int *[MAX_VQ];
  for (int i = 0; i < MAX_VQ; ++i)
    f[i] = new int[2];

  vtype root = 0;
  for (vtype u = 1; u < hq->num_v; ++u)
  {
    if (hq->outdeg_[u] > hq->outdeg_[root])
      root = u;
  }
  dfs_cvc(root, -1, f, hq);
  *num_vertex_covers = min(f[root][0], f[root][1]);

  // top-down select vertex cover
  int cover_index = 0;
  std::stack<vtype> st;
  std::stack<vtype> st_father;
  std::stack<bool> stf_selected;
  st.push(root);
  st_father.push(-1);
  stf_selected.push(false);
  while (!st.empty())
  {
    vtype cur_v = st.top();
    st.pop();
    vtype cur_v_father = st_father.top();
    st_father.pop();
    bool father_selected = stf_selected.top();
    stf_selected.pop();

    if (f[cur_v][1] <= f[cur_v][0])
    {
      vertex_cover_[cover_index++] = cur_v;

      for (offtype v_nbr_off = hq->offsets_[cur_v]; v_nbr_off < hq->offsets_[cur_v + 1]; ++v_nbr_off)
      {
        vtype v_nbr = hq->neighbors_[v_nbr_off];
        if (v_nbr == cur_v_father)
          continue;
        st.push(v_nbr);
        st_father.push(cur_v);
        stf_selected.push(true);
      }
    }
    else
    {
      if (father_selected)
        father_selected = false;
      else // father NOT selected
      {
        vertex_cover_[cover_index++] = cur_v;
        father_selected = true;
      }
      for (offtype v_nbr_off = hq->offsets_[cur_v]; v_nbr_off < hq->offsets_[cur_v + 1]; ++v_nbr_off)
      {
        vtype v_nbr = hq->neighbors_[v_nbr_off];
        if (v_nbr == cur_v_father)
          continue;
        st.push(v_nbr);
        st_father.push(cur_v);
        stf_selected.push(father_selected);
      }
    }
  }
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
    cpuCluster *&cpu_clusters_,
    encodingMeta *enc_meta)
{
  // get vertex cover
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

  // construct clusters first layer
  numtype &num_clusters = enc_meta->num_clusters; // use reference for simplicity.
  num_clusters = num_vertex_covers;

  cpu_clusters_ = new cpuCluster[MAX_CLUSTERS];

  for (int i = 0; i < num_clusters; ++i)
  {
    // aliases for simplicity.
    vtype core_u = vertex_cover_[i];
    cpuCluster &cluster = cpu_clusters_[i];

    cluster.num_query_us = hq->outdeg_[core_u] + 1;
    cluster.query_us_ = new vtype[cluster.num_query_us];
    cluster.query_us_[0] = core_u;
    memcpy(&cluster.query_us_[1],
           hq->neighbors_ + hq->offsets_[core_u],
           sizeof(vtype) * (cluster.num_query_us - 1));
  }

  // join clusters
  vtype connection_vertex;
  numtype num_new_clusters = 0;
  numtype num_actual_new_clusters = 0; // `new_cluster - combined_cluster`, assigned arbitrarily, larger than 1.
  int layer_index = 0;

  enc_meta->num_clusters_per_layer_[0] = num_clusters;

  int i = 0;

  do // while(num_actual_new_clusters > 1)
  {
    num_new_clusters = 0;
    num_actual_new_clusters = 0;
    // join clusters in layer `k-1` to form new clusters in layer `k`
    for (; i < num_clusters; ++i) // get cluster left
    {
      cpuCluster &cluster_i = cpu_clusters_[i];
      if (enc_meta->is_a_valid_cluster_[i] == false)
        continue;
      vtype core_i = cluster_i.query_us_[0];

      for (int j = i + 1; j < num_clusters; ++j) // get cluster right
      {
        cpuCluster &cluster_j = cpu_clusters_[j];
        if (enc_meta->is_a_valid_cluster_[j] == false)
          continue;
        vtype core_j = cluster_j.query_us_[0];

        // join cluster_i and cluster_j
        for (uint32_t i_ptr = 1; i_ptr < cluster_i.num_query_us; ++i_ptr) // iterate on cluster_i query vertices
        {
          vtype u_i = cluster_i.query_us_[i_ptr];

          for (uint32_t j_ptr = 0; j_ptr < cluster_j.num_query_us; ++j_ptr) // iterate on cluster_j query vertices
          {
            vtype u_j = cluster_j.query_us_[j_ptr];

            if (u_i == u_j)
            {
              connection_vertex = u_i;
              num_new_clusters++;
              int new_cluster_index = num_clusters + num_new_clusters - 1;

              enc_meta->is_a_valid_cluster_[new_cluster_index] = true;
              cpuCluster &new_cluster = cpu_clusters_[new_cluster_index];

              if (i_ptr && j_ptr) // both are not core vertex
              {
                new_cluster.num_query_us = 3;
                if (core_i == core_j)
                  new_cluster.num_query_us--;
                new_cluster.query_us_ = new vtype[new_cluster.num_query_us];
                new_cluster.query_us_[0] = connection_vertex;
                new_cluster.query_us_[1] = core_i; // core of i-th
                if (core_i != core_j)
                  new_cluster.query_us_[2] = core_j; // core of j-th
              }
              else if (!j_ptr) // u_j is the core vertex
              {
                new_cluster.num_query_us = 2;
                new_cluster.query_us_ = new vtype[2];
                new_cluster.query_us_[0] = connection_vertex;
                new_cluster.query_us_[1] = core_i; // core of i-th
              }
              else
              {
                std::cerr << "unexpected case" << std::endl;
                std::cerr << "i: " << i << " j: " << j << " i_ptr: " << i_ptr << " j_ptr: " << j_ptr << std::endl;
                std::cerr << "u_i: " << u_i << " u_j: " << u_j << std::endl;
                exit(1);
              }

              // #ifndef NDEBUG
              //               std::cout << "new cluster: ";
              //               std::cout << "num query us: " << new_cluster.num_query_us << " ";
              //               for (int k = 0; k < new_cluster.num_query_us; k++)
              //                 std::cout << new_cluster.query_us_[k] << " ";
              //               std::cout << std::endl;
              // #endif

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

    // combine

    layer_index++;
    num_actual_new_clusters = num_new_clusters;
    enc_meta->combine_checkpoints_[layer_index] = enc_meta->num_clusters + num_new_clusters;

    uint32_t &comb_cnt = enc_meta->combine_cnt;

    for (int new_cluster_out_ptr = num_clusters; new_cluster_out_ptr < num_clusters + num_new_clusters; ++new_cluster_out_ptr) // iterate on new clusters out
    {
      if (enc_meta->is_a_valid_cluster_[new_cluster_out_ptr] == false)
        continue;
      cpuCluster &cluster_out = cpu_clusters_[new_cluster_out_ptr];
      vtype core_u_out = cluster_out.query_us_[0];

      bool duplicate = false; // is the `core_u_out` the unique core in this layer?

      // scan for the inner, combine all clusters that have the same core vertex.
      std::set<int> to_combine_cluster_index_set;
      to_combine_cluster_index_set.insert(new_cluster_out_ptr);
      std::set<vtype> core_nbrs_set;
      for (int i = 1; i < cluster_out.num_query_us; ++i)
        core_nbrs_set.insert(cluster_out.query_us_[i]);

      for (int new_cluster_in_ptr = new_cluster_out_ptr + 1; new_cluster_in_ptr < num_clusters + num_new_clusters; ++new_cluster_in_ptr) // iterate on new clusters in
      {
        if (enc_meta->is_a_valid_cluster_[new_cluster_in_ptr] == false)
          continue;

        cpuCluster &cluster_in = cpu_clusters_[new_cluster_in_ptr];
        vtype core_u_in = cluster_in.query_us_[0];

        if (core_u_in == core_u_out)
        {
          duplicate = true;
          to_combine_cluster_index_set.insert(new_cluster_in_ptr);
          for (int i = 1; i < cluster_in.num_query_us; ++i)
            core_nbrs_set.insert(cluster_in.query_us_[i]);
        }
      }

      if (!duplicate)
        continue;

      bool if_create_new = true;
      auto set_iterator = to_combine_cluster_index_set.begin();
      while (if_create_new &&
             (set_iterator != to_combine_cluster_index_set.end()))
      {
        int num_query_us = cpu_clusters_[*set_iterator].num_query_us;
        if_create_new = if_create_new && (num_query_us != core_nbrs_set.size() + 1);
        set_iterator++;
      }
      int largest_cluster_index;
      extendArray(enc_meta->combine_type_, comb_cnt);

      if (!if_create_new) // There exists a cluster contains all.
      {
        set_iterator--;
        largest_cluster_index = *set_iterator;
        to_combine_cluster_index_set.erase(largest_cluster_index);
        enc_meta->combine_type_[comb_cnt] = 1;
      }
      else // create new cluster. no cluster contains all others.
      {
        num_new_clusters++;
        num_actual_new_clusters++;
        enc_meta->combine_type_[comb_cnt] = 0;

        int new_index = num_clusters + num_new_clusters - 1;

        enc_meta->is_a_valid_cluster_[new_index] = true;
        cpuCluster &new_cluster = cpu_clusters_[new_index];
        new_cluster.num_query_us = core_nbrs_set.size() + 1;
        new_cluster.query_us_ = new vtype[new_cluster.num_query_us];
        new_cluster.query_us_[0] = core_u_out;
        int i = 1;
        for (auto core_nbr : core_nbrs_set)
          new_cluster.query_us_[i++] = core_nbr;

        largest_cluster_index = new_index;

        // #ifndef NDEBUG
        //         std::cout << "new cluster combine: ";
        //         std::cout << "num query us: " << new_cluster.num_query_us << " ";
        //         for (int k = 0; k < new_cluster.num_query_us; k++)
        //           std::cout << new_cluster.query_us_[k] << " ";
        //         std::cout << std::endl;
        // #endif
      }

      set_iterator = to_combine_cluster_index_set.begin();
      extendArray(enc_meta->combine_cluster_out_, comb_cnt);
      extendArray(enc_meta->combine_clusters_other_, comb_cnt);
      enc_meta->combine_cluster_out_[comb_cnt] = largest_cluster_index;
      enc_meta->combine_clusters_other_[comb_cnt] = to_combine_cluster_index_set;

      num_actual_new_clusters -= to_combine_cluster_index_set.size();
      while (set_iterator != to_combine_cluster_index_set.end())
      {
        enc_meta->is_a_valid_cluster_[*set_iterator] = false;
        set_iterator++;
      }
      comb_cnt++;
    }

    num_clusters += num_new_clusters;
    enc_meta->num_clusters_per_layer_[layer_index] = num_new_clusters;
    std::cout << "layer " << layer_index << " num_clusters: " << num_clusters << " num_actual_new_clusters: " << num_actual_new_clusters << std::endl;
  } while (num_actual_new_clusters > 1);

  enc_meta->combine_checkpoints_[layer_index] = num_clusters;

  for (int l_index = 1; l_index <= layer_index; ++l_index)
  {
    enc_meta->layer_offsets_[l_index] = enc_meta->layer_offsets_[l_index - 1] + enc_meta->num_clusters_per_layer_[l_index - 1];
  }
  enc_meta->layer_offsets_[layer_index + 1] = num_clusters;

#ifndef NDEBUG
  std::cout << "see checkpoints: " << std::endl;
  for (int i = 0; i < layer_index + 1; ++i)
  {
    std::cout << "Layer " << i << " checkpoint: " << enc_meta->combine_checkpoints_[i] << std::endl;
  }

  std::cout << "see layer index: " << std::endl;
  for (int i = 0; i <= layer_index; ++i)
  {
    std::cout << "Layer " << i << " offset: " << enc_meta->layer_offsets_[i] << std::endl;
  }

#endif

  // construct meta
  enc_meta->init(cpu_clusters_);
  enc_meta->num_layers = layer_index + 1;

  delete[] vertex_cover_;
}

__global__ void
oneRoundFilterBidirectionKernel(
    // structure info
    vltype *query_vLabels_, degtype *query_out_degrees_,
    offtype *d_offsets_, vtype *d_nbrs_, vltype *d_v_labels_, degtype *d_v_degrees_,

    uint32_t *d_bitmap_, size_t bitmap_pitch,
    uint32_t *d_bitmap_reverse_,

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
  __shared__ vltype s_d_vlabels_part[WARP_PER_BLOCK][WARP_SIZE];

  if (tid < C_NUM_VQ)
  {
    s_q_degs[tid] = query_out_degrees_[tid];
    s_q_vlabels[tid] = query_vLabels_[tid];
  }
  if (tid < C_NUM_VQ * C_NUM_VLQ)
    s_q_nlc_table[tid / C_NUM_VLQ][tid % C_NUM_VLQ] = d_query_nlc_table_[tid];
  if (lid < MAX_VQ)
    warp_pos[lid][wid] = 0;
  if (lid < MAX_VQ)
    s_bitmap[lid][wid] = 0;
  s_bitmap_reverse[tid] = 0;
  if (idx < C_NUM_VD)
    s_d_vlabels_part[wid][lid] = d_v_labels_[idx];
  __syncthreads();

  vtype v = wid_g * warpSize;
  vtype v_end = min(v + 32, C_NUM_VD); // exclusive
  while (v < v_end)
  {
    if (lid < C_NUM_VLQ)
      s_d_nlc_table[wid][lid] = 0;
    __syncwarp();

    if (s_d_vlabels_part[wid][v % warpSize] >= C_NUM_VLQ)
    {
      ++v;
      continue;
    }

    // build data nlc table
    offtype v_nbr_off = d_offsets_[v] + lid;
    offtype v_nbr_off_end = d_offsets_[v + 1];
    while (v_nbr_off < v_nbr_off_end)
    {
      auto group = cooperative_groups::coalesced_threads();
      vtype v_nbr = d_nbrs_[v_nbr_off];
      vltype v_nbr_label = d_v_labels_[v_nbr];
      if (v_nbr_label < C_NUM_VLQ)
        atomicAdd(&s_d_nlc_table[wid][v_nbr_label], 1); // `wid` is `v`
      group.sync();
      v_nbr_off += warpSize;
    }
    __syncwarp();

    for (vtype u = 0; u < C_NUM_VQ; ++u)
    {
      // all lanes take the same branch
      if (s_q_degs[u] <= d_v_degrees_[v] &&
          s_q_vlabels[u] == s_d_vlabels_part[wid][v % warpSize])
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
    ++v;
  }
  __syncwarp();

  // read from shared, write to global memory.
  for (vtype u = 0; u < C_NUM_VQ; ++u)
  {
    // lid: v%warpSize
    if (s_bitmap[u][wid] & (1 << lid)) // if v is a candidate vertex for u
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
    uint32_t *d_bitmap_reverse_,

    vtype *d_u_candidate_vs_, numtype *d_num_u_candidate_vs_,
    vtype *d_v_candidate_us_, numtype *d_num_v_candidate_us_)
{
  numtype *h_query_nlc = nullptr;
  h_query_nlc = new numtype[NUM_VQ * NUM_VLQ];
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

  oneRoundFilterBidirectionKernel<<<GRID_DIM, BLOCK_DIM>>>(
      dq->vLabels_, dq->degree_,
      dg->offsets_, dg->neighbors_, dg->vLabels_, dg->degree_,
      d_bitmap_, bitmap_pitch,
      d_bitmap_reverse_,
      d_u_candidate_vs_, d_num_u_candidate_vs_,
      d_v_candidate_us_, d_num_v_candidate_us_,
      d_query_nlc);
  cuchk(cudaDeviceSynchronize());

  cuchk(cudaFree(d_query_nlc));
  delete[] h_query_nlc;
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
    numtype enc_num_total_us, numtype enc_num_blocks,
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

  if (wid_g < d_num_u_candidate_vs_[core_u]) // one warp one cluster.
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
    encodings_[core_v * enc_num_blocks + enc_pos / 32] |= 1 << (enc_pos % 32);
  __syncwarp();

  // encode core v's neighbors
  offtype nbr_off = d_offsets_[core_v] + lid;
  while (nbr_off < d_offsets_[core_v + 1])
  {
    auto group = cooperative_groups::coalesced_threads();
    vtype v_nbr = d_nbrs_[nbr_off];
    for (int i = 1; i < enc_num_query_us_[cluster_index]; ++i)
    {
      vtype tobe_map_u = enc_query_us_compact_[enc_pos + i];
      // TODO: optimize using bitmap. fast check.
      if (d_v_candidate_us_[v_nbr] & (1 << tobe_map_u))
        atomicOr(&encodings_[v_nbr * enc_num_blocks + (enc_pos + i) / 32], 1 << ((enc_pos + i) % 32));
      group.sync();
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
    numtype enc_num_total_us, numtype enc_num_blocks,
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

  __shared__ vtype s_core_v[WARP_PER_BLOCK]; // one warp one cluster.
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
  __syncwarp();

  // encode core vertex
  vtype core_v = s_core_v[wid];
  uint32_t enc_pos = enc_cluster_offsets_[cluster_index];

  if (lid == 0 && core_v != UINT32_MAX)
  {
    if (encodings_[core_v * enc_num_blocks + left_pos / 32] & (1 << (left_pos % 32)) &&
        encodings_[core_v * enc_num_blocks + right_pos / 32] & (1 << (right_pos % 32)))
    {
      s_core_v_valid[wid] = true;
      encodings_[core_v * enc_num_blocks + enc_pos / 32] |= 1 << (enc_pos % 32);
    }
    else
    {
      s_core_v[wid] = UINT32_MAX;
      d_u_candidate_vs_[core_u * C_MAX_L_FREQ + wid_g] = UINT32_MAX;
    }
  }
  __syncwarp();

  core_v = s_core_v[wid];
  if (core_v != UINT32_MAX)
  {
    offtype nbr_off = d_offsets_[core_v] + lid;
    while (nbr_off < d_offsets_[core_v + 1])
    {
      auto group = cooperative_groups::coalesced_threads();
      vtype v_nbr = d_nbrs_[nbr_off];
      for (int i = 1; i < enc_num_query_us_[cluster_index]; ++i)
      {
        vtype tobe_map_u = enc_query_us_compact_[enc_pos + i];
        if (d_v_candidate_us_[v_nbr] & (1 << tobe_map_u))
          atomicOr(&encodings_[v_nbr * enc_num_blocks + (enc_pos + i) / 32], 1 << ((enc_pos + i) % 32));
        // TODO: optimize using bitmap. fast check.
        group.sync();
      }
      nbr_off += warpSize;
    }
  }
  __syncwarp();
}

__global__ void
combineMultipleClustersKernel(
    offtype *d_offsets_, vtype *d_nbrs_,
    vtype core_u, bool combine_type,
    int big_cluster, int *small_clusters_arr_, int num_small_clusters,
    uint32_t *d_encodings_,
    numtype num_clusters, numtype *num_query_us_,
    numtype num_total_us, numtype num_blocks,
    vtype *query_us_compact_, offtype *cluster_offsets_,
    vtype *d_u_candidate_vs_, numtype *d_num_u_candidate_vs_)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int idx = tid + bid * blockDim.x;
  int wid = tid / warpSize;
  int lid = tid % warpSize;
  int wid_g = idx / warpSize;

  __shared__ bool s_combine_type;
  __shared__ int s_big_pos;
  __shared__ int s_small_pos[32]; // I guess 32 is enough. Need test is for larger graphs.
  __shared__ vtype s_warp_v[WARP_PER_BLOCK];

  if (tid == 0)
    s_combine_type = combine_type;
  if (tid == 0)
    s_big_pos = cluster_offsets_[big_cluster];
  if (tid < num_small_clusters)
  {
    int small_cluster = small_clusters_arr_[tid];
    int pos = cluster_offsets_[small_cluster];
    s_small_pos[tid] = pos;
  }
  __syncthreads();

  // TODO: only pick core_u's candidates.
  if (wid_g < d_num_u_candidate_vs_[core_u])
  {
    if (lid == 0)
      s_warp_v[wid] = d_u_candidate_vs_[core_u * C_MAX_L_FREQ + wid_g];
  }
  else
    return;
  __syncwarp();

  if (s_warp_v[wid] == UINT32_MAX)
    return;

  // combine core_v
  if (lid < num_small_clusters) // one lid, one small cluster.core_u position.
  {
    auto group = cooperative_groups::coalesced_threads();
    int small_pos = s_small_pos[lid];
    int big_pos = s_big_pos;
    int enc = 1;
    enc = enc && (d_encodings_[s_warp_v[wid] * num_blocks + small_pos / 32] & (1 << (small_pos % 32)));

    uint32_t mask = group.all(enc);
    if (mask)
    {
      if (s_combine_type == 1) // old cluster.
        d_encodings_[s_warp_v[wid] * num_blocks + big_pos / 32] &= 1 << (big_pos % 32);
      else // new cluster
        d_encodings_[s_warp_v[wid] * num_blocks + big_pos / 32] |= 1 << (big_pos % 32);
    }
  }

  // combine core_v_nbrs

  offtype v_nbr_off = d_offsets_[s_warp_v[wid]] + lid;
  offtype v_nbr_off_end = d_offsets_[s_warp_v[wid] + 1];
  while (v_nbr_off < v_nbr_off_end)
  {
    auto group = cooperative_groups::coalesced_threads();
    vtype v = d_nbrs_[v_nbr_off];
    for (int i = 1; i < num_query_us_[big_cluster]; ++i)
    {
      int target_pos = s_big_pos + i;
      vtype target_u = query_us_compact_[target_pos];
      int final_value = 1;
      for (int small_cluster_index = 0; small_cluster_index < num_small_clusters; ++small_cluster_index)
      {
        int small_pos = s_small_pos[small_cluster_index];
        while (small_pos < s_small_pos[small_cluster_index] + num_query_us_[small_clusters_arr_[small_cluster_index]])
        {
          if (target_u == query_us_compact_[small_pos])
          {
            final_value = final_value && (d_encodings_[v * num_blocks + small_pos / 32] & (1 << (small_pos % 32)));
            break;
          }
          small_pos++;
        }
        if (!final_value)
          break;
      }
      group.sync();
      // no warp divergence here, all the threads take the same path.
      if (s_combine_type == 1)
        d_encodings_[v * num_blocks + target_pos / 32] &= final_value << (target_pos % 32);
      else
        d_encodings_[v * num_blocks + target_pos / 32] |= final_value << (target_pos % 32);
    }

    v_nbr_off += warpSize;
  }
}

__global__ void
compactCandidatesKernel(
    vtype *d_u_candidate_vs_, numtype *d_num_u_candidate_vs_,
    vtype *d_u_candidate_vs_new_, numtype *d_num_u_candidate_vs_new_)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int idx = tid + bid * blockDim.x;
  int wid = tid / warpSize;
  int lid = tid % warpSize;
  int wid_g = idx / warpSize;

  __shared__ int warp_pos[WARP_PER_BLOCK];

  for (vtype u = 0; u < C_NUM_VQ; ++u)
  {
    if (idx < d_num_u_candidate_vs_[u])
    {
      vtype can_v = d_u_candidate_vs_[u * C_MAX_L_FREQ + idx];
      if (can_v != UINT32_MAX)
      {
        auto g = cooperative_groups::coalesced_threads();
        int rank = g.thread_rank();
        if (rank == 0)
          warp_pos[wid] = atomicAdd(&d_num_u_candidate_vs_new_[u], g.size());
        g.sync();
        int my_pos = warp_pos[wid] + rank;
        d_u_candidate_vs_new_[u * C_MAX_L_FREQ + my_pos] = can_v;
      }
    }
    __syncwarp();
  }
}

void encode(
    gpuGraph *dg,
    cpuCluster *cpu_clusters_, gpuCluster *gpu_clusters_,
    uint32_t *h_encodings_, uint32_t *d_encodings_, encodingMeta *enc_meta,
    vtype *d_u_candidate_vs_, numtype *d_num_u_candidate_vs_,
    vtype *d_v_candidate_us_, numtype *d_num_v_candidate_us_)
{
  // TODO: organize as a gpu_encoding_meta class.
  // device encoding meta
  uint32_t *enc_meta_buffer_;

  numtype *enc_d_num_query_us_;
  vtype *enc_d_query_us_compact_;
  offtype *enc_d_cluster_offsets_;
  numtype *enc_d_clusters_per_layer_;
  numtype *enc_d_merged_cluster_left_;
  numtype *enc_d_merged_cluster_right_;
  vtype *enc_d_merged_cluster_vertex_;
  numtype *enc_d_merged_cluster_layer_;

  uint32_t tot_mem_cnt = 0;
  tot_mem_cnt += enc_meta->num_clusters;
  tot_mem_cnt += enc_meta->num_total_us;
  tot_mem_cnt += enc_meta->num_clusters + 1;
  tot_mem_cnt += enc_meta->num_layers;
  tot_mem_cnt += enc_meta->merge_count;
  tot_mem_cnt += enc_meta->merge_count;
  tot_mem_cnt += enc_meta->merge_count;
  tot_mem_cnt += enc_meta->merge_count;
  tot_mem_cnt *= sizeof(uint32_t);

  cuchk(cudaMalloc((void **)&enc_meta_buffer_, tot_mem_cnt));
  uint32_t *current_ptr = enc_meta_buffer_;

  enc_d_num_query_us_ = current_ptr;
  cuchk(cudaMemcpy(enc_d_num_query_us_, enc_meta->num_query_us_, sizeof(numtype) * enc_meta->num_clusters, cudaMemcpyHostToDevice));
  current_ptr += enc_meta->num_clusters;

  enc_d_query_us_compact_ = current_ptr;
  cuchk(cudaMemcpy(enc_d_query_us_compact_, enc_meta->query_us_compact_, sizeof(vtype) * enc_meta->num_total_us, cudaMemcpyHostToDevice));
  current_ptr += enc_meta->num_total_us;

  enc_d_cluster_offsets_ = current_ptr;
  cuchk(cudaMemcpy(enc_d_cluster_offsets_, enc_meta->cluster_offsets_, sizeof(offtype) * (enc_meta->num_clusters + 1), cudaMemcpyHostToDevice));
  current_ptr += (enc_meta->num_clusters + 1);

  enc_d_clusters_per_layer_ = current_ptr;
  cuchk(cudaMemcpy(enc_d_clusters_per_layer_, enc_meta->num_clusters_per_layer_, sizeof(numtype) * enc_meta->num_layers, cudaMemcpyHostToDevice));
  current_ptr += enc_meta->num_layers;

  enc_d_merged_cluster_left_ = current_ptr;
  cuchk(cudaMemcpy(enc_d_merged_cluster_left_, enc_meta->merged_cluster_left_, sizeof(numtype) * enc_meta->merge_count, cudaMemcpyHostToDevice));
  current_ptr += enc_meta->merge_count;

  enc_d_merged_cluster_right_ = current_ptr;
  cuchk(cudaMemcpy(enc_d_merged_cluster_right_, enc_meta->merged_cluster_right_, sizeof(numtype) * enc_meta->merge_count, cudaMemcpyHostToDevice));
  current_ptr += enc_meta->merge_count;

  enc_d_merged_cluster_vertex_ = current_ptr;
  cuchk(cudaMemcpy(enc_d_merged_cluster_vertex_, enc_meta->merged_cluster_vertex_, sizeof(vtype) * enc_meta->merge_count, cudaMemcpyHostToDevice));
  current_ptr += enc_meta->merge_count;

  enc_d_merged_cluster_layer_ = current_ptr;
  cuchk(cudaMemcpy(enc_d_merged_cluster_layer_, enc_meta->merged_cluster_layer_, sizeof(numtype) * enc_meta->merge_count, cudaMemcpyHostToDevice));
  current_ptr += enc_meta->merge_count;

  // encode the first layer.
  for (int cluster_index = 0; cluster_index < enc_meta->num_clusters_per_layer_[0]; ++cluster_index)
  {
    vtype core_u = cpu_clusters_[cluster_index].query_us_[0];

    encodeKernel<<<GRID_DIM, BLOCK_DIM>>>(
        dg->offsets_, dg->neighbors_,
        core_u, 0, cluster_index,
        d_u_candidate_vs_, d_num_u_candidate_vs_,
        d_v_candidate_us_, d_num_v_candidate_us_,

        d_encodings_,
        enc_meta->num_clusters, enc_d_num_query_us_,
        enc_meta->num_total_us, enc_meta->num_blocks,
        enc_d_query_us_compact_, enc_d_cluster_offsets_,

        enc_meta->num_layers, enc_d_clusters_per_layer_,

        enc_meta->merge_count,
        enc_d_merged_cluster_left_, enc_d_merged_cluster_right_,
        enc_d_merged_cluster_vertex_, enc_d_merged_cluster_layer_);
    cuchk(cudaDeviceSynchronize());
  }

#ifndef NDEBUG
  std::cout << "first layer encoding done" << std::endl;
#endif

  uint32_t layer_index = 1;
  uint32_t cluster_sum = enc_meta->num_clusters_per_layer_[0];
  uint32_t combine_ptr = 0;

  int *d_small_clusters_;
  cuchk(cudaMalloc((void **)&d_small_clusters_, sizeof(int) * enc_meta->num_clusters));

  /**
   * from now on, each cluster is
   * (1) obtained by merging
   * (2) obtained by combining and that's a new cluster.
   * if it is obtained by merging,
   *    do not check if it is still valid.
   *    do a two-way intersection.
   * if it is obtained by combining,
   *    if it is not a new cluster, in-place intersection
   *    if it is a new cluster, combine.
   */
  for (int cluster_index = enc_meta->num_clusters_per_layer_[0]; cluster_index < enc_meta->num_clusters; ++cluster_index)
  {
    if (cluster_index == enc_meta->combine_checkpoints_[layer_index])
    {
      // combine
      if (enc_meta->combine_cnt != 0)
      {
        while (enc_meta->combine_cluster_out_[combine_ptr] < cluster_index && combine_ptr < enc_meta->combine_cnt)
        {
          int big_cluster = enc_meta->combine_cluster_out_[combine_ptr];
          std::set<int> small_clusters = enc_meta->combine_clusters_other_[combine_ptr];

          std::vector<int> small_clusters_vec = std::vector<int>(small_clusters.begin(), small_clusters.end());
          int *small_clusters_arr = small_clusters_vec.data();
          int num_small_clusters = small_clusters.size();

          if (num_small_clusters > 32)
          {
            std::cerr << "encoding: combing: num small cluster: " << num_small_clusters << std::endl;
            std::cerr << "too many small clusters" << std::endl;
            exit(1);
          }

#ifndef NDEBUG
          std::cout << "layer: " << layer_index << " ";
          std::cout << "small clusters: ";
          for (int i = 0; i < num_small_clusters; ++i)
            std::cout << small_clusters_arr[i] << " ";
          std::cout << std::endl;
#endif

          cuchk(cudaMemcpy(d_small_clusters_, small_clusters_arr, sizeof(int) * num_small_clusters, cudaMemcpyHostToDevice));

          vtype core_u = cpu_clusters_[big_cluster].query_us_[0];

          combineMultipleClustersKernel<<<GRID_DIM, BLOCK_DIM>>>(
              dg->offsets_, dg->neighbors_,
              core_u, enc_meta->combine_type_[combine_ptr],
              big_cluster, d_small_clusters_, num_small_clusters,
              d_encodings_,
              enc_meta->num_clusters, enc_d_num_query_us_,
              enc_meta->num_total_us, enc_meta->num_blocks,
              enc_d_query_us_compact_, enc_d_cluster_offsets_,
              d_u_candidate_vs_, d_num_u_candidate_vs_);
          cuchk(cudaDeviceSynchronize());
          ++combine_ptr;
        }
#ifndef NDEBUG
        std::cout << "combine for layer " << layer_index << " done" << std::endl;
#endif
      }

      // goto next layer

      // TODO: fix it !!!!!!!!!
      if (cluster_index != enc_meta->layer_offsets_[layer_index + 1])
      {
        cluster_index = enc_meta->layer_offsets_[layer_index + 1] - 1;
        continue;
      }
    }

    if (cluster_index == cluster_sum + enc_meta->num_clusters_per_layer_[layer_index])
    {
      cluster_sum += enc_meta->num_clusters_per_layer_[layer_index];
      layer_index++;
    }

    //? what is it?
    // if (enc_meta->is_a_valid_cluster_[cluster_index] == false)
    //   continue;
    vtype core_u = cpu_clusters_[cluster_index].query_us_[0];

    // merge clusters.
    int merge_index = cluster_index - enc_meta->num_clusters_per_layer_[0];
    int left = enc_meta->merged_cluster_left_[merge_index];
    int left_position = enc_meta->cluster_offsets_[left];
    while (enc_meta->query_us_compact_[left_position] != core_u)
      left_position++;
    int right = enc_meta->merged_cluster_right_[merge_index];
    int right_position = enc_meta->cluster_offsets_[right];
    while (enc_meta->query_us_compact_[right_position] != core_u)
      right_position++;
    vtype vertex = enc_meta->merged_cluster_vertex_[merge_index];
    int layer = enc_meta->merged_cluster_layer_[merge_index];

    mergeKernel<<<GRID_DIM, BLOCK_DIM>>>(
        left_position, right_position,
        dg->offsets_, dg->neighbors_,
        core_u, layer, cluster_index,
        d_u_candidate_vs_, d_num_u_candidate_vs_,
        d_v_candidate_us_, d_num_v_candidate_us_,

        d_encodings_,
        enc_meta->num_clusters, enc_d_num_query_us_,
        enc_meta->num_total_us, enc_meta->num_blocks,
        enc_d_query_us_compact_, enc_d_cluster_offsets_,

        enc_meta->num_layers, enc_d_clusters_per_layer_,

        enc_meta->merge_count,
        enc_d_merged_cluster_left_, enc_d_merged_cluster_right_,
        enc_d_merged_cluster_vertex_, enc_d_merged_cluster_layer_);
    cuchk(cudaDeviceSynchronize());
  }

  // #ifndef NDEBUG
  //   cuchk(cudaMemcpy(h_encodings_, d_encodings_, sizeof(uint32_t) * NUM_VD * enc_meta->num_blocks, cudaMemcpyDeviceToHost));

  //   std::cout << std::hex;
  //   for (vtype v = 0; v < NUM_VD; ++v)
  //   {
  //     std::cout << "vertex " << v << ": ";
  //     for (int i = 0; i < enc_meta->num_blocks; i++)
  //       std::cout << (int)h_encodings_[v * enc_meta->num_blocks + i] << " ";
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
    uint32_t *&h_encodings_, uint32_t *&d_encodings_,
    encodingMeta *enc_meta,

    // return
    vtype *&h_u_candidate_vs_, numtype *&h_num_u_candidate_vs_,
    vtype *&d_u_candidate_vs_, numtype *&d_num_u_candidate_vs_,
    vtype *&h_v_candidate_us_, numtype *&h_num_v_candidate_us_,
    vtype *&d_v_candidate_us_, numtype *&d_num_v_candidate_us_)
{
  // clustering
  clustering(hq, cpu_clusters_, enc_meta);

  h_encodings_ = new uint32_t[NUM_VD * enc_meta->num_blocks];
  memset(h_encodings_, 0, sizeof(uint32_t) * NUM_VD * enc_meta->num_blocks);
  cuchk(cudaMalloc((void **)&d_encodings_, sizeof(uint32_t) * NUM_VD * enc_meta->num_blocks));
  cuchk(cudaMemset(d_encodings_, 0, sizeof(uint32_t) * NUM_VD * enc_meta->num_blocks));

  uint32_t *d_bitmap = nullptr;
  size_t bitmap_pitch = 0;
  cuchk(cudaMallocPitch((void **)&d_bitmap, &bitmap_pitch, sizeof(uint32_t) * (NUM_VD - 1) / 32 + 1, NUM_VQ));
  cuchk(cudaMemset2D(d_bitmap, bitmap_pitch, 0, sizeof(uint32_t) * (NUM_VD - 1) / 32 + 1, NUM_VQ));

  uint32_t *d_bitmap_reverse = nullptr;
  cuchk(cudaMalloc((void **)&d_bitmap_reverse, sizeof(uint32_t) * NUM_VD));
  cuchk(cudaMemset(d_bitmap_reverse, 0, sizeof(uint32_t) * NUM_VD));

  oneRoundFilterBidirection(
      hq_backup, hg,
      dq_backup, dg,
      d_bitmap, bitmap_pitch,
      d_bitmap_reverse,
      d_u_candidate_vs_, d_num_u_candidate_vs_,
      d_v_candidate_us_, d_num_v_candidate_us_);

  // vtype *d_query_us_compact_ = nullptr;
  // cuchk(cudaMalloc((void **)&d_query_us_compact_, sizeof(vtype) * enc_meta->num_total_us));
  // cuchk(cudaMemcpy(d_query_us_compact_, enc_meta->query_us_compact_, sizeof(vtype) * enc_meta->num_total_us, cudaMemcpyHostToDevice));

#ifndef NDEBUG
  std::cout << "one round filter done" << std::endl;
#endif

  encode(
      dg,
      cpu_clusters_, gpu_clusters_,
      h_encodings_, d_encodings_, enc_meta,
      d_u_candidate_vs_, d_num_u_candidate_vs_,
      d_v_candidate_us_, d_num_v_candidate_us_);

#ifndef NDEBUG
  std::cout << "encode done" << std::endl;
#endif

  numtype *d_num_u_candidate_vs_new_ = nullptr;
  cuchk(cudaMalloc((void **)&d_num_u_candidate_vs_new_, sizeof(numtype) * NUM_VQ));
  cuchk(cudaMemset(d_num_u_candidate_vs_new_, 0, sizeof(numtype) * NUM_VQ));

  vtype *d_u_candidate_vs_new_ = nullptr;
  cuchk(cudaMalloc((void **)&d_u_candidate_vs_new_, sizeof(vtype) * NUM_VQ * MAX_L_FREQ));
  cuchk(cudaMemset(d_u_candidate_vs_new_, UINT32_MAX, sizeof(vtype) * NUM_VQ * MAX_L_FREQ));

  compactCandidatesKernel<<<GRID_DIM, BLOCK_DIM>>>(
      d_u_candidate_vs_, d_num_u_candidate_vs_,
      d_u_candidate_vs_new_, d_num_u_candidate_vs_new_);
  cuchk(cudaDeviceSynchronize());
  std::swap(d_num_u_candidate_vs_, d_num_u_candidate_vs_new_);
  std::swap(d_u_candidate_vs_, d_u_candidate_vs_new_);
  cuchk(cudaFree(d_num_u_candidate_vs_new_));
  cuchk(cudaFree(d_u_candidate_vs_new_));

  cuchk(cudaMemcpy(h_encodings_, d_encodings_, sizeof(uint32_t) * NUM_VD * enc_meta->num_blocks, cudaMemcpyDeviceToHost));

  cuchk(cudaMemcpy(h_u_candidate_vs_, d_u_candidate_vs_, sizeof(vtype) * NUM_VQ * MAX_L_FREQ, cudaMemcpyDeviceToHost));
  cuchk(cudaMemcpy(h_num_u_candidate_vs_, d_num_u_candidate_vs_, sizeof(numtype) * NUM_VQ, cudaMemcpyDeviceToHost));
  cuchk(cudaMemcpy(h_v_candidate_us_, d_v_candidate_us_, sizeof(vtype) * NUM_VD, cudaMemcpyDeviceToHost));
  cuchk(cudaMemcpy(h_num_v_candidate_us_, d_num_v_candidate_us_, sizeof(numtype) * NUM_VD, cudaMemcpyDeviceToHost));
  // #ifndef NDEBUG
  //   std::cout << std::hex;
  //   for (vtype v = 0; v < NUM_VD; ++v)
  //   {
  //     std::cout << "vertex " << v << ": ";
  //     for (int i = 0; i < enc_meta->num_blocks; i++)
  //       std::cout << (int)h_encodings_[v * enc_meta->num_blocks + i] << " ";
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
  //     for (int i = enc_meta->num_total_us - 1; ~i; --i)
  //     {
  //       vtype u = enc_meta->query_us_compact_[i];
  //       if (h_encodings_[v * enc_meta->num_blocks + i / 32] & (1 << (i % 32)))
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
