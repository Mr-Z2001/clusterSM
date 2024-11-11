#include "globals.cuh"
#include "cpuGraph.h"
#include "gpuGraph.h"
#include "join.cuh"
#include "structure.cuh"
#include "cuda_helpers.h"
#include "order.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cooperative_groups.h>

__global__ void
selectPartialMatchingsKernel(
    offtype *offsets_, vtype *nbrs_,
    vtype u, vtype u_matched,
    vtype *d_res_table_old_, numtype num_res_old,
    vtype *d_res_table_, numtype *num_res_new,

    uint32_t *d_encodings_, numtype num_blocks,
    uint32_t enc_pos_u, uint32_t enc_pos_u_matched)
{
  uint32_t tid = threadIdx.x;
  uint32_t bid = blockIdx.x;
  uint32_t idx = tid + bid * blockDim.x;
  uint32_t wid = tid / warpSize;
  uint32_t lid = tid % warpSize;
  uint32_t wid_g = idx / warpSize;

  __shared__ int warp_pos[WARP_PER_BLOCK];

  vtype v, v_nbr;

  int block_iter_cnt = 0;
  int grid_size = blockDim.x * gridDim.x;

  while (block_iter_cnt * grid_size < num_res_old)
  {
    idx = tid + bid * blockDim.x + block_iter_cnt * grid_size;
    wid_g = idx / warpSize;

    bool keep = false;
    if (idx < num_res_old)
    {
      auto group = cooperative_groups::coalesced_threads();
      v = d_res_table_old_[idx * C_NUM_VQ + u];
      v_nbr = d_res_table_old_[idx * C_NUM_VQ + u_matched];

      for (offtype v_off = offsets_[v]; !keep && v_off < offsets_[v + 1]; ++v_off)
        keep = keep || (nbrs_[v_off] == v_nbr);
      group.sync();

      if (keep)
      {
        auto g = cooperative_groups::coalesced_threads();
        if (g.thread_rank() == 0)
          warp_pos[wid] = atomicAdd(num_res_new, g.size());
        g.sync();
        int my_pos = warp_pos[wid] + g.thread_rank();
        for (int i = 0; i < C_NUM_VQ; ++i)
          d_res_table_[my_pos * C_NUM_VQ + i] = d_res_table_old_[idx * C_NUM_VQ + i];
      }
    }
    __syncthreads();
    block_iter_cnt++;
  }
}
__global__ void
firstJoinKernel(
    vtype u,
    vtype *d_u_candidate_vs_, numtype num_u_candidate_vs,
    vtype *d_res_table_)
{
  uint32_t tid = threadIdx.x;
  uint32_t bid = blockIdx.x;
  uint32_t idx = tid + bid * blockDim.x;

  int block_iter_cnt = 0;
  int grid_size = blockDim.x * gridDim.x;

  while (block_iter_cnt * grid_size < num_u_candidate_vs)
  {
    idx = tid + bid * blockDim.x + block_iter_cnt * grid_size;

    if (idx < num_u_candidate_vs)
    {
      vtype v = d_u_candidate_vs_[u * C_MAX_L_FREQ + idx];
      d_res_table_[idx * C_NUM_VQ + u] = v;
    }

    __syncthreads();

    block_iter_cnt++;
  }
}

__global__ void
joinOneEdgeKernel(
    // structure
    offtype *offsets_, vtype *nbrs_,

    vtype u, vtype u_matched,
    vtype *d_res_table_old_, numtype num_res_old,
    vtype *d_res_table_, numtype *num_res_new,

    uint32_t *encodings_, numtype num_blocks,
    int enc_pos)
{
  uint32_t tid = threadIdx.x;
  uint32_t bid = blockIdx.x;
  uint32_t idx = tid + bid * blockDim.x;
  uint32_t wid = tid / warpSize;
  uint32_t lid = tid % warpSize;
  uint32_t wid_g = idx / warpSize;

  __shared__ vtype s_v[WARP_PER_BLOCK];
  __shared__ int warp_pos[WARP_PER_BLOCK];

  int block_iter_cnt = 0;
  int grid_size = blockDim.x * gridDim.x;

  while (block_iter_cnt * (grid_size / warpSize) < num_res_old)
  {
    idx = tid + bid * blockDim.x + block_iter_cnt * grid_size;
    wid_g = idx / warpSize;

    // one warp one row
    if (lid == 0)
      if (wid_g < num_res_old)
        s_v[wid] = d_res_table_old_[wid_g * C_NUM_VQ + u_matched];
      else
        s_v[wid] = UINT32_MAX;
    __syncwarp();

    if (wid_g < num_res_old && s_v[wid] != UINT32_MAX)
    {
      vtype v = s_v[wid];
      int row = wid_g;

      offtype v_nbr_off = offsets_[v] + lid;
      offtype v_nbr_off_end = offsets_[v + 1];
      while (v_nbr_off < v_nbr_off_end)
      {
        auto group = cooperative_groups::coalesced_threads();
        vtype v_nbr = nbrs_[v_nbr_off];
        // if (v_candidate_us_[v_nbr] & (1u << u))
        if (encodings_[v_nbr * num_blocks + enc_pos / ENC_SIZE] & (1u << (enc_pos % ENC_SIZE)))
        {
          bool same_flag = false;
          for (int i = 0; i < C_NUM_VQ; ++i)
          {
            same_flag = same_flag || ((d_res_table_old_[row * C_NUM_VQ + i] == v_nbr));
          }
          if (!same_flag)
          {
            auto g = cooperative_groups::coalesced_threads();
            if (g.thread_rank() == 0)
              warp_pos[wid] = atomicAdd(num_res_new, g.size());
            g.sync();
            int pos = warp_pos[wid] + g.thread_rank();
            for (int i = 0; i < C_NUM_VQ; ++i)
              d_res_table_[pos * C_NUM_VQ + i] = d_res_table_old_[row * C_NUM_VQ + i];
            d_res_table_[pos * C_NUM_VQ + u] = v_nbr;
          }
        }
        group.sync();
        v_nbr_off += warpSize;
      }
      __syncwarp();
    }
    __syncthreads();

    block_iter_cnt++;
  }
}

__global__ void
collectMappedVs(
    vtype u_matched,
    bool *d_flag_,
    vtype *d_res_table_old_, numtype num_res_old)
{
  uint32_t tid = threadIdx.x;
  uint32_t bid = blockIdx.x;
  uint32_t idx = tid + bid * blockDim.x;
  uint32_t wid = tid / warpSize;
  uint32_t lid = tid % warpSize;

  // TODO: optimize this. reduce write conflicts.
  // __shared__ vtype mapped_vs[WARP_PER_BLOCK][WARP_SIZE];

  // mapped_vs[wid][lid] = UINT32_MAX;

  // if (idx < num_res_old)
  // {
  //   vtype v_mapped = d_res_table_old_[idx * C_NUM_VQ + u_matched];
  //   mapped_vs[wid][lid] = v_mapped;
  // }
  // __syncwarp();
  // if (lid == 0 && idx < num_res_old)
  // {

  // }

  if (idx < num_res_old)
  {
    vtype v_mapped = d_res_table_old_[idx * C_NUM_VQ + u_matched];
    d_flag_[v_mapped] = true;
  }
}

void joinOneEdge(
    cpuGraph *hq, cpuGraph *hg,
    gpuGraph *dq, gpuGraph *dg,

    vtype u, vtype u_matched,
    vtype *d_res_table_old_, numtype &num_res_old,
    vtype *d_res_table_, numtype &num_res_new,

    vtype *h_u_candidate_vs_, numtype *h_num_u_candidate_vs_,
    vtype *d_u_candidate_vs_, numtype *d_num_u_candidate_vs_,
    vtype *h_v_candidate_us_, numtype *h_num_v_candidate_us_,
    vtype *d_v_candidate_us_, numtype *d_num_v_candidate_us_,

    uint32_t *d_encodings_,

    encodingMeta *enc_meta)
{
  bool *d_flag_ = nullptr;
  num_res_new = 0;

  cuchk(cudaMalloc((void **)&d_flag_, sizeof(bool) * NUM_VD));
  cuchk(cudaMemset(d_flag_, false, sizeof(bool) * NUM_VD));

  vtype *d_candidate_v_buffer_;
  cuchk(cudaMalloc((void **)&d_candidate_v_buffer_, sizeof(vtype) * NUM_VD));
  cuchk(cudaMemset(d_candidate_v_buffer_, -1, sizeof(vtype) * NUM_VD));
  // TODO: revise this.

  int enc_pos = -1;
  for (int i = enc_meta->num_total_us - 1; ~i; --i)
  {
    if (enc_meta->query_us_compact_[i] == u)
    {
      enc_pos = i;
      break;
    }
  }
  if (enc_pos == -1)
  {
    std::cerr << "u: " << u << ", u_matched: " << u_matched << std::endl;
    std::cerr << "Error! Enc_pos == -1, unexpected." << std::endl;
    exit(1);
  }

  int *d_num_candidates_in_buffer;
  cuchk(cudaMalloc((void **)&d_num_candidates_in_buffer, sizeof(int)));
  cuchk(cudaMemset(d_num_candidates_in_buffer, 0, sizeof(int)));

  uint32_t *d_num_new_res;
  cuchk(cudaMalloc((void **)&d_num_new_res, sizeof(uint32_t)));
  cuchk(cudaMemset(d_num_new_res, 0, sizeof(uint32_t)));

  dim3 joe_block = 512;
  int N = num_res_old * 32;
  dim3 joe_grid = std::min(GRID_DIM, calc_grid_dim(N, joe_block.x));

  joinOneEdgeKernel<<<joe_grid, joe_block>>>(
      dg->offsets_, dg->neighbors_,
      u, u_matched,
      d_res_table_old_, num_res_old,
      d_res_table_, d_num_new_res,
      d_encodings_, enc_meta->num_blocks,
      enc_pos);
  cuchk(cudaDeviceSynchronize());

  cuchk(cudaMemcpy(&num_res_new, d_num_new_res, sizeof(uint32_t), cudaMemcpyDeviceToHost));

#ifndef NDEBUG
  std::cout << "After join one edge, num_res_new: " << num_res_new << std::endl;
#endif

  cuchk(cudaFree(d_num_new_res));
  cuchk(cudaFree(d_flag_));
  cuchk(cudaFree(d_candidate_v_buffer_));
  cuchk(cudaFree(d_num_candidates_in_buffer));
}
void join(
    cpuGraph *hq, cpuGraph *hg,
    gpuGraph *dq, gpuGraph *dg,
    etype *order,
    vtype *h_res_table_, numtype &num_res,

    encodingMeta *enc_meta,
    uint32_t *h_encodings_, uint32_t *d_encodings_,

    vtype *h_u_candidate_vs_, numtype *h_num_u_candidate_vs_,
    vtype *d_u_candidate_vs_, numtype *d_num_u_candidate_vs_,
    vtype *h_v_candidate_us_, numtype *h_num_v_candidate_us_,
    vtype *d_v_candidate_us_, numtype *d_num_v_candidate_us_,

    cpuCluster *cpu_clusters_, gpuCluster *gpu_clusters_)
{

  bool *mapped_us_ = new bool[NUM_VQ];
  memset(mapped_us_, false, sizeof(bool) * NUM_VQ);

  // make sure all the swap operations are done in this function, not inside sub-functions.

  numtype num_res_old = 0;
  vtype *d_res_table_ = nullptr;
  cuchk(cudaMalloc((void **)&d_res_table_, sizeof(vtype) * NUM_VQ * MAX_RES));
  cuchk(cudaMemset(d_res_table_, -1, sizeof(vtype) * NUM_VQ * MAX_RES));

  vtype *d_res_table_old_ = nullptr;
  cuchk(cudaMalloc((void **)&d_res_table_old_, sizeof(vtype) * NUM_VQ * MAX_RES));
  cuchk(cudaMemset(d_res_table_old_, -1, sizeof(vtype) * NUM_VQ * MAX_RES));

  // first Join initialize res_table.
  vtype u = hq->evv[order[0]].first;

  // #ifndef NDEBUG
  //   std::cout << "first join u: " << u << std::endl;
  //   std::cout << "num candidates: " << h_num_u_candidate_vs_[u] << std::endl;
  //   for (int i = 0; i < h_num_u_candidate_vs_[u]; ++i)
  //   {
  //     std::cout << h_u_candidate_vs_[u * C_MAX_L_FREQ + i] << " ";
  //   }
  //   std::cout << std::endl;
  // #endif

  dim3 fj_block = 512;
  dim3 fj_grid = std::min(GRID_DIM, calc_grid_dim(h_num_u_candidate_vs_[u], fj_block.x));
  firstJoinKernel<<<fj_grid, fj_block>>>(
      u,
      d_u_candidate_vs_, h_num_u_candidate_vs_[u],
      d_res_table_old_);
  num_res_old = h_num_u_candidate_vs_[u];
  mapped_us_[u] = true;

#ifndef NDEBUG
  std::cout << "first join done" << std::endl;
  std::cout << "num_res: " << num_res_old << std::endl;
#endif

  for (offtype e_off = 0; e_off < NUM_EQ; ++e_off)
  {
    etype e = order[e_off];
    vtype u = hq->evv[e].second;
    vtype u_matched = hq->evv[e].first;

    if (!mapped_us_[u])
    {
      joinOneEdge(
          hq, hg, dq, dg,
          u, u_matched,
          d_res_table_old_, num_res_old,
          d_res_table_, num_res,

          h_u_candidate_vs_, h_num_u_candidate_vs_,
          d_u_candidate_vs_, d_num_u_candidate_vs_,
          h_v_candidate_us_, h_num_v_candidate_us_,
          d_v_candidate_us_, d_num_v_candidate_us_,

          d_encodings_,

          enc_meta);
    }
    else // both matched.
    {
      numtype *d_num_res_new;
      cuchk(cudaMalloc((void **)&d_num_res_new, sizeof(numtype)));
      cuchk(cudaMemset(d_num_res_new, 0, sizeof(numtype)));

      int enc_pos_u = -1, enc_pos_u_matched = -1;
      // for (int cluster_index = enc_meta->num_clusters - 1; ~cluster_index; --cluster_index)
      // {
      //   enc_pos_u = -1, enc_pos_u_matched = -1;
      //   bool found_u = false, found_u_matched = false;
      //   for (int i = 0; i < enc_meta->num_query_us_[cluster_index]; ++i)
      //   {
      //     found_u = found_u || (cpu_clusters_[cluster_index].query_us_[i] == u);
      //     if (found_u)
      //       enc_pos_u = enc_meta->cluster_offsets_[cluster_index] + i;
      //     found_u_matched = found_u_matched || (cpu_clusters_[cluster_index].query_us_[i] == u_matched);
      //     if (found_u_matched)
      //       enc_pos_u_matched = enc_meta->cluster_offsets_[cluster_index] + i;
      //   }
      //   if (found_u && found_u_matched)
      //     break;
      // }
      // if (enc_pos_u == -1 || enc_pos_u_matched == -1)
      // {
      //   std::cerr << "Error: enc_pos_u == -1 || enc_pos_u_matched == -1" << std::endl;
      //   exit(1);
      // }

      dim3 spm_block = 512;
      int N = num_res_old;
      dim3 spm_grid = std::min(GRID_DIM, calc_grid_dim(N, spm_block.x));
      selectPartialMatchingsKernel<<<spm_grid, spm_block>>>(
          dg->offsets_, dg->neighbors_,
          u, u_matched,
          d_res_table_old_, num_res_old,
          d_res_table_, d_num_res_new,

          d_encodings_, enc_meta->num_blocks,
          enc_pos_u, enc_pos_u_matched);
      cuchk(cudaDeviceSynchronize());
      cuchk(cudaMemcpy(&num_res, d_num_res_new, sizeof(int), cudaMemcpyDeviceToHost));
#ifndef NDEBUG
      std::cout << "After selectPartialMatchingsKernel, num_res: " << num_res << std::endl;
#endif
    }
    std::swap(d_res_table_old_, d_res_table_);
    num_res_old = num_res;
    num_res = 0;
    mapped_us_[u] = true;

    // #ifndef NDEBUG
    //     std::cout << "join edge " << e_off << ": " << e << " done" << std::endl;
    //     cuchk(cudaMemcpy(h_res_table_, d_res_table_old_, sizeof(vtype) * NUM_VQ * num_res_old, cudaMemcpyDeviceToHost));
    //     for (int i = 0; i < num_res_old; ++i)
    //     {
    //       for (int j = 0; j < NUM_VQ; ++j)
    //       {
    //         std::cout << h_res_table_[i * NUM_VQ + j] << " ";
    //       }
    //       std::cout << std::endl;
    //     }
    // #endif
  }
  num_res = num_res_old;
}