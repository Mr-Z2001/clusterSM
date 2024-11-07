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

// __global__ void
// firstJoinCGKernel(
//     etype e, vtype u, vtype u_nbr,
//     numtype num_res,
//     vtype *candidate_vs_1_, vtype *candidate_vs_2_,
//     vtype *d_res_table, size_t d_res_table_pitch)
// {
//   int tid = threadIdx.x;
//   int bid = blockIdx.x;
//   int idx = tid + bid * blockDim.x;
//   if (idx < num_res)
//   {
//     vtype v = candidate_vs_1_[idx];
//     vtype v_nbr = candidate_vs_2_[idx];
//     d_res_table[idx * d_res_table_pitch / sizeof(uint32_t) + u] = v;
//     d_res_table[idx * d_res_table_pitch / sizeof(uint32_t) + u_nbr] = v_nbr;
//   }
// }

// void firstJoinCG(
//     cpuGraph *hq,
//     etype e,
//     cpuRelation *cpu_relations_, gpuRelation *gpu_relations_,
//     numtype *num_res,
//     vtype *d_res_table, size_t d_res_table_pitch)
// {
//   *num_res = cpu_relations_[e].num_candidates_[0];

//   vtype u, u_nbr;
//   u = hq->evv[e].first;
//   u_nbr = hq->evv[e].second;

//   firstJoinCGKernel<<<GRID_DIM, BLOCK_DIM>>>(
//       e, u, u_nbr,
//       *num_res,
//       gpu_relations_[e].candidate_vs_[0],
//       gpu_relations_[e].candidate_vs_[1],
//       d_res_table, d_res_table_pitch);
//   cuchk(cudaDeviceSynchronize());
// }

// // TODO: optimize:
// //  each thread counts the number of matches it finds.
// //  finally, device-wide reduction.
// //  avoid atomicAdd.
// __global__ void
// joinOneEdgeCountCGKernel(
//     vtype *keys_,
//     numtype num_candidates,
//     vtype *candidate_vs_1_, vtype *candidate_vs_2_,
//     vtype *d_res_table, size_t d_res_table_pitch, numtype num_res_old,
//     numtype *num_res_new)
// {
//   int tid = threadIdx.x;
//   int bid = blockIdx.x;
//   int idx = tid + bid * blockDim.x;
//   int wid = tid / warpSize;
//   int wid_global = idx / warpSize;
//   int lid = tid % warpSize;

//   vtype u = keys_[0], u_nbr = keys_[1];
//   vtype v = UINT32_MAX, v_nbr = UINT32_MAX;
//   vtype u_matched, u_nbr_matched;
//   if (idx < num_candidates)
//   {
//     v = candidate_vs_1_[idx];
//     v_nbr = candidate_vs_2_[idx];
//   }
//   for (int row = 0; row < num_res_old; ++row)
//   {
//     u_matched = d_res_table[row * d_res_table_pitch / sizeof(uint32_t) + u];
//     if (u_matched == v)
//     {
//       auto group = cooperative_groups::coalesced_threads();
//       int rank = group.thread_rank();
//       if (rank == 0)
//         atomicAdd(num_res_new, group.size());
//       group.sync();
//     }
//     __syncwarp();
//   }
// }

// void joinOneEdgeCountCG(
//     etype e, vtype u, vtype u_nbr,
//     numtype h_num_res_old,
//     cpuRelation *cpu_relations_, gpuRelation *gpu_relations_,
//     vtype *d_res_table, size_t d_res_table_pitch,
//     vtype *d_temp_res_table, size_t d_temp_res_table_pitch,
//     numtype *h_num_res)
// {
//   numtype *d_num_res_new = nullptr;
//   cuchk(cudaMalloc((void **)&d_num_res_new, sizeof(numtype)));
//   cuchk(cudaMemset(d_num_res_new, 0, sizeof(numtype)));

// #ifndef NDEBUG
//   std::cout << gpu_relations_[e].keys_ << std::endl;
//   std::cout << gpu_relations_[e].num_candidates_ << std::endl;
//   std::cout << gpu_relations_[e].candidate_vs_[0] << std::endl;
//   std::cout << gpu_relations_[e].candidate_vs_[1] << std::endl;
// #endif
//   joinOneEdgeCountCGKernel<<<GRID_DIM, BLOCK_DIM>>>(
//       gpu_relations_[e].keys_,
//       cpu_relations_[e].num_candidates_[0],
//       gpu_relations_[e].candidate_vs_[0],
//       gpu_relations_[e].candidate_vs_[1],
//       d_res_table, d_res_table_pitch, h_num_res_old,
//       d_num_res_new);
//   cuchk(cudaDeviceSynchronize());
//   cuchk(cudaMemcpy(h_num_res, d_num_res_new, sizeof(numtype), cudaMemcpyDeviceToHost));
//   cuchk(cudaFree(d_num_res_new));
// }

// __global__ void joinOneEdgeWriteCGKernel(
//     vtype *keys_,
//     numtype num_candidates,
//     vtype *candidate_vs_1_, vtype *candidate_vs_2_,
//     vtype *d_res_table, size_t d_res_table_pitch, numtype num_res_old,
//     vtype *d_temp_res_table, size_t d_temp_res_table_pitch,
//     numtype *num_res_new)
// {
//   int tid = threadIdx.x;
//   int bid = blockIdx.x;
//   int idx = tid + bid * blockDim.x;
//   int wid = tid / warpSize;
//   int wid_global = idx / warpSize;
//   int lid = tid % warpSize;

//   __shared__ int s_warp_pos[WARP_PER_BLOCK];

//   if (lid == 0)
//     s_warp_pos[wid] = 0;
//   __syncwarp();

//   vtype u = keys_[0], u_nbr = keys_[1];
//   vtype v = UINT32_MAX, v_nbr = UINT32_MAX;
//   vtype u_matched, u_nbr_matched;
//   if (idx < num_candidates)
//   {
//     v = candidate_vs_1_[idx];
//     v_nbr = candidate_vs_2_[idx];
//   }

//   for (int row = 0; row < num_res_old; ++row)
//   {
//     u_matched = d_res_table[row * d_res_table_pitch / sizeof(uint32_t) + u];
//     if (u_matched == v)
//     {
//       auto group = cooperative_groups::coalesced_threads();
//       int rank = group.thread_rank();
//       if (rank == 0)
//         s_warp_pos[wid] = atomicAdd(num_res_new, group.size());
//       group.sync();
//       int my_pos = s_warp_pos[wid] + rank;
//       for (int i = 0; i < C_NUM_VQ; ++i)
//         d_temp_res_table[my_pos * d_temp_res_table_pitch / sizeof(uint32_t) + i] = d_res_table[row * d_res_table_pitch / sizeof(uint32_t) + i];
//       d_temp_res_table[my_pos * d_temp_res_table_pitch / sizeof(uint32_t) + u_nbr] = v_nbr;
//     }
//     __syncwarp();
//   }
// }

// void joinOneEdgeWriteCG(
//     etype e, vtype u, vtype u_nbr,
//     numtype num_res_old,
//     cpuRelation *cpu_relations_, gpuRelation *gpu_relations_,
//     vtype *d_res_table, size_t d_res_table_pitch,
//     vtype *d_temp_res_table, size_t d_temp_res_table_pitch,
//     numtype *num_res)
// {
//   vtype *num_res_new;
//   cuchk(cudaMalloc((void **)&num_res_new, sizeof(vtype)));
//   cuchk(cudaMemset(num_res_new, 0, sizeof(vtype)));

//   joinOneEdgeWriteCGKernel<<<GRID_DIM, BLOCK_DIM>>>(
//       gpu_relations_[e].keys_,
//       cpu_relations_[e].num_candidates_[0],
//       gpu_relations_[e].candidate_vs_[0],
//       gpu_relations_[e].candidate_vs_[1],
//       d_res_table, d_res_table_pitch, num_res_old,
//       d_temp_res_table, d_temp_res_table_pitch,
//       num_res_new);
//   cuchk(cudaDeviceSynchronize());

//   cuchk(cudaFree(num_res_new));
// }

// __global__ void
// selectPartialMatchingsCGKernel(
//     vtype *keys_,
//     numtype num_candidates,
//     vtype *candidate_vs_1_, vtype *candidate_vs_2_,
//     vtype *d_res_table, size_t d_res_table_pitch,
//     vtype *d_temp_res_table, size_t d_temp_res_table_pitch,
//     numtype num_res_old,
//     numtype *num_res_new)
// {
//   int tid = threadIdx.x;
//   int bid = blockIdx.x;
//   int idx = tid + bid * blockDim.x;
//   int wid = tid / warpSize;
//   int wid_global = idx / warpSize;
//   int lid = tid % warpSize;

//   __shared__ int s_warp_pos[WARP_PER_BLOCK];

//   if (lid == 0)
//     s_warp_pos[wid] = 0;
//   __syncwarp();

//   vtype u = keys_[0], u_nbr = keys_[1];
//   vtype v = UINT32_MAX, v_nbr = UINT32_MAX;
//   vtype u_matched, u_nbr_matched;
//   if (idx < num_candidates)
//   {
//     v = candidate_vs_1_[idx];
//     v_nbr = candidate_vs_2_[idx];
//   }
//   if (wid_global * 32 < num_candidates)
//     for (int row = 0; row < num_res_old; ++row)
//     {
//       u_matched = d_res_table[row * d_res_table_pitch / sizeof(uint32_t) + u];
//       u_nbr_matched = d_res_table[row * d_res_table_pitch / sizeof(uint32_t) + u_nbr];
//       if (v == u_matched && v_nbr == u_nbr_matched)
//       {
//         auto group = cooperative_groups::coalesced_threads();
//         int rank = group.thread_rank();
//         if (rank == 0)
//           s_warp_pos[wid] = atomicAdd(num_res_new, group.size());
//         group.sync();
//         int my_pos = s_warp_pos[wid] + rank;
//         for (int i = 0; i < C_NUM_VQ; ++i)
//           d_temp_res_table[my_pos * d_temp_res_table_pitch / sizeof(uint32_t) + i] = d_res_table[row * d_res_table_pitch / sizeof(uint32_t) + i];
//       }
//       __syncwarp();
//     }
// }

// void selectPartialMatchingsCG(
//     etype e, vtype u, vtype u_nbr,
//     numtype num_res_old,
//     cpuRelation *cpu_relations_, gpuRelation *gpu_relations_,
//     vtype *d_res_table, size_t d_res_table_pitch,
//     vtype *d_temp_res_table, size_t d_temp_res_table_pitch,
//     numtype *num_res)
// {
//   numtype *d_num_res_new = nullptr;
//   cuchk(cudaMalloc((void **)&d_num_res_new, sizeof(numtype)));
//   cuchk(cudaMemset(d_num_res_new, 0, sizeof(numtype)));
//   selectPartialMatchingsCGKernel<<<GRID_DIM, BLOCK_DIM>>>(
//       gpu_relations_[e].keys_,
//       cpu_relations_[e].num_candidates_[0],
//       gpu_relations_[e].candidate_vs_[0],
//       gpu_relations_[e].candidate_vs_[1],
//       d_res_table, d_res_table_pitch,
//       d_temp_res_table, d_temp_res_table_pitch,
//       num_res_old,
//       d_num_res_new);
//   cuchk(cudaDeviceSynchronize());

//   cuchk(cudaMemcpy(num_res, d_num_res_new, sizeof(numtype), cudaMemcpyDeviceToHost));
//   cuchk(cudaFree(d_num_res_new));
// }

// // res table: column-first will be much better. (in terms of performance)
// void joinCG(
//     cpuGraph *hq, cpuGraph *hg,
//     gpuGraph *dq,
//     etype *order,
//     cpuRelation *cpu_relations_, gpuRelation *gpu_relations_,
//     numtype *h_num_res, vtype *h_res_table,
//     vtype *d_res_table, size_t d_res_table_pitch,
//     vtype *d_temp_res_table, size_t d_temp_res_table_pitch)
// {
//   bool *if_u_matched = new bool[hq->num_v];
//   memset(if_u_matched, false, sizeof(bool) * hq->num_v);

//   // first join
//   firstJoinCG(hq, order[0], cpu_relations_, gpu_relations_, h_num_res, d_res_table, d_res_table_pitch);

// #ifndef NDEBUG
//   std::cout << "first join done" << std::endl;
//   std::cout << "h_num_res: " << *h_num_res << std::endl;
//   vtype *h_res_debug_temp = new vtype[NUM_VQ * (*h_num_res)];
//   cuchk(cudaMemcpy2D(h_res_debug_temp, sizeof(vtype) * NUM_VQ, d_res_table, d_res_table_pitch, sizeof(vtype) * NUM_VQ, *h_num_res, cudaMemcpyDeviceToHost));
//   for (int row = 0; row < *h_num_res; ++row)
//   {
//     for (int col = 0; col < NUM_VQ; ++col)
//     {
//       std::cout << h_res_debug_temp[row * NUM_VQ + col] << " ";
//     }
//     std::cout << std::endl;
//   }
// #endif

//   etype e = order[0];
//   vtype u, u_nbr;
//   u = hq->evv[e].first;
//   u_nbr = hq->evv[e].second;
//   if_u_matched[u] = true;
//   if_u_matched[u_nbr] = true;

//   // join the rest
//   for (offtype order_off = 1; order_off < hq->num_e; ++order_off)
//   {
//     e = order[order_off];
//     // join PM and Relation[e]
//     numtype h_num_res_old = *h_num_res;
//     *h_num_res = 0;
//     u = hq->evv[e].first;
//     u_nbr = hq->evv[e].second;

//     if (if_u_matched[u] == true && if_u_matched[u_nbr] == false)
//     {
//       joinOneEdgeCountCG(
//           e, u, u_nbr,
//           h_num_res_old,
//           cpu_relations_, gpu_relations_,
//           d_res_table, d_res_table_pitch,
//           d_temp_res_table, d_temp_res_table_pitch,
//           h_num_res);
//       joinOneEdgeWriteCG(
//           e, u, u_nbr,
//           h_num_res_old,
//           cpu_relations_, gpu_relations_,
//           d_res_table, d_res_table_pitch,
//           d_temp_res_table, d_temp_res_table_pitch,
//           h_num_res);
//       if_u_matched[u] = if_u_matched[u_nbr] = true;
//     }
//     else if (if_u_matched[u] && if_u_matched[u_nbr]) // select compromising partial matches.
//     {
//       selectPartialMatchingsCG(
//           e, u, u_nbr,
//           h_num_res_old,
//           cpu_relations_, gpu_relations_,
//           d_res_table, d_res_table_pitch,
//           d_temp_res_table, d_temp_res_table_pitch,
//           h_num_res);
//     }
//     else
//     {
//       std::cerr << "Error: u is not matched" << std::endl;
//       exit(EXIT_FAILURE);
//     }
//     cuchk(cudaMemcpy2D(d_res_table, d_res_table_pitch, d_temp_res_table, d_temp_res_table_pitch, sizeof(vtype) * NUM_VQ, *h_num_res, cudaMemcpyDeviceToDevice));

// #ifndef NDEBUG
//     std::cout << "join edge " << order_off << ": " << order[order_off] << " done" << std::endl;
// #endif
//   }
//   cuchk(cudaMemcpy2D(h_res_table, sizeof(vtype) * NUM_VQ, d_res_table, d_res_table_pitch, sizeof(vtype) * NUM_VQ, *h_num_res, cudaMemcpyDeviceToHost));

//   delete[] if_u_matched;
// }

__global__ void
firstJoinKernel(
    vtype u,
    vtype *d_u_candidate_vs_, numtype num_u_candidate_vs_,
    vtype *d_res_table_)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int idx = tid + bid * blockDim.x;
  if (idx < num_u_candidate_vs_)
  {
    vtype v = d_u_candidate_vs_[idx];
    d_res_table_[idx * C_NUM_VQ + u] = v;
  }
}

__global__ void
joinOneEdgeKernel(
    // structure
    offtype *offsets_, vtype *nbrs_,

    vtype u, vtype u_matched,
    vtype *d_res_table_old_, numtype num_res_old,
    vtype *d_res_table_, numtype num_res_new,
    vtype *candidate_v_buffer_,

    uint32_t *v_candidate_us_, // bitmap reverse

    uint32_t *encodings_, numtype num_bytes,
    int enc_pos,

    int *num_candidates_in_buffer,
    bool *flag_)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int idx = tid + bid * blockDim.x;
  int wid = tid / warpSize;
  int lid = tid % warpSize;
  int wid_g = idx / warpSize;

  __shared__ vtype s_v[WARP_PER_BLOCK];
  __shared__ int s_row[WARP_PER_BLOCK];

  auto grid_group = cooperative_groups::this_grid();

  if (idx < C_NUM_VD)
    flag_[idx] = true;
  __syncthreads();

  vtype v = UINT32_MAX;
  if (idx < num_res_old)
  {
    v = d_res_table_old_[idx * C_NUM_VQ + u];
    flag_[v] = true;
  }
  grid_group.sync();

  if (idx < C_NUM_VD)
  {
    if (flag_[idx])
    {
      int pos = atomicAdd(num_candidates_in_buffer, 1);
      candidate_v_buffer_[pos] = idx;
    }
  }
  grid_group.sync();

  // one warp one row
  if (wid_g < num_res_old)
  {
    if (lid == 0)
    {
      s_row[wid] = wid_g;
      s_v[wid] = d_res_table_old_[wid_g * C_NUM_VQ + u];
    }
  }
  __syncwarp();
  if (wid_g < num_res_old)
  {
    vtype v = s_v[wid];
    int row = s_row[wid];

    offtype v_nbr_off = offsets_[v] + lid;
    offtype v_nbr_off_end = offsets_[v + 1];
    while (v_nbr_off < v_nbr_off_end)
    {
      vtype v_nbr = nbrs_[v_nbr_off];
      if (v_candidate_us_[v_nbr] & (1 << u))
      {
        int pos = atomicAdd(&num_res_new, 1);
        for (int i = 0; i < C_NUM_VQ; ++i)
          d_res_table_[pos * C_NUM_VQ + i] = d_res_table_old_[row * C_NUM_VQ + i];
        d_res_table_[pos * C_NUM_VQ + u] = v;
      }
      v_nbr_off += warpSize;
    }
    __syncwarp();
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
  bool *d_flag_;
  num_res_new = 0;
  cuchk(cudaMalloc((void **)&d_flag_, sizeof(bool) * NUM_VD));

  vtype *d_candidate_v_buffer_;
  cuchk(cudaMalloc((void **)&d_candidate_v_buffer_, sizeof(vtype) * NUM_VD));
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
    std::cerr << "Error! Enc_pos == -1, unexpected." << std::endl;
    exit(1);
  }

  int *d_num_candidates_in_buffer;
  cuchk(cudaMalloc((void **)&d_num_candidates_in_buffer, sizeof(int)));
  cuchk(cudaMemset(d_num_candidates_in_buffer, 0, sizeof(int)));

  void *kernelArgs[] = {
      (void *)&dg->offsets_, (void *)&dg->neighbors_,
      (void *)&u, (void *)&u_matched,
      (void *)&d_res_table_old_, (void *)&num_res_old,
      (void *)&d_res_table_, (void *)&num_res_new,
      (void *)&d_candidate_v_buffer_,
      (void *)&d_v_candidate_us_,
      (void *)&d_encodings_, (void *)&(enc_meta->num_bytes),
      (void *)&enc_pos,
      (void *)&d_flag_};

  cuchk(cudaLaunchCooperativeKernel(
      (void *)joinOneEdgeKernel,
      GRID_DIM, BLOCK_DIM,
      kernelArgs));
  cuchk(cudaDeviceSynchronize());
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
  numtype MAX_RES = 100000;
  numtype num_res_old = 0;
  vtype *d_res_table_;
  cuchk(cudaMalloc((void **)&d_res_table_, sizeof(vtype) * NUM_VQ * MAX_RES));
  cuchk(cudaMemset(d_res_table_, -1, sizeof(vtype) * NUM_VQ * MAX_RES));

  vtype *d_res_table_old_;
  cuchk(cudaMalloc((void **)&d_res_table_old_, sizeof(vtype) * NUM_VQ * MAX_RES));
  cuchk(cudaMemset(d_res_table_old_, -1, sizeof(vtype) * NUM_VQ * MAX_RES));

  // first Join
  vtype u = hq->evv[order[0]].first;
  firstJoinKernel<<<GRID_DIM, BLOCK_DIM>>>(
      u,
      d_u_candidate_vs_, h_num_u_candidate_vs_[u],
      d_res_table_old_);
  num_res_old = h_num_u_candidate_vs_[u];

  for (offtype e_off = 1; e_off < NUM_EQ; ++e_off)
  {
    etype e = order[e_off];
    vtype u = hq->evv[e].second;
    vtype u_matched = hq->evv[e].first;

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
    std::swap(d_res_table_old_, d_res_table_);
    num_res_old = num_res;
    num_res = 0;
  }
}