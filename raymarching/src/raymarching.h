#pragma once

#include <stdint.h>
#include <torch/torch.h>


void near_far_from_aabb(const at::Tensor rays_o, const at::Tensor rays_d, const at::Tensor aabb, const uint32_t N, const float min_near, at::Tensor nears, at::Tensor fars);
void sph_from_ray(const at::Tensor rays_o, const at::Tensor rays_d, const float radius, const uint32_t N, at::Tensor coords);
void morton3D(const at::Tensor coords, const uint32_t N, at::Tensor indices);
void morton3D_invert(const at::Tensor indices, const uint32_t N, at::Tensor coords);
void packbits(const at::Tensor grid, const uint32_t N, const float density_thresh, at::Tensor bitfield);

void march_rays_train(const at::Tensor rays_o, const at::Tensor rays_d, const at::Tensor grid, const float bound, const float dt_gamma, const uint32_t max_steps, const uint32_t N, const uint32_t C, const uint32_t H, const uint32_t M, const at::Tensor nears, const at::Tensor fars, at::Tensor xyzs, at::Tensor dirs, at::Tensor deltas, at::Tensor z_vals, at::Tensor rays, at::Tensor counter, at::Tensor noises);
void composite_rays_train_forward(const at::Tensor sigmas, const at::Tensor rgbs, const at::Tensor deltas, const at::Tensor rays, const uint32_t M, const uint32_t N, const float T_thresh,  at::Tensor alphas, at::Tensor Ts, at::Tensor weights, at::Tensor depth, at::Tensor image);
void composite_rays_train_backward(const at::Tensor grad_weights, const at::Tensor grad_image, const at::Tensor sigmas,  const at::Tensor rgbs, const at::Tensor deltas, const at::Tensor rays, const at::Tensor alphas, const at::Tensor Ts, const at::Tensor weights,  const at::Tensor image, const uint32_t M, const uint32_t N, const float T_thresh, at::Tensor grad_helper, at::Tensor grad_sigmas, at::Tensor grad_rgbs);

void get_weights_train_forward(const at::Tensor sigmas,  const at::Tensor deltas, const at::Tensor rays, const uint32_t M, const uint32_t N, const float T_thresh, at::Tensor weights, at::Tensor Ts, at::Tensor alphas);
void get_weights_train_backward(const at::Tensor grad_weights, const at::Tensor sigmas, const at::Tensor deltas, const at::Tensor rays, const at::Tensor weights, const at::Tensor Ts, const at::Tensor alphas, const uint32_t M, const uint32_t N, const float T_thresh, at::Tensor grad_helper, at::Tensor grad_sigmas);


void weighted_sum_train_forward(const at::Tensor weights, const at::Tensor values, const at::Tensor rays, const uint32_t M, const uint32_t N, const uint32_t D,  at::Tensor weighted_sum);
void weighted_sum_train_backward(const at::Tensor grad_weighted_sum, const at::Tensor weights, const at::Tensor values, const at::Tensor rays, const uint32_t M, const uint32_t N, const uint32_t D, const at::Tensor grad_weights, const at::Tensor grad_values);

void weighted_sum_cuda_forward(const at::Tensor weights, const at::Tensor values, const uint32_t N, const uint32_t Ns, const uint32_t D,  at::Tensor weighted_sum);
void weighted_sum_cuda_backward(const at::Tensor grad_weighted_sum, const at::Tensor weights, const at::Tensor values, const uint32_t N, const uint32_t Ns, const uint32_t D, const at::Tensor grad_weights, const at::Tensor grad_values);


void march_rays(const uint32_t n_alive, const uint32_t n_step, const at::Tensor rays_alive, const at::Tensor rays_t, const at::Tensor rays_o, const at::Tensor rays_d, const float bound, const float dt_gamma, const uint32_t max_steps, const uint32_t C, const uint32_t H, const at::Tensor grid, const at::Tensor nears, const at::Tensor fars, at::Tensor xyzs, at::Tensor dirs, at::Tensor deltas, at::Tensor noises);
void composite_rays(const uint32_t n_alive, const uint32_t n_step, const float T_thresh, at::Tensor rays_alive, at::Tensor rays_t, at::Tensor sigmas, at::Tensor rgbs, at::Tensor deltas, at::Tensor weights_sum, at::Tensor depth, at::Tensor image);