#include <cuda_runtime.h>
#include <iostream>

#include "matrix.h"

template <int TILE_Y = 32, int TILE_X = 32>
__global__ void swizzle_transpose(float *A, int rows, int cols, float *B) {
  __shared__ float tile[TILE_Y][TILE_X];
  int r = blockDim.y * blockIdx.y + threadIdx.y; // row = (32 * by) + ty
  int c = blockDim.x * blockIdx.x + threadIdx.x; // col = (32 * bx) + tx
  int A_linear_idx = r * cols + c;               // linear 1-D index

  int swizzle_idx = (r ^ c) % TILE_X; // mouldo swizzle index by TILE_X to ensure valid indices
  int tile_x = r % TILE_Y;            // modulo rows by TILE_Y to ensure valid indices

  tile[tile_x][swizzle_idx] = A[A_linear_idx];

  __syncthreads();

  int r_out = blockDim.x * blockIdx.x + threadIdx.y; // row = (32 * bx) + ty
  int c_out = blockDim.y * blockIdx.y + threadIdx.x; // col = (32 * by) + tx
  int B_linear_idx = c_out * rows + r_out; // tranpose linear 1-D index

  B[B_linear_idx] = tile[tile_x][swizzle_idx];
}

int div_up(int a, int b) { return (a + b - 1) / b; }

int main() {
  constexpr int rows = 64;
  constexpr int cols = 64;
  constexpr int size = rows * cols;
  constexpr int size_bytes = size * sizeof(float);

  constexpr int TILE_X = 32;
  constexpr int TILE_Y = 32;

  float A[size];
  float B[size];

  initialize_matrix(A, rows, cols);
  std::cout << "Matrix A: " << std::endl;
  print_matrix<float, TILE_Y>(A, rows, cols);

  float *A_device, *B_device;

  cudaMalloc((void **)&A_device, size_bytes);
  cudaMalloc((void **)&B_device, size_bytes);

  cudaMemcpy(A_device, A, size_bytes, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(TILE_Y, TILE_X);
  dim3 blocksPerGrid(div_up(rows, threadsPerBlock.y),
                     div_up(cols, threadsPerBlock.x));

  swizzle_transpose<TILE_Y, TILE_X>
      <<<blocksPerGrid, threadsPerBlock>>>(A_device, rows, cols, B_device);

  cudaMemcpy(B, B_device, size_bytes, cudaMemcpyDeviceToHost);

  std::cout << "Matrix B: " << std::endl;
  print_swizzle_matrix<float, TILE_X>(B, cols, rows);

  cudaFree(A_device);
  cudaFree(B_device);

  return 0;
}