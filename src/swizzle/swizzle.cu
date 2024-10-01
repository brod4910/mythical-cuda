#include <cuda_runtime.h>
#include <iostream>

#include "matrix.h"
#include "cuutil.h"

template <int TILE_X = 32, int TILE_Y = 32>
__global__ void swizzle_transpose(float *A, int rows, int cols, float *B) {
  __shared__ float tile[TILE_Y][TILE_X];
  int r = blockDim.y * blockIdx.y + threadIdx.y; // row = (32 * by) + ty
  int c = blockDim.x * blockIdx.x + threadIdx.x; // col = (32 * bx) + tx
  int A_linear_idx = r * cols + c;               // linear 1-D index

  int swizzle_idx =
      (r ^ c) %
      TILE_X; // mouldo swizzle index by TILE_X to ensure valid indices
  int tile_y = r % TILE_Y; // modulo rows by TILE_Y to ensure valid indices

  tile[tile_y][swizzle_idx] =
      A[A_linear_idx]; // bank-conflict free swizzled store

  __syncthreads();

  // TODO: Fix indices of destination, currently transposing the tile 
  // and not the whole matrix
  int r_out = blockDim.x * blockIdx.x + threadIdx.y; // row = (32 * bx) + ty
  int c_out = blockDim.y * blockIdx.y + threadIdx.x; // col = (32 * by) + tx
  int B_linear_idx = c_out * rows + r_out; // tranpose linear 1-D index

  B[B_linear_idx] =
      tile[tile_y][swizzle_idx]; // bank-conflict free swizzled load
}

template <int Rows, int Cols, int TILE_X = 32, int TILE_Y = 32>
void launch_swizzle_transpose() {
  constexpr int size = Rows * Cols;
  constexpr int size_bytes = size * sizeof(float);

  float A[size];
  float B[size];

  initialize_matrix(A, Rows, Cols);
  std::cout << "Matrix A: " << std::endl;
  print_matrix<float, TILE_Y>(A, Rows, Cols);

  float *A_device, *B_device;

  cudaMalloc((void **)&A_device, size_bytes);
  cudaMalloc((void **)&B_device, size_bytes);

  cudaMemcpy(A_device, A, size_bytes, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(TILE_X, TILE_Y);
  dim3 blocksPerGrid(div_up(Rows, threadsPerBlock.y),
                     div_up(Cols, threadsPerBlock.x));

  swizzle_transpose<TILE_X, TILE_Y>
      <<<blocksPerGrid, threadsPerBlock>>>(A_device, Rows, Cols, B_device);

  cudaMemcpy(B, B_device, size_bytes, cudaMemcpyDeviceToHost);

  std::cout << "Matrix B: " << std::endl;
  print_swizzle_matrix<float, TILE_X>(B, Cols, Rows);

  cudaFree(A_device);
  cudaFree(B_device);
}

int main() {
  constexpr int rows = 128;
  constexpr int cols = 128;

  constexpr int TILE_X = 32;
  constexpr int TILE_Y = 32;

  launch_swizzle_transpose<rows, cols, TILE_X, TILE_Y>();

  return 0;
}