#include <cuda_runtime.h>
#include <iostream>

#include "matrix.h"

int div_up(int a, int b) { return (a + b - 1) / b; }

template <int TILE_Y = 32, int TILE_X = 32>
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

  int r_out = blockDim.x * blockIdx.x + threadIdx.y; // row = (32 * bx) + ty
  int c_out = blockDim.y * blockIdx.y + threadIdx.x; // col = (32 * by) + tx
  int B_linear_idx = c_out * rows + r_out; // tranpose linear 1-D index

  B[B_linear_idx] =
      tile[tile_y][swizzle_idx]; // bank-conflict free swizzled load
}

template <typename T, int TILE_Y = 32, int TILE_X = 32>
__global__ void
swizzle_bilinear_interpolation(const T *__restrict__ A, int input_rows,
                               int input_cols, T *__restrict__ B, int out_rows,
                               int out_cols, float scale_y, float scale_x) {
  __shared__ T ATile[TILE_Y][TILE_X];
  // __shared__ BTile[32][32];

  int r = blockDim.y * blockIdx.y + threadIdx.y;
  int c = blockDim.x * blockIdx.x + threadIdx.x;


  // int swizzle_idx =
  //     (r ^ c) %
  //     TILE_X; // mouldo swizzle index by TILE_X to ensure valid indices
  int tile_x = c % TILE_X;
  int tile_y = r % TILE_Y; // modulo rows by TILE_Y to ensure valid indices

  // ATile[tile_y][tile_x] = __ldg(&A[r * input_cols + c]); // possibly remove ldg
  ATile[tile_y][tile_x] = A[r * input_cols + c]; // possibly remove ldg

  __syncthreads();

  float r_offset = r * scale_y;
  float c_offset = c * scale_x;

  int x1 = floor(c_offset);
  int y1 = floor(r_offset);
  int x2 = ceil(c_offset);
  int y2 = ceil(r_offset);

  float dx = c_offset - (float)x1;
  float dy = r_offset - (float)y1;

  x1 %= TILE_X;
  y1 %= TILE_Y;
  x2 %= TILE_X;
  y2 %= TILE_Y;

  float dxp = (1 - dx);
  float dyp = (1 - dy);

  T ap = ATile[y1][x1];
  T bp = ATile[y1][x2];
  T cp = ATile[y2][x1];
  T dp = ATile[y2][x2];

  // The Parallel Method to Reduce Individual Elements Products into a Final Sum???
  T pixel = ap * dxp * dyp + bp * dx * dyp + cp * dy * dxp + dp * dx * dy;
  int B_linear_idx = r * out_cols + c;

  B[B_linear_idx] = pixel;
}

template <int Rows, int Cols, int TILE_Y = 32, int TILE_X = 32>
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

  dim3 threadsPerBlock(TILE_Y, TILE_X);
  dim3 blocksPerGrid(div_up(Rows, threadsPerBlock.y),
                     div_up(Cols, threadsPerBlock.x));

  swizzle_transpose<TILE_Y, TILE_X>
      <<<blocksPerGrid, threadsPerBlock>>>(A_device, Rows, Cols, B_device);

  cudaMemcpy(B, B_device, size_bytes, cudaMemcpyDeviceToHost);

  std::cout << "Matrix B: " << std::endl;
  print_swizzle_matrix<float, TILE_X>(B, Cols, Rows);

  cudaFree(A_device);
  cudaFree(B_device);
}

template <typename T, int Rows, int Cols, int Out_Rows, int Out_Cols,
          int TILE_Y = 32, int TILE_X = 32>
void launch_swizzle_bilinear() {
  constexpr int size = Rows * Cols;
  constexpr int size_bytes = size * sizeof(T);

  constexpr int out_size = Out_Rows * Out_Cols;
  constexpr int out_size_bytes = out_size * sizeof(T);

  T A[size];
  T B[out_size];

  initialize_matrix(A, Rows, Cols);
  std::cout << "Matrix A: " << std::endl;
  print_matrix<T, TILE_Y>(A, Rows, Cols);

  T *A_device, *B_device;

  cudaMalloc((void **)&A_device, size_bytes);
  cudaMalloc((void **)&B_device, out_size_bytes);

  cudaMemcpy(A_device, A, size_bytes, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(TILE_Y, TILE_X);
  dim3 blocksPerGrid(div_up(Out_Rows, threadsPerBlock.y),
                     div_up(Out_Cols, threadsPerBlock.x));

  constexpr float scale_x = (Cols - 1.) / (Out_Cols - 1.);
  constexpr float scale_y = (Rows - 1.) / (Out_Rows - 1.);

  swizzle_bilinear_interpolation<T, TILE_Y, TILE_X>
      <<<blocksPerGrid, threadsPerBlock>>>(
          A_device, Rows, Cols, B_device, Out_Rows, Out_Cols, scale_y, scale_x);

  cudaMemcpy(B, B_device, out_size_bytes, cudaMemcpyDeviceToHost);

  std::cout << "Matrix B: " << std::endl;
  print_swizzle_matrix<T, TILE_X>(B, Out_Rows, Out_Cols);

  cudaFree(A_device);
  cudaFree(B_device);
}

int main() {
  constexpr int rows = 128;
  constexpr int cols = 128;

  constexpr int out_rows = 1024;
  constexpr int out_cols = 1024;

  constexpr int TILE_X = 32;
  constexpr int TILE_Y = 32;

  launch_swizzle_transpose<rows, cols, TILE_Y, TILE_X>();
  launch_swizzle_bilinear<float, rows, cols, out_rows, out_cols, TILE_Y,
                          TILE_X>();
  return 0;
}