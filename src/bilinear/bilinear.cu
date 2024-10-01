#include "matrix.h"
#include "cuutil.h"

template <typename T, int TILE_X = 32, int TILE_Y = 32>
__global__ void
bilinear_interpolation_v1(const T *__restrict__ A, int input_rows,
                               int input_cols, T *__restrict__ B, int out_rows,
                               int out_cols, float scale_y, float scale_x) {
  int r = blockDim.y * blockIdx.y + threadIdx.y;
  int c = blockDim.x * blockIdx.x + threadIdx.x;

  float x = c * scale_x;
  float y = r * scale_y;

  int x_floor = floor(x);
  int y_floor = floor(y);
  int x_ceil = x_floor + 1;
  int y_ceil = y_floor + 1;

  float d_x = x - x_floor;
  float d_y = y - y_floor;

  // The Parallel Method to Reduce Individual Elements Products into a Final Sum???
  float w00 = (1 - d_x) * (1 - d_y);
  float w01 = d_x * (1 - d_y);
  float w10 = (1 - d_x) * d_y;
  float w11 = d_x * d_y;

  T ap = A[y_floor * input_cols + x_floor];
  T bp = A[y_floor * input_cols + x_ceil];;
  T cp = A[y_ceil * input_cols + x_floor];;
  T dp = A[y_ceil * input_cols + x_ceil];;

  T q00 = ap * w00;
  T q01 = bp * w01;
  T q10 = cp * w10;
  T q11 = dp * w11;

  T pixel = q00 + q01 + q10 + q11;
  B[r * out_cols + c] = pixel;
}

template <typename T, int Rows, int Cols, int Out_Rows, int Out_Cols,
          int TILE_X = 32, int TILE_Y = 32>
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

  // dim3 threadsPerBlock(TILE_X, TILE_Y);
  // dim3 blocksPerGrid(div_up(Out_Rows, threadsPerBlock.y),
  //                    div_up(Out_Cols, threadsPerBlock.x));

  constexpr float scale_x = (Cols - 1.) / (Out_Cols - 1.);
  constexpr float scale_y = (Rows - 1.) / (Out_Rows - 1.);
  
  // TODO: only works in the upsampling case
  constexpr int thread_x = (Out_Cols - 1.) / (Cols - 1.);
  constexpr int thread_y = (Out_Rows - 1.) / (Rows - 1.);

  dim3 threadsPerBlock(TILE_X, TILE_Y);
  dim3 blocksPerGrid(div_up(Out_Rows, threadsPerBlock.y),
                     div_up(Out_Cols, threadsPerBlock.x));


  bilinear_interpolation_v1<T, TILE_X, TILE_Y>
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

  launch_swizzle_bilinear<float, rows, cols, out_rows, out_cols, TILE_X,
                          TILE_Y>();
  return 0;
}