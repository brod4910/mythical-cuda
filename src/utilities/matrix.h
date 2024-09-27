#include <array>
#include <iostream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

// Function to create and return an array of fully constructed ANSI escape codes
constexpr std::array<const char *, 33> createColorLUT() {
  return {{
      "\033[38;2;255;0;0m",     // Red
      "\033[38;2;0;255;0m",     // Green
      "\033[38;2;0;0;255m",     // Blue
      "\033[38;2;255;255;0m",   // Yellow
      "\033[38;2;255;0;255m",   // Magenta
      "\033[38;2;0;255;255m",   // Cyan
      "\033[38;2;128;0;0m",     // Dark Red
      "\033[38;2;0;128;0m",     // Dark Green
      "\033[38;2;0;0;128m",     // Dark Blue
      "\033[38;2;128;128;0m",   // Olive
      "\033[38;2;128;0;128m",   // Purple
      "\033[38;2;0;128;128m",   // Teal
      "\033[38;2;192;192;192m", // Silver
      "\033[38;2;128;128;128m", // Gray
      "\033[38;2;64;0;0m",      // Maroon
      "\033[38;2;0;64;0m",      // Dark Green 2
      "\033[38;2;0;0;64m",      // Dark Navy
      "\033[38;2;64;64;0m",     // Dark Olive Green
      "\033[38;2;64;0;64m",     // Dark Slate Blue
      "\033[38;2;0;64;64m",     // Teal 2
      "\033[38;2;255;128;0m",   // Orange
      "\033[38;2;128;255;0m",   // Lime
      "\033[38;2;128;0;255m",   // Violet
      "\033[38;2;0;255;128m",   // Spring Green
      "\033[38;2;0;128;255m",   // Azure
      "\033[38;2;255;0;128m",   // Hot Pink
      "\033[38;2;192;64;0m",    // Burnt Orange
      "\033[38;2;64;192;0m",    // Dark Lime
      "\033[38;2;0;64;192m",    // Dodger Blue
      "\033[38;2;192;0;64m",    // Crimson
      "\033[38;2;64;0;192m",    // Medium Purple
      "\033[38;2;192;192;64m",  // Light Yellow Green
      "\033[0m"                 // Reset
  }};
}

template <typename T> void print_matrix(T *matrix, int rows, int cols) {
  constexpr auto colorLUT = createColorLUT();
  auto RESET = colorLUT[32];

  for (int r = 0; r < rows; ++r) {
    auto COLOR = colorLUT[r % 32];

    std::cout << COLOR << "[";
    for (int c = 0; c < cols; ++c) {
      std::cout << matrix[r * cols + c] << " ";
    }
    std::cout << "]" << RESET << std::endl;
  }
}

template <typename T> void print_swizzle_matrix(T *matrix, int rows, int cols) {
  constexpr auto colorLUT = createColorLUT();
  auto RESET = colorLUT[32];

  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      int swizzle_idx = (r ^ c) % 32;
      auto COLOR = colorLUT[swizzle_idx];

      std::cout << COLOR << "[";
      std::cout << matrix[r * cols + c] << " ";
    }
    std::cout << "]" << RESET << std::endl;
  }
}

template <typename T> void initialize_matrix(T *matrix, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      matrix[r * cols + c] = c;
    }
  }
}