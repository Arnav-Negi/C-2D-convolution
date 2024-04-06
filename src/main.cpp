#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")

#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>
#include <immintrin.h>
#include <omp.h>

namespace solution {
    std::string compute(const std::string &bitmap_path, const float kernel[3][3], const std::int32_t num_rows,
                        const std::int32_t num_cols) {
        std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.bmp";
        std::ofstream sol_fs(sol_path, std::ios::binary);
        std::ifstream bitmap_fs(bitmap_path, std::ios::binary);

//        const auto img = std::make_unique<float[]>(num_rows * num_cols);
//        const auto padded_img = std::make_unique<float[]>((num_rows + 2) * (num_cols + 2));
        // try raw pointer
        auto *padded_img = static_cast<float *>(malloc((num_rows + 2) * (num_cols + 2) * sizeof(float)));
        auto *output_img = static_cast<float *>(malloc((num_rows + 2) * (num_cols + 2) * sizeof(float)));
//        const auto horizontal_conv_img = std::make_unique<float[]>((num_rows + 2) * (num_cols + 2));
//        bitmap_fs.read(reinterpret_cast<char *>(img.get()), sizeof(float) * num_rows * num_cols);

        // Padding
        for (std::int32_t i = 0; i < num_rows; i++) {
            bitmap_fs.read(reinterpret_cast<char *>(padded_img + (i + 1) * (num_cols + 2) + 1),
                           sizeof(float) * num_cols);
        }
        bitmap_fs.close();
        // pad with zeros
        for (std::int32_t i = 0; i < num_rows + 2; i++) {
            padded_img[i * (num_cols + 2)] = 0.0;
            padded_img[i * (num_cols + 2) + num_cols + 1] = 0.0;
        }
        for (std::int32_t j = 0; j < num_cols + 2; j++) {
            padded_img[j] = 0.0;
            padded_img[(num_rows + 1) * (num_cols + 2) + j] = 0.0;
        }

        __m512 kernel_vec[3][3];
        for (std::int32_t i = 0; i < 3; i++) {
            for (std::int32_t j = 0; j < 3; j++) {
                kernel_vec[i][j] = _mm512_set1_ps(kernel[i][j]);
            }
        }

#pragma omp parallel for collapse(1) schedule(static) num_threads(8) shared(padded_img, output_img, kernel_vec)
        for (std::int32_t i = 1; i < num_rows + 1; i++) {
            for (std::int32_t j = 1; j < num_cols + 1; j += 16) {
                __m512 sum = _mm512_setzero_ps();
                for (std::int32_t di = -1; di <= 1; di++) {
                    for (std::int32_t dj = -1; dj <= 1; dj++) {
//                            sum += kernel[di + 1][dj + 1] * padded_img[(i + di) * (num_cols + 2) + j + dj];
                        __m512 img_val = _mm512_loadu_ps(padded_img + (i + di) * (num_cols + 2) + j + dj);
                        sum = _mm512_fmadd_ps(kernel_vec[di + 1][dj + 1], img_val, sum);
                    }
                }
                // store the sum
//                sol_fs.write(reinterpret_cast<char *>(&sum), sizeof(sum));
                _mm512_storeu_ps(output_img + i * (num_cols + 2) + j, sum);
            }
        }

        // write the output image
        for (std::int32_t i = 1; i < num_rows + 1; i++) {
            sol_fs.write(reinterpret_cast<char *>(output_img + i * (num_cols + 2) + 1), sizeof(float) * num_cols);
        }

        sol_fs.close();
        free(padded_img);
        free(output_img);
        return sol_path;
    }
};