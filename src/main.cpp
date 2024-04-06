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
//        std::ofstream sol_fs(sol_path, std::ios::binary);
//        std::ifstream bitmap_fs(bitmap_path, std::ios::binary);

        const float kernel1d[3] = {0.25f, 0.5f, 0.25f};

//        const auto img = std::make_unique<float[]>(num_rows * num_cols);
//        const auto padded_img = std::make_unique<float[]>((num_rows + 2) * (num_cols + 2));
        // try raw pointer
        auto *padded_img = static_cast<float *>(malloc((num_rows + 2) * (num_cols + 2) * sizeof(float)));
        auto *output_img = static_cast<float *>(malloc((num_rows) * (num_cols) * sizeof(float)));
//        const auto horizontal_conv_img = std::make_unique<float[]>((num_rows + 2) * (num_cols + 2));
//        bitmap_fs.read(reinterpret_cast<char *>(img.get()), sizeof(float) * num_rows * num_cols);

        std::FILE *file = std::fopen(bitmap_path.c_str(), "rb");
        // Padding
        for (std::int32_t i = 0; i < num_rows; i++) {
//            bitmap_fs.read(reinterpret_cast<char *>(padded_img + (i + 1) * (num_cols + 2) + 1),
//                           sizeof(float) * num_cols);
// use fread
            std::fread(padded_img + (i + 1) * (num_cols + 2) + 1, sizeof(float), num_cols, file);
        }
        std::fclose(file);
//        bitmap_fs.close();
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
        const std::int32_t VEC_SIZE = 16;
        const std::int32_t BLOCK_SIZE = 512;

        std::FILE *sol_fs = std::fopen(sol_path.c_str(), "wb");
#pragma omp parallel for num_threads(24) schedule(static) collapse(2) shared(padded_img, output_img, kernel_vec)
        // blocking the image
        for (std::int32_t ii = 1; ii < num_rows + 1; ii += BLOCK_SIZE) {
            for (std::int32_t jj = 1; jj < num_cols + 1; jj += BLOCK_SIZE) {
                for (std::int32_t i = ii; i < ii + BLOCK_SIZE; i++) {
                    for (std::int32_t j = jj; j < jj + BLOCK_SIZE; j += VEC_SIZE) {
                        __m512 sum = _mm512_setzero_ps();
                        for (std::int32_t di = -1; di <= 1; di++) {
                            for (std::int32_t dj = -1; dj <= 1; dj++) {
                                __m512 img_val = _mm512_loadu_ps(padded_img + (i + di) * (num_cols + 2) + j + dj);
                                sum = _mm512_fmadd_ps(kernel_vec[di + 1][dj + 1], img_val, sum);
                            }
                        }

                        // store the sum
                        _mm512_storeu_ps(output_img + (i-1) * (num_cols) + j-1, sum);
                    }
                }
            }
        }
//        for (std::int32_t i = 1; i < num_rows + 1; i++) {
//            for (std::int32_t j = 1; j < num_cols + 1; j += 8) {
//                __m512 sum = _mm512_setzero_ps();
//                for (std::int32_t di = -1; di <= 1; di++) {
//                    for (std::int32_t dj = -1; dj <= 1; dj++) {
//                        __m512 img_val = _mm512_loadu_ps(padded_img + (i + di) * (num_cols + 2) + j + dj);
//                        sum = _mm512_fmadd_ps(kernel_vec[di + 1][dj + 1], img_val, sum);
//                    }
//                }
//
//                // store the sum
//                _mm512_storeu_ps(output_img + (i-1) * (num_cols) + j-1, sum);
//            }
//        }

        // write the output image
//        for (std::int32_t i = 1; i < num_rows + 1; i++) {
////            sol_fs.write(reinterpret_cast<char *>(output_img + i * (num_cols + 2) + 1), sizeof(float) * num_cols);
//            std::fwrite(output_img + i * (num_cols + 2) + 1, sizeof(float), num_cols, sol_fs);
//        }
        std::fwrite(output_img, sizeof(float), num_rows * num_cols, sol_fs);

        std::fclose(sol_fs);
        free(padded_img);
        free(output_img);
        return sol_path;
    }
};