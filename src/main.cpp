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

//        const float kernel1d[3] = {0.25f, 0.5f, 0.25f};

        // try raw pointer
        auto *padded_img = static_cast<float *>(malloc((num_rows + 2) * (num_cols + 2) * sizeof(float)));
        auto *output_img = static_cast<float *>(malloc((num_rows + 2) * (num_cols + 2) * sizeof(float)));

        std::FILE *file = std::fopen(bitmap_path.c_str(), "rb");
        // Padding
        for (std::int32_t i = 0; i < num_rows; i++) {
            // use fread
            std::fread(padded_img + (i + 1) * (num_cols + 2) + 1, sizeof(float), num_cols, file);
        }
        std::fclose(file);

        // pad with zeros
        for (std::int32_t i = 0; i < num_rows + 2; i++) {
            padded_img[i * (num_cols + 2)] = 0.0;
            padded_img[i * (num_cols + 2) + num_cols + 1] = 0.0;
        }
        for (std::int32_t j = 0; j < num_cols + 2; j++) {
            padded_img[j] = 0.0;
            padded_img[(num_rows + 1) * (num_cols + 2) + j] = 0.0;
        }

        __m256 kernel_vec[3][3];
        for (std::int32_t i = 0; i < 3; i++) {
            for (std::int32_t j = 0; j < 3; j++) {
                kernel_vec[i][j] = _mm256_set1_ps(kernel[i][j]);
            }
        }
        std::int32_t reduced_cols = num_cols / 8;
#pragma omp parallel for collapse(1) schedule(static) num_threads(16) shared(padded_img, output_img, kernel_vec)
        for (std::int32_t k = 0; k < num_rows * reduced_cols; k++) {
//            for (std::int32_t j = 1; j < num_cols + 1; j += 16) {
            std::int32_t i = k / reduced_cols + 1;
            std::int32_t j = (k % reduced_cols) * 16 + 1;
            __m256 sum = _mm256_setzero_ps();
            for (std::int32_t di = -1; di <= 1; di++) {
                for (std::int32_t dj = -1; dj <= 1; dj++) {
                    __m256 img_val = _mm256_loadu_ps(padded_img + (i + di) * (num_cols + 2) + j + dj);
                    sum = _mm256_fmadd_ps(kernel_vec[di + 1][dj + 1], img_val, sum);
                }
            }
            // store the sum
            _mm256_storeu_ps(output_img + (i - 1) * num_cols + j - 1, sum);
        }
//                _mm256_storeu_ps(output_img + i * (num_cols + 2) + j, sum);
//            }

//        __m256 kernel_vec[3];
//        for (std::int32_t i = 0; i < 3; i++) {
//            kernel_vec[i] = _mm256_set1_ps(kernel1d[i]);
//        }

        // horizontal pass
//#pragma omp parallel for collapse(1) num_threads(8) shared(padded_img, output_img, kernel_vec)
//        for (std::int32_t i = 1; i < num_rows + 1; i++) {
//            for (std::int32_t j = 1; j < num_cols + 1; j+=16) {
//                __m256 sum = _mm256_setzero_ps();
//                    __m256 img_val = _mm256_loadu_ps(padded_img + i * (num_cols + 2) + j -1);
//                    sum = _mm256_fmadd_ps(kernel_vec[0], img_val, sum);
//                    img_val = _mm256_loadu_ps(padded_img + i * (num_cols + 2) + j + 0);
//                    sum = _mm256_fmadd_ps(kernel_vec[1], img_val, sum);
//                    img_val = _mm256_loadu_ps(padded_img + i * (num_cols + 2) + j + 1);
//                    sum = _mm256_fmadd_ps(kernel_vec[2], img_val, sum);
//
//                _mm256_storeu_ps(output_img + i * (num_cols + 2) + j, sum);
//            }
//        }
//
//        // vertical pass - from output_img to padded_img
//#pragma omp parallel for collapse(1) num_threads(8) shared(padded_img, output_img, kernel_vec)
//        for (std::int32_t i = 1; i < num_rows + 1; i++) {
//            for (std::int32_t j = 1; j < num_cols + 1; j+=16) {
//                __m256 sum = _mm256_setzero_ps();
//                    __m256 img_val = _mm256_loadu_ps(output_img + (i -1) * (num_cols + 2) + j);
//                    sum = _mm256_fmadd_ps(kernel_vec[0], img_val, sum);
//                    img_val = _mm256_loadu_ps(output_img + (i ) * (num_cols + 2) + j);
//                    sum = _mm256_fmadd_ps(kernel_vec[1], img_val, sum);
//                    img_val = _mm256_loadu_ps(output_img + (i + 1) * (num_cols + 2) + j);
//                    sum = _mm256_fmadd_ps(kernel_vec[2], img_val, sum);
//
////                store in padded_img without padding
//                _mm256_storeu_ps(padded_img + (i - 1) * num_cols + j-1, sum);
//            }
//        }

        std::FILE *sol_fs = std::fopen(sol_path.c_str(), "wb");
        std::fwrite(padded_img, sizeof(float), num_rows * num_cols, sol_fs);
        std::fclose(sol_fs);

        free(padded_img);
        free(output_img);
        return sol_path;
    }
};