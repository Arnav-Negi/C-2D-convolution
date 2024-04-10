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
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <sys/stat.h>
#include <stdlib.h>

namespace solution {
    std::string compute(const std::string &bitmap_path, const float kernel[3][3], const std::int32_t num_rows,
                        const std::int32_t num_cols) {
        std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.bmp";

        constexpr std::int32_t VEC_SIZE = 16;
        constexpr std::int32_t NUM_THREADS = 4;


//         mmap
        int infd = open(bitmap_path.c_str(), O_RDONLY);
        auto input_img = static_cast<float *>(mmap(nullptr, sizeof(float) * num_cols * num_rows, PROT_READ, MAP_PRIVATE,
                                                   infd, 0));

        int outfd = open(sol_path.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
        ftruncate(outfd, sizeof(float) * num_cols * num_rows);
        auto output_img = static_cast<float *>(mmap(nullptr, sizeof(float) * num_cols * num_rows,
                                                    PROT_READ | PROT_WRITE, MAP_SHARED, outfd, 0));

        __m512 kernel_vec[3][3];
        for (std::int32_t i = 0; i < 3; i++) {
            for (std::int32_t j = 0; j < 3; j++) {
                kernel_vec[i][j] = _mm512_set1_ps(kernel[i][j]);
            }
        }

        __m256 kernel_vec8[3][3];
        for (std::int32_t i = 0; i < 3; i++) {
            for (std::int32_t j = 0; j < 3; j++) {
                kernel_vec8[i][j] = _mm256_set1_ps(kernel[i][j]);
            }
        }

        __m128 kernel_vec4[3][3];
        for (std::int32_t i = 0; i < 3; i++) {
            for (std::int32_t j = 0; j < 3; j++) {
                kernel_vec4[i][j] = _mm_set1_ps(kernel[i][j]);
            }
        }

        setenv("OMP_NUM_THREADS", std::to_string(NUM_THREADS).c_str(), 1);
        setenv("OMP_PROC_BIND", "true", 1);
#pragma omp parallel num_threads(NUM_THREADS) default(none) shared(input_img, output_img) firstprivate(kernel, kernel_vec, kernel_vec8, kernel_vec4, num_cols, num_rows)
        {
            // top row
#pragma omp single nowait
            {
                // top left corner
                output_img[0] = kernel[1][1] * input_img[0] + kernel[1][2] * input_img[1] +
                                kernel[2][1] * input_img[num_cols] + kernel[2][2] * input_img[num_cols + 1];

                // 1 - 2 no vectorise
#pragma unroll 2
                for (std::int32_t j = 1; j < 3; j++) {
                    output_img[j] = kernel[1][0] * input_img[j - 1] + kernel[1][1] * input_img[j] +
                                    kernel[1][2] * input_img[j + 1] +
                                    kernel[2][0] * input_img[num_cols + j - 1] +
                                    kernel[2][1] * input_img[num_cols + j] +
                                    kernel[2][2] * input_img[num_cols + j + 1];
                }

                // 3 - 6 using kernel_vec4
                for (std::int32_t j = 3; j < 7; j += 4) {
                    __m128 sum = _mm_setzero_ps();
#pragma unroll 2
                    for (std::int32_t di = 0; di <= 1; di++) {
#pragma unroll 3
                        for (std::int32_t dj = -1; dj <= 1; dj++) {
                            __m128 img_val = _mm_loadu_ps(input_img + di * num_cols + j + dj);
                            sum = _mm_fmadd_ps(kernel_vec4[di + 1][dj + 1], img_val, sum);
                        }
                    }

                    // store the sum
                    _mm_storeu_ps(output_img + j, sum);
                }

                // 7 - 14 using kernel_vec8
                for (std::int32_t j = 7; j < 15; j += 8) {
                    __m256 sum = _mm256_setzero_ps();
#pragma unroll 2
                    for (std::int32_t di = 0; di <= 1; di++) {
#pragma unroll 3
                        for (std::int32_t dj = -1; dj <= 1; dj++) {
                            __m256 img_val = _mm256_loadu_ps(input_img + di * num_cols + j + dj);
                            sum = _mm256_fmadd_ps(kernel_vec8[di + 1][dj + 1], img_val, sum);
                        }
                    }

                    // store the sum
                    _mm256_storeu_ps(output_img + j, sum);
                }

                // need to start from 15 and go till num_cols - 2
                for (std::int32_t j = 15; j < num_cols - 1; j += VEC_SIZE) {
                    __m512 sum = _mm512_setzero_ps();
#pragma unroll 2
                    for (std::int32_t di = 0; di <= 1; di++) {
#pragma unroll 3
                        for (std::int32_t dj = -1; dj <= 1; dj++) {
                            __m512 img_val = _mm512_loadu_ps(input_img + di * num_cols + j + dj);
                            sum = _mm512_fmadd_ps(kernel_vec[di + 1][dj + 1], img_val, sum);
                        }
                    }

                    // store the sum
                    _mm512_storeu_ps(output_img + j, sum);
                }

                // top right corner
                output_img[num_cols - 1] =
                        kernel[1][0] * input_img[num_cols - 2] + kernel[1][1] * input_img[num_cols - 1] +
                        kernel[2][0] * input_img[2 * num_cols - 2] +
                        kernel[2][1] * input_img[2 * num_cols - 1];
            };

            // for each row 1 to num_rows - 2, column 0 seperately, then 1 to 14, then 15 to num_cols - 2, then num_cols - 1
#pragma omp for schedule(static) collapse(1) nowait
            for (int i = 1; i < num_rows - 1; ++i) {
                // column 0
                output_img[i * num_cols] = kernel[0][1] * input_img[(i - 1) * num_cols] +
                                           kernel[0][2] * input_img[(i - 1) * num_cols + 1] +
                                           kernel[1][1] * input_img[i * num_cols] +
                                           kernel[1][2] * input_img[i * num_cols + 1] +
                                           kernel[2][1] * input_img[(i + 1) * num_cols] +
                                           kernel[2][2] * input_img[(i + 1) * num_cols + 1];

                // 1 - 2 no vectorize
#pragma unroll 2
                for (int j = 1; j < 3; j++) {
                    output_img[i * num_cols + j] =
                            kernel[0][0] * input_img[(i - 1) * num_cols + j - 1] +
                            kernel[0][1] * input_img[(i - 1) * num_cols + j] +
                            kernel[0][2] * input_img[(i - 1) * num_cols + j + 1] +
                            kernel[1][0] * input_img[i * num_cols + j - 1] +
                            kernel[1][1] * input_img[i * num_cols + j] +
                            kernel[1][2] * input_img[i * num_cols + j + 1] +
                            kernel[2][0] * input_img[(i + 1) * num_cols + j - 1] +
                            kernel[2][1] * input_img[(i + 1) * num_cols + j] +
                            kernel[2][2] * input_img[(i + 1) * num_cols + j + 1];
                }

                // 3 - 6 using kernel_vec4
                for (int j = 3; j < 7; j += 4) {
                    __m128 sum = _mm_setzero_ps();
#pragma unroll 3
                    for (int di = -1; di <= 1; di++) {
#pragma unroll 3
                        for (int dj = -1; dj <= 1; dj++) {
                            __m128 img_val = _mm_loadu_ps(input_img + (i + di) * num_cols + j + dj);
                            sum = _mm_fmadd_ps(kernel_vec4[di + 1][dj + 1], img_val, sum);
                        }
                    }

                    // store the sum
                    _mm_storeu_ps(output_img + i * num_cols + j, sum);
                }

                // 7 - 14 using kernel_vec8
                for (int j = 7; j < 15; j += 8) {
                    __m256 sum = _mm256_setzero_ps();
#pragma unroll 3
                    for (int di = -1; di <= 1; di++) {
#pragma unroll 3
                        for (int dj = -1; dj <= 1; dj++) {
                            __m256 img_val = _mm256_loadu_ps(input_img + (i + di) * num_cols + j + dj);
                            sum = _mm256_fmadd_ps(kernel_vec8[di + 1][dj + 1], img_val, sum);
                        }
                    }

                    // store the sum
                    _mm256_storeu_ps(output_img + i * num_cols + j, sum);
                }

                // need to start from 15 and go till num_cols - 2
                for (int j = 15; j < num_cols - 1; j += VEC_SIZE) {
                    __m512 sum = _mm512_setzero_ps();
#pragma unroll 3
                    for (int di = -1; di <= 1; ++di) {
#pragma unroll 3
                        for (int dj = -1; dj <= 1; ++dj) {
                            __m512 img_val = _mm512_loadu_ps(input_img + (i + di) * num_cols + j + dj);
                            sum = _mm512_fmadd_ps(kernel_vec[di + 1][dj + 1], img_val, sum);
                        }
                    }

                    // store the sum
                    _mm512_storeu_ps(output_img + i * num_cols + j, sum);
                }

                // last column
                output_img[i * num_cols + num_cols - 1] =
                        kernel[0][0] * input_img[(i - 1) * num_cols + num_cols - 2] +
                        kernel[0][1] * input_img[(i - 1) * num_cols + num_cols - 1] +
                        kernel[1][0] * input_img[i * num_cols + num_cols - 2] +
                        kernel[1][1] * input_img[i * num_cols + num_cols - 1] +
                        kernel[2][0] * input_img[(i + 1) * num_cols + num_cols - 2] +
                        kernel[2][1] * input_img[(i + 1) * num_cols + num_cols - 1];
            }

            // bottom row
#pragma omp single nowait
            {
                // bottom left corner
                output_img[(num_rows - 1) * num_cols] = kernel[0][1] * input_img[(num_rows - 2) * num_cols] +
                                                        kernel[0][2] * input_img[(num_rows - 2) * num_cols + 1] +
                                                        kernel[1][1] * input_img[(num_rows - 1) * num_cols] +
                                                        kernel[1][2] * input_img[(num_rows - 1) * num_cols + 1];

                // 1 - 2 no vectorize
#pragma unroll 2
                for (std::int32_t j = 1; j < 3; j++) {
                    output_img[(num_rows - 1) * num_cols + j] =
                            kernel[0][0] * input_img[(num_rows - 2) * num_cols + j - 1] +
                            kernel[0][1] * input_img[(num_rows - 2) * num_cols + j] +
                            kernel[0][2] * input_img[(num_rows - 2) * num_cols + j + 1] +
                            kernel[1][0] * input_img[(num_rows - 1) * num_cols + j - 1] +
                            kernel[1][1] * input_img[(num_rows - 1) * num_cols + j] +
                            kernel[1][2] * input_img[(num_rows - 1) * num_cols + j + 1];
                }

                // 3 - 6 using kernel_vec4
                for (std::int32_t j = 3; j < 7; j += 4) {
                    __m128 sum = _mm_setzero_ps();
#pragma unroll 2
                    for (std::int32_t di = -1; di <= 0; di++) {
#pragma unroll 3
                        for (std::int32_t dj = -1; dj <= 1; dj++) {
                            __m128 img_val = _mm_loadu_ps(input_img + (num_rows + di - 1) * num_cols + j + dj);
                            sum = _mm_fmadd_ps(kernel_vec4[di + 1][dj + 1], img_val, sum);
                        }
                    }

                    // store the sum
                    _mm_storeu_ps(output_img + (num_rows - 1) * num_cols + j, sum);
                }

                // 7 - 14 using kernel_vec8
                for (std::int32_t j = 7; j < 15; j += 8) {
                    __m256 sum = _mm256_setzero_ps();
#pragma unroll 2
                    for (std::int32_t di = -1; di <= 0; di++) {
#pragma unroll 3
                        for (std::int32_t dj = -1; dj <= 1; dj++) {
                            __m256 img_val = _mm256_loadu_ps(input_img + (num_rows + di - 1) * num_cols + j + dj);
                            sum = _mm256_fmadd_ps(kernel_vec8[di + 1][dj + 1], img_val, sum);
                        }
                    }

                    // store the sum
                    _mm256_storeu_ps(output_img + (num_rows - 1) * num_cols + j, sum);
                }

                // need to start from 15 and go till num_cols - 2
                for (std::int32_t j = 15; j < num_cols - 1; j += VEC_SIZE) {
                    __m512 sum = _mm512_setzero_ps();
#pragma unroll 2
                    for (std::int32_t di = -1; di <= 0; di++) {
#pragma unroll 3
                        for (std::int32_t dj = -1; dj <= 1; dj++) {
                            __m512 img_val = _mm512_loadu_ps(input_img + (num_rows + di - 1) * num_cols + j + dj);
                            sum = _mm512_fmadd_ps(kernel_vec[di + 1][dj + 1], img_val, sum);
                        }
                    }

                    // store the sum
                    _mm512_storeu_ps(output_img + (num_rows - 1) * num_cols + j, sum);
                }

                // bottom right corner
                output_img[num_rows * num_cols - 1] = kernel[0][0] * input_img[(num_rows - 1) * num_cols - 2] +
                                                      kernel[0][1] * input_img[(num_rows - 1) * num_cols - 1] +
                                                      kernel[1][0] * input_img[num_rows * num_cols - 2] +
                                                      kernel[1][1] * input_img[num_rows * num_cols - 1];
            };
//# pragma omp barrier
        }
//    std::cout << "convolution successful" << std::endl;
        return sol_path;
    }
};


//// Blocking code

//#pragma omp parallel for num_threads(NUM_THREADS) schedule(static) collapse(2) shared(padded_img, output_img)  \
//firstprivate(kernel_vec, num_cols, num_rows) default(none)
//         // blocking the image
//        for (std::int32_t ii = 1; ii < num_rows + 1; ii += BLOCK_SIZE) {
//            for (std::int32_t jj = 1; jj < num_cols + 1; jj += BLOCK_SIZE) {
//                for (std::int32_t i = ii; i < ii + BLOCK_SIZE; i++) {
//                    for (std::int32_t j = jj; j < jj + BLOCK_SIZE; j += VEC_SIZE) {
//                        __m512 sum = _mm512_setzero_ps();
//                        for (std::int32_t di = -1; di <= 1; di++) {
//                            for (std::int32_t dj = -1; dj <= 1; dj++) {
//                                __m512 img_val = _mm512_loadu_ps(padded_img + (i + di) * (num_cols + 2) + j + dj);
//                                sum = _mm512_fmadd_ps(static_cast<__m512>(kernel_vec[di + 1][dj + 1]), img_val, sum);
//                            }
//                        }
//
//                        // store the sum
//                        _mm512_storeu_ps(output_img + (i - 1) * (num_cols) + j - 1, sum);
//                    }
//                }
//            }
//        }

//#pragma omp parallel for num_threads(NUM_THREADS) schedule(static) collapse(2) shared(padded_img, output_img)  \
//firstprivate(kernel_vec, num_cols, num_rows) default(none)
//        for (std::int32_t i = 1; i < num_rows + 1; i++) {
//            for (std::int32_t j = 1; j < num_cols + 1; j += VEC_SIZE) {
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