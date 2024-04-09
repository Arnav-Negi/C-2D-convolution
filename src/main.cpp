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

namespace solution {
    std::string compute(const std::string &bitmap_path, const float kernel[3][3], const std::int32_t num_rows,
                        const std::int32_t num_cols) {
        std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.bmp";

        constexpr std::int32_t VEC_SIZE = 16;
        constexpr std::int32_t NUM_THREADS = 4;


        // mmap
        int infd = open(bitmap_path.c_str(), O_RDONLY);
        auto input_img = static_cast<float *>(mmap(nullptr, sizeof(float) * num_cols * num_rows, PROT_READ, MAP_PRIVATE,
                                                   infd, 0));

        int outfd = open(sol_path.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
        ftruncate(outfd, sizeof(float) * num_cols * num_rows);
        auto output_img = static_cast<float *>(mmap(nullptr, sizeof(float) * num_cols * num_rows,
                                                    PROT_READ | PROT_WRITE, MAP_SHARED, outfd, 0));

        // direct IO
//        int infd = open(bitmap_path.c_str(), O_RDONLY | O_DIRECT);
//        float *input_img, *output_img;
//        posix_memalign((void **) &input_img, 512, sizeof(float) * num_cols * num_rows);
//        read(infd, input_img, sizeof(float) * num_cols * num_rows);
//        posix_memalign((void **) &output_img, 512, sizeof(float) * num_cols * num_rows);

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

#pragma omp parallel num_threads(NUM_THREADS) default(none) shared(input_img, output_img) firstprivate(kernel, kernel_vec, kernel_vec8, num_cols, num_rows)
        {

            // all corners along with 8 pixels along each side
#pragma omp single nowait
            {
                // non vectorized code for corners
                output_img[0] = kernel[1][1] * input_img[0] + kernel[1][2] * input_img[1] +
                                kernel[2][1] * input_img[num_cols] + kernel[2][2] * input_img[num_cols + 1];
                output_img[num_cols - 1] =
                        kernel[1][0] * input_img[num_cols - 2] + kernel[1][1] * input_img[num_cols - 1] +
                        kernel[2][0] * input_img[2 * num_cols - 2] +
                        kernel[2][1] * input_img[2 * num_cols - 1];
                output_img[num_cols * (num_rows - 1)] = kernel[0][1] * input_img[num_cols * (num_rows - 2)] +
                                                        kernel[0][2] * input_img[num_cols * (num_rows - 2) + 1] +
                                                        kernel[1][1] * input_img[num_cols * (num_rows - 1)] +
                                                        kernel[1][2] * input_img[num_cols * (num_rows - 1) + 1];
                output_img[num_cols * num_rows - 1] = kernel[0][0] * input_img[num_cols * (num_rows - 1) - 2] +
                                                      kernel[0][1] * input_img[num_cols * (num_rows - 1) - 1] +
                                                      kernel[1][0] * input_img[num_cols * num_rows - 2] +
                                                      kernel[1][1] * input_img[num_cols * num_rows - 1];
            }

            // top row
#pragma omp single nowait
            {
                // start from 1 to 14 no vectorize
                for (std::int32_t j = 1; j < 15; j++) {
                    output_img[j] = kernel[1][0] * input_img[j - 1] + kernel[1][1] * input_img[j] +
                                    kernel[1][2] * input_img[j + 1] + kernel[2][0] * input_img[num_cols + j - 1] +
                                    kernel[2][1] * input_img[num_cols + j] + kernel[2][2] * input_img[num_cols + j + 1];
                }

                // need to start from 15 and go till num_cols - 2
                for (std::int32_t j = 15; j < num_cols - 1; j += VEC_SIZE) {
                    __m512 sum = _mm512_setzero_ps();
                    for (std::int32_t di = 0; di <= 1; di++) {
                        for (std::int32_t dj = -1; dj <= 1; dj++) {
                            __m512 img_val = _mm512_loadu_ps(input_img + di * num_cols + j + dj);
                            sum = _mm512_fmadd_ps(kernel_vec[di + 1][dj + 1], img_val, sum);
                        }
                    }

                    // store the sum
                    _mm512_storeu_ps(output_img + j, sum);
                }
            };

            // bottom row
#pragma omp single nowait
            {
                // start from 1 to 14 no vectorize
                for (std::int32_t j = 1; j < 15; j++) {
                    output_img[(num_rows - 1) * num_cols + j] =
                            kernel[0][0] * input_img[(num_rows - 2) * num_cols + j - 1] +
                            kernel[0][1] * input_img[(num_rows - 2) * num_cols + j] +
                            kernel[0][2] * input_img[(num_rows - 2) * num_cols + j + 1] +
                            kernel[1][0] * input_img[(num_rows - 1) * num_cols + j - 1] +
                            kernel[1][1] * input_img[(num_rows - 1) * num_cols + j] +
                            kernel[1][2] * input_img[(num_rows - 1) * num_cols + j + 1];
                }
                // need to start from 15 and go till num_cols - 2
                for (std::int32_t j = 15; j < num_cols - 1; j += VEC_SIZE) {
                    __m512 sum = _mm512_setzero_ps();
                    for (std::int32_t di = -1; di <= 0; di++) {
                        for (std::int32_t dj = -1; dj <= 1; dj++) {
                            __m512 img_val = _mm512_loadu_ps(input_img + (num_rows + di - 1) * num_cols + j + dj);
                            sum = _mm512_fmadd_ps(kernel_vec[di + 1][dj + 1], img_val, sum);
                        }
                    }

                    // store the sum
                    _mm512_storeu_ps(output_img + (num_rows - 1) * num_cols + j, sum);
                }
            };

            // rightmost column - single thread no vectorise
#pragma omp single nowait
            {
                for (std::int32_t i = 1; i < num_rows - 1; i++) { // 6 multiply and adds
                    output_img[i * num_cols + num_cols - 1] =
                            kernel[0][0] * input_img[(i - 1) * num_cols + num_cols - 2] +
                            kernel[0][1] * input_img[(i - 1) * num_cols + num_cols - 1] +
                            kernel[1][0] * input_img[i * num_cols + num_cols - 2] +
                            kernel[1][1] * input_img[i * num_cols + num_cols - 1] +
                            kernel[2][0] * input_img[(i + 1) * num_cols + num_cols - 2] +
                            kernel[2][1] * input_img[(i + 1) * num_cols + num_cols - 1];
                }
            };

            // leftmost column - single thread no vectorise
#pragma omp single nowait
            {
                for (std::int32_t i = 1; i < num_rows - 1; i++) { // 6 multiply and adds
                    output_img[i * num_cols] = kernel[0][1] * input_img[(i - 1) * num_cols] +
                                               kernel[0][2] * input_img[(i - 1) * num_cols + 1] +
                                               kernel[1][1] * input_img[i * num_cols] +
                                               kernel[1][2] * input_img[i * num_cols + 1] +
                                               kernel[2][1] * input_img[(i + 1) * num_cols] +
                                               kernel[2][2] * input_img[(i + 1) * num_cols + 1];
                }
            };

            // for each row 1 to num_rows - 2, column 1 to 14
#pragma omp for schedule(static) collapse(1) nowait
            for (int i = 1; i < num_rows - 1; ++i) {
                // start from 1 to 6, no vectorize
                for (int j = 1; j <= 6; ++j) {
                    for (int di = -1; di <= 1; ++di) {
                        for (int dj = -1; dj <= 1; ++dj) {
                            output_img[i * num_cols + j] +=
                                    kernel[di + 1][dj + 1] * input_img[(i + di) * num_cols + j + dj];
                        }
                    }
                }

                // 7 to 14, vectorize
                __m256 sum = _mm256_setzero_ps();
                for (int di = -1; di <= 1; ++di) {
                    for (int dj = -1; dj <= 1; ++dj) {
                        __m256 img_val = _mm256_loadu_ps(input_img + (i + di) * num_cols + 7 + dj);
                        sum = _mm256_fmadd_ps(kernel_vec8[di + 1][dj + 1], img_val, sum);
                    }
                }
                // store
                _mm256_storeu_ps(output_img + i * num_cols + 7, sum);
            }

            // for each row 1 to num_rows - 2, column 15 to num_cols - 2 - main pixels
#pragma omp for schedule(static) collapse(2) nowait
            for (int i = 1; i < num_rows - 1; ++i) {
                for (int j = 15; j < num_cols - 1; j += VEC_SIZE) {
                    __m512 sum = _mm512_setzero_ps();
                    for (int di = -1; di <= 1; ++di) {
                        for (int dj = -1; dj <= 1; ++dj) {
                            __m512 img_val = _mm512_loadu_ps(input_img + (i + di) * num_cols + j + dj);
                            sum = _mm512_fmadd_ps(kernel_vec[di + 1][dj + 1], img_val, sum);
                        }
                    }

                    // store the sum
                    _mm512_storeu_ps(output_img + i * num_cols + j, sum);
                }
            }
# pragma omp barrier
        }
//    std::cout << "convolution successful" << std::endl;

        // write the output image
//        int outfd = open(sol_path.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_DIRECT, S_IRWXU);
//        write(outfd, output_img, sizeof(float) * num_cols * num_rows);
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