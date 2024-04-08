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
        constexpr std::int32_t BLOCK_SIZE = 64;
        constexpr std::int32_t NUM_THREADS = 8;

//        const float kernel1d[3] = {0.25f, 0.5f, 0.25f};

        // try raw pointer
        auto *padded_img = static_cast<float *>(malloc((num_rows + 2) * (num_cols + 2) * sizeof(float)));

        float *output_img;
        // direct io
        int fd = open(bitmap_path.c_str(), O_RDONLY);

        // mmap
        output_img = static_cast<float *>(mmap(nullptr, sizeof(float) * num_cols * num_rows, PROT_READ, MAP_PRIVATE, fd, 0));
//         using direct io
//        output_img = static_cast<float *>(malloc(sizeof(float) * num_cols * num_rows));
//        int read_err = read(fd, output_img, sizeof(float) * num_cols * num_rows);
//        if (read_err == -1) {
//            std::cout << "read error" << std::endl;
//            exit(1);
//        }
//        std::cout << "mmap read successful" << std::endl;

        // Padding
#pragma omp parallel for num_threads(NUM_THREADS) schedule(static) collapse(1) shared(padded_img, output_img)
        for (std::int32_t i = 0; i < num_rows; i++) {
            // use memcpy
            memcpy(padded_img + (i + 1) * (num_cols + 2) + 1, output_img + i * num_cols, sizeof(float) * num_cols);
        }
        close(fd);
        munmap(output_img, sizeof(float) * num_cols * num_rows);

//        std::cout << "memcpy successful" << std::endl;

        // mmap output_img with sol file
        fd = open(sol_path.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
        ftruncate(fd, sizeof(float) * num_cols * num_rows);
        output_img = static_cast<float *>(mmap(nullptr, sizeof(float) * num_cols * num_rows, PROT_READ | PROT_WRITE,
                                                  MAP_SHARED, fd, 0));

//        std::cout << "mmap write successful" << std::endl;

        // pad with zeros
        for (std::int32_t i = 0; i < num_rows + 2; i++) {
            padded_img[i * (num_cols + 2)] = 0.0;
            padded_img[i * (num_cols + 2) + num_cols + 1] = 0.0;
        }
        for (std::int32_t j = 0; j < num_cols + 2; j++) {
            padded_img[j] = 0.0;
            padded_img[(num_rows + 1) * (num_cols + 2) + j] = 0.0;
        }

//        std::cout << "padding successful" << std::endl;

        __m512 kernel_vec[3][3];
        for (std::int32_t i = 0; i < 3; i++) {
            for (std::int32_t j = 0; j < 3; j++) {
                kernel_vec[i][j] = _mm512_set1_ps(kernel[i][j]);
            }
        }

//        // kernel vec for 3x1 vector
//        __m512 kernel_vec[3];
//        for (std::int32_t i = 0; i < 3; i++) {
//            kernel_vec[i] = _mm512_set1_ps(kernel1d[i]);
//        }

//#pragma omp parallel for num_threads(NUM_THREADS) schedule(static) collapse(2) shared(padded_img, output_img, kernel_vec)
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

//        std::cout << "convolution successful" << std::endl;

#pragma omp parallel for num_threads(NUM_THREADS) schedule(static) collapse(2) shared(padded_img, output_img)  \
firstprivate(kernel_vec, num_cols, num_rows) default(none)
        for (std::int32_t i = 1; i < num_rows + 1; i++) {
            for (std::int32_t j = 1; j < num_cols + 1; j += VEC_SIZE) {
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
//    std::cout << "convolution successful" << std::endl;
        // unmap
        munmap(output_img, sizeof(float) * num_cols * num_rows);
//        close(fd);

//        free(padded_img);
        return sol_path;
    }
};