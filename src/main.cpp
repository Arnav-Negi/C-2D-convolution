#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")

#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>

namespace solution {
    std::string compute(const std::string &bitmap_path, const float kernel[3][3], const std::int32_t num_rows,
                        const std::int32_t num_cols) {
        std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.bmp";
        std::ofstream sol_fs(sol_path, std::ios::binary);
        std::ifstream bitmap_fs(bitmap_path, std::ios::binary);
        const auto img = std::make_unique<float[]>(num_rows * num_cols);
//		const auto ouptput = std::make_unique<float[]>(num_rows * num_cols);
        bitmap_fs.read(reinterpret_cast<char *>(img.get()), sizeof(float) * num_rows * num_cols);
        bitmap_fs.close();
        std::int32_t i, j, di, dj, ni, nj;
        float sum;

        // top left corner
        sum = 0.0;
        for (di = 0; di <= 1; di++)
            for (dj = 0; dj <= 1; dj++) {
                sum += kernel[di + 1][dj + 1] * img[di * num_cols + dj];
            }
        sol_fs.write(reinterpret_cast<char *>(&sum), sizeof(sum));

        // top row
        for (j = 1; j < num_cols - 1; j++) {
            sum = 0.0;
            for (di = 0; di <= 1; di++)
                for (dj = -1; dj <= 1; dj++) {
                    int ni = di, nj = j + dj;
                    sum += kernel[di + 1][dj + 1] * img[ni * num_cols + nj];
                }
            sol_fs.write(reinterpret_cast<char *>(&sum), sizeof(sum));
        }

        // top right corner
        sum = 0.0;
        for (di = 0; di <= 1; di++)
            for (dj = -1; dj <= 0; dj++) {
                int ni = di, nj = num_cols - 1 + dj;
                sum += kernel[di + 1][dj + 1] * img[ni * num_cols + nj];
            }
        sol_fs.write(reinterpret_cast<char *>(&sum), sizeof(sum));

        // center square
        for (i = 1; i < num_rows-1 ; i++) {
            // left cell
            sum = 0.0;
            for (di = -1; di <= 1; di++)
                for (dj = 0; dj <= 1; dj++) {
                    int ni = i + di, nj = dj;
                    sum += kernel[di + 1][dj + 1] * img[ni * num_cols + nj];
                }
            sol_fs.write(reinterpret_cast<char *>(&sum), sizeof(sum));

            // center
            for (j = 1; j < num_cols-1; j++) {
                sum = 0.0;
                for (int di = -1; di <= 1; di++)
                    for (int dj = -1; dj <= 1; dj++) {
                        int ni = i + di, nj = j + dj;
//                        if (ni >= 0 and ni < num_rows and nj >= 0 and nj < num_cols)
                            sum += kernel[di + 1][dj + 1] * img[ni * num_cols + nj];
                    }
                sol_fs.write(reinterpret_cast<char *>(&sum), sizeof(sum));
            }

            // right cell
            sum = 0.0;
            for (di = -1; di <= 1; di++)
                for (dj = -1; dj <= 0; dj++) {
                    int ni = i + di, nj = num_cols-1 + dj;
                    sum += kernel[di + 1][dj + 1] * img[ni * num_cols + nj];
                }
            sol_fs.write(reinterpret_cast<char *>(&sum), sizeof(sum));
        }

        // bottom left corner
        sum = 0.0;
        for (di = -1; di <= 0; di++)
            for (dj = 0; dj <= 1; dj++) {
                int ni = num_rows-1 + di, nj = dj;
                sum += kernel[di + 1][dj + 1] * img[ni * num_cols + nj];
            }
        sol_fs.write(reinterpret_cast<char *>(&sum), sizeof(sum));

        // bottom row
        for (j = 1; j < num_cols - 1; j++) {
            sum = 0.0;
            for (di = -1; di <= 0; di++)
                for (dj = -1; dj <= 1; dj++) {
                    int ni = num_rows-1 + di, nj = j + dj;
                    sum += kernel[di + 1][dj + 1] * img[ni * num_cols + nj];
                }
            sol_fs.write(reinterpret_cast<char *>(&sum), sizeof(sum));
        }

        // bottom right corner
        sum = 0.0;
        for (di = -1; di <= 0; di++)
            for (dj = -1; dj <= 0; dj++) {
                int ni = num_rows-1 + di, nj = num_cols-1 + dj;
                sum += kernel[di + 1][dj + 1] * img[ni * num_cols + nj];
            }
        sol_fs.write(reinterpret_cast<char *>(&sum), sizeof(sum));

        sol_fs.close();
        return sol_path;
    }
};