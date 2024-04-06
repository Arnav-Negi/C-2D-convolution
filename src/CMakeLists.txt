add_library(studentlib main.cpp)
target_include_directories(studentlib PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_options(studentlib PRIVATE  -fopenmp -march=native -Wall -Wextra -mavx2 -mfma -ffast-math -Ofast)
target_link_libraries(studentlib PUBLIC -Ofast -march=native -Wall -Wextra -fopenmp -mavx2 -mfma -ffast-math)
