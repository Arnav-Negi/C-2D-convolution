add_library(studentlib main.cpp)
target_include_directories(studentlib PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_options(studentlib PRIVATE -Ofast -march=native)
target_link_libraries(studentlib PUBLIC -fopenmp -mavx2 -mfma)
