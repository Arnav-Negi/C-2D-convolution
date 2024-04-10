add_library(studentlib main.cpp
        check.cpp)
target_include_directories(studentlib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Define the directory where your files are located
set(MY_DIRECTORY ${CMAKE_BINARY_DIR})

# Define the file extension you're looking for
set(FILE_EXTENSION ".gcda")

# Use file(GLOB ...) to find files with the specified extension
file(GLOB FILES "${MY_DIRECTORY}/*${FILE_EXTENSION}")

# Check if the profile data exists
if(FILES)
    set(PROFILE_OPTIMIZATION_FLAGS "-fprofile-use")
else()
    set(PROFILE_OPTIMIZATION_FLAGS "-fprofile-generate")
endif()

#target_compile_options(studentlib PRIVATE -Ofast -march=native -g  -ftree-vectorize -funroll-loops -Wall -Wextra -fopenmp)
#target_link_libraries(studentlib PUBLIC -fopenmp -mavx2 -mavx -mfma -fsanitize=address -ffast-math)
target_compile_options(studentlib PRIVATE -O3 ${PROFILE_OPTIMIZATION_FLAGS} -march=native -funroll-loops -fopenmp)
target_link_libraries(studentlib PUBLIC -O3 ${PROFILE_OPTIMIZATION_FLAGS} -march=native -funroll-loops -fopenmp -mavx2 -mavx -mfma -ffast-math)
