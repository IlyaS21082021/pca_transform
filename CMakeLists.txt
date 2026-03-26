project(pca_transform)
include_directories(pca_transform /usr/local/include/eigen3)
add_executable(pca_transform main.cpp pca_transform.cpp)