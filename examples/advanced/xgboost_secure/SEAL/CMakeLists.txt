cmake_minimum_required(VERSION 3.13)
project(HistHE VERSION 4.1.1 LANGUAGES CXX)

# Import Microsoft SEAL
find_package(SEAL 4.1.1 EXACT REQUIRED)

add_executable(HistHE hist_he.cpp)
target_link_libraries(HistHE PRIVATE SEAL::seal)
