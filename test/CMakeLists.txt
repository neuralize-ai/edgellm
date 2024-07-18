cmake_minimum_required(VERSION 3.14)

project(edgellmTests LANGUAGES CXX)

include(../cmake/project-is-top-level.cmake)
include(../cmake/folders.cmake)

# ---- Dependencies ----

if(PROJECT_IS_TOP_LEVEL)
  find_package(edgellm REQUIRED)
  enable_testing()
endif()

find_package(Catch2 REQUIRED)
include(Catch)

# ---- Tests ----

add_executable(edgellm_test source/edgellm_test.cpp)
target_link_libraries(
    edgellm_test PRIVATE
    edgellm::edgellm
    Catch2::Catch2WithMain
)
target_compile_features(edgellm_test PRIVATE cxx_std_17)

catch_discover_tests(edgellm_test)

# ---- End-of-file commands ----

add_folders(Test)
