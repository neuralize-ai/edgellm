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

# ---- Test data ----

if(ANDROID)
    foreach(dir ${CONAN_RUNTIME_LIB_DIRS})
        file(GLOB_RECURSE shared_libs "${dir}/*.so")
        file(COPY ${shared_libs} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    endforeach()
else()
    set(MODELS_DIR "${CMAKE_SOURCE_DIR}/models")
    set(MODELS_DEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/models")
    if(UNIX)
        execute_process(COMMAND ln -sfn ${MODELS_DIR} ${MODELS_DEST_DIR})
    elseif(WIN32)
        execute_process(
            COMMAND cmd.exe /c mklink ${MODELS_DEST_DIR} ${MODELS_DIR}
        )
    endif()
endif()

# ---- Tests ----

add_executable(edgellm_test source/edgellm_test.cpp source/tokenizer_test.cpp)
target_link_libraries(
    edgellm_test PRIVATE edgellm::edgellm fmt::fmt Catch2::Catch2WithMain
)
target_compile_features(edgellm_test PRIVATE cxx_std_17)

if(ANDROID)
    add_custom_target(
        test-android
        COMMAND "${CMAKE_SOURCE_DIR}/scripts/run_with_adb.sh" -b
                "${CMAKE_CURRENT_BINARY_DIR}" -e "edgellm_test"
        VERBATIM
    )
    add_dependencies(test-android edgellm_test)
else()
    catch_discover_tests(edgellm_test)
endif()

# ---- End-of-file commands ----

add_folders(Test)
