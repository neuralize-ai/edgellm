cmake_minimum_required(VERSION 3.14)

include(cmake/prelude.cmake)

project(
    edgellm
    VERSION 0.1.0
    DESCRIPTION "ML pipeline for LLM inference on edge devices"
    HOMEPAGE_URL "https://runlocal.ai"
    LANGUAGES CXX
)

include(cmake/project-is-top-level.cmake)
include(cmake/variables.cmake)

# ---- Declare library ----

add_library(
    edgellm_edgellm source/edgellm.cpp source/tokenizer.cpp source/sampler.cpp
)
add_library(edgellm::edgellm ALIAS edgellm_edgellm)

include(GenerateExportHeader)
generate_export_header(
    edgellm_edgellm
    BASE_NAME
    edgellm
    EXPORT_FILE_NAME
    export/edgellm/edgellm_export.hpp
    CUSTOM_CONTENT_FROM_VARIABLE
    pragma_suppress_c4251
)

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(edgellm_edgellm PUBLIC EDGELLM_STATIC_DEFINE)
endif()

set_target_properties(
    edgellm_edgellm
    PROPERTIES CXX_VISIBILITY_PRESET hidden
               VISIBILITY_INLINES_HIDDEN YES
               VERSION "${PROJECT_VERSION}"
               SOVERSION "${PROJECT_VERSION_MAJOR}"
               EXPORT_NAME edgellm
               OUTPUT_NAME edgellm
)

target_include_directories(
    edgellm_edgellm ${warning_guard}
    PUBLIC "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
)

target_include_directories(
    edgellm_edgellm SYSTEM
    PUBLIC "\$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/export>"
)

target_compile_features(edgellm_edgellm PUBLIC cxx_std_17)

find_package(fmt REQUIRED)
target_link_libraries(edgellm_edgellm PRIVATE fmt::fmt)

find_package(edgerunner REQUIRED)
target_link_libraries(edgellm_edgellm PUBLIC edgerunner::edgerunner)

# ---- Install rules ----

if(NOT CMAKE_SKIP_INSTALL_RULES)
    include(cmake/install-rules.cmake)
endif()

# ---- Examples ----

if(PROJECT_IS_TOP_LEVEL)
    option(BUILD_EXAMPLES "Build examples tree." "${edgellm_DEVELOPER_MODE}")
    if(BUILD_EXAMPLES)
        add_subdirectory(example)
    endif()
endif()

# ---- Developer mode ----

if(NOT edgellm_DEVELOPER_MODE)
    return()
elseif(NOT PROJECT_IS_TOP_LEVEL)
    message(
        AUTHOR_WARNING "Developer mode is intended for developers of edgellm"
    )
endif()

include(cmake/dev-mode.cmake)
