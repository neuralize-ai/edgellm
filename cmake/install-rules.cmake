if(PROJECT_IS_TOP_LEVEL)
  set(
      CMAKE_INSTALL_INCLUDEDIR "include/edgellm-${PROJECT_VERSION}"
      CACHE STRING ""
  )
  set_property(CACHE CMAKE_INSTALL_INCLUDEDIR PROPERTY TYPE PATH)
endif()

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# find_package(<package>) call for consumers to find this project
set(package edgellm)

install(
    DIRECTORY
    include/
    "${PROJECT_BINARY_DIR}/export/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT edgellm_Development
)

install(
    TARGETS edgellm_edgellm
    EXPORT edgellmTargets
    RUNTIME #
    COMPONENT edgellm_Runtime
    LIBRARY #
    COMPONENT edgellm_Runtime
    NAMELINK_COMPONENT edgellm_Development
    ARCHIVE #
    COMPONENT edgellm_Development
    INCLUDES #
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

write_basic_package_version_file(
    "${package}ConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion
)

# Allow package maintainers to freely override the path for the configs
set(
    edgellm_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/${package}"
    CACHE STRING "CMake package config location relative to the install prefix"
)
set_property(CACHE edgellm_INSTALL_CMAKEDIR PROPERTY TYPE PATH)
mark_as_advanced(edgellm_INSTALL_CMAKEDIR)

install(
    FILES cmake/install-config.cmake
    DESTINATION "${edgellm_INSTALL_CMAKEDIR}"
    RENAME "${package}Config.cmake"
    COMPONENT edgellm_Development
)

install(
    FILES "${PROJECT_BINARY_DIR}/${package}ConfigVersion.cmake"
    DESTINATION "${edgellm_INSTALL_CMAKEDIR}"
    COMPONENT edgellm_Development
)

install(
    EXPORT edgellmTargets
    NAMESPACE edgellm::
    DESTINATION "${edgellm_INSTALL_CMAKEDIR}"
    COMPONENT edgellm_Development
)

if(PROJECT_IS_TOP_LEVEL)
  include(CPack)
endif()
