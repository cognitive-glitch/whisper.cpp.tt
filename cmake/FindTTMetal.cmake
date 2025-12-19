# - Try to find TT-Metalium (TTNN) SDK
#
# Once done, this will define:
#  TTMetal_FOUND      - system has TT-Metalium
#  TT_INCLUDE_DIRS    - include directories
#  TT_LIBRARIES       - libraries to link

include(FindPackageHandleStandardArgs)

set(TT_HOME $ENV{TT_METAL_HOME})

find_path(TTNN_INCLUDE_DIR
    NAMES ttnn/device.hpp
    PATHS ${TT_HOME}/include
    NO_DEFAULT_PATH)

find_library(TTNN_LIBRARY
    NAMES ttnn
    PATHS ${TT_HOME}/build/lib
    NO_DEFAULT_PATH)

find_library(TT_METAL_LIBRARY
    NAMES tt_metal
    PATHS ${TT_HOME}/build/lib
    NO_DEFAULT_PATH)

find_library(YAML_CPP_LIBRARY
    NAMES yaml-cpp
    PATHS ${TT_HOME}/build/lib
    NO_DEFAULT_PATH)

find_package_handle_standard_args(TTMetal DEFAULT_MSG
    TTNN_LIBRARY
    TT_METAL_LIBRARY
    TTNN_INCLUDE_DIR)

if (TTMetal_FOUND)
    set(TT_LIBRARIES ${TTNN_LIBRARY} ${TT_METAL_LIBRARY} ${YAML_CPP_LIBRARY})
    set(TT_INCLUDE_DIRS ${TTNN_INCLUDE_DIR})
endif()

mark_as_advanced(
    TTNN_INCLUDE_DIR
    TTNN_LIBRARY
    TT_METAL_LIBRARY
    YAML_CPP_LIBRARY)
