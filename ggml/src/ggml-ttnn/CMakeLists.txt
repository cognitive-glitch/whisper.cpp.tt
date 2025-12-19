list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../../cmake")

find_package(TTMetal REQUIRED)

file(GLOB GGML_SOURCES_TTNN "*.cpp")

ggml_add_backend_library(ggml-ttnn ${GGML_SOURCES_TTNN})
target_link_libraries(ggml-ttnn PRIVATE ${TT_LIBRARIES})
target_include_directories(ggml-ttnn PRIVATE ${TT_INCLUDE_DIRS})
set_target_properties(ggml-ttnn PROPERTIES CXX_STANDARD 17)

message(STATUS "TTNN: TT_INCLUDE_DIRS = ${TT_INCLUDE_DIRS}")
message(STATUS "TTNN: TT_LIBRARIES   = ${TT_LIBRARIES}")
