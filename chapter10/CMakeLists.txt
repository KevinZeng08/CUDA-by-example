cmake_minimum_required(VERSION 3.10)

project(chapter10)

find_package(OpenMP REQUIRED)

set(SOURCE_FILES
    basic_single_stream.cu
    basic_double_stream.cu
    basic_double_stream_correct.cu
)

set(SOURCE_FILES_WITH_OMP
    vectorAdd.cu
    vectorAdd_pin_memory.cu
    vectorAdd_single_stream.cu
    vectorAdd_double_stream.cu
    vectorAdd_triple_stream.cu
)

foreach(SOURCE_FILE IN LISTS SOURCE_FILES)
    get_filename_component(TARGET_NAME ${SOURCE_FILE} NAME_WE)

    add_executable(${TARGET_NAME} ${SOURCE_FILE})

    set_target_properties(${TARGET_NAME} PROPERTIES
        CUDA_ARCHITECTURES "86;80;75;72")
endforeach()

foreach(SOURCE_FILE IN LISTS SOURCE_FILES_WITH_OMP)
    get_filename_component(TARGET_NAME ${SOURCE_FILE} NAME_WE)

    add_executable(${TARGET_NAME} ${SOURCE_FILE})

    set_target_properties(${TARGET_NAME} PROPERTIES
        CUDA_ARCHITECTURES "86;80;75;72")
    
    if (OpenMP_CXX_FOUND)
        target_link_libraries(${TARGET_NAME} PRIVATE OpenMP::OpenMP_CXX)
    endif()
endforeach()