cmake_minimum_required(VERSION 3.20)
project(ds_onnx_infer)


set(CMAKE_CXX_STANDARD 17)

option(ENABLE_DML "Enable DirectML support" off)
option(ENABLE_CUDA "Enable CUDA support" off)

if(ENABLE_DML)
    message("-- ONNX Runtime DirectML is enabled")
endif()

if(ENABLE_CUDA)
    message("-- ONNX Runtime CUDA is enabled")
endif()

if(NOT DEFINED ONNXRUNTIME_INCLUDE_PATH)  # ONNX Runtime include path
    message("-- Notice: ONNXRUNTIME_INCLUDE_PATH is not set. The build system will use system default paths.")
else()
    message("-- ONNXRUNTIME_INCLUDE_PATH is set to \"${ONNXRUNTIME_INCLUDE_PATH}\"")
endif()

if(NOT DEFINED ONNXRUNTIME_LIB_PATH)  # ONNX Runtime library path
    message("-- Notice: ONNXRUNTIME_LIB_PATH is not set. The build system will use system default paths.")
else()
    message("-- ONNXRUNTIME_LIB_PATH is set to \"${ONNXRUNTIME_LIB_PATH}\"")
endif()


if(ENABLE_DML)
    if(NOT DEFINED DML_INCLUDE_PATH)  # DirectML include path
        message("-- Notice: DML_INCLUDE_PATH is not set. The build system will use system default paths.")
    else()
        message("-- DML_INCLUDE_PATH is set to \"${DML_INCLUDE_PATH}\"")
    endif()

    if(NOT DEFINED DML_LIB_PATH)  # DirectML library path
        message("-- Notice: DML_LIB_PATH is not set. The build system will use system default paths.")
    else()
        message("-- DML_LIB_PATH is set to \"${DML_LIB_PATH}\"")
    endif()
endif()

set(THIRDPARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty")

add_executable(${PROJECT_NAME}
        main.cpp
        TString.h
        TString.cpp
        PowerManagement.cpp
        PowerManagement.h
        ArrayUtil.hpp
        Preprocess.cpp
        Preprocess.h
        DsProject.h
        DsConfig.cpp
        DsConfig.h
        DsProject.cpp
        SampleCurve.cpp
        SampleCurve.h
        ModelData.h
        SpeakerEmbed.cpp
        SpeakerEmbed.h
        Inference/Inference.cpp
        Inference/Inference.h
        Inference/AcousticModelFlags.h
        Inference/InferenceUtils.hpp
        Inference/AcousticInference.cpp
        Inference/AcousticInference.h
        Inference/VocoderInference.cpp
        Inference/VocoderInference.h
        Inference/LinguisticInference.cpp
        Inference/LinguisticInference.h
        Inference/PitchInference.cpp
        Inference/PitchInference.h
        Inference/VarianceInference.cpp
        Inference/VarianceInference.h
        Inference/PhonemeDurInference.cpp
        Inference/PhonemeDurInference.h
)

# Add current directory to include path
target_include_directories(${PROJECT_NAME} PRIVATE .)


find_package(SndFile CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE SndFile::sndfile)

find_package(RapidJSON CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE rapidjson)

find_package(yaml-cpp CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE yaml-cpp)

find_package(argparse CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE argparse::argparse)

# ONNX Runtime libraries

target_include_directories(${PROJECT_NAME} PRIVATE
        ${ONNXRUNTIME_INCLUDE_PATH}
)

target_link_directories(${PROJECT_NAME} PRIVATE
        ${ONNXRUNTIME_LIB_PATH}
)

if(ENABLE_DML)
    if(DEFINED DML_INCLUDE_PATH)
        target_include_directories(${PROJECT_NAME} PRIVATE
                ${DML_INCLUDE_PATH}
        )
    endif()
    if(DEFINED DML_LIB_PATH)
        target_link_directories(${PROJECT_NAME} PRIVATE
                ${DML_LIB_PATH}
        )
    endif()
    target_compile_definitions(${PROJECT_NAME} PRIVATE
            ONNXRUNTIME_ENABLE_DML
    )
endif()

if(ENABLE_CUDA)
    target_compile_definitions(${PROJECT_NAME} PRIVATE
            ONNXRUNTIME_ENABLE_CUDA
    )
endif()

if(WIN32 AND MSVC)
    target_link_libraries(${PROJECT_NAME} PRIVATE
            "user32.lib" "gdi32.lib" "onnxruntime.lib")
else()
    target_link_libraries(${PROJECT_NAME} PRIVATE
            "-lonnxruntime")
endif()

# fix gcc build
if (CMAKE_COMPILER_IS_GNUCC
        AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 8.0
        AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(${PROJECT_NAME} PRIVATE
            "stdc++fs")
endif()


#[[
if(WIN32)
    set(microsoft_wil_SOURCE_DIR "${THIRDPARTY_DIR}/wil")
    add_library(wil INTERFACE)
    target_include_directories(wil INTERFACE ${microsoft_wil_SOURCE_DIR}/include)
    set(WIL_LIB wil)
endif()
#]]

# On Linux the samples use libjpeg and libpng for decoding images.
# On Windows they use Windows Image Component(WIC)
if(NOT WIN32)
    find_package(JPEG)
    if(LIBPNG_ROOTDIR)
        set(PNG_FOUND true)
        set(PNG_LIBRARIES png16)
        set(PNG_INCLUDE_DIRS "${LIBPNG_ROOTDIR}/include")
        set(PNG_LIBDIR "${LIBPNG_ROOTDIR}/lib")
    else()
        find_package(PNG)
    endif()
endif()

# Windows might have an onnxruntime.dll in the system directory so it's more robust to manually copy the dlls to
# the output dir. Define a function to do so. This is called from the cmake file in the subdirectories.
function(copy_dlls target_name)
    if (MSVC)
        if(ONNXRUNTIME_LIB_PATH)
            file(GLOB ORT_DLLS ${ONNXRUNTIME_LIB_PATH}/*.dll)
            foreach(ORT_DLL ${ORT_DLLS})
                add_custom_command(TARGET ${target_name} POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy ${ORT_DLL}  $<TARGET_FILE_DIR:${target_name}>)
            endforeach()
        endif()
        if(ENABLE_DML AND DML_LIB_PATH)
            list(APPEND DML_DLLS "DirectML.dll")
            if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
                list(APPEND DML_DLLS "DirectML.Debug.dll")
            endif()
            foreach(DML_DLL ${DML_DLLS})
                add_custom_command(TARGET ${target_name} POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy ${DML_LIB_PATH}/${DML_DLL}  $<TARGET_FILE_DIR:${target_name}>)
            endforeach()
        endif()
    endif()
endfunction()

copy_dlls(${PROJECT_NAME})
