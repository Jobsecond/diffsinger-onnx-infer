cmake_minimum_required(VERSION 3.20)
project(ds_onnx_infer)


set(CMAKE_CXX_STANDARD 17)

set(THIRDPARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty")
set(NUGET_ROOT "${THIRDPARTY_DIR}/nuget_packages")
set(DML_ROOT "${NUGET_ROOT}/microsoft.ai.directml/1.12.1")
set(DML_INCLUDE "${DML_ROOT}/include")

set(ONNX_ROOT "${NUGET_ROOT}/microsoft.ml.onnxruntime.directml/1.15.1")
set(ONNX_INCLUDE "${ONNX_ROOT}/build/native/include")
set(ONNX_LIB "${ONNX_ROOT}/runtimes/win-x64/native")


add_executable(${PROJECT_NAME}
        main.cpp
        TString.h
        TString.cpp
        ArrayUtil.hpp
        Preprocess.cpp
        Preprocess.h
        Inference.cpp
        Inference.h
        DsProject.h
        DsConfig.cpp
        DsConfig.h
        DsProject.cpp
        SampleCurve.cpp
        SampleCurve.h
        ModelData.h)


find_package(SndFile CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE SndFile::sndfile)

find_package(RapidJSON CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE rapidjson)

find_package(yaml-cpp CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE yaml-cpp)

find_package(argparse CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE argparse::argparse)

target_include_directories(${PROJECT_NAME} PUBLIC
        ${DML_INCLUDE} ${ONNX_INCLUDE}
)

target_link_directories(${PROJECT_NAME} PUBLIC
        ${ONNX_LIB}
)

target_link_libraries(${PROJECT_NAME} PUBLIC
        "user32.lib" "gdi32.lib" "onnxruntime.lib")

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
function(copy_ort_dlls target_name)
    if (MSVC)
        file(GLOB ORT_DLLS ${ONNXRUNTIME_ROOTDIR}/lib/*.dll)
        foreach(ORT_DLL ${ORT_DLLS})
            add_custom_command(TARGET ${target_name} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy ${ORT_DLL}  $<TARGET_FILE_DIR:${target_name}>)
        endforeach()
    endif()
endfunction()

