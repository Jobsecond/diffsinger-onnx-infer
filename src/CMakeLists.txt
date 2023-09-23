cmake_minimum_required(VERSION 3.20)
project(ds_onnx_infer)


set(CMAKE_CXX_STANDARD 17)

set(THIRDPARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty")
set(NUGET_ROOT "${THIRDPARTY_DIR}/nuget_packages")

# Load NuGet packages
file(READ "../nuget_packages.txt" nuget_packages)
string(REPLACE "\n" ";" pkg_list ${nuget_packages})
foreach(_item ${pkg_list})
    string(REPLACE "=" ";" pair ${_item})
    list(GET pair 0 pkg_name)
    string(STRIP ${pkg_name} pkg_name)
    list(GET pair 1 pkg_ver)
    string(STRIP pkg_ver ${pkg_ver})

    set(package_root "${NUGET_ROOT}/${pkg_name}.${pkg_ver}")
    if(pkg_name STREQUAL "Microsoft.AI.DirectML")
        message("-- NuGet ${pkg_name} (${pkg_ver})")
        set(DML_ROOT "${package_root}")
        set(DML_INCLUDE "${DML_ROOT}/include")
        set(DML_LIB "${DML_ROOT}/bin/x64-win")
    elseif(pkg_name STREQUAL "Microsoft.ML.OnnxRuntime.DirectML")
        message("-- NuGet ${pkg_name} (${pkg_ver})")
        set(ONNX_DML_ROOT "${package_root}")
        set(ONNX_DML_INCLUDE "${ONNX_DML_ROOT}/build/native/include")
        set(ONNX_DML_LIB "${ONNX_DML_ROOT}/runtimes/win-x64/native")
    endif()
endforeach()


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
        ModelData.h
        SpeakerEmbed.cpp
        SpeakerEmbed.h)


find_package(SndFile CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE SndFile::sndfile)

find_package(RapidJSON CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE rapidjson)

find_package(yaml-cpp CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE yaml-cpp)

find_package(argparse CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE argparse::argparse)

target_include_directories(${PROJECT_NAME} PUBLIC
        ${DML_INCLUDE} ${ONNX_DML_INCLUDE}
)

target_link_directories(${PROJECT_NAME} PUBLIC
        ${ONNX_DML_LIB}
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
function(copy_dlls target_name)
    if (MSVC)
        file(GLOB ORT_DLLS ${ONNX_DML_LIB}/*.dll)
        foreach(ORT_DLL ${ORT_DLLS})
            add_custom_command(TARGET ${target_name} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy ${ORT_DLL}  $<TARGET_FILE_DIR:${target_name}>)
        endforeach()
        list(APPEND DML_DLLS "DirectML.dll")
        if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
            list(APPEND DML_DLLS "DirectML.Debug.dll")
        endif()
        foreach(DML_DLL ${DML_DLLS})
            add_custom_command(TARGET ${target_name} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy ${DML_LIB}/${DML_DLL}  $<TARGET_FILE_DIR:${target_name}>)
        endforeach()
    endif()
endfunction()

copy_dlls(${PROJECT_NAME})
