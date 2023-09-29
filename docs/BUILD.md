## Build Instructions

### Requirements

- Toolchains
  - A compiler that supports C++17
  - CMake
  - vcpkg
  - NuGet (optional)
- Third-party libraries:
  - ONNX Runtime
  - yaml-cpp
  - rapidjson
  - libsndfile
  - argparse

### Steps

#### 1. Download ONNX Runtime libraries

You can download pre-built ONNX Runtime binaries from [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) or NuGet. ONNX Runtime has three official builds for inference (see [here](https://onnxruntime.ai/docs/get-started/with-cpp.html) for details):

- CPU
- GPU (supports CUDA)
- DirectML (Windows only).

You can download ONNX Runtime using our PowerShell script (see step 1a), or download manually (see step 1b).

##### 1a. Download using PowerShell script

We provided a PowerShell script to fetch ONNX Runtime from NuGet. 

Usage:

```bash
powershell -ExecutionPolicy Bypass -File scripts/download_nuget.ps1
    [-OrtBuild <build>] [-Platform <platform>]
    [-OutConfig <filepath>] [-Overwrite]
```

**Parameters:**

- **`-OrtBuild <build>`**
  Specify ONNX Runtime build type. The values are case-insensitive.
  
  | Build Type     | Values                                            |
  |----------------|---------------------------------------------------|
  | CPU            | `cpu`<br>This is the default and fallback option. |
  | GPU (CUDA)     | `gpu` `cuda`                                      |
  | GPU (DirectML) | `dml` `directml`                                  |
  
  If omitted or provided invalid values, this parameter will be set to `cpu`.

- **`-Platform <platform>`**
  Specifiy ONNX Runtime platform. For example: `win-x64`, `osx-arm64`, `osx-x64`, `linux-x64`.
  If omitted, this parameter will be set to `win-x64`.

- **`-OutConfig <filepath>`** **`-Overwrite`**
  
  If `-OutConfig` parameter is provided, the script will write CMake configure command arguments to `<filepath>`.
  
  If `-Overwrite`, the script will overwrite the existing file in `<filepath>`, otherwise, the script will append to the existing file. After executing the script, it will output CMake configure options, which will be used for configuring CMake project in step 3.

##### 1b. Download manually

You can download from [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) releases or NuGet manually.

If you choose DirectML version of ONNX Runtime, you also need to download the [DirectML library](https://www.nuget.org/packages/Microsoft.AI.DirectML/).

You will need to specify **include path** and **lib path** for each library in step 3.

#### 2. Download other third-party libraries using vcpkg

##### Windows

```bash
git clone https://github.com/microsoft/vcpkg
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg install --triplet=x64-windows "libsndfile[core]" rapidjson yaml-cpp argparse
```

##### Linux and macOS

```bash
git clone https://github.com/microsoft/vcpkg
cd vcpkg
./bootstrap-vcpkg.sh
./vcpkg install --triplet=<TRIPLET> "libsndfile[core]" rapidjson yaml-cpp argparse
```

Replace `<TRIPLET>` with your platform triplet. For Linux, it is usually `x64-linux`, and for macOS, it can be `arm64-osx` (Apple Silicon) or `x64-osx` (Intel).

#### 3. Configure and build using CMake

CMake optional configure options:

| Variable                   | Type   | Description                                                                            |
| -------------------------- | ------ | -------------------------------------------------------------------------------------- |
| `ONNXRUNTIME_INCLUDE_PATH` | `PATH` | ONNX Runtime include path                                                              |
| `ONNXRUNTIME_LIB_PATH`     | `PATH` | ONNX Runtime library path                                                              |
| `DML_INCLUDE_PATH`         | `PATH` | DirectML include path<br>(required if configuring with DirectML build of ONNX Runtime) |
| `DML_LIB_PATH`             | `PATH` | DirectML library path<br>(required if configuring with DirectML build of ONNX Runtime) |
| `ENABLE_CUDA`              | `BOOL` | Enable CUDA execution provider support.                                                |
| `ENABLE_DML`               | `BOOL` | Enable DirectML execution provider support.                                            |

##### Configure

In the project root directory, execute CMake commands to configure.

```bash
cmake -G Ninja -B your_build_dir -S . \
  -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DVCPKG_TARGET_TRIPLET=<triplet> \
  <options>
```

Make sure to append the optional configure options to the command line.

If you used PowerShell to download packages, the CMake command line options are generated for you, just append it to the command line.

For example, if you want to build with DirectML support, you can execute:

```bash
cmake -G Ninja -B your_build_dir -S . \
  -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DVCPKG_TARGET_TRIPLET=x64-windows \
  -DONNXRUNTIME_INCLUDE_PATH="path/to/ort/include" \
  -DONNXRUNTIME_LIB_PATH="path/to/ort/lib" \
  -DDML_INCLUDE_PATH="path/to/dml/include" \
  -DDML_LIB_PATH="path/to/dml/lib" \
  -DENABLE_DML:BOOL=ON
```

If you want to build with CUDA support, you can execute:

```bash
cmake -G Ninja -B your_build_dir -S . \
  -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DVCPKG_TARGET_TRIPLET=x64-windows \
  -DONNXRUNTIME_INCLUDE_PATH="path/to/ort/include" \
  -DONNXRUNTIME_LIB_PATH="path/to/ort/lib" \
  -DENABLE_CUDA:BOOL=ON
```

(Assuming using bash, the line continuation character is backslash `\`. For Windows Command Prompt (cmd) it is caret `^`, and for PowerShell it is backtick <code>`</code>).

##### Build

```bash
cmake --build your_build_dir
```
