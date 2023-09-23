# DiffSinger ONNX Inference
**Note: This project is still working in progress. Tested on Windows only.**

## Command Line Options
```
Usage: ds_onnx_infer [-h] --ds-file VAR --acoustic-config VAR --vocoder-config VAR [--spk VAR] --out VAR [--speedup VAR]

Optional arguments:
  -h, --help            shows help message and exits
  -v, --version         prints version information and exits
  --ds-file             Path to .ds file [required]
  --acoustic-config     Path to acoustic dsconfig.yaml [required]
  --vocoder-config      Path to vocoder.yaml [required]
  --spk                 Speaker Mixture (e.g. "name" or "name1|name2" or "name1:0.25|name2:0.75") [default: ""]
  --out                 Output Audio Filename (*.wav) [required]
  --speedup             PNDM speedup ratio [default: 10]
```

## Build instructions
### Requirements
* Toolchains
  * A compiler that supports C++17
  * CMake
  * NuGet
  * vcpkg
* Third-party libraries:
  * ONNX Runtime
  * yaml-cpp
  * rapidjson
  * libsndfile
  * argparse

### Steps
#### Fetch onnxruntime (DirectML version) from NuGet
Simply run `scripts/download_nuget.bat`. It will download `nuget` and fetch required packages.

#### Download other third-party libraries using vcpkg
```bash
git clone https://github.com/microsoft/vcpkg
cd vcpkg
bootstrap-vcpkg.bat
vcpkg install --triplet=x64-windows libsndfile[core] rapidjson yaml-cpp argparse
```

#### Configure and build using CMake
```bash
cmake -G Ninja -B build -S . -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build build
```
