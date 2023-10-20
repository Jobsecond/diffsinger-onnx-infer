# DiffSinger ONNX Inference

**Note: This project is still working in progress. Tested on Windows only.**

## Command Line Options

```
Usage:
    ds_onnx_infer --ds-file <file> --acoustic-config <file> --vocoder-config <file> --out <file> [options]

Required Options:
    --ds-file <file>            Path to .ds file
    --acoustic-config <file>    Path to acoustic dsconfig.yaml
    --vocoder-config <file>     Path to vocoder.yaml
    --out <file>                Output Audio Filename (*.wav)

Options:
    --spk <spk>               Speaker Mixture (e.g. "name" or "name1|name2" or "name1:0.25|name2:0.75")
    --speedup <rate>          PNDM speedup ratio [default: 10]
    --depth <depth>           Shallow diffusion depth (needs acoustic model support) [default: 1000]
    --ep <ep>                 Execution Provider for audio inference. (cpu/directml/cuda) [default: "cpu"]
    --device-index <index>    GPU device index [default: "cpu"]
    -v, --version             Show version information
    -h, --help                Show help information
```

## Build instructions

See [docs/BUILD.md](docs/BUILD.md) for detailed build instructions.

## Frequently Asked Questions

See [docs/FAQ.md](docs/FAQ.md).
