# DiffSinger ONNX Inference

**Note: This project is still working in progress. Tested on Windows only.**

## Command Line Options

```
Usage: ds_onnx_infer [-h] --ds-file VAR --acoustic-config VAR --vocoder-config VAR
       [--spk VAR] --out VAR [--speedup VAR] [--depth VAR]
       [--ep VAR] [--device-index VAR]

Optional arguments:
  -h, --help            shows help message and exits
  -v, --version         prints version information and exits
  --ds-file             Path to .ds file [required]
  --acoustic-config     Path to acoustic dsconfig.yaml [required]
  --vocoder-config      Path to vocoder.yaml [required]
  --spk                 Speaker Mixture (e.g. "name" or "name1|name2" or "name1:0.25|name2:0.75")
                        [default: ""]
  --out                 Output Audio Filename (*.wav) [required]
  --speedup             PNDM speedup ratio [default: 10]
  --depth               Shallow diffusion depth (needs acoustic model support) [default: 1000]
  --ep                  Execution Provider for audio inference. (cpu/directml/cuda)
                        [default: "cpu"]
  --device-index        GPU device index [default: 0]
```

## Build instructions

See [docs/BUILD.md](docs/BUILD.md) for detailed build instructions.

## Frequently Asked Questions

See [docs/FAQ.md](docs/FAQ.md).
