
#ifndef DS_ONNX_INFER_INFERENCE_H
#define DS_ONNX_INFER_INFERENCE_H

#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "TString.h"
#include "DsCommon.h"

namespace diffsinger {
    Ort::Session createSession(const TString &modelPath,
                               bool useDml);

    Ort::Value acousticInfer(const TString &model, const PreprocessedData &pd, int speedup);
    std::vector<float> vocoderInfer(const TString& model, Ort::Value& mel, const std::vector<double>& f0);

}

#endif //DS_ONNX_INFER_INFERENCE_H
