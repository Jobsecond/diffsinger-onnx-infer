
#ifndef DS_ONNX_INFER_INFERENCE_H
#define DS_ONNX_INFER_INFERENCE_H

#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "TString.h"
#include "ModelData.h"

namespace diffsinger {

    struct InferenceSettings {
        int speedup = 10;
        int depth = 1000;
    };

    std::vector<float> vocoderInfer(const TString& model, Ort::Value& mel, const std::vector<double>& f0);

    class AcousticInference {
    public:
        explicit AcousticInference(const TString &modelPath);
        void initSession(bool useDml, int deviceIndex = 0);
        void endSession();
        bool hasSession();
        TString getModelPath();
        static std::vector<float> ortValueToVector(const Ort::Value &value);
        std::vector<float> infer(const PreprocessedData &pd, const InferenceSettings &inferSettings);
        Ort::Value inferToOrtValue(const PreprocessedData &pd, const InferenceSettings &inferSettings);
    private:
        TString m_modelPath;
        Ort::Session m_session;
        Ort::Env m_env;
        OrtApi const& ortApi; // Uses ORT_API_VERSION
    };

    class VocoderInference {
    public:
        explicit VocoderInference(const TString &modelPath);
        void initSession();
        void endSession();
        bool hasSession();
        TString getModelPath();

    };
}

#endif //DS_ONNX_INFER_INFERENCE_H
