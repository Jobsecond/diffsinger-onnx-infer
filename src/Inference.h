
#ifndef DS_ONNX_INFER_INFERENCE_H
#define DS_ONNX_INFER_INFERENCE_H

#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "TString.h"
#include "ModelData.h"
#include "InferenceFlags.h"

namespace diffsinger {

    struct InferenceSettings {
        int speedup = 10;
        int depth = 1000;
    };

    enum class ExecutionProvider {
        CPU,
        CUDA,
        DirectML
    };

    std::vector<float> vocoderInfer(const TString& model, Ort::Value& mel, const std::vector<double>& f0);

    class AcousticInference {
    public:
        explicit AcousticInference(const TString &modelPath);
        bool initSession(ExecutionProvider ep = ExecutionProvider::CPU, int deviceIndex = 0);
        void endSession();
        bool hasSession();
        TString getModelPath();
        static std::vector<float> ortValueToVector(const Ort::Value &value);
        std::vector<float> infer(const PreprocessedData &pd, const InferenceSettings &inferSettings);
        Ort::Value inferToOrtValue(const PreprocessedData &pd, const InferenceSettings &inferSettings);
    private:
        TString m_modelPath;

        // Ort::Env must be initialized before Ort::Session.
        // (In this class, it should be defined before Ort::Session)
        // Otherwise, access violation will occur when Ort::Session destructor is called.
        Ort::Env m_env;
        Ort::Session m_session;
        OrtApi const &ortApi; // Uses ORT_API_VERSION
        AcousticModelFlags m_modelFlags;
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
