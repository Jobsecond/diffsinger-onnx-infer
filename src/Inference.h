
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
    };  // struct InferenceSettings

    enum class ExecutionProvider {
        CPU,
        CUDA,
        DirectML
    };  // enum class ExecutionProvider


    class BaseInference {
    public:
        explicit BaseInference(const TString &modelPath);
        bool initSession(ExecutionProvider ep = ExecutionProvider::CPU, int deviceIndex = 0);
        void endSession();
        bool hasSession();
        TString getModelPath();
    protected:
        TString m_modelPath;

        // Ort::Env must be initialized before Ort::Session.
        // (In this class, it should be defined before Ort::Session)
        // Otherwise, access violation will occur when Ort::Session destructor is called.
        Ort::Env m_env;
        Ort::Session m_session;
        OrtApi const &ortApi; // Uses ORT_API_VERSION
    protected:
        virtual bool postInitCheck();
        virtual void postCleanup();
    };  // class BaseInference


    class AcousticInference : public BaseInference {
    public:
        explicit AcousticInference(const TString &modelPath);
        void printModelFeatures();
        static std::vector<float> ortValueToVector(const Ort::Value &value);
        std::vector<float> infer(const PreprocessedData &pd, const InferenceSettings &inferSettings);
        Ort::Value inferToOrtValue(const PreprocessedData &pd, const InferenceSettings &inferSettings);
    private:
        AcousticModelFlags m_modelFlags;
    private:
        void updateFlags();
    protected:
        bool postInitCheck() override;
        void postCleanup() override;
    };  // class AcousticInference


    class VocoderInference : public BaseInference {
    public:
        explicit VocoderInference(const TString &modelPath);
        std::vector<float> infer(Ort::Value& mel, const std::vector<double>& f0);
    };  // class VocoderInference

}  // namespace diffsinger

#endif //DS_ONNX_INFER_INFERENCE_H
