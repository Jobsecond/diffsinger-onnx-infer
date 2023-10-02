
#ifndef DS_ONNX_INFER_INFERENCE_H
#define DS_ONNX_INFER_INFERENCE_H

#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "TString.h"

namespace diffsinger {

    enum class ExecutionProvider {
        CPU,
        CUDA,
        DirectML
    };  // enum class ExecutionProvider


    class Inference {
    public:
        explicit Inference(const TString &modelPath);

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
    };  // class Inference

}  // namespace diffsinger

#endif //DS_ONNX_INFER_INFERENCE_H
