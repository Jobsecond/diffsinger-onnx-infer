#include <unordered_map>
#include <iostream>

#ifdef ONNXRUNTIME_ENABLE_DML
#include <dml_provider_factory.h>
#endif

#include "Inference.h"
#include "InferenceUtils.hpp"

namespace diffsinger {

    Inference::Inference(const TString &modelPath)
            : m_modelPath(modelPath),
              m_env(ORT_LOGGING_LEVEL_ERROR, "DiffSinger"),
              m_session(nullptr),
              ortApi(Ort::GetApi()) {}

    TString Inference::getModelPath() {
        return m_modelPath;
    }

    bool Inference::initSession(ExecutionProvider ep, int deviceIndex) {
        try {
            auto options = Ort::SessionOptions();
            switch (ep) {
                case ExecutionProvider::DirectML:
#ifdef ONNXRUNTIME_ENABLE_DML
                {
                    std::cout << "Try DirectML...\n";
                    const OrtDmlApi *ortDmlApi;
                    auto getApiStatus = Ort::Status(ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void **>(&ortDmlApi)));
                    if (getApiStatus.IsOK()) {
                        std::cout << "Successfully got DirectML API.\n";
                        options.DisableMemPattern();
                        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

                        auto status = Ort::Status(ortDmlApi->SessionOptionsAppendExecutionProvider_DML(options, deviceIndex));
                        if (status.IsOK()) {
                            std::cout << "Successfully appended DirectML Execution Provider.\n";
                        }
                        else {
                            std::cout << "Failed to append DirectML Execution Provider. Use CPU instead. Error code: "
                                      << status.GetErrorCode()
                                      << ", Reason: " << status.GetErrorMessage() << "\n";
                        }
                    }
                    else {
                        std::cout << "Failed to get DirectML API. Now use CPU instead.\n";
                    }
                }
#else
                    std::cout << "The software is not built with DirectML support. Use CPU instead.\n";
#endif
                    break;
                case ExecutionProvider::CUDA:
#ifdef ONNXRUNTIME_ENABLE_CUDA
                {
                    std::cout << "Try CUDA...\n";
                    OrtCUDAProviderOptionsV2 *cudaOptions = nullptr;
                    ortApi.CreateCUDAProviderOptions(&cudaOptions);

                    // The following block of code sets device_id
                    {
                        // Device ID from int to string
                        auto cudaDeviceIdStr = std::to_string(deviceIndex);
                        auto cudaDeviceIdStr_cstyle = cudaDeviceIdStr.c_str();

                        constexpr int CUDA_OPTIONS_SIZE = 2;
                        const char *cudaOptionsKeys[CUDA_OPTIONS_SIZE] = { "device_id", "cudnn_conv_algo_search" };
                        const char *cudaOptionsValues[CUDA_OPTIONS_SIZE] = { cudaDeviceIdStr_cstyle, "DEFAULT" };
                        auto updateStatus = Ort::Status(
                                ortApi.UpdateCUDAProviderOptions(cudaOptions, cudaOptionsKeys, cudaOptionsValues,
                                                                 CUDA_OPTIONS_SIZE));
                        if (!updateStatus.IsOK()) {
                            std::cout << "Failed to update CUDA Execution Provider options. Error code: "
                                      << updateStatus.GetErrorCode()
                                      << ", Reason: " << updateStatus.GetErrorMessage() << "\n";
                        }
                    }

                    auto status = Ort::Status(
                            ortApi.SessionOptionsAppendExecutionProvider_CUDA_V2(options, cudaOptions));
                    ortApi.ReleaseCUDAProviderOptions(cudaOptions);

                    if (status.IsOK()) {
                        std::cout << "Successfully appended CUDA Execution Provider.\n";
                    }
                    else {
                        std::cout << "Failed to append CUDA Execution Provider. Use CPU instead. Error code: "
                                  << status.GetErrorCode()
                                  << ", Reason: " << status.GetErrorMessage() << "\n";
                    }
                }
#else
                    std::cout << "The software is not built with CUDA support. Use CPU instead.\n";
#endif
                    break;
                default:
                    // CPU and other
                    std::cout << "Use CPU.\n";
                    break;
            }

            //options.AppendExecutionProvider_CUDA(options1);
            m_session = Ort::Session(m_env, m_modelPath.c_str(), options);

            return postInitCheck();
        }
        catch (const Ort::Exception &ortException) {
            printOrtError(ortException);
        }
        return false;
    }


    bool Inference::hasSession() {
        return m_session;
    }

    void Inference::endSession() {
        {
            Ort::Session emptySession(nullptr);
            std::swap(m_session, emptySession);
        }
        postCleanup();
    }

    bool Inference::postInitCheck() {
        return true;
    }

    void Inference::postCleanup() {}

}  // namespace diffsinger
