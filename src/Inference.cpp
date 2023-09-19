#include <unordered_map>

#include "cpu_provider_factory.h"
#include "dml_provider_factory.h"

#include "Inference.h"

namespace diffsinger {

    template<class T_vector, class T_tensor = T_vector>
    inline Ort::Value vectorToTensor(const std::vector<T_vector> &vec);

    template<class T_scalar, class T_tensor = T_scalar>
    inline Ort::Value scalarToTensor(const T_scalar &scalar);

    template<class T_vector, class T_tensor = T_vector>
    inline void appendVectorToInputTensors(const char *inputName,
                                           const std::vector<T_vector> &vec,
                                           std::vector<const char *> &inputNames,
                                           std::vector<Ort::Value> &inputTensors);

    template<class T_scalar, class T_tensor = T_scalar>
    inline void appendScalarToInputTensors(const char *inputName,
                                           const T_scalar &scalar,
                                           std::vector<const char *> &inputNames,
                                           std::vector<Ort::Value> &inputTensors);

    /* IMPLEMENTATION BELOW */

    Ort::Session createSession(const TString &modelPath,
                               bool useDml) {

        OrtApi const& ortApi = Ort::GetApi(); // Uses ORT_API_VERSION
        auto options = Ort::SessionOptions();
        if (useDml) {
            const OrtDmlApi *ortDmlApi;
            ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void **>(&ortDmlApi));

            options.DisableMemPattern();
            options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

            ortDmlApi->SessionOptionsAppendExecutionProvider_DML(options, /*device index*/ 0);
        }

        //options.AppendExecutionProvider_CUDA(options1);
        Ort::Env env(/*ORT_LOGGING_LEVEL_WARNING*/ ORT_LOGGING_LEVEL_ERROR, "DiffSinger");
        auto session = Ort::Session(env, modelPath.c_str(), options);
        return session;
    }

    Ort::Value acousticInfer(const TString &model, const PreprocessedData &pd, int speedup) {
        auto session = createSession(model, true);

        std::vector<const char *> inputNames;
        std::vector<Ort::Value> inputTensors;

        // tokens
        appendVectorToInputTensors<int64_t, int64_t>("tokens", pd.tokens, inputNames, inputTensors);
        // durations
        appendVectorToInputTensors<int64_t, int64_t>("durations", pd.phDur, inputNames, inputTensors);
        // F0
        appendVectorToInputTensors<double, float>("f0", pd.f0Seq, inputNames, inputTensors);
        // Speedup
        appendScalarToInputTensors<int64_t, int64_t>("speedup", speedup, inputNames, inputTensors);

        // Create output names
        const char *outputNames[] = { "mel" };
        auto outputNamesSize = sizeof(outputNames) / sizeof(outputNames[0]);

        // Run the session
        std::vector<Ort::Value> outputTensors = session.Run(
                Ort::RunOptions{},
                inputNames.data(),
                inputTensors.data(),
                inputNames.size(),
                outputNames,
                outputNamesSize);

        // Get the output tensor
        Ort::Value& outputTensor = outputTensors[0];

        return std::move(outputTensor);
    }

    std::vector<float> vocoderInfer(const TString& model, Ort::Value& mel, const std::vector<double>& f0) {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "VocoderInfer");
        Ort::SessionOptions sessionOptions;

        Ort::Session session(env, model.c_str(), sessionOptions);

        Ort::AllocatorWithDefaultOptions allocator;

        std::vector<const char *> inputNames;
        std::vector<Ort::Value> inputTensors;

        // f0
        appendVectorToInputTensors<double, float>("f0", f0, inputNames, inputTensors);

        // mel
        inputNames.push_back("mel");

        // TODO: Why not omit std::move? Why not omit const in function parameter of mel?
        inputTensors.push_back(std::move(mel));

        std::vector<const char*> outputNames = {"waveform"};
        std::vector<Ort::Value> outputTensors = session.Run(Ort::RunOptions{}, inputNames.data(), inputTensors.data(), inputNames.size(), outputNames.data(), outputNames.size());

        Ort::Value& waveformOutput = outputTensors[0];
        auto waveformBuffer = waveformOutput.GetTensorMutableData<float>();
        std::vector<float> waveform(waveformBuffer, waveformBuffer + waveformOutput.GetTensorTypeAndShapeInfo().GetElementCount());

        return waveform;
    }

    template<class T_vector, class T_tensor>
    Ort::Value vectorToTensor(const std::vector<T_vector> &vec) {
        int64_t shape[] = { 1, static_cast<int64_t>(vec.size()) };  // shape = {1, N}
        auto shapeSize = sizeof(shape) / sizeof(shape[0]);

        Ort::AllocatorWithDefaultOptions allocator;
        auto tensor = Ort::Value::CreateTensor<T_tensor>(allocator, shape, shapeSize);
        auto buffer = tensor.template GetTensorMutableData<T_tensor>();
        for (size_t i = 0; i < vec.size(); i++) {
            buffer[i] = static_cast<T_tensor>(vec[i]);
        }

        return tensor;
    }

    template<class T_scalar, class T_tensor>
    Ort::Value scalarToTensor(const T_scalar &scalar) {
        int64_t shape[] = { 1 };
        auto shapeSize = sizeof(shape) / sizeof(shape[0]);

        Ort::AllocatorWithDefaultOptions allocator;
        auto tensor = Ort::Value::CreateTensor<T_tensor>(allocator, shape, shapeSize);
        auto buffer = tensor.template GetTensorMutableData<T_tensor>();
        buffer[0] = static_cast<T_tensor>(scalar);

        return tensor;
    }

    template<class T_vector, class T_tensor>
    void appendVectorToInputTensors(const char *inputName,
                                    const std::vector<T_vector> &vec,
                                    std::vector<const char *> &inputNames,
                                    std::vector<Ort::Value> &inputTensors) {
        inputNames.push_back(inputName);
        auto inputTensor = vectorToTensor<T_vector, T_tensor>(vec);
        inputTensors.push_back(std::move(inputTensor));
    }

    template<class T_scalar, class T_tensor>
    void appendScalarToInputTensors(const char *inputName,
                                    const T_scalar &scalar,
                                    std::vector<const char *> &inputNames,
                                    std::vector<Ort::Value> &inputTensors) {
        inputNames.push_back(inputName);
        auto inputTensor = scalarToTensor<T_scalar, T_tensor>(scalar);
        inputTensors.push_back(std::move(inputTensor));
    }
}