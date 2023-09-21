#include <unordered_map>
#include <unordered_set>
#include <iostream>

#include "cpu_provider_factory.h"
#include "dml_provider_factory.h"

#include "Inference.h"

namespace diffsinger {

    inline std::unordered_set<std::string> getSupportedInputNames(const Ort::Session &session);

    inline std::unordered_set<std::string> getSupportedOutputNames(const Ort::Session &session);

    template<class T_vector, class T_tensor = T_vector>
    inline Ort::Value vectorToTensor(const std::vector<T_vector> &vec);

    template<class T_vector, class T_tensor = T_vector>
    inline Ort::Value vectorToTensorWithShape(const std::vector<T_vector> &vec, const std::vector<int64_t> &shape);

    template<class T_scalar, class T_tensor = T_scalar>
    inline Ort::Value scalarToTensor(const T_scalar &scalar);

    template<class T_vector, class T_tensor = T_vector>
    inline void appendVectorToInputTensors(const char *inputName,
                                           const std::vector<T_vector> &vec,
                                           std::vector<const char *> &inputNames,
                                           std::vector<Ort::Value> &inputTensors);

    template<class T_vector, class T_tensor = T_vector>
    inline void appendVectorToInputTensorsWithShape(const char *inputName,
                                                    const std::vector<T_vector> &vec,
                                                    const std::vector<int64_t> &shape,
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

        auto supportedInputNames = getSupportedInputNames(session);
        auto supportedOutputNames = getSupportedOutputNames(session);

        // Basic validation
        bool isValidModel = true;
        // Required input names
        isValidModel &= (supportedInputNames.find("tokens") != supportedInputNames.end());
        isValidModel &= (supportedInputNames.find("durations") != supportedInputNames.end());
        isValidModel &= (supportedInputNames.find("f0") != supportedInputNames.end());
        isValidModel &= (supportedInputNames.find("speedup") != supportedInputNames.end());
        // Required Output names
        isValidModel &= (supportedOutputNames.find("mel") != supportedOutputNames.end());
        isValidModel &= (supportedOutputNames.size() == 1);
        if (!isValidModel) {
            return Ort::Value(nullptr);
        }

        // Parameters that the model may support
        bool supportsVelocity = supportedInputNames.find("velocity") != supportedInputNames.end();
        bool supportsGender = supportedInputNames.find("gender") != supportedInputNames.end();
        bool supportsSpeakers = supportedInputNames.find("spk_embed") != supportedInputNames.end();
        bool supportsEnergy = supportedInputNames.find("energy") != supportedInputNames.end();
        bool supportsBreathiness = supportedInputNames.find("breathiness") != supportedInputNames.end();

        std::cout << "Velocity: " << (supportsVelocity ? "Yes" : "No") << "\n";
        std::cout << "Gender: " << (supportsGender ? "Yes" : "No") << "\n";
        std::cout << "Speakers: " << (supportsSpeakers ? "Yes" : "No") << "\n";
        std::cout << "Energy: " << (supportsEnergy ? "Yes" : "No") << "\n";
        std::cout << "Breathiness: " << (supportsBreathiness ? "Yes" : "No") << "\n";

        // tokens
        appendVectorToInputTensors<int64_t, int64_t>("tokens", pd.tokens, inputNames, inputTensors);
        // durations
        appendVectorToInputTensors<int64_t, int64_t>("durations", pd.durations, inputNames, inputTensors);
        // F0
        appendVectorToInputTensors<double, float>("f0", pd.f0, inputNames, inputTensors);
        // Speedup
        appendScalarToInputTensors<int64_t, int64_t>("speedup", speedup, inputNames, inputTensors);
        // Velocity
        if (supportsVelocity) {
            appendVectorToInputTensors<double, float>("velocity", pd.velocity, inputNames, inputTensors);
        }
        // Gender
        if (supportsGender) {
            appendVectorToInputTensors<double, float>("gender", pd.gender, inputNames, inputTensors);
        }
        // Speakers Embed
        if (supportsSpeakers) {
            auto spkEmbedFrames = static_cast<int64_t>(pd.spk_embed.size()) / spkEmbedLastDimension;
            appendVectorToInputTensorsWithShape<double, float>("spk_embed", pd.spk_embed,
                                                               {1, spkEmbedFrames, spkEmbedLastDimension},
                                                               inputNames, inputTensors);
        }
        // Energy
        if (supportsEnergy) {
            appendVectorToInputTensors<double, float>("energy", pd.energy, inputNames, inputTensors);
        }
        // Breathiness
        if (supportsBreathiness) {
            appendVectorToInputTensors<double, float>("breathiness", pd.breathiness, inputNames, inputTensors);
        }

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
        Ort::Value &outputTensor = outputTensors[0];

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

    std::unordered_set<std::string> getSupportedInputNames(const Ort::Session &session) {
        auto inputCount = session.GetInputCount();
        std::unordered_set<std::string> supportedInputNames;
        supportedInputNames.reserve(inputCount);

        Ort::AllocatorWithDefaultOptions allocator;

        for (size_t i = 0; i < inputCount; i++) {
            auto inputNamePtr = session.GetInputNameAllocated(i, allocator);
            supportedInputNames.emplace(inputNamePtr.get());
        }

        return supportedInputNames;
    }

    std::unordered_set<std::string> getSupportedOutputNames(const Ort::Session &session) {
        auto outputCount = session.GetOutputCount();
        std::unordered_set<std::string> supportedOutputNames;
        supportedOutputNames.reserve(outputCount);

        Ort::AllocatorWithDefaultOptions allocator;

        for (size_t i = 0; i < outputCount; i++) {
            auto outputNamePtr = session.GetOutputNameAllocated(i, allocator);
            supportedOutputNames.emplace(outputNamePtr.get());
        }

        return supportedOutputNames;
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

    template<class T_vector, class T_tensor>
    Ort::Value vectorToTensorWithShape(const std::vector<T_vector> &vec, const std::vector<int64_t> &shape) {
        auto shapeSize = shape.size();

        Ort::AllocatorWithDefaultOptions allocator;
        auto tensor = Ort::Value::CreateTensor<T_tensor>(allocator, shape.data(), shapeSize);
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

    template<class T_vector, class T_tensor>
    void appendVectorToInputTensorsWithShape(const char *inputName,
                                             const std::vector<T_vector> &vec,
                                             const std::vector<int64_t> &shape,
                                             std::vector<const char *> &inputNames,
                                             std::vector<Ort::Value> &inputTensors) {
        inputNames.push_back(inputName);
        auto inputTensor = vectorToTensorWithShape<T_vector, T_tensor>(vec, shape);
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