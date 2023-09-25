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

    inline bool hasKey(const std::unordered_set<std::string> &container, const std::string &key);

    inline void printOrtError(const Ort::Exception &err);

    /* IMPLEMENTATION BELOW */

    AcousticInference::AcousticInference(const TString &modelPath)
            : m_modelPath(modelPath), m_session(nullptr), ortApi(Ort::GetApi()),
              m_env(ORT_LOGGING_LEVEL_ERROR, "DiffSinger"),
              m_modelFlags() {}

    TString AcousticInference::getModelPath() {
        return m_modelPath;
    }

    bool AcousticInference::initSession(bool useDml, int deviceIndex) {
        m_modelFlags.reset();
        try {
            auto options = Ort::SessionOptions();
            if (useDml) {
                const OrtDmlApi *ortDmlApi;
                ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void **>(&ortDmlApi));

                options.DisableMemPattern();
                options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

                ortDmlApi->SessionOptionsAppendExecutionProvider_DML(options, deviceIndex);
            }

            //options.AppendExecutionProvider_CUDA(options1);
            m_session = Ort::Session(m_env, m_modelPath.c_str(), options);
            return true;
        }
        catch (const Ort::Exception &ortException) {
            printOrtError(ortException);
        }
        return false;
    }


    Ort::Value AcousticInference::inferToOrtValue(const PreprocessedData &pd, const InferenceSettings &inferSettings) {
        if (!m_session) {
            std::cout << "Session is not initialized!\n";
            return Ort::Value(nullptr);
        }

        std::vector<const char *> inputNames;
        std::vector<Ort::Value> inputTensors;

        auto supportedInputNames = getSupportedInputNames(m_session);
        auto supportedOutputNames = getSupportedOutputNames(m_session);

        // Basic validation
        m_modelFlags.unset(AcousticModelFlags::Valid);
        bool isValidModel = true;
        // Required input names
        isValidModel &= hasKey(supportedInputNames, "tokens");
        isValidModel &= hasKey(supportedInputNames, "durations");
        isValidModel &= hasKey(supportedInputNames, "f0");
        isValidModel &= hasKey(supportedInputNames, "speedup");
        // Required Output names
        isValidModel &= hasKey(supportedOutputNames, "mel");
        isValidModel &= (supportedOutputNames.size() == 1);
        if (!isValidModel) {
            return Ort::Value(nullptr);
        }
        m_modelFlags.set(AcousticModelFlags::Valid);

        // Parameters that the model may support
        m_modelFlags.setIf(AcousticModelFlags::Velocity, hasKey(supportedInputNames, "velocity"));
        m_modelFlags.setIf(AcousticModelFlags::Gender, hasKey(supportedInputNames, "gender"));
        m_modelFlags.setIf(AcousticModelFlags::MultiSpeakers, hasKey(supportedInputNames, "spk_embed"));
        m_modelFlags.setIf(AcousticModelFlags::Energy, hasKey(supportedInputNames, "energy"));
        m_modelFlags.setIf(AcousticModelFlags::Breathiness, hasKey(supportedInputNames, "breathiness"));
        m_modelFlags.setIf(AcousticModelFlags::ShallowDiffusion, hasKey(supportedInputNames, "depth"));

        std::cout << "Supported features:\n"
                  << "Velocity="
                  << (m_modelFlags.check(AcousticModelFlags::Velocity) ? "Yes" : "No") << "; "
                  << "Gender="
                  << (m_modelFlags.check(AcousticModelFlags::Gender) ? "Yes" : "No") << "; "
                  << "Multi_Speakers="
                  << (m_modelFlags.check(AcousticModelFlags::MultiSpeakers) ? "Yes" : "No") << "; "
                  << "Energy="
                  << (m_modelFlags.check(AcousticModelFlags::Energy) ? "Yes" : "No") << "; "
                  << "Breathiness="
                  << (m_modelFlags.check(AcousticModelFlags::Breathiness) ? "Yes" : "No") << "; "
                  << "Shallow_Diffusion="
                  << (m_modelFlags.check(AcousticModelFlags::ShallowDiffusion) ? "Yes" : "No") << '\n';

        // tokens
        appendVectorToInputTensors<int64_t, int64_t>("tokens", pd.tokens, inputNames, inputTensors);
        // durations
        appendVectorToInputTensors<int64_t, int64_t>("durations", pd.durations, inputNames, inputTensors);
        // F0
        appendVectorToInputTensors<double, float>("f0", pd.f0, inputNames, inputTensors);
        // Speedup
        appendScalarToInputTensors<decltype(inferSettings.speedup), int64_t>(
                "speedup", inferSettings.speedup, inputNames, inputTensors);
        // Velocity
        if (m_modelFlags.check(AcousticModelFlags::Velocity)) {
            appendVectorToInputTensors<double, float>("velocity", pd.velocity, inputNames, inputTensors);
        }
        // Gender
        if (m_modelFlags.check(AcousticModelFlags::Gender)) {
            appendVectorToInputTensors<double, float>("gender", pd.gender, inputNames, inputTensors);
        }
        // Speakers Embed
        if (m_modelFlags.check(AcousticModelFlags::MultiSpeakers)) {
            auto spkEmbedFrames = static_cast<int64_t>(pd.spk_embed.size()) / spkEmbedLastDimension;
            appendVectorToInputTensorsWithShape<float, float>("spk_embed", pd.spk_embed,
                                                               {1, spkEmbedFrames, spkEmbedLastDimension},
                                                               inputNames, inputTensors);
        }
        // TODO: If energy and breathiness are not supplied but required by the acoustic model,
        //       they should be inferred by the variance model.
        bool isVarianceError = false;
        // Energy
        if (m_modelFlags.check(AcousticModelFlags::Energy)) {
            if (pd.energy.empty()) {
                std::cout << "ERROR: The acoustic model required energy input, but such parameter is not supplied.\n";
                isVarianceError = true;
            }
            appendVectorToInputTensors<double, float>("energy", pd.energy, inputNames, inputTensors);
        }
        // Breathiness
        if (m_modelFlags.check(AcousticModelFlags::Breathiness)) {
            if (pd.breathiness.empty()) {
                std::cout << "ERROR: The acoustic model required breathiness input, but such parameter is not supplied.\n";
                isVarianceError = true;
            }
            appendVectorToInputTensors<double, float>("breathiness", pd.breathiness, inputNames, inputTensors);
        }

        if (isVarianceError) {
            return Ort::Value(nullptr);
        }

        // Shallow Diffusion depth
        if (m_modelFlags.check(AcousticModelFlags::ShallowDiffusion)) {
            if (inferSettings.depth < 0) {
                std::cout << "ERROR: The model supports shallow diffusion, but depth is unset or negative.\n";
                return Ort::Value(nullptr);
            }
            appendScalarToInputTensors<decltype(inferSettings.depth), int64_t>(
                    "depth", inferSettings.depth, inputNames, inputTensors);
        }

        // Create output names
        const char *outputNames[] = { "mel" };
        auto outputNamesSize = sizeof(outputNames) / sizeof(outputNames[0]);

        try {
            // Run the session
            auto outputTensors = m_session.Run(
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
        catch (const Ort::Exception &ortException) {
            printOrtError(ortException);
            return Ort::Value(nullptr);
        }
    }

    std::vector<float> AcousticInference::ortValueToVector(const Ort::Value &value) {
        auto buffer = value.GetTensorData<float>();
        std::vector<float> output(buffer, buffer + value.GetTensorTypeAndShapeInfo().GetElementCount());
        return output;
    }

    std::vector<float> AcousticInference::infer(const PreprocessedData &pd, const InferenceSettings &inferSettings) {
        return ortValueToVector(inferToOrtValue(pd, inferSettings));
    }

    bool AcousticInference::hasSession() {
        return m_session;
    }

    void AcousticInference::endSession() {
        {
            Ort::Session emptySession(nullptr);
            std::swap(m_session, emptySession);
        }
        m_modelFlags.reset();
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

    bool hasKey(const std::unordered_set<std::string> &container, const std::string &key) {
        return container.find(key) != container.end();
    }

    void printOrtError(const Ort::Exception &err) {
        std::cout << "[ONNXRuntimeError] : "
                  << err.GetOrtErrorCode() << " : "
                  << err.what() << '\n';
    }
}