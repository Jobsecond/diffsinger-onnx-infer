#include "AcousticInference.h"
#include "InferenceUtils.hpp"
#include "ModelData.h"


namespace diffsinger {

    AcousticInference::AcousticInference(const TString &modelPath) : Inference(modelPath), m_modelFlags() {}

    bool AcousticInference::postInitCheck() {
        updateFlags();
        if (!m_modelFlags.check(AcousticModelFlags::Valid)) {
            std::cout << "Invalid acoustic model!\n";
            endSession();
            return false;
        }

        return true;
    }

    void AcousticInference::postCleanup() {
        m_modelFlags.reset();
    }

    void AcousticInference::printModelFeatures() {
        if (!m_modelFlags.check(AcousticModelFlags::Valid)) {
            std::cout << "The acoustic model is invalid.\n";
            return;
        }

        std::cout << "Acoustic Model supported features:\n"
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
    }

    Ort::Value AcousticInference::inferToOrtValue(const PreprocessedData &pd, const AcousticInferenceSettings &inferSettings) {
        if (!m_session) {
            std::cout << "Session is not initialized!\n";
            return Ort::Value(nullptr);
        }

        if (!m_modelFlags.check(AcousticModelFlags::Valid)) {
            std::cout << "Invalid acoustic model!\n";
            return Ort::Value(nullptr);
        }

        std::vector<const char *> inputNames;
        std::vector<Ort::Value> inputTensors;

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
        }
        return Ort::Value(nullptr);
    }

    std::vector<float> AcousticInference::ortValueToVector(const Ort::Value &value) {
        auto buffer = value.GetTensorData<float>();
        std::vector<float> output(buffer, buffer + value.GetTensorTypeAndShapeInfo().GetElementCount());
        return output;
    }

    std::vector<float> AcousticInference::infer(const PreprocessedData &pd, const AcousticInferenceSettings &inferSettings) {
        return ortValueToVector(inferToOrtValue(pd, inferSettings));
    }

    void AcousticInference::updateFlags() {
        m_modelFlags.reset();
        if (!m_session) {
            return;
        }

        auto supportedInputNames = getSupportedInputNames(m_session);
        auto supportedOutputNames = getSupportedOutputNames(m_session);

        // Basic validation
        bool isValidModel = true;
        // Required input names
        isValidModel &= hasKey(supportedInputNames, "tokens");
        isValidModel &= hasKey(supportedInputNames, "durations");
        isValidModel &= hasKey(supportedInputNames, "f0");
        isValidModel &= hasKey(supportedInputNames, "speedup");
        // Required Output names
        isValidModel &= hasKey(supportedOutputNames, "mel");
        isValidModel &= (supportedOutputNames.size() == 1);
        m_modelFlags.setIf(AcousticModelFlags::Valid, isValidModel);
        if (!isValidModel) {
            return;
        }

        // Parameters that the model may support
        m_modelFlags.setIf(AcousticModelFlags::Velocity, hasKey(supportedInputNames, "velocity"));
        m_modelFlags.setIf(AcousticModelFlags::Gender, hasKey(supportedInputNames, "gender"));
        m_modelFlags.setIf(AcousticModelFlags::MultiSpeakers, hasKey(supportedInputNames, "spk_embed"));
        m_modelFlags.setIf(AcousticModelFlags::Energy, hasKey(supportedInputNames, "energy"));
        m_modelFlags.setIf(AcousticModelFlags::Breathiness, hasKey(supportedInputNames, "breathiness"));
        m_modelFlags.setIf(AcousticModelFlags::ShallowDiffusion, hasKey(supportedInputNames, "depth"));
    }
} // namespace diffsinger
