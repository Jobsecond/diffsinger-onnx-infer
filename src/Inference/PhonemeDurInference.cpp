#include "PhonemeDurInference.h"
#include "InferenceUtils.hpp"
#include "ModelData.h"

namespace diffsinger {
    PhonemeDurInference::PhonemeDurInference(const TString &modelPath) : Inference(modelPath) {}

    std::vector<float>
    PhonemeDurInference::infer(
            const LinguisticEncodedData &linguisticEncodedData,
            const std::vector<int> &ph_midi) {
        if (!m_session) {
            return {};
        }

        auto supportedInputNames = getSupportedInputNames(m_session);
        auto supportedOutputNames = getSupportedOutputNames(m_session);

        bool isValidModel = true;
        isValidModel &= hasKey(supportedInputNames, "encoder_out");
        isValidModel &= hasKey(supportedInputNames, "x_masks");
        isValidModel &= hasKey(supportedInputNames, "ph_midi");
        isValidModel &= hasKey(supportedOutputNames, "ph_dur_pred");

        if (!isValidModel) {
            std::cout << "Invalid Dur predictor model! "
                         "Must have inputs: encoder_out, x_masks, ph_midi; "
                         "outputs: ph_dur_pred\n";
            return {};
        }

        std::vector<const char *> inputNames;
        std::vector<Ort::Value> inputTensors;

        appendVectorToInputTensors<char, bool>("x_masks", linguisticEncodedData.x_masks, inputNames, inputTensors);
        auto sp = static_cast<int64_t>(linguisticEncodedData.encoder_out.size()) / linguisticEncodedData.hidden_size;
        appendVectorToInputTensorsWithShape<float, float>(
                "encoder_out",
                linguisticEncodedData.encoder_out,
                {
                    1,
                    static_cast<int64_t>(linguisticEncodedData.encoder_out.size()) / linguisticEncodedData.hidden_size,
                    linguisticEncodedData.hidden_size
                },
                inputNames, inputTensors);
        appendVectorToInputTensors<int, int64_t>("ph_midi", ph_midi, inputNames, inputTensors);

        std::vector<const char*> outputNames = {"ph_dur_pred"};
        std::vector<Ort::Value> outputTensors = m_session.Run(
                Ort::RunOptions{}, inputNames.data(), inputTensors.data(),
                inputNames.size(), outputNames.data(), outputNames.size());

        Ort::Value &phDurPredOutput = outputTensors[0];
        auto phDurPredBuffer = phDurPredOutput.GetTensorData<float>();

        return {phDurPredBuffer, phDurPredBuffer + phDurPredOutput.GetTensorTypeAndShapeInfo().GetElementCount()};
    }
} // namespace diffsinger
