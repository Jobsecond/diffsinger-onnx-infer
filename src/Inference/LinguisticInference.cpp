#include "LinguisticInference.h"
#include "InferenceUtils.hpp"

namespace diffsinger {
    LinguisticInference::LinguisticInference(const TString &modelPath) : Inference(modelPath) {}

    LinguisticEncodedData LinguisticInference::infer(const LinguisticInput &input) {
        if (!m_session) {
            return {};
        }

        auto supportedInputNames = getSupportedInputNames(m_session);
        auto supportedOutputNames = getSupportedOutputNames(m_session);

        bool isValidModel = true;
        isValidModel &= hasKey(supportedInputNames, "tokens");
        isValidModel &= hasKey(supportedInputNames, "word_div");
        isValidModel &= hasKey(supportedInputNames, "word_dur");
        isValidModel &= hasKey(supportedOutputNames, "encoder_out");
        isValidModel &= hasKey(supportedOutputNames, "x_masks");

        if (!isValidModel) {
            std::cout << "Invalid Linguistic predictor model! "
                         "Must have inputs: tokens, word_div, word_dur; "
                         "outputs: encoder_out, x_masks\n";
            return {};
        }

        std::vector<const char *> inputNames;
        std::vector<Ort::Value> inputTensors;

        appendVectorToInputTensors<int64_t, int64_t>("tokens", input.tokens, inputNames, inputTensors);
        appendVectorToInputTensors<int64_t, int64_t>("word_div", input.word_div, inputNames, inputTensors);
        appendVectorToInputTensors<int64_t, int64_t>("word_dur", input.word_dur, inputNames, inputTensors);

        std::vector<const char*> outputNames = {"encoder_out", "x_masks"};
        std::vector<Ort::Value> outputTensors = m_session.Run(
                Ort::RunOptions{}, inputNames.data(), inputTensors.data(),
                inputNames.size(), outputNames.data(), outputNames.size());

        if (outputTensors.size() < 2) {
            return {};
        }
        Ort::Value &encoderOutOutput = outputTensors[0];
        auto encoderOutBuffer = encoderOutOutput.GetTensorData<float>();
        Ort::Value &xMasksOutput = outputTensors[1];
        auto xMasksBuffer = xMasksOutput.GetTensorData<bool>();

        LinguisticEncodedData out;
        out.encoder_out = std::vector<float>(
                encoderOutBuffer,
                encoderOutBuffer + encoderOutOutput.GetTensorTypeAndShapeInfo().GetElementCount());
        out.x_masks = std::vector<char>(
                xMasksBuffer,
                xMasksBuffer + xMasksOutput.GetTensorTypeAndShapeInfo().GetElementCount());
        auto encoderOutShape = encoderOutOutput.GetTensorTypeAndShapeInfo().GetShape();
        out.hidden_size = encoderOutShape[encoderOutShape.size() - 1];
        return out;
    }

} // namespace diffsinger
