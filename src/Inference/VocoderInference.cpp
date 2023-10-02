#include "VocoderInference.h"
#include "InferenceUtils.hpp"

namespace diffsinger {

    VocoderInference::VocoderInference(const TString &modelPath) : Inference(modelPath) {}

    std::vector<float> VocoderInference::infer(Ort::Value &mel, const std::vector<double> &f0) {
        std::vector<const char *> inputNames;
        std::vector<Ort::Value> inputTensors;

        // f0
        appendVectorToInputTensors<double, float>("f0", f0, inputNames, inputTensors);

        // mel
        inputNames.push_back("mel");

        // TODO: Why not omit std::move? Why not omit const in function parameter of mel?
        inputTensors.push_back(std::move(mel));

        std::vector<const char*> outputNames = {"waveform"};
        std::vector<Ort::Value> outputTensors = m_session.Run(
                Ort::RunOptions{}, inputNames.data(), inputTensors.data(),
                inputNames.size(), outputNames.data(), outputNames.size());

        Ort::Value& waveformOutput = outputTensors[0];
        auto waveformBuffer = waveformOutput.GetTensorMutableData<float>();
        std::vector<float> waveform(waveformBuffer, waveformBuffer + waveformOutput.GetTensorTypeAndShapeInfo().GetElementCount());

        return waveform;
    }

}  // namespace diffsinger
