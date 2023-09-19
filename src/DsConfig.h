
#ifndef DS_ONNX_INFER_DSCONFIG_H
#define DS_ONNX_INFER_DSCONFIG_H

#include <vector>
#include <filesystem>

#include "TString.h"

namespace diffsinger {
    struct DsVocoderConfig {
        std::string name;

        std::filesystem::path model;

        int numMelBins = 128;
        int hopSize = 512;
        int sampleRate = 44100;
    };

    struct RandomPitchShifting {
        float rangeLow = -5.0f;
        float rangeHigh = 5.0f;
        float scale = 1.5f;
    };

    struct RandomTimeShifting {
        std::string domain = "log";
        float rangeLow = 0.5f;
        float rangeHigh = 2.0f;
        float scale = 1.5f;
    };

    struct DsConfig {
        std::filesystem::path phonemes;
        std::filesystem::path acoustic;
        std::string vocoder;
        std::vector<std::string> speakers;

        RandomTimeShifting randomTimeShifting;
        RandomPitchShifting randomPitchShifting;

        int hiddenSize = 256;
        int hopSize = 512;
        int sampleRate = 44100;
        bool useKeyShiftEmbed = false;
        bool useSpeedEmbed = false;
        bool useEnergyEmbed = false;
        bool useBreathinessEmbed = false;
    };

    DsConfig loadDsConfig(const TString &dsConfigPath, bool *ok = nullptr);
    DsVocoderConfig loadDsVocoderConfig(const TString &dsVocoderConfigPath, bool *ok = nullptr);
}

#endif //DS_ONNX_INFER_DSCONFIG_H
