
#ifndef DS_ONNX_INFER_DSCONFIG_H
#define DS_ONNX_INFER_DSCONFIG_H

#include <vector>
#include <filesystem>

#include "TString.h"
#include "SpeakerEmbed.h"

namespace diffsinger {
    enum class AxisDomain {
        Linear,
        Log
    };

    struct DsVocoderConfig {
        std::string name;

        std::filesystem::path model;

        int numMelBins = 128;
        int hopSize = 512;
        int sampleRate = 44100;

        static DsVocoderConfig fromYAML(const TString &dsVocoderConfigPath, bool *ok = nullptr);
    };


    struct AugmentationArgs {
        float rangeLow = 0.0f;
        float rangeHigh = 0.0f;
        float scale = 0.0f;
        AxisDomain domain = AxisDomain::Linear;
    };

    struct DsConfig {
        std::filesystem::path phonemes;
        std::filesystem::path acoustic;
        std::string vocoder;
        std::vector<std::string> speakers;
        SpeakerEmbed spkEmb;

        AugmentationArgs randomTimeShifting {0.5f, 2.0f, 1.5f, AxisDomain::Log};
        AugmentationArgs randomPitchShifting{-5.0f, 5.0f, 1.5f, AxisDomain::Linear};

        int hiddenSize = 256;
        int hopSize = 512;
        int sampleRate = 44100;
        int maxDepth = -1;
        bool useKeyShiftEmbed = false;
        bool useSpeedEmbed = false;
        bool useEnergyEmbed = false;
        bool useBreathinessEmbed = false;
        bool useShallowDiffusion = false;

        static DsConfig fromYAML(const TString &dsConfigPath, bool *ok = nullptr);
    };
}

#endif //DS_ONNX_INFER_DSCONFIG_H
