#include <fstream>
#include <filesystem>

#include <yaml-cpp/yaml.h>

#include "DsConfig.h"

#ifdef _WIN32
#define DS_STRING_CONVERT(x) diffsinger::MBStringToWString((x), 65001)
#else
#define DS_STRING_CONVERT(x) (x)
#endif

namespace diffsinger {
    DsConfig DsConfig::fromYAML(const TString &dsConfigPath, bool *ok) {
        DsConfig dsConfig;

        std::ifstream fileStream(dsConfigPath);
        if (!fileStream.is_open()) {
            if (ok) {
                *ok = false;
            }
            return dsConfig;
        }

        auto dsConfigPathFs = std::filesystem::path(dsConfigPath);
        auto dsConfigDir = dsConfigPathFs.parent_path();
        YAML::Node config = YAML::Load(fileStream);
        if (config["phonemes"]) {
            auto phonemesFilename = DS_STRING_CONVERT(config["phonemes"].as<std::string>());
            dsConfig.phonemes = dsConfigDir / phonemesFilename;
        }

        if (config["acoustic"]) {
            auto acousticFilename = DS_STRING_CONVERT(config["acoustic"].as<std::string>());
            dsConfig.acoustic = dsConfigDir / acousticFilename;
        }

        if (config["vocoder"]) {
            dsConfig.vocoder = config["vocoder"].as<std::string>();
        }

        if (config["augmentation_args"]) {
            auto augmentation_args_node = config["augmentation_args"];

            auto random_pitch_shifting_node = augmentation_args_node["random_pitch_shifting"];
            auto pitch_range_node = random_pitch_shifting_node["range"];
            dsConfig.randomPitchShifting.domain = AxisDomain::Linear;
            dsConfig.randomPitchShifting.rangeLow = pitch_range_node[0].as<float>();
            dsConfig.randomPitchShifting.rangeHigh = pitch_range_node[1].as<float>();
            dsConfig.randomPitchShifting.scale = random_pitch_shifting_node["scale"].as<float>();

            auto random_time_stretching_node = augmentation_args_node["random_time_stretching"];
            dsConfig.randomTimeShifting.domain = random_time_stretching_node["domain"].as<std::string>() == "log" ? AxisDomain::Log : AxisDomain::Linear;
            auto time_range_node = random_time_stretching_node["range"];
            dsConfig.randomTimeShifting.rangeLow = time_range_node[0].as<float>();
            dsConfig.randomTimeShifting.rangeHigh = time_range_node[1].as<float>();
            dsConfig.randomTimeShifting.scale = random_time_stretching_node["scale"].as<float>();
        }

        if (config["use_key_shift_embed"]) {
            dsConfig.useKeyShiftEmbed = config["use_key_shift_embed"].as<bool>();
        }

        if (config["use_speed_embed"]) {
            dsConfig.useSpeedEmbed = config["use_speed_embed"].as<bool>();
        }

        if (config["use_energy_embed"]) {
            dsConfig.useEnergyEmbed = config["use_energy_embed"].as<bool>();
        }

        if (config["use_breathiness_embed"]) {
            dsConfig.useBreathinessEmbed = config["use_breathiness_embed"].as<bool>();
        }

        if (config["use_shallow_diffusion"]) {
            dsConfig.useShallowDiffusion = config["use_shallow_diffusion"].as<bool>();
        }

        if (config["max_depth"]) {
            dsConfig.maxDepth = config["max_depth"].as<int>();
        }

        if (config["speakers"]) {
            dsConfig.speakers = config["speakers"].as<std::vector<std::string>>();
            dsConfig.spkEmb.loadSpeakers(dsConfig.speakers, dsConfigDir);
        }

        if (ok) {
            *ok = true;
        }
        return dsConfig;
    }

    DsVocoderConfig DsVocoderConfig::fromYAML(const TString &dsVocoderConfigPath, bool *ok) {
        DsVocoderConfig dsVocoderConfig;

        std::ifstream fileStream(dsVocoderConfigPath);
        if (!fileStream.is_open()) {
            if (ok) {
                *ok = false;
            }
            return dsVocoderConfig;
        }

        auto dsVocoderConfigPathFs = std::filesystem::path(dsVocoderConfigPath);
        auto dsVocoderConfigDir = dsVocoderConfigPathFs.parent_path();
        YAML::Node config = YAML::Load(fileStream);
        if (config["name"]) {
            dsVocoderConfig.name = config["name"].as<std::string>();
        }

        if (config["model"]) {
            auto model = DS_STRING_CONVERT(config["model"].as<std::string>());
            dsVocoderConfig.model = dsVocoderConfigDir / model;
        }

        if (config["num_mel_bins"]) {
            dsVocoderConfig.numMelBins = config["num_mel_bins"].as<int>();
        }

        if (config["hop_size"]) {
            dsVocoderConfig.hopSize = config["hop_size"].as<int>();
        }

        if (config["sample_rate"]) {
            dsVocoderConfig.sampleRate = config["sample_rate"].as<int>();
        }

        if (ok) {
            *ok = true;
        }
        return dsVocoderConfig;
    }

    DsDurConfig DsDurConfig::fromYAML(const TString &dsDurConfigPath, bool *ok) {
        DsDurConfig dsDurConfig;

        std::ifstream fileStream(dsDurConfigPath);
        if (!fileStream.is_open()) {
            if (ok) {
                *ok = false;
            }
            return dsDurConfig;
        }

        auto dsDurConfigPathFs = std::filesystem::path(dsDurConfigPath);
        auto dsDurConfigDir = dsDurConfigPathFs.parent_path();
        YAML::Node config = YAML::Load(fileStream);
        if (config["phonemes"]) {
            auto model = DS_STRING_CONVERT(config["phonemes"].as<std::string>());
            dsDurConfig.phonemes = dsDurConfigDir / model;
        }

        if (config["linguistic"]) {
            auto model = DS_STRING_CONVERT(config["linguistic"].as<std::string>());
            dsDurConfig.linguistic = dsDurConfigDir / model;
        }

        if (config["dur"]) {
            auto model = DS_STRING_CONVERT(config["dur"].as<std::string>());
            dsDurConfig.dur = dsDurConfigDir / model;
        }

        if (config["hop_size"]) {
            dsDurConfig.hopSize = config["hop_size"].as<int>();
        }

        if (config["sample_rate"]) {
            dsDurConfig.sampleRate = config["sample_rate"].as<int>();
        }

        if (config["predict_dur"]) {
            dsDurConfig.predict_dur = config["predict_dur"].as<bool>();
        }

        if (ok) {
            *ok = true;
        }
        return dsDurConfig;
    }
}
