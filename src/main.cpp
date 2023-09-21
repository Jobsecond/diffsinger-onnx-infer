#include <iostream>
#include <fstream>

#include <onnxruntime_cxx_api.h>

#include <argparse/argparse.hpp>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>

#include <sndfile.hh>

#ifdef WIN32
#include <Windows.h>
#endif

#include "TString.h"
#include "DsCommon.h"
#include "DsConfig.h"
#include "ArrayUtil.hpp"
#include "Preprocess.h"
#include "Inference.h"


namespace diffsinger {
    void run(const TString &dsFilePath,
             const TString &dsConfigPath,
             const TString &vocoderConfigPath,
             const TString &outputWavePath,
             int acousticSpeedup = 10);

    std::vector<DsSegment> loadDsProject(const TString &dsFilePath);
}

using diffsinger::MBStringToWString;

int main(int argc, char *argv[]) {

    argparse::ArgumentParser program("DiffSinger");
    program.add_argument("--ds-file").help("Path to .ds file");
    program.add_argument("--acoustic-config").help("Path to acoustic dsconfig.yaml");
    program.add_argument("--vocoder-config").help("Path to vocoder.yaml");
    program.add_argument("--title").help("Output Audio File Title");
    program.add_argument("--speedup").scan<'i', int>().default_value(10).help("PNDM speedup ratio");

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    auto dsPath = program.get("--ds-file");
    auto dsConfigPath = program.get("--acoustic-config");
    auto vocoderConfigPath = program.get("--vocoder-config");
    auto outputAudioTitle = program.get("--title");
    auto speedup = program.get<int>("--speedup");
    if (speedup < 1 || speedup > 1000) {
        speedup = 10;
    }

#ifdef WIN32
    auto currentCodePage = ::GetACP();
    diffsinger::run(MBStringToWString(dsPath, currentCodePage),
                    MBStringToWString(dsConfigPath, currentCodePage),
                    MBStringToWString(vocoderConfigPath, currentCodePage),
                    MBStringToWString(outputAudioTitle, currentCodePage),
                    speedup);
#else
    diffsinger::run(dsPath, dsConfigPath, vocoderConfigPath, outputAudioTitle, speedup);
#endif

    return 0;
}


namespace diffsinger {
    void run(const TString &dsFilePath,
             const TString &dsConfigPath,
             const TString &vocoderConfigPath,
             const TString &outputWavePath,
             int acousticSpeedup) {

        // Get the available providers
        auto availableProviders = Ort::GetAvailableProviders();

        // Print the available providers
        std::cout << "Available Providers:" << std::endl;
        for (const auto &provider: availableProviders) {
            std::cout << provider << std::endl;
        }

        auto dsProject = loadDsProject(dsFilePath);

        auto dsConfig = loadDsConfig(dsConfigPath);
        std::unordered_map<std::string, int64_t> name2token;
        std::string line;
        std::ifstream phonemesFile(dsConfig.phonemes);

        int64_t token = 0;
        while (std::getline(phonemesFile, line)) {
            name2token.emplace(line, token);
            ++token;
        }
        phonemesFile.close();

        auto vocoderConfig = loadDsVocoderConfig(vocoderConfigPath);
        int sampleRate = vocoderConfig.sampleRate;
        int hopSize = vocoderConfig.hopSize;
        double frameLength = 1.0 * hopSize / sampleRate;
        size_t numSegments = dsProject.size();
        for (size_t i = 0; i <= numSegments; i++) {
            std::cout << i << " of " << numSegments << "\n";
            auto pd = diffsinger::acousticPreprocess(name2token, dsProject[i], frameLength);

            std::cout << "Mel" << "\n";
            auto mel = diffsinger::acousticInfer(dsConfig.acoustic, pd, acousticSpeedup);

            // TODO: mel will be `std::move`d in the next step, so it will not be usable after that.
            std::cout << "Waveform" << "\n";
            auto waveform = diffsinger::vocoderInfer(vocoderConfig.model, mel, pd.f0);

            std::basic_stringstream<TChar> ss;
            ss << outputWavePath << DS_T("_") << i << DS_T(".wav");
            SndfileHandle audioFile(ss.str().c_str(), SFM_WRITE, SF_FORMAT_WAV | SF_FORMAT_FLOAT, 1, sampleRate);
            auto numFrames = static_cast<sf_count_t>(waveform.size());
            audioFile.write(waveform.data(), numFrames);
        }
    }

    std::vector<DsSegment> loadDsProject(const TString &dsFilePath) {
        std::ifstream dsFile(dsFilePath);

        if (!dsFile.is_open()) {
            std::cout << "Failed to open file!\n";
            return {};
        }

        rapidjson::IStreamWrapper streamWrapper(dsFile);
        rapidjson::Document data;
        data.ParseStream(streamWrapper);

        if (!data.IsArray()) {
            std::cout << "Invalid ds file format!\n";
            dsFile.close();
            return {};
        }

        std::vector<DsSegment> result;

        auto numSegments = data.Size();
        for (rapidjson::SizeType i = 0; i < numSegments; i++) {
            DsSegment dsSegment{};
            const auto &segment = data[i];
            if (!segment.IsObject()) {
                std::cout << "Segment at index " << i << " is not an object!\n";
                continue;
            }

            // TODO: ph_dur and f0 curve can be inferred using rhythmizers and autopitch models.
            //       In this case, these parameters can be omitted from .ds files, but note sequences
            //       must be supplied.
            if (!segment.HasMember("ph_seq")
                || !segment.HasMember("ph_dur")
                || !segment.HasMember("f0_seq")
                || !segment.HasMember("f0_timestep")) {
                std::cout << "Segment at index " << i
                          << " must contain required keys (ph_seq, ph_dur, f0_seq, f0_timestep)!\n";
                continue;
            }
            if (!segment["ph_seq"].IsString()
                || !segment["ph_dur"].IsString()
                || !segment["f0_seq"].IsString()
                || !(segment["f0_timestep"].IsNumber() || segment["f0_timestep"].IsString())) {
                std::cout << "Segment at index " << i
                          << " must contain valid keys (ph_seq, ph_dur, f0_seq, f0_timestep)!\n";
                continue;
            }

            dsSegment.ph_seq = splitString<std::string>(segment["ph_seq"].GetString());
            dsSegment.ph_dur = splitString<double>(segment["ph_dur"].GetString());

            auto loadSampleCurve = [&segment](
                    const char *sampleKey, const char *timestepKey,
                    SampleCurve *sampleCurve) {
                if (!sampleCurve) {
                    return;
                }
                if (!segment.HasMember(sampleKey) || !segment.HasMember(timestepKey)) {
                    return;
                }
                if (!segment[sampleKey].IsString()) {
                    return;
                }
                bool timestepIsString = segment[timestepKey].IsString();
                bool timestepIsNumber = segment[timestepKey].IsNumber();
                if (!timestepIsString && !timestepIsNumber) {
                    return;
                }

                sampleCurve->samples = splitString<double>(segment[sampleKey].GetString());

                if (timestepIsString) {
                    sampleCurve->timestep = std::stod(segment[timestepKey].GetString());
                } else {
                    sampleCurve->timestep = segment[timestepKey].GetDouble();
                }
            };

            loadSampleCurve("f0_seq", "f0_timestep", &dsSegment.f0);
            loadSampleCurve("gender", "gender_timestep", &dsSegment.gender);
            loadSampleCurve("velocity", "velocity_timestep", &dsSegment.velocity);
            loadSampleCurve("energy", "energy_timestep", &dsSegment.energy);
            loadSampleCurve("breathiness", "breathiness_timestep", &dsSegment.breathiness);

            // TODO: spk_mix

            dsSegment.offset = segment.HasMember("offset") ? segment["offset"].GetDouble() : 0.0;
            dsFile.close();

            result.push_back(dsSegment);
        }
        return result;
    }
}