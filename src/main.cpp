#include <iostream>
#include <fstream>
#include <filesystem>
#include <utility>

#include <onnxruntime_cxx_api.h>

#include <argparse/argparse.hpp>

#include <sndfile.hh>

#ifdef WIN32
#include <Windows.h>
#endif

#include "TString.h"
#include "DsProject.h"
#include "DsConfig.h"
#include "Preprocess.h"
#include "Inference.h"


namespace diffsinger {
    void run(const TString &dsFilePath,
             const TString &dsConfigPath,
             const TString &vocoderConfigPath,
             const TString &outputWavePath,
             const std::string &spkMixStr = "",
             int acousticSpeedup = 10);
}

using diffsinger::MBStringToWString;

int main(int argc, char *argv[]) {

    argparse::ArgumentParser program("DiffSinger");
    program.add_argument("--ds-file").required().help("Path to .ds file");
    program.add_argument("--acoustic-config").required().help("Path to acoustic dsconfig.yaml");
    program.add_argument("--vocoder-config").required().help("Path to vocoder.yaml");
    program.add_argument("--spk").default_value(std::string())
            .help(R"(Speaker Mixture (e.g. "name" or "name1|name2" or "name1:0.25|name2:0.75"))");
    program.add_argument("--out").required().help("Output Audio Filename (*.wav)");
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
    auto spkMixStr = program.get("--spk");
    auto outputAudioTitle = program.get("--out");
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
                    spkMixStr,
                    speedup);
#else
    diffsinger::run(dsPath, dsConfigPath, vocoderConfigPath, outputAudioTitle, spkMixStr, speedup);
#endif

    return 0;
}


namespace diffsinger {
    void run(const TString &dsFilePath,
             const TString &dsConfigPath,
             const TString &vocoderConfigPath,
             const TString &outputWavePath,
             const std::string &spkMixStr,
             int acousticSpeedup) {

        // Get the available providers
        auto availableProviders = Ort::GetAvailableProviders();

        // Print the available providers
        std::cout << "Available Providers:" << std::endl;
        for (const auto &provider: availableProviders) {
            std::cout << provider << std::endl;
        }

        auto dsProject = loadDsProject(dsFilePath, spkMixStr);

        auto dsConfig = DsConfig::fromYAML(dsConfigPath);
        std::unordered_map<std::string, int64_t> name2token;
        std::string line;
        std::ifstream phonemesFile(dsConfig.phonemes);

        int64_t token = 0;
        while (std::getline(phonemesFile, line)) {
            name2token.emplace(line, token);
            ++token;
        }
        phonemesFile.close();

        auto vocoderConfig = DsVocoderConfig::fromYAML(vocoderConfigPath);
        int sampleRate = vocoderConfig.sampleRate;
        int hopSize = vocoderConfig.hopSize;
        double frameLength = 1.0 * hopSize / sampleRate;
        size_t numSegments = dsProject.size();

        AcousticInference acousticInference(dsConfig.acoustic);
        acousticInference.initSession(true);

        std::vector< std::pair<int64_t, std::vector<float>> > waveformArr{};
        waveformArr.reserve(numSegments);

        for (size_t i = 0; i < numSegments; i++) {
            std::cout << i + 1 << " of " << numSegments << "\n";
            std::cout << "Preprocessing input" << "\n";
            auto offsetInSamples = static_cast<int64_t>(std::ceil(dsProject[i].offset * vocoderConfig.sampleRate));

            auto pd = acousticPreprocess(name2token, dsProject[i], dsConfig, frameLength);

            std::cout << "Mel" << "\n";
            auto mel = acousticInference.inferToOrtValue(pd, acousticSpeedup);

            if (mel == Ort::Value(nullptr)) {
                std::cout << "ERROR: Acoustic Infer failed.\n";
                waveformArr.emplace_back(offsetInSamples, std::vector<float>{});
                continue;
            }
            // mel will be `std::move`d in the next steps, so it will not be usable after that.
            std::cout << "Waveform" << "\n";
            auto waveform = vocoderInfer(vocoderConfig.model, mel, pd.f0);

            waveformArr.emplace_back(offsetInSamples, std::move(waveform));
        }

        std::cout << "Concatenating and saving wave file...\n";
        int64_t totalSamples = 0;
        for (const auto& [offsetInSamples, waveform] : waveformArr) {
            auto currentSamples = offsetInSamples + static_cast<int64_t>(waveform.size());
            totalSamples = (totalSamples < currentSamples) ? currentSamples : totalSamples;
        }
        std::vector<float> wavBuffer(totalSamples, 0.0f);
        for (const auto& [offsetInSamples, waveform] : waveformArr) {
            auto currentSamples = offsetInSamples + static_cast<int64_t>(waveform.size());
            std::transform(
                wavBuffer.begin() + offsetInSamples,
                wavBuffer.begin() + currentSamples,
                waveform.begin(),
                wavBuffer.begin() + offsetInSamples,
                std::plus<>()
            );
        }

        SndfileHandle audioFile(outputWavePath.c_str(), SFM_WRITE, SF_FORMAT_WAV | SF_FORMAT_FLOAT, 1, sampleRate);
        auto numFrames = static_cast<sf_count_t>(wavBuffer.size());
        audioFile.write(wavBuffer.data(), numFrames);
    }
}