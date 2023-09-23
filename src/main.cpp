#include <iostream>
#include <fstream>
#include <filesystem>

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
             int acousticSpeedup = 10);
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
        for (size_t i = 0; i <= numSegments; i++) {
            std::cout << i << " of " << numSegments << "\n";
            std::cout << "Preprocessing input" << "\n";
            auto pd = diffsinger::acousticPreprocess(name2token, dsProject[i], dsConfig, frameLength);

            std::cout << "Mel" << "\n";
            auto mel = diffsinger::acousticInfer(dsConfig.acoustic, pd, acousticSpeedup);

            if (mel == Ort::Value(nullptr)) {
                std::cout << "ERROR: Acoustic Infer failed.\n";
                continue;
            }
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
}