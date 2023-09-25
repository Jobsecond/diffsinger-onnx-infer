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
             int acousticSpeedup = 10,
             int shallowDiffusionDepth = 1000,
             bool cpuOnly = false,
             int deviceIndex = 0);
}

#ifdef WIN32
using diffsinger::MBStringToWString;
#endif

int main(int argc, char *argv[]) {

    argparse::ArgumentParser program("DiffSinger");
    program.add_argument("--ds-file").required().help("Path to .ds file");
    program.add_argument("--acoustic-config").required().help("Path to acoustic dsconfig.yaml");
    program.add_argument("--vocoder-config").required().help("Path to vocoder.yaml");
    program.add_argument("--spk").default_value(std::string())
            .help(R"(Speaker Mixture (e.g. "name" or "name1|name2" or "name1:0.25|name2:0.75"))");
    program.add_argument("--out").required().help("Output Audio Filename (*.wav)");
    program.add_argument("--speedup").scan<'i', int>().default_value(10).help("PNDM speedup ratio");
    program.add_argument("--depth").scan<'i', int>().default_value(1000).help("Shallow diffusion depth (needs acoustic model support)");
    program.add_argument("--cpu-only").default_value(false).implicit_value(true).help("Use CPU for audio inference");
    program.add_argument("--device-index").scan<'i', int>().default_value(0).help("GPU device index");

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
    auto depth = program.get<int>("--depth");
    auto cpuOnly = program.get<bool>("--cpu-only");
    auto deviceIndex = program.get<int>("--device-index");

#ifdef WIN32
    auto currentCodePage = ::GetACP();
    diffsinger::run(MBStringToWString(dsPath, currentCodePage),
                    MBStringToWString(dsConfigPath, currentCodePage),
                    MBStringToWString(vocoderConfigPath, currentCodePage),
                    MBStringToWString(outputAudioTitle, currentCodePage),
                    spkMixStr,
                    speedup,
                    depth,
                    cpuOnly,
                    deviceIndex);
#else
    diffsinger::run(dsPath, dsConfigPath, vocoderConfigPath, outputAudioTitle, spkMixStr, speedup, depth, cpuOnly, deviceIndex);
#endif

    return 0;
}


namespace diffsinger {
    void run(const TString &dsFilePath,
             const TString &dsConfigPath,
             const TString &vocoderConfigPath,
             const TString &outputWavePath,
             const std::string &spkMixStr,
             int acousticSpeedup,
             int shallowDiffusionDepth,
             bool cpuOnly,
             int deviceIndex) {

        // Get the available providers
        auto availableProviders = Ort::GetAvailableProviders();

        // Print the available providers
        std::cout << "Available Providers:" << std::endl;
        for (const auto &provider: availableProviders) {
            std::cout << provider << std::endl;
        }

        auto dsConfig = DsConfig::fromYAML(dsConfigPath);

        if (acousticSpeedup < 1 || acousticSpeedup > 1000) {
            std::cout << "WARNING: speedup must be in range [1, 1000]. Falling back to 10.\n";
            acousticSpeedup = 10;
        }

        if (dsConfig.useShallowDiffusion) {
            if (dsConfig.maxDepth < 0) {
                std::cout << "ERROR: max_depth is unset or negative in acoustic configuration.\n";
                return;
            }
            if (shallowDiffusionDepth > dsConfig.maxDepth) {
                shallowDiffusionDepth = dsConfig.maxDepth;
            }
            // make sure depth can be divided by speedup
            shallowDiffusionDepth = shallowDiffusionDepth / acousticSpeedup * acousticSpeedup;
        }

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

        auto dsProject = loadDsProject(dsFilePath, spkMixStr);
        size_t numSegments = dsProject.size();

        AcousticInference acousticInference(dsConfig.acoustic);
        bool isSessionInitOk = acousticInference.initSession(!cpuOnly, deviceIndex);
        if (!isSessionInitOk) {
            std::cout << "ERROR: Session initialization failed.\n";
            return;
        }

        std::vector< std::pair<int64_t, std::vector<float>> > waveformArr{};
        waveformArr.reserve(numSegments);

        InferenceSettings inferSettings{};
        inferSettings.speedup = acousticSpeedup;
        inferSettings.depth = shallowDiffusionDepth;

        for (size_t i = 0; i < numSegments; i++) {
            std::cout << i + 1 << " of " << numSegments << "\n";
            std::cout << "Preprocessing input" << "\n";
            auto offsetInSamples = static_cast<int64_t>(std::ceil(dsProject[i].offset * vocoderConfig.sampleRate));

            auto pd = acousticPreprocess(name2token, dsProject[i], dsConfig, frameLength);

            std::cout << "Mel" << "\n";
            auto mel = acousticInference.inferToOrtValue(pd, inferSettings);

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
        auto numWritten = audioFile.write(wavBuffer.data(), numFrames);
        if ((audioFile.error() != SF_ERR_NO_ERROR) || (numWritten == 0)) {
            std::cout << "ERROR: audio write failed. Reason: " << audioFile.strError() << '\n';
        }
    }
}