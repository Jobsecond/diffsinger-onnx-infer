#include <iostream>
#include <fstream>
#include <sstream>
#include <cctype>
#include <filesystem>
#include <chrono>
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
             ExecutionProvider ep = ExecutionProvider::CPU,
             int deviceIndex = 0);

    ExecutionProvider parseEPFromString(const std::string &ep);
    std::string millisecondsToSecondsString(long long milliseconds);
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
    program.add_argument("--ep").default_value("cpu").help(
            "Execution Provider for audio inference. Supported: cpu (CPUExecutionProvider)"
#ifdef ONNXRUNTIME_ENABLE_CUDA
            ", cuda (CUDAExecutionProvider)"
#endif
#ifdef ONNXRUNTIME_ENABLE_DML
            ", directml, dml (DmlExecutionProvider)"
#endif
            );
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
    auto ep = program.get("--ep");
    auto deviceIndex = program.get<int>("--device-index");

    auto epEnum = diffsinger::parseEPFromString(ep);

#ifdef WIN32
    auto currentCodePage = ::GetACP();
    diffsinger::run(MBStringToWString(dsPath, currentCodePage),
                    MBStringToWString(dsConfigPath, currentCodePage),
                    MBStringToWString(vocoderConfigPath, currentCodePage),
                    MBStringToWString(outputAudioTitle, currentCodePage),
                    spkMixStr,
                    speedup,
                    depth,
                    epEnum,
                    deviceIndex);
#else
    diffsinger::run(dsPath, dsConfigPath, vocoderConfigPath, outputAudioTitle, spkMixStr, speedup, depth, epEnum, deviceIndex);
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
             ExecutionProvider ep,
             int deviceIndex) {

        // Get the available providers
        auto availableProviders = Ort::GetAvailableProviders();

        // Print the available providers
        std::cout << "Available Providers:" << std::endl;
        for (const auto &provider: availableProviders) {
            std::cout << '-' << ' ' << provider << std::endl;
        }

        auto dsConfig = DsConfig::fromYAML(dsConfigPath);

        if (acousticSpeedup < 1 || acousticSpeedup > 1000) {
            std::cout << "!! WARNING: speedup must be in range [1, 1000]. Falling back to 10.\n";
            acousticSpeedup = 10;
        }

        if (dsConfig.useShallowDiffusion) {
            if (dsConfig.maxDepth < 0) {
                std::cout << "!! ERROR: max_depth is unset or negative in acoustic configuration.\n";
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
            // handle CRLF line endings on Linux and macOS
            if (!line.empty() && line[line.size() - 1] == '\r')
                line.erase(line.size() - 1);

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

        std::cout << '\n';
        std::cout << "Initializing acoustic inference session...\n";
        AcousticInference acousticInference(dsConfig.acoustic);

        bool isAcousticSessionInitOk = acousticInference.initSession(ep, deviceIndex);
        if (!isAcousticSessionInitOk) {
            std::cout << "!! ERROR: Acoustic Session initialization failed.\n";
            return;
        }
        std::cout << "Successfully created acoustic inference session.\n";
        acousticInference.printModelFeatures();

        std::cout << '\n';
        std::cout << "Initializing vocoder inference session...\n";
        VocoderInference vocoderInference(vocoderConfig.model);

        bool isVocoderSessionInitOk = vocoderInference.initSession(ExecutionProvider::CPU, 0);
        if (!isVocoderSessionInitOk) {
            std::cout << "!! ERROR: Vocoder Session initialization failed.\n";
            return;
        }
        std::cout << "Successfully created vocoder inference session.\n";
        std::cout << '\n';

        std::vector< std::pair<int64_t, std::vector<float>> > waveformArr{};
        waveformArr.reserve(numSegments);

        InferenceSettings inferSettings{};
        inferSettings.speedup = acousticSpeedup;
        inferSettings.depth = shallowDiffusionDepth;

        for (size_t i = 0; i < numSegments; i++) {
            std::cout << i + 1 << " of " << numSegments << "\n";
            auto timeStart = std::chrono::steady_clock::now();
            std::cout << ">> Preprocessing input" << "\n";
            auto offsetInSamples = static_cast<int64_t>(std::ceil(dsProject[i].offset * vocoderConfig.sampleRate));

            auto pd = acousticPreprocess(name2token, dsProject[i], dsConfig, frameLength);

            std::cout << ">> Acoustic infer -> Mel" << "\n";
            auto mel = acousticInference.inferToOrtValue(pd, inferSettings);

            if (mel == Ort::Value(nullptr)) {
                std::cout << "!! ERROR: Acoustic Infer failed.\n";
                waveformArr.emplace_back(offsetInSamples, std::vector<float>{});
                continue;
            }
            // mel will be `std::move`d in the next steps, so it will not be usable after that.
            std::cout << ">> Vocoder infer -> Waveform" << "\n";
            auto waveform = vocoderInference.infer(mel, pd.f0);

            waveformArr.emplace_back(offsetInSamples, std::move(waveform));
            auto timeEnd = std::chrono::steady_clock::now();
            auto timeSpent = std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart).count();
            std::cout << ">> Time Elapsed: " << millisecondsToSecondsString(timeSpent) << " seconds\n";
        }

        std::cout << "Inference finished.\n";
        std::cout << ">> Concatenating and saving wave file...\n";
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
            std::cout << "!! ERROR: audio write failed. Reason: " << audioFile.strError() << '\n';
        }
    }

    ExecutionProvider parseEPFromString(const std::string &ep) {
        std::string epLower;
        epLower.resize(ep.size());
        std::transform(ep.begin(), ep.end(), epLower.begin(), [](unsigned char c){ return std::tolower(c); });
        if (ep == "cuda" || ep == "cudaexecutionprovider") {
            return ExecutionProvider::CUDA;
        }
        else if (ep == "dml" || ep == "directml" || ep == "dmlexecutionprovider") {
            return ExecutionProvider::DirectML;
        }
        return ExecutionProvider::CPU;
    }

    std::string millisecondsToSecondsString(long long milliseconds) {
        auto integerPart = milliseconds / 1000;
        auto decimalPart = milliseconds % 1000;
        std::stringstream ss;
        ss << integerPart << '.';
        if (decimalPart < 100) {
            ss << '0';
        }
        if (decimalPart < 10) {
            ss << '0';
        }
        if (decimalPart == 0) {
            ss << '0';
        }
        ss << decimalPart;
        return ss.str();
    }
}
