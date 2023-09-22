#include <iostream>
#include <iterator>
#include <numeric>
#include <unordered_map>

#include "DsProject.h"
#include "ArrayUtil.hpp"
#include "SampleCurve.h"
#include "Preprocess.h"


namespace diffsinger {

    inline std::vector<int64_t> phonemesToTokens(const std::unordered_map<std::string, int64_t> &name2token,
                                          const std::vector<std::string> &phonemes);
    inline std::vector<int64_t> phonemeDurationToFrames(const std::vector<double> &durations,
                                                 double frameLength);


    /* IMPLEMENTATION BELOW */

    PreprocessedData acousticPreprocess(
            const std::unordered_map<std::string, int64_t> &name2token,
            const DsSegment &dsSegment,
            double frameLength) {

        PreprocessedData pd{};

        pd.tokens = phonemesToTokens(name2token, dsSegment.ph_seq);
        pd.durations = phonemeDurationToFrames(dsSegment.ph_dur, frameLength);

        int64_t targetLength = std::accumulate(pd.durations.begin(), pd.durations.end(), static_cast<int64_t>(0));

        pd.f0 = dsSegment.f0.resample(frameLength, targetLength);
        pd.velocity = dsSegment.velocity.resample(frameLength, targetLength);
        if (pd.velocity.empty()) {
            pd.velocity.resize(targetLength, 1.0);
        }

        pd.gender = dsSegment.gender.resample(frameLength, targetLength);
        if (pd.gender.empty()) {
            pd.gender.resize(targetLength, 0.0);
        }

        pd.energy = dsSegment.energy.resample(frameLength, targetLength);
        pd.breathiness = dsSegment.breathiness.resample(frameLength, targetLength);

        // TODO: spk_mix

        return pd;
    }

    std::vector<int64_t> phonemesToTokens(const std::unordered_map<std::string, int64_t> &name2token,
                                             const std::vector<std::string> &phonemes) {
        std::vector<int64_t> tokens;
        tokens.reserve(phonemes.size());

        for (const auto &ph: phonemes) {
            auto it = name2token.find(ph);
            if (it != name2token.end()) {
                // If phoneme is found in name2token, push back value (token).
                tokens.push_back(it->second);
            } else {
                // Handle error: phoneme not found in name2token
                tokens.push_back(0);
            }
        }
        return tokens;
    }

    std::vector<int64_t> phonemeDurationToFrames(const std::vector<double> &durations,
                                                  double frameLength) {
        // Converts phoneme durations' units from seconds to frames
        std::vector<int64_t> phDurations;

        phDurations.reserve(durations.size());
        std::vector<double> phAccumulate;
        phAccumulate.reserve(durations.size());

        std::partial_sum(durations.begin(), durations.end(), std::back_inserter(phAccumulate));
        std::transform(phAccumulate.begin(), phAccumulate.end(), std::back_inserter(phDurations),
                       [frameLength](double value) { return std::llround(value / frameLength); });

        std::adjacent_difference(phDurations.begin(), phDurations.end(), phDurations.begin());

        return phDurations;
    }
}