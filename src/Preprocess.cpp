//
// Created by zhang on 2023/9/16.
//
#include <iostream>
#include <iterator>
#include <numeric>
#include <unordered_map>

#include "DsCommon.h"
#include "ArrayUtil.hpp"
#include "Preprocess.h"


namespace diffsinger {

    /**
     * @brief Transform curve samples to target time step and length using interpolation.
     *
     * @param inputSamples    The original curve samples.
     * @param inputTimestep   The original curve time step.
     * @param targetTimestep  The target curve time step.
     * @param targetLength    The target length of sample points.
     * @return                The target curve samples
     *
     * This function transforms original curve to target time step and length. The original curve
     * will be interpolated, and then resized to target length. If the size of interpolated vector is
     * smaller than target length, it will be truncated; otherwise, it will be expanded using the last value.
     */
    inline std::vector<double> curveTransform(const std::vector<double> &inputSamples, double inputTimestep,
                                       double targetTimestep, int64_t targetLength);
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

        pd.f0 = curveTransform(dsSegment.f0.samples, dsSegment.f0.timestep, frameLength, targetLength);
        pd.velocity = curveTransform(dsSegment.velocity.samples, dsSegment.velocity.timestep, frameLength, targetLength);
        pd.gender = curveTransform(dsSegment.gender.samples, dsSegment.gender.timestep, frameLength, targetLength);
        pd.energy = curveTransform(dsSegment.energy.samples, dsSegment.energy.timestep, frameLength, targetLength);
        pd.breathiness = curveTransform(dsSegment.breathiness.samples, dsSegment.breathiness.timestep, frameLength, targetLength);

        // TODO: spk_mix

        return pd;
    }

    std::vector<double> curveTransform(const std::vector<double> &inputSamples, double inputTimestep, double targetTimestep, int64_t targetLength) {
        if (inputSamples.empty() || inputTimestep == 0 || targetTimestep == 0 || targetLength == 0) {
            return {};
        }
        // Find the time duration of input samples in seconds.
        auto tMax = static_cast<double>(inputSamples.size() - 1) * inputTimestep;

        // Construct target time axis for interpolation.
        auto targetTimeAxis = arange(0.0, tMax, targetTimestep);

        // Construct input time axis (for interpolation).
        auto inputTimeAxis = arange(0.0, static_cast<double>(inputSamples.size()), 1.0);
        std::transform(inputTimeAxis.begin(), inputTimeAxis.end(), inputTimeAxis.begin(),
                       [inputTimestep](double value) {return value * inputTimestep; });

        // Interpolate sample curve to target time axis
        auto targetSamples = interpolate(targetTimeAxis, inputTimeAxis, inputSamples);

        // Resize the interpolated curve vector to target length
        auto actualLength = static_cast<int64_t>(targetSamples.size());

        if (actualLength > targetLength) {
            // Truncate vector to target length
            targetSamples.resize(targetLength);
        } else if (actualLength < targetLength) {
            // Expand vector to target length, filling last value
            auto lastValue = targetSamples.back();
            targetSamples.resize(targetLength, lastValue);
        }
        return targetSamples;
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