#include <iterator>
#include <numeric>
#include <unordered_map>

#include "DsConfig.h"
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
            const DsConfig &dsConfig,
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

        // DONE: static spk_mix
        // TODO: curve spk_mix
        if (!dsConfig.speakers.empty()) {
            // Required to choose a speaker.
            int64_t spkEmbedArraySize = targetLength * SPK_EMBED_SIZE;
            pd.spk_embed.resize(spkEmbedArraySize);
            if (dsSegment.spk_mix.empty()) {
                // Use the first one by default.
                auto emb = dsConfig.spkEmb.getMixedEmb({{dsConfig.speakers[0], 1.0}});
                for (size_t i = 0; i < spkEmbedArraySize; ++i) {
                    pd.spk_embed[i] = emb[i % SPK_EMBED_SIZE];
                }
            } else {
                auto spkMixResampled = dsSegment.spk_mix.resample(frameLength, targetLength);
                for (int64_t i = 0; i < targetLength; ++i) {
                    std::unordered_map<std::string, double> mix;
                    int64_t speakerIndex = 0;
                    for (const auto &speakerItem : spkMixResampled.spk) {
                        // If SampleCurve::resample guarantees the size of returned array is at least `targetLength`,
                        // subscripting will not go out of range here.
                        mix[speakerItem.first] = speakerItem.second.samples[i];
                        ++speakerIndex;
                    }
                    auto emb = dsConfig.spkEmb.getMixedEmb(mix);
                    int64_t y = i * SPK_EMBED_SIZE;
                    for (int64_t j = 0; j < SPK_EMBED_SIZE; ++j) {
                        pd.spk_embed[y + j] = emb[j];
                    }
                }
            }
        }

        return pd;
    }

    LinguisticInput linguisticPreprocess(
            const std::unordered_map<std::string, int64_t> &name2token,
            const DsSegment &dsSegment,
            double frameLength) {
        LinguisticInput li{};
        li.tokens = phonemesToTokens(name2token, dsSegment.ph_seq);
        li.word_div = std::vector<int64_t>(dsSegment.ph_num.begin(), dsSegment.ph_num.end());
        li.word_dur = phonemeDurationToFrames(dsSegment.note_dur, frameLength);

        return li;
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

    std::vector<int> noteMidiToDurMidi(const std::vector<int> &noteMidi, const std::vector<int> &phNum) {
        std::vector<int> out;
        auto newSize = std::accumulate(phNum.begin(), phNum.end(), 0);
        out.reserve(newSize);
        auto n = noteMidi.size();
        if (n > phNum.size()) {
            n = phNum.size();
        }

        for (size_t i = 0; i < n; ++i) {
            for (int j = 0; j < phNum[i]; ++j) {
                out.push_back(noteMidi[i]);
            }
        }
        return out;
    }

    void fillZeroMidiWithNearestInPlace(std::vector<int> &src) {
        auto not_zero = [](int x) constexpr { return x != 0; };
        auto it = std::find(src.begin(), src.end(), 0);
        auto it_left = std::find_if(src.begin(), it, not_zero);
        auto it_right = std::find_if(it, src.end(), not_zero);

        if (it == src.end() || it_right == src.end()) {
            return;
        }

        // fill zero values at beginning
        if (it_left == it) {
            std::fill(src.begin(), it_right, *it_right);
            it = it_right;
        }

        // middle and end
        while (it != src.end() || it_right != src.end()) {
            auto it_prev = it;
            it = std::find(it_prev, src.end(), 0);
            it_left = it - 1;
            it_right = std::find_if(it, src.end(), not_zero);
            if (it_right == src.end()) {
                // end
                std::fill(it, it_right, *it_left);
                break;
            }
            // middle
            auto dist = std::distance(it_left, it_right);
            auto left_fills = dist / 2;
            auto right_fills = dist - left_fills - 1;
            std::fill(it, it + left_fills, *it_left);
            std::fill(it_right - right_fills, it_right, *it_right);
        }
    }

    std::vector<int> fillZeroMidiWithNearest(const std::vector<int> &src) {
        std::vector<int> dst(src.begin(), src.end());
        fillZeroMidiWithNearestInPlace(dst);
        return dst;
    }
}
