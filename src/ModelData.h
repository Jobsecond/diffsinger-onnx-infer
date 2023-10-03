#ifndef DS_ONNX_INFER_MODELDATA_H
#define DS_ONNX_INFER_MODELDATA_H

#include <cstdint>
#include <vector>

namespace diffsinger {
    constexpr int spkEmbedLastDimension = 256;

    struct PreprocessedData {
        std::vector<int64_t> tokens;
        std::vector<int64_t> durations;
        std::vector<double> f0;
        std::vector<double> velocity;
        std::vector<double> gender;
        std::vector<float> spk_embed;
        std::vector<double> energy;
        std::vector<double> breathiness;
    };

    struct LinguisticInput {
        std::vector<int64_t> tokens;
        std::vector<int64_t> word_div;
        std::vector<int64_t> word_dur;
    };

    struct LinguisticOut {
        std::vector<float> encoder_out;

        // x_masks should be bool vector, however vector<bool> is not a container storing bool
        std::vector<char> x_masks;

        bool empty() const {
            return encoder_out.empty() && x_masks.empty();
        }
    };
}

#endif //DS_ONNX_INFER_MODELDATA_H
