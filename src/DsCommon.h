//
// Created by zhang on 2023/9/18.
//

#ifndef DS_ONNX_INFER_NAMESPACE_H
#define DS_ONNX_INFER_NAMESPACE_H


#include <cstdint>
#include <vector>

#include "TString.h"

namespace diffsinger {

    constexpr int spkEmbedLastDimension = 256;

    struct PreprocessedData {
        std::vector<int64_t> tokens;
        std::vector<int64_t> durations;
        std::vector<double> f0;
        std::vector<double> velocity;
        std::vector<double> gender;
        std::vector<double> spk_embed;
        std::vector<double> energy;
        std::vector<double> breathiness;
    };

}

#endif //DS_ONNX_INFER_NAMESPACE_H
