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

    struct SampleCurve {
        std::vector<double> samples;
        double timestep = 0.0;
    };

    // TODO: still figuring out the format of spk_mix
    struct SpeakerMixCurve {
        std::unordered_map<std::string, std::vector<double>> spk;
        double timestep = 0.0;
    };

    struct DsSegment {
        double offset = 0.0;
        std::vector<std::string> ph_seq;
        std::vector<double> ph_dur;
        SampleCurve f0;
        SampleCurve gender;
        SampleCurve velocity;
        SampleCurve energy;
        SampleCurve breathiness;
        SpeakerMixCurve spk_mix;
    };
}

#endif //DS_ONNX_INFER_NAMESPACE_H
