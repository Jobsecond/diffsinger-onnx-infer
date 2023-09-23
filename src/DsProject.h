#ifndef DS_ONNX_INFER_NAMESPACE_H
#define DS_ONNX_INFER_NAMESPACE_H


#include <cstdint>
#include <vector>

#include "TString.h"
#include "SampleCurve.h"

namespace diffsinger {

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

    std::vector<DsSegment> loadDsProject(const TString &dsFilePath, const std::string &spkMixStr = "");

}

#endif //DS_ONNX_INFER_NAMESPACE_H
