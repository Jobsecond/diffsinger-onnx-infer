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
        std::vector<int> ph_num;
        std::vector<int> note_seq;  // MIDI note number
        std::vector<double> note_dur;
        SampleCurve f0;
        SampleCurve gender;
        SampleCurve velocity;
        SampleCurve energy;
        SampleCurve breathiness;
        SpeakerMixCurve spk_mix;
    };

    std::vector<DsSegment> loadDsProject(const TString &dsFilePath, const std::string &spkMixStr = "");

    int noteNameToMidi(const std::string &note);

}

#endif //DS_ONNX_INFER_NAMESPACE_H
