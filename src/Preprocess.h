

#ifndef DS_ONNX_INFER_PREPROCESS_H
#define DS_ONNX_INFER_PREPROCESS_H

#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <cmath>

#include "ModelData.h"

namespace diffsinger {

    struct DsSegment;
    struct DsConfig;

    PreprocessedData acousticPreprocess(
            const std::unordered_map<std::string, int64_t> &name2token,
            const DsSegment &dsSegment,
            const DsConfig &dsConfig,
            double frameLength);

    LinguisticInput linguisticPreprocess(
            const std::unordered_map<std::string, int64_t> &name2token,
            const DsSegment &dsSegment,
            double frameLength);

    std::vector<int> noteMidiToDurMidi(const std::vector<int> &noteMidi, const std::vector<int> &noteNum);

    std::vector<int> fillZeroMidiWithNearest(const std::vector<int> &src);

    void fillZeroMidiWithNearestInPlace(std::vector<int> &src);
}
#endif //DS_ONNX_INFER_PREPROCESS_H
