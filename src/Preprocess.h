

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

    struct DsConfig;

    PreprocessedData acousticPreprocess(
            const std::unordered_map<std::string, int64_t> &name2token,
            const DsSegment &dsSegment,
            const DsConfig &dsConfig,
            double frameLength);

}
#endif //DS_ONNX_INFER_PREPROCESS_H
