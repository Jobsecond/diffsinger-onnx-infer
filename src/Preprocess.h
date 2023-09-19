

#ifndef DS_ONNX_INFER_PREPROCESS_H
#define DS_ONNX_INFER_PREPROCESS_H

#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <cmath>

#include "DsCommon.h"

namespace diffsinger {


    PreprocessedData acousticPreprocess(
            const std::unordered_map<std::string, int64_t> &name2token,
            const std::vector<std::string> &phonemes,
            const std::vector<double> &durations,
            const std::vector<double> &f0,
            double frameLength,
            double f0Timestep);

}
#endif //DS_ONNX_INFER_PREPROCESS_H
