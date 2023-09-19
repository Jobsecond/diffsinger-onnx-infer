//
// Created by zhang on 2023/9/18.
//

#ifndef DS_ONNX_INFER_NAMESPACE_H
#define DS_ONNX_INFER_NAMESPACE_H


#include <cstdint>
#include <vector>

#include "TString.h"

namespace diffsinger {

    struct PreprocessedData {
        std::vector<int64_t> tokens;
        std::vector<int64_t> phDur;
        std::vector<double> f0Seq;
    };

}

#endif //DS_ONNX_INFER_NAMESPACE_H
