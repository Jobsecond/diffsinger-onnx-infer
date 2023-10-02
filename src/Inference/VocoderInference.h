#ifndef DS_ONNX_INFER_VOCODERINFERENCE_H
#define DS_ONNX_INFER_VOCODERINFERENCE_H


#include <vector>

#include "TString.h"
#include "Inference.h"

namespace diffsinger {

    class VocoderInference : public Inference {
    public:
        explicit VocoderInference(const TString &modelPath);

        std::vector<float> infer(Ort::Value &mel, const std::vector<double> &f0);
    };  // class VocoderInference

}  // namespace diffsinger

#endif //DS_ONNX_INFER_VOCODERINFERENCE_H
