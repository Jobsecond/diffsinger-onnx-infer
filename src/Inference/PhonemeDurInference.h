#ifndef DS_ONNX_INFER_PHONEMEDURINFERENCE_H
#define DS_ONNX_INFER_PHONEMEDURINFERENCE_H

#include "Inference.h"

namespace diffsinger {

    struct LinguisticEncodedData;

    class PhonemeDurInference : public Inference {
    public:
        explicit PhonemeDurInference(const TString &modelPath);

        std::vector<float> infer(const LinguisticEncodedData &linguisticEncodedData,
                                 const std::vector<int> &ph_midi);
    };

} // namespace diffsinger

#endif //DS_ONNX_INFER_PHONEMEDURINFERENCE_H
