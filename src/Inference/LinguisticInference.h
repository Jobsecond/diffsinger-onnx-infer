#ifndef DS_ONNX_INFER_LINGUISTICINFERENCE_H
#define DS_ONNX_INFER_LINGUISTICINFERENCE_H

#include "TString.h"
#include "Inference.h"
#include "ModelData.h"

namespace diffsinger {

    struct LinguisticInput;
    struct LinguisticOut;

    class LinguisticInference : public Inference {
    public:
        explicit LinguisticInference(const TString &modelPath);

        LinguisticOut infer(const LinguisticInput &input);
    };

} // diffsinger

#endif //DS_ONNX_INFER_LINGUISTICINFERENCE_H
