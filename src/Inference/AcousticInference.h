#ifndef DS_ONNX_INFER_ACOUSTICINFERENCE_H
#define DS_ONNX_INFER_ACOUSTICINFERENCE_H

#include <vector>

#include "TString.h"
#include "Inference.h"
#include "AcousticModelFlags.h"

namespace diffsinger {

    struct PreprocessedData;
    struct AcousticInferenceSettings;

    struct AcousticInferenceSettings {
        int speedup = 10;
        int depth = 1000;
    };  // struct AcousticInferenceSettings


    class AcousticInference : public Inference {
    public:
        explicit AcousticInference(const TString &modelPath);

        void printModelFeatures();

        static std::vector<float> ortValueToVector(const Ort::Value &value);

        std::vector<float> infer(const PreprocessedData &pd, const AcousticInferenceSettings &inferSettings);

        Ort::Value inferToOrtValue(const PreprocessedData &pd, const AcousticInferenceSettings &inferSettings);

    private:
        AcousticModelFlags m_modelFlags;
    private:
        void updateFlags();

    protected:
        bool postInitCheck() override;

        void postCleanup() override;
    };  // class AcousticInference

} // namespace diffsinger

#endif //DS_ONNX_INFER_ACOUSTICINFERENCE_H
