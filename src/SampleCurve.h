
#ifndef DS_ONNX_INFER_SAMPLECURVE_H
#define DS_ONNX_INFER_SAMPLECURVE_H

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace diffsinger {
    struct SampleCurve {
        std::vector<double> samples;
        double timestep = 0.0;

        /**
         * @brief Resamples curve to target time step and length using interpolation.
         *
         * @param targetTimestep  The target curve time step.
         * @param targetLength    The target length of sample points.
         * @return                The target curve samples
         *
         * This function resamples original curve to target time step and length. The original curve
         * will be interpolated, and then resized to target length. If the size of interpolated vector is
         * smaller than target length, it will be truncated; otherwise, it will be expanded using the last value.
         */
        std::vector<double> resample(double targetTimestep, int64_t targetLength) const;
    };

    // TODO: still figuring out the format of spk_mix
    struct SpeakerMixCurve {
        std::unordered_map<std::string, std::vector<float>> spk;
        double timestep = 0.0;
    };
}


#endif //DS_ONNX_INFER_SAMPLECURVE_H
