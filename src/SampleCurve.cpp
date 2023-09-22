#include "ArrayUtil.hpp"
#include "SampleCurve.h"

namespace diffsinger {
    std::vector<double>
    SampleCurve::resample(double targetTimestep, int64_t targetLength) const {
        if (samples.empty() || timestep == 0 || targetTimestep == 0 || targetLength == 0) {
            return {};
        }
        // Find the time duration of input samples in seconds.
        auto tMax = static_cast<double>(samples.size() - 1) * timestep;

        // Construct target time axis for interpolation.
        auto targetTimeAxis = arange(0.0, tMax, targetTimestep);

        // Construct input time axis (for interpolation).
        auto inputTimeAxis = arange(0.0, static_cast<double>(samples.size()), 1.0);
        std::transform(inputTimeAxis.begin(), inputTimeAxis.end(), inputTimeAxis.begin(),
                       [this](double value) { return value * timestep; });

        // Interpolate sample curve to target time axis
        auto targetSamples = interpolate(targetTimeAxis, inputTimeAxis, samples);

        // Resize the interpolated curve vector to target length
        auto actualLength = static_cast<int64_t>(targetSamples.size());

        if (actualLength > targetLength) {
            // Truncate vector to target length
            targetSamples.resize(targetLength);
        } else if (actualLength < targetLength) {
            // Expand vector to target length, filling last value
            auto lastValue = targetSamples.back();
            targetSamples.resize(targetLength, lastValue);
        }
        return targetSamples;
    }
}