#ifndef DS_ONNX_INFER_INFERENCEUTILS_HPP
#define DS_ONNX_INFER_INFERENCEUTILS_HPP

#include <string>
#include <vector>
#include <iostream>
#include <unordered_set>
#include <cstdint>

#include <onnxruntime_cxx_api.h>


namespace diffsinger {
    inline std::unordered_set<std::string> getSupportedInputNames(const Ort::Session &session);

    inline std::unordered_set<std::string> getSupportedOutputNames(const Ort::Session &session);

    template<class T_vector, class T_tensor = T_vector>
    inline Ort::Value vectorToTensor(const std::vector<T_vector> &vec);

    template<class T_vector, class T_tensor = T_vector>
    inline Ort::Value vectorToTensorWithShape(const std::vector<T_vector> &vec, const std::vector<int64_t> &shape);

    template<class T_scalar, class T_tensor = T_scalar>
    inline Ort::Value scalarToTensor(const T_scalar &scalar);

    template<class T_vector, class T_tensor = T_vector>
    inline void appendVectorToInputTensors(const char *inputName,
                                           const std::vector<T_vector> &vec,
                                           std::vector<const char *> &inputNames,
                                           std::vector<Ort::Value> &inputTensors);

    template<class T_vector, class T_tensor = T_vector>
    inline void appendVectorToInputTensorsWithShape(const char *inputName,
                                                    const std::vector<T_vector> &vec,
                                                    const std::vector<int64_t> &shape,
                                                    std::vector<const char *> &inputNames,
                                                    std::vector<Ort::Value> &inputTensors);

    template<class T_scalar, class T_tensor = T_scalar>
    inline void appendScalarToInputTensors(const char *inputName,
                                           const T_scalar &scalar,
                                           std::vector<const char *> &inputNames,
                                           std::vector<Ort::Value> &inputTensors);

    inline bool hasKey(const std::unordered_set<std::string> &container, const std::string &key);

    inline void printOrtError(const Ort::Exception &err);


    /* IMPLEMENTATION BELOW */

    std::unordered_set<std::string> getSupportedInputNames(const Ort::Session &session) {
        auto inputCount = session.GetInputCount();
        std::unordered_set<std::string> supportedInputNames;
        supportedInputNames.reserve(inputCount);

        Ort::AllocatorWithDefaultOptions allocator;

        for (size_t i = 0; i < inputCount; i++) {
            auto inputNamePtr = session.GetInputNameAllocated(i, allocator);
            supportedInputNames.emplace(inputNamePtr.get());
        }

        return supportedInputNames;
    }

    std::unordered_set<std::string> getSupportedOutputNames(const Ort::Session &session) {
        auto outputCount = session.GetOutputCount();
        std::unordered_set<std::string> supportedOutputNames;
        supportedOutputNames.reserve(outputCount);

        Ort::AllocatorWithDefaultOptions allocator;

        for (size_t i = 0; i < outputCount; i++) {
            auto outputNamePtr = session.GetOutputNameAllocated(i, allocator);
            supportedOutputNames.emplace(outputNamePtr.get());
        }

        return supportedOutputNames;
    }

    template<class T_vector, class T_tensor>
    Ort::Value vectorToTensor(const std::vector<T_vector> &vec) {
        int64_t shape[] = { 1, static_cast<int64_t>(vec.size()) };  // shape = {1, N}
        auto shapeSize = sizeof(shape) / sizeof(shape[0]);

        Ort::AllocatorWithDefaultOptions allocator;
        auto tensor = Ort::Value::CreateTensor<T_tensor>(allocator, shape, shapeSize);
        auto buffer = tensor.template GetTensorMutableData<T_tensor>();
        for (size_t i = 0; i < vec.size(); i++) {
            buffer[i] = static_cast<T_tensor>(vec[i]);
        }

        return tensor;
    }

    template<class T_vector, class T_tensor>
    Ort::Value vectorToTensorWithShape(const std::vector<T_vector> &vec, const std::vector<int64_t> &shape) {
        auto shapeSize = shape.size();

        Ort::AllocatorWithDefaultOptions allocator;
        auto tensor = Ort::Value::CreateTensor<T_tensor>(allocator, shape.data(), shapeSize);
        auto buffer = tensor.template GetTensorMutableData<T_tensor>();
        for (size_t i = 0; i < vec.size(); i++) {
            buffer[i] = static_cast<T_tensor>(vec[i]);
        }

        return tensor;
    }

    template<class T_scalar, class T_tensor>
    Ort::Value scalarToTensor(const T_scalar &scalar) {
        int64_t shape[] = { 1 };
        auto shapeSize = sizeof(shape) / sizeof(shape[0]);

        Ort::AllocatorWithDefaultOptions allocator;
        auto tensor = Ort::Value::CreateTensor<T_tensor>(allocator, shape, shapeSize);
        auto buffer = tensor.template GetTensorMutableData<T_tensor>();
        buffer[0] = static_cast<T_tensor>(scalar);

        return tensor;
    }

    template<class T_vector, class T_tensor>
    void appendVectorToInputTensors(const char *inputName,
                                    const std::vector<T_vector> &vec,
                                    std::vector<const char *> &inputNames,
                                    std::vector<Ort::Value> &inputTensors) {
        inputNames.push_back(inputName);
        auto inputTensor = vectorToTensor<T_vector, T_tensor>(vec);
        inputTensors.push_back(std::move(inputTensor));
    }

    template<class T_vector, class T_tensor>
    void appendVectorToInputTensorsWithShape(const char *inputName,
                                             const std::vector<T_vector> &vec,
                                             const std::vector<int64_t> &shape,
                                             std::vector<const char *> &inputNames,
                                             std::vector<Ort::Value> &inputTensors) {
        inputNames.push_back(inputName);
        auto inputTensor = vectorToTensorWithShape<T_vector, T_tensor>(vec, shape);
        inputTensors.push_back(std::move(inputTensor));
    }

    template<class T_scalar, class T_tensor>
    void appendScalarToInputTensors(const char *inputName,
                                    const T_scalar &scalar,
                                    std::vector<const char *> &inputNames,
                                    std::vector<Ort::Value> &inputTensors) {
        inputNames.push_back(inputName);
        auto inputTensor = scalarToTensor<T_scalar, T_tensor>(scalar);
        inputTensors.push_back(std::move(inputTensor));
    }

    bool hasKey(const std::unordered_set<std::string> &container, const std::string &key) {
        return container.find(key) != container.end();
    }

    void printOrtError(const Ort::Exception &err) {
        std::cout << "[ONNXRuntimeError] : "
                  << err.GetOrtErrorCode() << " : "
                  << err.what() << '\n';
    }
}  // namespace diffsinger

#endif //DS_ONNX_INFER_INFERENCEUTILS_HPP
