
#ifndef DS_ONNX_INFER_TSTRING_H
#define DS_ONNX_INFER_TSTRING_H


#include <string>

namespace diffsinger {
#if defined(WIN32)
    #define DS_T(x) L##x
    using TString = std::wstring;
    using TChar = wchar_t;

    std::wstring MBStringToWString(const std::string &mbStr, unsigned int codePage);
#else
    #define DS_T(x) x
    using TString = std::string;
    using TChar = char;
#endif
}

#endif //DS_ONNX_INFER_TSTRING_H
