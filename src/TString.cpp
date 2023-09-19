#include "TString.h"

#if defined(WIN32)
#include <Windows.h>
#endif

namespace diffsinger {
#if defined(WIN32)
    std::wstring MBStringToWString(const std::string &mbStr, unsigned int codePage) {
        int len = MultiByteToWideChar(codePage, 0, mbStr.c_str(), mbStr.size(), nullptr, 0);
        std::wstring buffer;
        buffer.resize(len);
        MultiByteToWideChar(codePage, 0, mbStr.c_str(), mbStr.size(), buffer.data(), len);
        return buffer;
    }
#endif
}