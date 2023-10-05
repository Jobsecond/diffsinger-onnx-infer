#include "PowerManagement.h"

#ifdef _WIN32
#include <Windows.h>
#endif

namespace diffsinger {

    void keepSystemAwake() {
#ifdef _WIN32
        ::SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED);
#endif
    }

    void restorePowerState() {
#ifdef _WIN32
        ::SetThreadExecutionState(ES_CONTINUOUS);
#endif
    }

}  // namespace diffsinger
