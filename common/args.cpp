#include "args.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>
#endif

std::vector<std::string> utf8_args(int argc, char ** argv) {
    std::vector<std::string> args;
#ifdef _WIN32
    (void) argc;
    (void) argv;
    int n = 0;
    LPWSTR * wide = CommandLineToArgvW(GetCommandLineW(), &n);
    for (int i = 0; i < n; i++) {
        const int size = WideCharToMultiByte(CP_UTF8, 0, wide[i], -1, nullptr, 0, nullptr, nullptr);
        std::string a(size > 0 ? size - 1 : 0, '\0');
        WideCharToMultiByte(CP_UTF8, 0, wide[i], -1, a.data(), size, nullptr, nullptr);
        args.push_back(a);
    }
    LocalFree(wide);
#else
    for (int i = 0; i < argc; i++) args.push_back(argv[i]);
#endif
    return args;
}
