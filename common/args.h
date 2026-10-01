#pragma once

#include <string>
#include <vector>

/** The command line as UTF-8. On Windows argv arrives in the ANSI code page, which cannot hold Japanese on most systems. */
std::vector<std::string> utf8_args(int argc, char ** argv);
