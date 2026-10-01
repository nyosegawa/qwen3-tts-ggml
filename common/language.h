#pragma once

#include <string>

/**
 * Whether the BCP 47 tag `tag` names `language` itself or a region or script of it (`ja-JP` for `ja`,
 * `zh-Hant` for `zh`), compared without regard to case as BCP 47 asks.
 */
bool bcp47_matches(const std::string & tag, const std::string & language);
