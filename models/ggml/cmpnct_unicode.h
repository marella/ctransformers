#ifndef CNCT_UNICODE_H
#define CNCT_UNICODE_H

#include <string>
#include <vector>
#include "cmpnct_unicode.h"
enum CNCTCharType {
    DIGIT,          // a numerical char in any language
    LETTER,         // a letter in any language
    WHITESPACE,     // any form of whitespace
    ACCENT_MARK,    // letter modifiers like ´ in é
    PUNCTUATION,    // punctuation including brackets
    SYMBOL,         // math, currency, other symbols
    CONTROL,        // control characters
    MIXED,          // a mix of the above
    UNIDENTIFIED    // something more exotic like emoji or separators
};

struct CNCTUnicode;

struct CNCTString {
    std::string str;
    size_t utf8_chars;

    CNCTCharType char_type=UNIDENTIFIED;
    bool is_sequential=false;

    size_t seq_offset_bytes=0;
    size_t seq_offset_utf8_chars=0;

    bool operator==(const std::string &other) const;
    bool operator==(const char other) const;
    bool operator==(const CNCTString &other) const;
    CNCTString &operator+=(const std::string &other);
    CNCTString &operator+=(const char other);
    friend CNCTString operator+(CNCTString lhs, const std::string &rhs);
    friend CNCTString operator+(CNCTString lhs, const char rhs);
    CNCTString& operator+=(const CNCTString& other);
    friend CNCTString operator+(CNCTString lhs, const CNCTString& rhs);
};



struct CNCTUnicode {
    static bool check_code_range(int c, const std::vector<std::pair<int, int>>& ranges);
    static CNCTCharType get_code_type(int c);
    static CNCTCharType get_code_type(const std::string &utf8_char);
    static int utf8_len(const char c);
    static int strlen_utf8(std::string src);
    static std::vector<std::string> split_utf8(const std::string &src);
    static std::vector<CNCTString> split_utf8_enhanced(const std::string &src);
    static CNCTCharType string_identify(const std::string& str);
    static bool string_test(const std::string& str, CNCTCharType chartype);
};

#endif // CNCT_UNICODE_H