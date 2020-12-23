#pragma once

#include "my_utils.hpp"

// From https://stackoverflow.com/questions/8629382/debug-macro-for-c-with-variable-arguments-without-the-format-string/8629465

class VariadicToOutputStream {
public:
    VariadicToOutputStream(std::ostream& s, const std::string& separator = "") : m_stream(s), m_hasEntries(false), m_separator(separator) {}
    template<typename ObjectType>
    VariadicToOutputStream& operator,(const ObjectType& v) {
        if (m_hasEntries) m_stream << m_separator;
        m_stream << v;
        m_hasEntries=true;
        return *this;
    }
    ~VariadicToOutputStream() {
        //m_stream << std::endl;
    }

private:
    std::ostream& m_stream;
    bool m_hasEntries;
    std::string m_separator;
};

#define debug_cout(debug, ...) { \
   if (debug) { \
      VariadicToOutputStream(std::cout),__VA_ARGS__; \
   } \
}

