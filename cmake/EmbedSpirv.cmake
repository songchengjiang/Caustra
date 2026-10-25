file(READ "${INPUT}" spirv HEX)
string(LENGTH "${spirv}" hex_length)
math(EXPR remainder "${hex_length} % 8")
if(NOT remainder EQUAL 0)
  message(FATAL_ERROR "SPIR-V must contain complete 32-bit words")
endif()
# Convert all little-endian words in one native regex pass. A CMake foreach
# repeatedly copies the growing source/hex strings and becomes quadratic for
# the larger surface/volume integrator. Each generated line holds one word.
string(REGEX REPLACE "([0-9a-f][0-9a-f])([0-9a-f][0-9a-f])([0-9a-f][0-9a-f])([0-9a-f][0-9a-f])"
  "0x\\4\\3\\2\\1,\n" words "${spirv}")
if(NOT DEFINED SYMBOL)
  set(SYMBOL pathtraceSpv)
endif()
file(WRITE "${OUTPUT}" "#pragma once\n#include <cstdint>\ninline constexpr uint32_t ${SYMBOL}[] = {\n${words}};\n")
