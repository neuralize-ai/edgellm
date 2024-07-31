#include <filesystem>
#include <vector>

#include "edgellm/edgellm.hpp"

#include <edgerunner/edgerunner.hpp>
#include <fmt/core.h>
#include <fmt/ranges.h>

namespace edgellm {

EdgeLLM::EdgeLLM(std::vector<std::filesystem::path>&& promptProcessorPaths,
                 std::vector<std::filesystem::path>&& tokenGeneratorPaths,
                 const std::filesystem::path& tokenizerPath)
    : m_promptProcessorPaths(std::move(promptProcessorPaths))
    , m_tokenGeneratorPaths(std::move(tokenGeneratorPaths))
    , m_creationSuccess(m_tokenizer.load(tokenizerPath))
    , m_vocabSize(m_tokenizer.getVocabSize())
    , m_bosId(m_tokenizer.getBosTok())
    , m_eosId(m_tokenizer.getEosTok())
    , m_sampler(m_vocabSize, MTemperature, MTopp) {}

}  // namespace edgellm
