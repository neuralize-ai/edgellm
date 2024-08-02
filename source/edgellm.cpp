#include <filesystem>
#include <vector>

#include "edgellm/edgellm.hpp"

#include <edgerunner/edgerunner.hpp>
#include <edgerunner/model.hpp>

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
    , m_sampler(m_vocabSize, MTemperature, MTopp)
    , m_ropeEmbedding(128, MMaxSequenceLength) {}

auto EdgeLLM::generate(const std::string& prompt) -> std::vector<std::string> {
    const auto inputTokens = m_tokenizer.encode(prompt, MNBos, MNEos);

    if (inputTokens.size() > MMaxSequenceLength) {
        return {};
    }

    const auto startIndex = MMaxSequenceLength - inputTokens.size();

    const auto [positionIdsCos, positionIdsSin] =
        m_ropeEmbedding.getEmbedding(inputTokens);

    return {};
}

}  // namespace edgellm
