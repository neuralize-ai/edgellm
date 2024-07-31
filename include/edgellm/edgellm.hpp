#pragma once

#include <filesystem>
#include <queue>
#include <vector>

#include <edgerunner/model.hpp>

#include "edgellm/edgellm_export.hpp"
#include "edgellm/sampler.hpp"
#include "edgellm/tokenizer.hpp"

namespace edgellm {

class EDGELLM_EXPORT EdgeLLM {
  public:
    EdgeLLM(std::vector<std::filesystem::path>&& promptProcessorPaths,
            std::vector<std::filesystem::path>&& tokenGeneratorPaths,
            const std::filesystem::path& tokenizerPath);

    auto getCreationStatus() const -> bool { return m_creationSuccess; }

  private:
    std::vector<std::filesystem::path> m_promptProcessorPaths;
    std::vector<std::filesystem::path> m_tokenGeneratorPaths;
    Tokenizer m_tokenizer;

    bool m_creationSuccess {};

    std::queue<std::unique_ptr<edge::Model>> m_promptProcessor;
    std::queue<std::unique_ptr<edge::Model>> m_tokenGenerator;

    static constexpr float MTemperature = 0.9F;
    static constexpr float MTopp = 0.9F;

    static constexpr float MLogitsScale = 0.0F;
    static constexpr int32_t MLogitsOffset = 0;
    static constexpr int32_t MEvalMode = 0;
    static constexpr int32_t MMaxSequenceLength = 1024;

    size_t m_vocabSize {};
    size_t m_bosId {};
    size_t m_eosId {};
    static constexpr int32_t MNBos = 1;
    static constexpr int32_t MNEos = 1;

    Sampler m_sampler;
};

}  // namespace edgellm
