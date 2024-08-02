#pragma once

#include <filesystem>
#include <queue>
#include <vector>

#include <edgerunner/model.hpp>

#include "edgellm/edgellm_export.hpp"
#include "edgellm/ropeEmbedding.hpp"
#include "edgellm/sampler.hpp"
#include "edgellm/tokenizer.hpp"

namespace edgellm {

class EDGELLM_EXPORT EdgeLLM {
  public:
    EdgeLLM(std::vector<std::filesystem::path>&& promptProcessorPaths,
            std::vector<std::filesystem::path>&& tokenGeneratorPaths,
            const std::filesystem::path& tokenizerPath);

    auto getCreationStatus() const -> bool { return m_creationSuccess; }

    auto generate(const std::string& prompt) -> std::vector<std::string>;

  private:
    std::vector<std::filesystem::path> m_promptProcessorPaths;
    std::vector<std::filesystem::path> m_tokenGeneratorPaths;
    Tokenizer m_tokenizer;

    bool m_creationSuccess {};

    std::queue<std::unique_ptr<edge::Model>> m_promptProcessor;
    std::queue<std::unique_ptr<edge::Model>> m_tokenGenerator;

    static constexpr float MTemperature = 0.8F;
    static constexpr float MTopp = 0.9F;

    static constexpr float MLogitsScale = 0.0F;
    static constexpr int64_t MLogitsOffset = 0;
    static constexpr size_t MEvalMode = 0;
    static constexpr size_t MMaxSequenceLength = 1024;

    size_t m_vocabSize {};
    size_t m_bosId {};
    size_t m_eosId {};
    static constexpr size_t MNBos = 1;
    static constexpr size_t MNEos = 1;

    Sampler m_sampler;

    RopeEmbedding m_ropeEmbedding;

    static constexpr size_t MNumKVHeads = 32;
    static constexpr size_t MNumLayersPerSplit = 8;
    static constexpr size_t MAttentionHiddenDimension = 4096;
    static constexpr size_t MMaxHiddenLayers = 32;
};

}  // namespace edgellm
