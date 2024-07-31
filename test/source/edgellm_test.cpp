#include <filesystem>
#include <utility>
#include <vector>

#include "edgellm/edgellm.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Models load", "[models][load]") {
    constexpr size_t NumSplits = 4;

    std::vector<std::filesystem::path> promptProcessorPaths;
    promptProcessorPaths.reserve(NumSplits);
    for (size_t i = 0; i < NumSplits; ++i) {
        promptProcessorPaths.emplace_back(
            "models/llama_v2_7b_chat_quantized/"
            "llama_v2_7b_chat_quantized_PromptProcessor_"
            + std::to_string(i + 1) + "_Quantized.bin"

        );
    }

    std::vector<std::filesystem::path> tokenGeneratorPaths;
    tokenGeneratorPaths.reserve(NumSplits);
    for (size_t i = 0; i < NumSplits; ++i) {
        tokenGeneratorPaths.emplace_back(
            "models/llama_v2_7b_chat_quantized/"
            "llama_v2_7b_chat_quantized_TokenGenerator_"
            + std::to_string(i + 1) + "_Quantized.bin"

        );
    }

    std::filesystem::path tokenizerPath =
        "models/llama_v2_7b_chat_quantized/tokenizer.bin";

    edgellm::EdgeLLM llm(std::move(promptProcessorPaths),
                         std::move(tokenGeneratorPaths),
                         tokenizerPath);

    REQUIRE(llm.getCreationStatus() == true);
}
