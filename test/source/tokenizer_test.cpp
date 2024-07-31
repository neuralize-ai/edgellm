#include <filesystem>
#include <vector>

#include "edgellm/tokenizer.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Tokenizer load", "[tokenizer][load]") {
    std::filesystem::path tokenizerPath =
        "models/llama_v2_7b_chat_quantized/tokenizer.bin";

    edgellm::Tokenizer tokenizer;
    const auto loadSuccess = tokenizer.load(tokenizerPath);
    REQUIRE(loadSuccess);

    const std::string input = "once upon a time <0x01>";

    const auto tokens = tokenizer.encode(input, 1, 1);

    std::string result;
    const auto eosToken = tokenizer.getEosTok();
    for (auto token = tokens.cbegin() + 1; token != tokens.cend(); ++token) {
        if (*token == eosToken) {
            break;
        }

        const auto word = tokenizer.decode(*(token - 1), *token);

        result += word;
    }

    REQUIRE(result == input);
}
