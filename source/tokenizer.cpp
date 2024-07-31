#include <algorithm>
#include <array>
#include <climits>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>

#include "edgellm/tokenizer.hpp"

#include <fmt/core.h>

namespace edgellm {

auto Tokenizer::load(const std::filesystem::path& tokenizerPath) -> bool {
    std::ifstream file(tokenizerPath, std::ios::binary);
    if (!file) {
        return false;
    }
    std::array<int32_t, 4> metadata {};
    for (auto& field : metadata) {
        if (!file.read(reinterpret_cast<char*> /* NOLINT */ (&field),
                       sizeof(int32_t)))
        {
            return false;
        }
    }

    m_vocabSize = static_cast<size_t>(metadata[0]);
    m_bosTok = static_cast<size_t>(metadata[1]);
    m_eosTok = static_cast<size_t>(metadata[2]);
    m_maxTokenLength = static_cast<size_t>(metadata[3]);

    // allocate space for the vocabulary
    m_vocab.resize(m_vocabSize);
    m_vocabScores.resize(m_vocabSize);
    m_sortedVocabIndices.resize(m_vocabSize);

    // read in the vocabulary
    for (size_t i = 0; i < m_vocabSize; ++i) {
        if (!file.read(reinterpret_cast<char*> /* NOLINT */ (&m_vocabScores[i]),
                       sizeof(float)))
        {
            // This is allowed, we just pad the rest of the vocab with <pad>
            // strings
            m_vocab[i] = "<pad>";
            continue;
        }
        int32_t len = 0;
        if (!file.read(reinterpret_cast<char*> /* NOLINT */ (&len),
                       sizeof(int32_t)))
        {
            return false;
        }
        m_vocab[i].resize(static_cast<size_t>(len));
        if (!file.read(m_vocab[i].data(), len)) {
            return false;
        }
    }

    std::iota(m_sortedVocabIndices.begin(), m_sortedVocabIndices.end(), 0);

    std::sort(m_sortedVocabIndices.begin(),
              m_sortedVocabIndices.end(),
              [this](auto index1, auto index2) {
                  return m_vocab[index1] < m_vocab[index2];
              });

    return true;
}

auto Tokenizer::decode(size_t prevToken, /* NOLINT */
                       size_t token) const -> std::string {
    if (!Tokenizer::decodeVerify(token)) {
        return {};
    }

    auto piece = m_vocab.begin()
        + static_cast<std::vector<std::string>::difference_type>(token);

    // following BOS token, sentencepiece decoder strips any leading
    // whitespace
    if (prevToken == m_bosTok && (*piece)[0] == ' ') {
        piece++;
    }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byteVal {};
    std::string result;
    if (sscanf /* NOLINT */ (piece->data(), "<0x%02hhX>", &byteVal) == 1) {
        result.push_back(static_cast<char>(byteVal));
    } else {
        result = *piece;
    }

    return result;
}

auto Tokenizer::strLookup(const std::string& str) const -> int32_t {
    auto index = std::lower_bound(m_sortedVocabIndices.begin(),
                                  m_sortedVocabIndices.end(),
                                  str,
                                  [this](auto index, const auto& value) {
                                      return m_vocab[index] < value;
                                  });

    return (index != m_sortedVocabIndices.end() && m_vocab[*index] == str)
        ? static_cast<int32_t>(*index)
        : -1;
}

auto Tokenizer::encode(const std::string& input,
                       size_t numBos, /* NOLINT */
                       size_t numEos) const -> std::vector<size_t> {
    // encode the string text (input) into an upper-bound preallocated tokens[]
    // array bos != 0 means prepend the BOS token (=1), eos != 0 means
    // append the EOS token (=2)
    if (input.empty()) {
        return {};
    }

    std::vector<size_t> tokens;

    // add optional BOS tokens, if desired
    tokens.resize(static_cast<size_t>(numBos), m_bosTok);

    // prepend a dummy prefix token to the input string, but only if input is
    // not empty
    // TODO: pretty sure this isn't correct in the general case (correct source:
    // sentencepiece)
    const std::string space = " ";
    if (input.empty()) {
        auto dummyPrefix = strLookup(space);
        tokens.push_back(static_cast<size_t>(dummyPrefix));
    }

    // Wikipedia: Code point ↔ UTF-8 conversion
    //
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    //
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // temporary buffer that will store merge candidates of always two
    // consecutive tokens
    std::string strBuffer;

    // process the raw (UTF-8) byte sequence of the input string
    for (auto byte = input.begin(); byte < input.end(); ++byte) {
        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the
        // rest 0x80 is 10000000 in UTF-8, all continuation bytes start with
        // "10" in first two bits so in English this is: "if this byte is
        // not a
        // continuation byte"
        constexpr uint8_t TwoLeadingBitsMask =
            std::numeric_limits<uint8_t>::max() << (CHAR_BIT - 2) /* NOLINT */;
        constexpr uint8_t LeadingBitMask = std::numeric_limits<uint8_t>::max()
            << (CHAR_BIT - 1) /* NOLINT */;
        if ((*byte & TwoLeadingBitsMask) /* NOLINT */ != LeadingBitMask) {
            // this byte must be either a leading byte (11...) or an ASCII char
            // (0x...)
            // => reset our location, as we're starting a new UTF-8
            // codepoint
            strBuffer.clear();
        }

        // append the current byte to the buffer
        strBuffer.push_back(*byte);

        // while the next character is a continuation byte, continue appending
        // up to 4 bytes
        if ((*(byte + 1) & TwoLeadingBitsMask) /* NOLINT */ == LeadingBitMask
            && strBuffer.size() < 4)
        {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        auto index = strLookup(strBuffer);
        if (index != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens.push_back(static_cast<size_t>(index));
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>,
            // </s> so the individual bytes only start at index 3
            for (auto strByte : strBuffer) {
                tokens.push_back(static_cast<unsigned char>(strByte) + 3);
            }
        }
        strBuffer.clear();
    }

    // merge the best consecutive pair each iteration, according the scores in
    // vocab_scores
    while (true) {
        float bestScore = std::numeric_limits<float>::lowest();
        int bestId = -1;
        size_t bestIdIndex {};

        for (size_t i = 0; i < tokens.size() - 1; ++i) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            strBuffer = m_vocab[tokens[i]] + m_vocab[tokens[i + 1]];
            auto index = strLookup(strBuffer);
            if (index != -1
                && m_vocabScores[static_cast<size_t>(index)] > bestScore)
            {
                // this merge pair exists in vocab! record its score and
                // position
                bestScore = m_vocabScores[static_cast<size_t>(index)];
                bestId = index;
                bestIdIndex = i;
            }
        }

        if (bestId == -1) {
            break;  // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (bestIdIndex, bestIdIndex+1) into new
        // token best_id
        tokens[static_cast<size_t>(bestIdIndex)] = static_cast<size_t>(bestId);
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (size_t i = bestIdIndex + 1; i < tokens.size() - 1; ++i) {
            tokens[i] = tokens[i + 1];
        }
        tokens.pop_back();  // token length decreased
    }

    // add optional EOS (=2) token, if desired

    tokens.insert(tokens.end(), numEos, m_eosTok);

    return tokens;
}

}  // namespace edgellm
