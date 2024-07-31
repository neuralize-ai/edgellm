#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>
namespace edgellm {

class Tokenizer {
  public:
    explicit Tokenizer() = default;
    Tokenizer(const Tokenizer&) = default;
    Tokenizer(Tokenizer&&) = delete;
    auto operator=(const Tokenizer&) -> Tokenizer& = default;
    auto operator=(Tokenizer&&) -> Tokenizer& = delete;
    ~Tokenizer() = default;

    auto load(const std::filesystem::path& tokenizerPath) -> bool;

    auto encode(const std::string& input,
                size_t numBos,
                size_t numEos) const -> std::vector<size_t>;

    auto decodeVerify(size_t token) const -> bool {
        return token < m_vocabSize;
    }

    auto decode(size_t prevToken, size_t token) const -> std::string;

    auto getVocabSize() const -> size_t { return m_vocabSize; }

    auto getBosTok() const -> size_t { return m_bosTok; }

    auto getEosTok() const -> size_t { return m_eosTok; }

  private:
    auto strLookup(const std::string& str) const -> int32_t;

    size_t m_vocabSize = 0;
    size_t m_bosTok = 0;
    size_t m_eosTok = 0;

    std::vector<std::string> m_vocab;
    std::vector<float> m_vocabScores;
    std::vector<size_t> m_sortedVocabIndices;
    size_t m_maxTokenLength = 0;
};

}  // namespace edgellm
