#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

class RopeEmbedding {
    /*
    Compute Rotary Position Embedding
    Ref: https://arxiv.org/pdf/2104.09864

    Compute RopeEmbedding outside model to simplify model quantization
    */

  public:
    explicit RopeEmbedding(size_t headDim = MHeadDim,
                           size_t maxLength = MMaxLength)
        : m_maxLength(maxLength) {
        /*
        head_dim: dimension size of head
        max_length: max sequence length to expect
        */

        precomputeFreqsCis(headDim, maxLength * 2);
    }

    void precomputeFreqsCis(size_t dim, size_t end, float theta = MTheta) {
        /*
        Precompute embedding matrix
        */
        const auto planeSize = dim / 2;
        std::vector<float> freqs(planeSize);
        for (size_t i = 0; i < planeSize; ++i) {
            freqs[i] = 1.0F
                / std::pow(theta,
                           static_cast<float>(i * 2) / static_cast<float>(dim));
        }

        std::vector<float> indices(end);
        std::iota(indices.begin(), indices.end(), 0.0);

        std::vector<float> freqsCis(end * planeSize);
        for (size_t i = 0; i < end; ++i) {
            for (size_t j = 0; j < planeSize; ++j) {
                freqsCis[i * planeSize + j] = indices[i] * freqs[j];
            }
        }

        m_cos.resize(m_maxLength * planeSize);
        m_sin.resize(m_maxLength * planeSize);
        for (size_t i = 0; i < m_maxLength; ++i) {
            for (size_t j = 0; j < planeSize; ++j) {
                m_cos[i * planeSize + j] =
                    std::cos(freqsCis[i * planeSize + j]);
                m_sin[i * planeSize + j] =
                    std::sin(freqsCis[i * planeSize + j]);
            }
        }
    }

    auto getEmbedding(const std::vector<size_t>& positionIds)
        -> std::pair<std::vector<float>, std::vector<float>> {
        /*
        position_ids: [batch_size, sequence_length]
        return [batch_size, 1, sequence_length, head_dim//2][2]
        */
        const auto cosWidth = m_cos.size() / m_maxLength;
        const auto sinWidth = m_sin.size() / m_maxLength;

        std::vector<float> cosEmbedding(positionIds.size() * cosWidth);
        std::vector<float> sinEmbedding(positionIds.size() * sinWidth);

        for (size_t i = 0; i < positionIds.size(); ++i) {
            const auto pos = positionIds[i];
            std::copy(m_cos.begin() + pos * cosWidth,
                      m_cos.begin() + (pos + 1) * cosWidth,
                      cosEmbedding.begin() + i * cosWidth);
            std::copy(m_sin.begin() + pos * sinWidth,
                      m_sin.begin() + (pos + 1) * sinWidth,
                      sinEmbedding.begin() + i * sinWidth);
        }

        return {cosEmbedding, sinEmbedding};
    }

  private:
    size_t m_maxLength;
    std::vector<float> m_cos, m_sin;

    static constexpr float MTheta = 10000.0;
    static constexpr size_t MHeadDim = 128;
    static constexpr size_t MMaxLength = 1024;
};
