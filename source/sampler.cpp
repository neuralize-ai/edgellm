#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "edgellm/sampler.hpp"

namespace edgellm {

template<typename T>
struct ProbIndex {
    T prob;
    size_t index;
};  // struct used when sorting probabilities during top-p sampling

Sampler::Sampler(size_t vocabSize, /* NOLINT */ float temperature, float topp)
    : m_vocabSize(vocabSize)
    , m_temperature(temperature)
    , m_topP(topp) {
    m_gen = std::mt19937(m_rd());
    m_dist = std::uniform_real_distribution<float>(0.0, 1.0);
}

template<typename T>
auto Sampler::sampleArgmax(std::vector<T>& probabilities) -> size_t {
    // return the index that has the highest probability
    size_t maxI = 0;
    T maxP = probabilities[0];
    for (size_t i = 1; i < m_vocabSize; ++i) {
        if (probabilities[i] > maxP) {
            maxI = i;
            maxP = probabilities[i];
        }
    }
    return maxI;
}

template<typename T>
auto Sampler::sampleMult(std::vector<T>& probabilities, float coin) -> size_t {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1)
    T cdf {};
    for (size_t i = 0; i < m_vocabSize; ++i) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return m_vocabSize - 1;  // in case of rounding errors
}

template<typename T>
auto Sampler::sampleTopP(std::vector<T>& probabilities, float coin) -> size_t {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topP. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1)

    const float cutoff = (1.0F - m_topP) / static_cast<float>(m_vocabSize - 1);
    std::vector<size_t> indices(probabilities.size());
    std::iota(indices.begin(), indices.end(), T {});
    auto cutoffIt = std::partition(
        indices.begin(), indices.end(), [cutoff, &probabilities](auto index) {
            return probabilities[index] > cutoff;
        });

    const auto numHighProb =
        static_cast<size_t>(std::distance(indices.begin(), cutoffIt));

    std::sort(
        indices.begin(), cutoffIt, [&probabilities](auto index1, auto index2) {
            return probabilities[index1] > probabilities[index2];
        });

    // truncate the list where cumulative probability exceeds topp
    T cumulativeProb = 0;
    auto lastIndex =
        numHighProb - 1;  // in case of rounding errors consider all elements
    for (size_t i = 0; i < numHighProb; ++i) {
        cumulativeProb += probabilities[indices[i]];
        if (cumulativeProb > m_topP) {
            lastIndex = i;
            break;  // we've exceeded topp by including lastIndex
        }
    }

    // sample from the truncated list
    const T freq = coin * cumulativeProb;
    T cdf = 0;
    for (size_t i = 0; i <= lastIndex; ++i) {
        cdf += probabilities[indices[i]];
        if (freq < cdf) {
            return indices[i];
        }
    }
    return indices[lastIndex];  // in case of rounding errors
}

template<typename T>
static void softmax(std::vector<T>& values) {
    const auto maxVal = *std::max_element(values.cbegin(), values.cend());

    const auto sum = std::transform_reduce(
        values.begin(),
        values.end(),
        static_cast<float>(0),
        std::plus<>(),
        [maxVal](auto& value) { return value = expf(value - maxVal); });

    std::transform(
        values.cbegin(), values.cend(), values.begin(), [sum](auto value) {
            return value / sum;
        });
}

template<typename T>
auto Sampler::sample(std::vector<T>& logits) -> size_t {
    // sample the token given the logits and some hyperparameters
    if (m_temperature == 0.0F) {
        // greedy argmax sampling: take the token with the highest probability
        return sampleArgmax(logits);
    }
    size_t next = 0;
    // apply the temperature to the logits
    for (size_t i = 0; i < m_vocabSize; ++i) {
        logits[i] /= m_temperature;
    }
    // apply softmax to the logits to get the probabilities for next token
    softmax(logits);
    // flip a (float) coin (this is our source of entropy for sampling)
    const float coin = m_dist(m_gen);
    // we sample from this distribution to get the next token
    if (m_topP <= 0 || m_topP >= 1) {
        // simply sample from the predicted probability distribution
        next = sampleMult(logits, coin);
    } else {
        // top-p (nucleus) sampling, clamping the least likely tokens to
        // zero
        next = sampleTopP(logits, coin);
    }
    return next;
}

template size_t Sampler::sample<float>(std::vector<float>& logits);

}  // namespace edgellm
