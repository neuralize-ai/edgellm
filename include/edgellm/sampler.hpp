#include <random>
#include <vector>

namespace edgellm {

class Sampler {
  public:
    Sampler(size_t vocabSize, float temperature, float topp);

    template<typename T>
    auto sample(std::vector<T>& logits) -> size_t;

  private:
    template<typename T>
    auto sampleTopP(std::vector<T>&, float coin) -> size_t;

    template<typename T>
    auto sampleMult(std::vector<T>&, float coin) -> size_t;

    template<typename T>
    auto sampleArgmax(std::vector<T>& probabilities) -> size_t;

    size_t m_vocabSize;
    float m_temperature;
    float m_topP;

    std::random_device m_rd;
    std::mt19937 m_gen;
    std::uniform_real_distribution<float> m_dist;
};

}  // namespace edgellm
