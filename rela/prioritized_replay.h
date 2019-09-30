#pragma once

#include <random>
#include <torch/script.h>

#include "rela/sum_tree.h"

namespace rela {

template <class DataType>
class PrioritizedReplay {
 public:
  PrioritizedReplay(int capacity, int seed, float alpha, float beta)
      : alpha_(alpha)  // priority exponent
      , beta_(beta)    // importance sampling exponent
      , sumTree_(capacity)
      , numAdd_(0) {
    rng_.seed(seed);
  }

  void add(const DataType& sample, const torch::Tensor& priority) {
    assert(priority.dim() == 1);
    sumTree_.append(sample, torch::pow(priority, alpha_));
    numAdd_ += priority.size(0);
  }

  void add(const std::vector<DataType>& sample, const torch::Tensor& priority) {
    assert(priority.dim() == 1);
    sumTree_.append(sample, torch::pow(priority, alpha_));
    numAdd_ += priority.size(0);
  }

  // returns {sample, w}
  std::tuple<DataType, torch::Tensor> sample(int batchsize) {
    if (!sampledIndices_.empty()) {
      std::cout << "Error: previous samples' priority are not updated"
                << std::endl;
      assert(false);
    }

    float pTotal = 0;
    int n = 0;
    std::tie(pTotal, n) = sumTree_.total();
    std::uniform_real_distribution<float> dist(0.0, pTotal);

    // sample random points to sample from the priority buffer
    std::vector<float> rands(batchsize);
    for (int i = 0; i < batchsize; i++) {
      rands[i] = dist(rng_);
    }

    torch::Tensor weights;
    DataType samples;
    std::tie(samples, weights, sampledIndices_) = sumTree_.find(rands);

    // convert probs to weights via importance sampling and normalize by max
    // float n = (float)sumTree_.size();
    auto probs = weights / pTotal;
    {
      // sanity check
      auto absdiff = std::abs(1 - probs.sum().item<float>());
      if (absdiff > 1 + 1e-5) {
        std::cout << "probs sum: " << probs.sum().item<float>() << std::endl;
        assert(false);
      }
    }
    auto ws = torch::pow(n * probs, -beta_);
    ws /= ws.max();
    return std::make_tuple(samples, ws);
  }

  void updatePriority(const torch::Tensor& priority) {
    assert(priority.dim() == 1);
    assert((int)sampledIndices_.size() == priority.size(0));

    sumTree_.update(sampledIndices_, torch::pow(priority, alpha_));
    sampledIndices_.clear();
  }

  int size() const {
    return sumTree_.size();
  }

  bool full() const {
    return sumTree_.size() == sumTree_.capacity;
  }

  int numAdd() const {
    return numAdd_;
  }

 private:
  const float alpha_;
  const float beta_;

  SumTree<DataType> sumTree_;
  std::vector<int> sampledIndices_;
  std::mt19937 rng_;

  std::atomic<int> numAdd_;
};

using FFPrioritizedReplay = PrioritizedReplay<FFTransition>;
using RNNPrioritizedReplay = PrioritizedReplay<RNNTransition>;
}  // namespace rela