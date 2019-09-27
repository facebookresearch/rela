#include "rela/prioritized_replay.h"

using namespace rela;

void test() {
  int capacity = 32;
  int numbatch = 10000;
  int batchsize = 40;

  torch::Tensor t = torch::arange(capacity).to(torch::kInt32);
  TensorDict t2 = {};
  FFTransition transition(t2, t2, t, t, t, t2);
  torch::Tensor priorities = torch::arange(capacity).to(torch::kFloat32);

  std::unordered_map<int, float> counts;
  for (int i = 0; i < numbatch; i++) {
    FFPrioritizedReplay buffer(capacity, 42+i, 1.0, 1.0, false);
    buffer.add(transition, priorities);

    FFTransition samples;
    torch::Tensor ws;
    std::tie(samples, ws) = buffer.sample(batchsize, "cpu");
    // buffer.updatePriority(ws);
    // std::cout << "reward " << samples.reward[0].item<int>()
    //           << ", weight: " << ws[0].item<float>() << std::endl;
    for (int j = 0; j < batchsize; j++) {
      int x = samples.reward[j].item<int>();
      auto it = counts.find(x);
      if (it == counts.end()) {
        counts.insert({x, 1});
      } else {
        it->second++;
      }
    }
    // std::cout << "==========" << std::endl;
  }

  // std::cout << priorities.sizes() << std::endl;
  float total = numbatch * batchsize;
  float maxDiff = 0;
  for (int i = 0; i < capacity; i++) {
    auto it = counts.find(i);
    float ratio = 0;
    if (it != counts.end()) {
      ratio = (float) it->second / total;
    }
    float expected = (float) priorities[i].item<float>() / priorities.sum().item<float>();
    std::cout << i << ":" << ratio << " vs. " << expected << std::endl;
    maxDiff = std::max(maxDiff, std::abs(expected - ratio));
  }
  std::cout << "max diff: " << maxDiff << std::endl;
}

int main() {
  test();
  return 0;
}
