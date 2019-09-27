#include "rela/prioritized_replay.h"
#include <future>

using namespace rela;

void testSingleThread() {
  ConcurrentQueue<int> q(200);

  std::vector<int> block = {1, 2, 3};
  auto weight = torch::ones({3}, torch::kFloat32);

  int numBlocks = 50;
  for (int i = 0; i < numBlocks; ++i) {
    // std::cout << i << std::endl;
    q.blockAppend(block, weight);
  }

  {
    float sum;
    int size = q.safeSize(&sum);
    assert(size == numBlocks * (int)block.size());
    assert(sum == numBlocks * weight.sum().item<float>());
  }
  // std::cout << size << ", " << sum << std::endl;

  {
    q.blockPop(40);
    float sum;
    int size = q.safeSize(&sum);
    assert(40 + size == numBlocks * (int)block.size());
    assert(40 + sum == numBlocks * weight.sum().item<float>());
  }

  std::cout << "pass test1" << std::endl;
}

void testMultiAppend() {
  int blockSize = 10;
  int numBlocks = 5000;
  ConcurrentQueue<int> q(numBlocks * blockSize);

  std::vector<int> block;
  for (int i = 0; i < blockSize; ++i) {
    block.push_back(i);
  }
  auto weight = torch::ones({blockSize}, torch::kFloat32);

  std::vector<std::future<void>> futures;
  for (int i = 0; i < numBlocks; ++i) {
    // std::cout << "block " << i << std::endl;
    auto f = std::async(
        std::launch::async, &ConcurrentQueue<int>::blockAppend, &q, block, weight);
    futures.push_back(std::move(f));
  }

  for (int i = 0; i < numBlocks; ++i) {
    futures[i].get();
  }

  {
    float sum;
    int size = q.safeSize(&sum);
    assert(size == numBlocks * (int)block.size());
    assert(sum == numBlocks * weight.sum().item<float>());
  }

  std::cout << "pass test2" << std::endl;
}

void testMultiAppendPop() {
  int blockSize = 10;
  int numBlocks = 5000;
  ConcurrentQueue<int> q(numBlocks * blockSize / 2);

  std::vector<int> block;
  for (int i = 0; i < blockSize; ++i) {
    block.push_back(i);
  }
  auto weight = torch::ones({blockSize}, torch::kFloat32);

  std::vector<std::future<void>> futures;
  for (int i = 0; i < numBlocks; ++i) {
    // std::cout << "block " << i << std::endl;
    auto f1 = std::async(
        std::launch::async, &ConcurrentQueue<int>::blockAppend, &q, block, weight);
    futures.push_back(std::move(f1));
  }
  int k = 0;
  while (k < numBlocks) {
    while (q.safeSize(nullptr) < blockSize) {
    }
    // std::cout << "pop: " << k << std::endl;
    // std::cout << "before, safesize: " << q.safeSize(nullptr) << std::endl;
    q.blockPop(blockSize);
    // std::cout << "safesize: " << q.safeSize(nullptr) << std::endl;
    // std::cout << "size: " << q.size() << std::endl;
    ++k;
  }

  for (int i = 0; i < numBlocks; ++i) {
    futures[i].get();
  }

  {
    float sum;
    int size = q.safeSize(&sum);
    assert(size == 0);
    assert(sum == 0);
    assert(q.size() == 0);
    // std::cout << "size: " << size << std::endl;
    // std::cout << "sum: " << sum << std::endl;
    // assert(size == numBlocks * (int)block.size());
    // assert(sum == numBlocks * weight.sum().item<float>());
  }

  std::cout << "pass test3" << std::endl;
}

int main() {
  testSingleThread();
  testMultiAppend();
  testMultiAppendPop();
}
