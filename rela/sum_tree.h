#pragma once

#include <string>
#include <torch/extension.h>
#include <unordered_map>
#include <vector>

#include "rela/types.h"
#include "rela/utils.h"

namespace rela {

// a tree with log_2(capacity) layers
template <class DataType>
class SumTree {
 public:
  SumTree(int capacity)
      : capacity(capacity)
      , data_(capacity)
      , evicted_(capacity, false)
      , tree_(2 * capacity - 1)
      , currentIdx_(0)
      , full_(false) {
    // requires capacity to be a power of 2
    if (!(capacity > 0 && ((capacity & (capacity - 1)) == 0))) {
      std::cout << "Capacity: " << capacity << " is not a power of 2!"
                << std::endl;
      assert(false);
    }
  }

  // batched update
  void update(const std::vector<int>& indices, const torch::Tensor& weights) {
    auto weightAccessor = weights.accessor<float, 1>();
    assert(weightAccessor.size(0) == (int)indices.size());
    std::lock_guard<std::mutex> lk(mTree_);
    for (int i = 0; i < weightAccessor.size(0); i++) {
      update_(indices[i], weightAccessor[i]);
    }
  }

  // batched append, batch's first dim is batch dim
  void append(const DataType& batch, const torch::Tensor& weights) {
    auto weightAccessor = weights.accessor<float, 1>();
    std::lock_guard<std::mutex> lk(mTree_);
    for (int i = 0; i < weightAccessor.size(0); i++) {
      append_(batch.index(i), weightAccessor[i]);
    }
  }

  // batched append, vector version
  void append(const std::vector<DataType>& batch, const torch::Tensor& weights) {
    auto weightAccessor = weights.accessor<float, 1>();
    std::lock_guard<std::mutex> lk(mTree_);
    for (size_t i = 0; i < batch.size(); i++) {
      append_(batch[i], weightAccessor[i]);
    }
  }

  // batched find
  std::tuple<DataType, torch::Tensor, std::vector<int>> find(
      const std::vector<float>& values) {
    int batchsize = values.size();
    std::vector<int> indices(batchsize);
    auto weights = torch::zeros(batchsize, torch::kFloat32);
    auto weightAccessor = weights.accessor<float, 1>();
    std::vector<DataType> samples;

    std::lock_guard<std::mutex> lk(mTree_);
    for (int i = 0; i < batchsize; i++) {
      float weight;
      int idx;
      const auto& sample = find_(values[i], &weight, &idx);
      indices[i] = idx;
      weightAccessor[i] = weight;
      samples.push_back(sample);
    }
    auto batch = DataType::makeBatch(samples);
    return std::make_tuple(batch, weights, indices);
  }

  // return <sum, size>
  std::tuple<float, int> total() const {
    std::lock_guard<std::mutex> lk(mTree_);
    return std::make_tuple(tree_[0], size_());
  }

  int size() const {
    std::lock_guard<std::mutex> lk(mTree_);
    return size_();
  }

  const int capacity;

 private:
  int getParent(int treeIdx) const {
    return (treeIdx - 1) / 2;
  }

  int getLeftChild(int treeIdx) const {
    return 2 * treeIdx + 1;
  }

  int getRightChild(int treeIdx) const {
    return 2 * treeIdx + 2;
  }

  int toTreeIdx(int dataIdx) const {
    return dataIdx + capacity - 1;
  }

  int toDataIdx(int treeIdx) const {
    return treeIdx - capacity + 1;
  }

  int size_() const {
    return full_ ? capacity : currentIdx_;
  }

  // lockfree, single element update
  bool update_(int idx, float weight) {
    if (evicted_[idx]) {
      return false;
    }

    auto treeIdx = toTreeIdx(idx);
    tree_[treeIdx] = weight;
    propagate(treeIdx);
    return true;
    /*checkTree(0); // check tree from root*/
  }

  // lockfree, single element append
  void append_(const DataType& data, float weight) {
    data_[currentIdx_] = data;
    evicted_[currentIdx_] = false;  // make it updatable
    update_(currentIdx_, weight);
    evicted_[currentIdx_] = true;

    currentIdx_ = (currentIdx_ + 1) % capacity;
    full_ = full_ || currentIdx_ == 0;
  }

  // lockfree, single element find
  const DataType& find_(float value, float* weight, int* idx) {
    int treeIdx = locate(value);
    int dataIdx = toDataIdx(treeIdx);
    assert(dataIdx >= 0 && dataIdx < size_());

    assert(weight != nullptr && idx != nullptr);
    *weight = tree_[treeIdx];
    *idx = dataIdx;

    // data point is now available for updates
    evicted_[dataIdx] = false;

    return data_[dataIdx];
  }

  // update all the sum values from this tree node until root
  void propagate(int treeIdx) {
    int parent = getParent(treeIdx);
    int left = getLeftChild(parent);
    int right = getRightChild(parent);
    tree_[parent] = tree_[left] + tree_[right];
    if (parent > 0) {
      propagate(parent);
    }
  }

  int locate(float value) const {
    return locate_(0, value);
  }

  // find value in the subtree rooted at root
  int locate_(int root, float value) const {
    int left = getLeftChild(root);
    int right = getRightChild(root);
    if (left >= 2 * capacity - 1) {
      // root is on bottom
      return root;
    } else if (value <= tree_[left]) {
      // search left
      return locate_(left, value);
    } else {
      // search right
      return locate_(right, value - tree_[left]);
    }
  }

  // check tree for invariants
  // 1) check that sums of left and right is the parent
  // 2) check that all entries are positive
  void checkTree(int treeIdx) const {
    assert(tree_[treeIdx] > 0);
    int left = getLeftChild(treeIdx);
    int right = getRightChild(treeIdx);
    if (left < 2 * capacity - 1) {
      if (!(right < 2 * capacity - 1)) {
        std::cout << "At treeIdx " + std::to_string(treeIdx) << std::endl;
        std::cout << std::to_string(right) + " < " +
                         std::to_string(2 * capacity - 1)
                  << std::endl;
        assert(false);
      }
      if (!(tree_[treeIdx] == tree_[left] + tree_[right])) {
        std::cout << "At treeIdx " + std::to_string(treeIdx) << std::endl;
        std::cout << std::to_string(tree_[treeIdx]) +
                         " == " + std::to_string(tree_[left]) + " + " +
                         std::to_string(tree_[right])
                  << std::endl;
        assert(false);
      }
      checkTree(left);
      checkTree(right);
    }
  }

  std::vector<DataType> data_;
  std::vector<bool> evicted_;

  mutable std::mutex mTree_;
  std::vector<float> tree_;

  int currentIdx_;
  bool full_;
};
}  // namespace rela
