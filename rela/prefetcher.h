#pragma once

#include <atomic>
#include <memory>
#include <thread>
#include <vector>
#include <torch/script.h>

#include "rela/prioritized_replay.h"

namespace rela {

template <class PrioritizedReplay, class DataType>
class Prefetcher {
 public:
  Prefetcher(std::shared_ptr<PrioritizedReplay> replayer,
             int batchsize)
        : batchsize_(batchsize)
        , replayer_(std::move(replayer)) {
    //sampleThr_ = std::thread([this]() {
    //    workerThread();
    //});
    
  }

  std::tuple<DataType, torch::Tensor> sample() {
    return replayer_->sample(batchsize_);
  }

  void updatePriority(const torch::Tensor& priority) {
    replayer_->updatePriority(priority);
  }

//  void updatePriority(const torch::Tensor& priority) {
//    assert(priority.dim() == 1);
//    assert(std::abs((int)sampledIndices_.size() - (int) sample_buffer_.size()) < 3);
//
//    m_indices.lock();
//    std::vector<int> currSampledIndices = sampledIndices_.front();
//    sumTree_.update(currSampledIndices, torch::pow(priority, alpha_));
//    sampledIndices_.pop_front();
//    m_indices.unlock();
//  } 

 private:
//  void workerThread() {
//    while(true) {
//        wait_to_fill_buffer();
//        if (done_) {
//            cv_done.notify_one();
//            return;
//        }
//        sample_batch(batchsize_);
//    }
//  }

  size_t batchsize_;

  std::shared_ptr<PrioritizedReplay> replayer_;
  std::thread sampleThr_;

};

using FFPrefetcher = Prefetcher<FFPrioritizedReplay, FFTransition>;
using RNNPrefetcher = Prefetcher<RNNPrioritizedReplay, RNNTransition>;
}
