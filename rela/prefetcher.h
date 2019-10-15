#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <torch/script.h>
#include <vector>
#include <numeric>

#include "rela/prioritized_replay.h"

namespace rela {

template <class DataType>
class Prefetcher {
 public:
  Prefetcher(std::shared_ptr<PrioritizedReplay<DataType>> replayer,
             int batchsize,
             int bufferSize)
      : batchsize_(batchsize)
      , bufferSize_(bufferSize)
      , insertIdx_(0)
      , sampleIdx_(0)
      , done_(false)
      , replayBuffer_(std::move(replayer))
      , sampleIndicator_(bufferSize)
      , sampledIndices_(bufferSize)
      , sampleBuffer_(bufferSize) {
  }

  void start() {
    sampleThr_ = std::thread([this]() { workerThread(); });
  }

  std::tuple<DataType, torch::Tensor> sample() {
    waitForBufferToFill();
    std::lock_guard<std::mutex> lk(mData_);
    return sampleBuffer_[sampleIdx_];
  }

  void updatePriority(const torch::Tensor& priority) {
    assert(priority.dim() == 1);
//    std::vector<int> indices;
    {
      std::lock_guard<std::mutex> lk(mData_);
      assert(sampleIndicator_[sampleIdx_] == 1);
//      indices = sampledIndices_[sampleIdx_];
      replayBuffer_->updatePriority(priority, sampledIndices_[sampleIdx_]);

      // updating the sample indicator
      sampleIndicator_[sampleIdx_] = 0;
      sampleIdx_++;
      if (sampleIdx_ == bufferSize_) {
        sampleIdx_ = sampleIdx_ % bufferSize_;
      }
//      replayBuffer_->updatePriority(priority, sampledIndices_[sampleIdx_]);
    }

    cvData_.notify_one();
  }

  ~Prefetcher() {
    signalDone();
    wait();
    sampleThr_.join();
    sampledIndices_.clear();
    sampleBuffer_.clear();
    sampleIndicator_.clear();
  }

 private:
  void workerThread() {
    while (true) {
      waitToFillBuffer();
      if (done_) {
        cvDone_.notify_one();
        return;
      }
      sampleBatch();
    }
  }

  void waitToFillBuffer() {
    std::unique_lock<std::mutex> lk(mData_);
    cvData_.wait(lk, [this] {
      if (done_)
        return true;
//      return (size_t) std::accumulate(sampleIndicator_.begin(), sampleIndicator_.end(), 0) < bufferSize_/2;
      return sampleIndicator_[insertIdx_] == 0;
    });
  }

  void waitForBufferToFill() {
    std::unique_lock<std::mutex> lk(mData_);
    cvData_.wait(lk, [this] { return sampleIndicator_[sampleIdx_] == 1; });
  }

  void sampleBatch() {
    if (updateSecondTree_) {
        
    }
    std::tuple<DataType, torch::Tensor> batch =
        replayBufferSampler_->sample(batchsize_);
    std::vector<int> indices = replayBufferSampler_->getSampledIndices();

    {
      std::lock_guard<std::mutex> lk(mData_);
      sampledIndices_[insertIdx_] = indices;
      sampleBuffer_[insertIdx_] = batch;

      sampleIndicator_[insertIdx_] = 1;
      insertIdx_++;
      if (insertIdx_ == bufferSize_) {
        insertIdx_ = insertIdx_ % bufferSize_;
      }
    }

    replayBufferSampler_->clearSampledIndices();
    cvData_.notify_one();
  }

  void signalDone() {
    done_ = true;
    cvData_.notify_all();
  }

  void wait() {
    std::unique_lock<std::mutex> lg(mData_);
    while (!done_) {
      cvDone_.wait(lg);
    }
  }

  size_t batchsize_;
  size_t bufferSize_;

  size_t insertIdx_;
  size_t sampleIdx_;
  
  bool updateSecondTree_;
  bool done_;

  std::mutex mData_;

  std::condition_variable cvDone_;
  std::condition_variable cvData_;

  std::shared_ptr<PrioritizedReplay<DataType>> replayBuffer_;
  std::shared_ptr<PrioritizedReplay<DataType>> replayBufferSampler_;
  std::thread sampleThr_;

  std::vector<int> sampleIndicator_;
  std::vector<std::vector<int>> sampledIndices_;
  std::vector<std::tuple<DataType, torch::Tensor>> sampleBuffer_;
};

using FFPrefetcher = Prefetcher<FFTransition>;
using RNNPrefetcher = Prefetcher<RNNTransition>;
}
