#pragma once

#include <atomic>
#include <memory>
#include <thread>
#include <torch/script.h>
#include <vector>
#include <chrono>

#include "rela/prioritized_replay.h"


namespace rela {

template <class DataType>
class Prefetcher {
 public:
  Prefetcher(std::shared_ptr<PrioritizedReplay<DataType>> replayer, int batchsize, int bufferSize)
      : batchsize_(batchsize)
      , bufferSize_(bufferSize)
      , replayer_(std::move(replayer)) {
    done_ = false;
  }

  void start() {
    sampleThr_ = std::thread([this]() { workerThread(); });
  }

  std::tuple<DataType, torch::Tensor> sample() {
    waitForBufferToFill();
    std::lock_guard<std::mutex> lk (mData_);

    std::tuple<DataType, torch::Tensor> currSample = sampleBuffer_.front();
    sampleBuffer_.pop_front();
    cvData_.notify_one();
    return currSample;
  }

  void updatePriority(const torch::Tensor& priority) {
    assert(priority.dim() == 1);
    {
    std::lock_guard<std::mutex> lk (mData_);
    assert((int)sampledIndices_.size() - (int)sampleBuffer_.size() < 3);
    std::vector<int> currIndices = sampledIndices_.front();
    sampledIndices_.pop_front();
    replayer_->updatePrefetcherPriority(priority, currIndices);
    }
  }

  ~Prefetcher() {
    signalDone();
    wait();
    sampleThr_.join();
    sampledIndices_.clear();
    sampleBuffer_.clear();
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
    cvData_.wait(lk, [this]{
        if(done_) return false;
        return (size_t) sampleBuffer_.size() < bufferSize_;
    });
  }

  void waitForBufferToFill() {
    std::unique_lock<std::mutex> lk(mData_);
    cvData_.wait(lk, [this]{
        return sampleBuffer_.size() != 0;
    });
  }

  void sampleBatch() {
    std::tuple<DataType, torch::Tensor> batch = replayer_->sample(batchsize_);
    std::vector<int> indices = replayer_->getSampledIndices();

    {
    std::lock_guard<std::mutex> lk(mData_);
    sampledIndices_.push_back(indices);
    sampleBuffer_.push_back(batch);
    }

    replayer_->clearSampledIndices();
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

  std::deque<std::vector<int>> sampledIndices_;
  std::deque<std::tuple<DataType, torch::Tensor>> sampleBuffer_;

  size_t batchsize_;
  size_t bufferSize_;

  bool done_;

  std::mutex mData_;

  std::condition_variable cvDone_;
  std::condition_variable cvData_;

  std::shared_ptr<PrioritizedReplay<DataType>> replayer_;
  std::thread sampleThr_;
};

using FFPrefetcher = Prefetcher<FFTransition>;
using RNNPrefetcher = Prefetcher<RNNTransition>;
}
