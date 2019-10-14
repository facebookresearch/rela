#pragma once

#include <atomic>
#include <memory>
#include <thread>
#include <torch/script.h>
#include <vector>

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
    std::lock_guard<std::mutex> lk (mBuffer);

    std::tuple<DataType, torch::Tensor> currSample = sampleBuffer_.front();
    sampleBuffer_.pop_front();
    return currSample;
  }

  void updatePriority(const torch::Tensor& priority) {
    assert(priority.dim() == 1);
    assert((int)sampledIndices_.size() - (int)sampleBuffer_.size() < 3);

    mIndices_.lock();
    std::vector<int> currIndices = sampledIndices_.front();
    sampledIndices_.pop_front();
    mIndices_.unlock();

    replayer_->updatePrefetcherPriority(priority, currIndices);
    cvFillBuffer_.notify_one();
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
    std::unique_lock<std::mutex> lk(mBuffer_);
    while ((size_t)sampleBuffer_.size() > bufferSize_ ||
           (size_t)replayer_->size() < 2 * batchsize_) {
      if (done_)
        break;
      cvFillBuffer_.wait(lg);
    }
  }

  void waitForBufferToFill() {
    if ((size_t)replayer_->size() >= 2 * batchsize_) {
      cvFillBuffer_.notify_one();
    }
    std::unique_lock<std::mutex> lg(mBuffer_);
    while (sampleBuffer_.size() == 0) {
      cvBufferEmpty_.wait(lg);
    }
  }

  void sampleBatch() {
    std::tuple<DataType, torch::Tensor> batch = replayer_->sample(batchsize_);
    std::vector<int> indices = replayer_->getSampledIndices();

    mIndices_.lock();
    sampledIndices_.push_back(indices);
    mIndices_.unlock();

    mBuffer_.lock();
    sampleBuffer_.push_back(batch);
    mBuffer_.unlock();

    replayer_->clearSampledIndices();
    cvBufferEmpty_.notify_one();
  }

  void signalDone() {
    done_ = true;
    cvFillBuffer_.notify_one();
  }

  void wait() {
    std::unique_lock<std::mutex> lg(mBuffer_);
    while (!done_) {
      cvDone_.wait(lg);
    }
  }

  std::deque<std::vector<int>> sampledIndices_;
  std::deque<std::tuple<DataType, torch::Tensor>> sampleBuffer_;

  size_t batchsize_;
  size_t bufferSize_;

  bool done_;

  std::mutex mIndices_;
  std::mutex mBuffer_;

  std::condition_variable cvBufferEmpty_;
  std::condition_variable cvFillBuffer_;
  std::condition_variable cvDone_;

  std::shared_ptr<PrioritizedReplay<DataType>> replayer_;
  std::thread sampleThr_;
};

using FFPrefetcher = Prefetcher<FFTransition>;
using RNNPrefetcher = Prefetcher<RNNTransition>;
}
