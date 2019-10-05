#pragma once

#include <atomic>
#include <memory>
#include <thread>
#include <torch/script.h>
#include <vector>

#include "rela/prioritized_replay.h"

namespace rela {

template <class PrioritizedReplay, class DataType>
class Prefetcher {
 public:
  Prefetcher(std::shared_ptr<PrioritizedReplay> replayer, int batchsize)
      : batchsize_(batchsize)
      , replayer_(std::move(replayer)) {
    sampleThr_ = std::thread([this]() { workerThread(); });
    done_ = false;
    bufferSize_ = 50;
  }

  std::tuple<DataType, torch::Tensor> sample() {
    wait_for_buffer_to_fill();

    mBuffer_.lock();
    std::tuple<DataType, torch::Tensor> currSample = sampleBuffer_.front();
    sampleBuffer_.pop_front();
    mBuffer_.unlock();
    return currSample;
  }

  void updatePriority(const torch::Tensor& priority) {
    assert(priority.dim() == 1);
    assert(std::abs((int)sampledIndices_.size() - (int)sampleBuffer_.size()) <
           3);

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
      wait_to_fill_buffer();
      if (done_) {
        cvDone_.notify_one();
        return;
      }
      sample_batch();
    }
  }

  void wait_to_fill_buffer() {
    std::unique_lock<std::mutex> lg(mBuffer_);
    while ((size_t)sampleBuffer_.size() > bufferSize_ ||
           (size_t)replayer_->size() < 2 * batchsize_) {
      if (done_)
        break;
      cvFillBuffer_.wait(lg);
    }
  }

  void wait_for_buffer_to_fill() {
    if ((size_t)replayer_->size() >= 2 * batchsize_)
      cvFillBuffer_.notify_one();
    std::unique_lock<std::mutex> lg(mBuffer_);
    while (sampleBuffer_.size() == 0) {
      cvBufferEmpty_.wait(lg);
    }
  }

  void sample_batch() {
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

  std::shared_ptr<PrioritizedReplay> replayer_;
  std::thread sampleThr_;
};

using FFPrefetcher = Prefetcher<FFPrioritizedReplay, FFTransition>;
using RNNPrefetcher = Prefetcher<RNNPrioritizedReplay, RNNTransition>;
}
