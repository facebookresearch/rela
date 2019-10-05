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

    m_buffer.lock();
    std::tuple<DataType, torch::Tensor> currSample = sampleBuffer_.front();
    sampleBuffer_.pop_front();
    m_buffer.unlock();
    return currSample;
  }

  void updatePriority(const torch::Tensor& priority) {
    assert(priority.dim() == 1);
    assert(std::abs((int)sampledIndices_.size() - (int)sampleBuffer_.size()) <
           3);

    m_indices.lock();
    std::vector<int> currIndices = sampledIndices_.front();
    sampledIndices_.pop_front();
    m_indices.unlock();
    replayer_->updatePrefetcherPriority(priority, currIndices);

    cv_fill_buffer.notify_one();
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
        cv_done.notify_one();
        return;
      }
      sample_batch();
    }
  }

  void wait_to_fill_buffer() {
    std::unique_lock<std::mutex> lg(m_buffer);
    while ((size_t)sampleBuffer_.size() > bufferSize_ ||
           (size_t)replayer_->size() < 2 * batchsize_) {
      if (done_)
        break;
      cv_fill_buffer.wait(lg);
    }
  }

  void wait_for_buffer_to_fill() {
    if ((size_t)replayer_->size() >= 2 * batchsize_)
      cv_fill_buffer.notify_one();
    std::unique_lock<std::mutex> lg(m_buffer);
    while (sampleBuffer_.size() == 0) {
      cv_buffer_empty.wait(lg);
    }
  }

  void sample_batch() {
    std::tuple<DataType, torch::Tensor> batch = replayer_->sample(batchsize_);
    std::vector<int> indices = replayer_->getSampledIndices();
    m_indices.lock();
    sampledIndices_.push_back(indices);
    m_indices.unlock();

    m_buffer.lock();
    sampleBuffer_.push_back(batch);
    m_buffer.unlock();

    replayer_->clearSampledIndices();
    cv_buffer_empty.notify_one();
  }

  void signalDone() {
    done_ = true;
    cv_fill_buffer.notify_one();
  }

  void wait() {
    std::unique_lock<std::mutex> lg(m_buffer);
    while (!done_) {
      cv_done.wait(lg);
    }
  }

  std::deque<std::vector<int>> sampledIndices_;
  std::deque<std::tuple<DataType, torch::Tensor>> sampleBuffer_;

  size_t batchsize_;
  size_t bufferSize_;

  bool done_;

  std::mutex m_indices;
  std::mutex m_buffer;

  std::condition_variable cv_buffer_empty;
  std::condition_variable cv_fill_buffer;
  std::condition_variable cv_done;

  std::shared_ptr<PrioritizedReplay> replayer_;
  std::thread sampleThr_;
};

using FFPrefetcher = Prefetcher<FFPrioritizedReplay, FFTransition>;
using RNNPrefetcher = Prefetcher<RNNPrioritizedReplay, RNNTransition>;
}
