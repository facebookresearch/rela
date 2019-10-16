#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <numeric>
#include <thread>
#include <torch/script.h>
#include <vector>

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
      , updateCounter_(0)
      , done_(false)
      , threadsDone_(0)
      , replayBuffer_(std::move(replayer))
      , sampleIndicator_(bufferSize)
      , sampledIndices_(bufferSize)
      , sampleBuffer_(bufferSize) {
  }

  void start() {
    sampleThr_ = std::thread([this]() { sampleWorkerThread(); });
    updateThr_ = std::thread([this]() { updateWorkerThread(); });
  }

  std::tuple<DataType, torch::Tensor> sample() {
    waitForBufferToFill();
    std::lock_guard<std::mutex> lk(mData_);

    auto batch = sampleBuffer_[sampleIdx_];

    {
      std::lock_guard<std::mutex> lk_u(mUpdate_);

      updateIndices_.push_back(sampledIndices_[sampleIdx_]);
      sampleIndicator_[sampleIdx_] = 0;
      cvData_.notify_all();
    }

    return batch;
  }

  void updatePriority(const torch::Tensor& priority) {
    assert(priority.dim() == 1);
    {
      std::lock_guard<std::mutex> lk_u(mUpdate_);
      std::lock_guard<std::mutex> lk_d(mData_);

      updateBuffer_.push_back(
          std::make_tuple(priority, updateIndices_.front()));
      updateIndices_.pop_front();

      updateCounter_++;
      sampleIdx_++;
      if (sampleIdx_ == bufferSize_) {
        sampleIdx_ = sampleIdx_ % bufferSize_;
      }

      cvUpdate_.notify_one();
    }
  }

  ~Prefetcher() {
    signalDone();
    wait();
    sampleThr_.join();
    updateThr_.join();
    sampledIndices_.clear();
    sampleBuffer_.clear();
    sampleIndicator_.clear();
    updateBuffer_.clear();
    updateIndices_.clear();
  }

 private:
  void sampleWorkerThread() {
    while (true) {
      waitToFillBuffer();
      if (done_) {
        threadsDone_++;
        cvDone_.notify_one();
        return;
      }
      sampleBatch();
    }
  }

  void updateWorkerThread() {
    while (true) {
      waitToUpdate();
      if (done_) {
        threadsDone_++;
        cvDone_.notify_one();
        return;
      }
      updateReplayBuffer();
    }
  }

  void waitToUpdate() {
    std::unique_lock<std::mutex> lk(mUpdate_);
    cvUpdate_.wait(lk, [this] {
      if (done_)
        return true;
      return updateCounter_ != 0;
    });
  }

  void waitToFillBuffer() {
    std::unique_lock<std::mutex> lk(mData_);
    cvData_.wait(lk, [this] {
      if (done_)
        return true;
      return sampleIndicator_[insertIdx_] == 0;
    });
  }

  void waitForBufferToFill() {
    std::unique_lock<std::mutex> lk(mData_);
    cvData_.wait(lk, [this] { return sampleIndicator_[sampleIdx_] == 1; });
  }

  void updateReplayBuffer() {
    torch::Tensor weights;
    std::vector<int> indices;

    {
      std::lock_guard<std::mutex> lk(mUpdate_);
      std::tie(weights, indices) = updateBuffer_.front();
      updateBuffer_.pop_front();
    }

    replayBuffer_->updatePriority(weights, indices);
    updateCounter_--;
  }

  void sampleBatch() {
    std::tuple<DataType, torch::Tensor> batch =
        replayBuffer_->sample(batchsize_);
    std::vector<int> indices = replayBuffer_->getSampledIndices();

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

    replayBuffer_->clearSampledIndices();
    cvData_.notify_one();
  }

  void signalDone() {
    done_ = true;
    cvData_.notify_all();
    cvUpdate_.notify_all();
  }

  void wait() {
    std::unique_lock<std::mutex> lk(mData_);
    cvDone_.wait(lk, [this] { return threadsDone_ == 2; });
  }

  size_t batchsize_;
  size_t bufferSize_;

  size_t insertIdx_;
  size_t sampleIdx_;

  size_t updateCounter_;
  bool done_;
  size_t threadsDone_;

  std::mutex mData_;
  std::mutex mUpdate_;

  std::condition_variable cvDone_;
  std::condition_variable cvData_;
  std::condition_variable cvUpdate_;

  std::shared_ptr<PrioritizedReplay<DataType>> replayBuffer_;

  std::thread sampleThr_;
  std::thread updateThr_;

  std::vector<int> sampleIndicator_;
  std::vector<std::vector<int>> sampledIndices_;

  std::deque<std::vector<int>> updateIndices_;
  std::deque<std::tuple<torch::Tensor, std::vector<int>>> updateBuffer_;

  std::vector<std::tuple<DataType, torch::Tensor>> sampleBuffer_;
};

using FFPrefetcher = Prefetcher<FFTransition>;
using RNNPrefetcher = Prefetcher<RNNTransition>;
}
