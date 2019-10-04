#pragma once

#include <random>
#include <torch/script.h>

#include "rela/sum_tree.h"

namespace rela {

template <class DataType>
class PrioritizedReplay {
 public:
  PrioritizedReplay(int capacity, int seed, float alpha, float beta,
                    int batchsize)
      : alpha_(alpha)  // priority exponent
      , beta_(beta)    // importance sampling exponent
      , sumTree_(capacity)
      , numAdd_(0)
      , batchsize_(batchsize) {
    rng_.seed(seed);
    sample_thr = std::thread([this]() {
        worker_thread();
    });
    done_ = false;
    bufferSize_ = 50;
  }

  void add(const DataType& sample, const torch::Tensor& priority) {
    assert(priority.dim() == 1);
    sumTree_.append(sample, torch::pow(priority, alpha_));
    numAdd_ += priority.size(0);
  }

  void add(const std::vector<DataType>& sample, const torch::Tensor& priority) {
    assert(priority.dim() == 1);
    sumTree_.append(sample, torch::pow(priority, alpha_));
    numAdd_ += priority.size(0);
  }

  // returns {sample, w}
  std::tuple<DataType, torch::Tensor> sample() {
    wait_for_buffer_to_fill();
    m_buffer.lock();
    std::tuple<DataType, torch::Tensor> sampled = sample_buffer_.front();
    sample_buffer_.pop_front();
    m_buffer.unlock();
    cv_fill_buffer.notify_one();
    return sampled;
  }

  void updatePriority(const torch::Tensor& priority) {
    assert(priority.dim() == 1);
    assert(std::abs((int)sampledIndices_.size() - (int) sample_buffer_.size()) < 3);

    m_indices.lock();
    std::vector<int> currSampledIndices = sampledIndices_.front();
    sumTree_.update(currSampledIndices, torch::pow(priority, alpha_));
    sampledIndices_.pop_front();
    m_indices.unlock();
  }

  int size() const {
    return sumTree_.size();
  }

  bool full() const {
    return sumTree_.size() == sumTree_.capacity;
  }

  int numAdd() const {
    return numAdd_;
  }

  void signalDone() {
    done_ = true;
    cv_fill_buffer.notify_one();
  }

  ~PrioritizedReplay() {
    wait();
    sample_thr.join();
    sampledIndices_.clear();
    sample_buffer_.clear();
  }

 private:
  void sample_batch(int batchsize) {  
    if (!((int) sampledIndices_.size() == (int) sample_buffer_.size() || 
          (int) sampledIndices_.size() -1 == (int) sample_buffer_.size())) {
      std::cout << "Error: previous samples' priority are not updated"
                << std::endl;
      assert(false);
    }

    float pTotal = 0;
    int n = 0;
    std::tie(pTotal, n) = sumTree_.total();
    std::uniform_real_distribution<float> dist(0.0, pTotal);

    // sample random points to sample from the priority buffer
    std::vector<float> rands(batchsize);
    for (int i = 0; i < batchsize; i++) {
      rands[i] = dist(rng_);
    }

    torch::Tensor weights;
    DataType samples;
    std::vector<int> currSampledIndices;
    std::tie(samples, weights, currSampledIndices) = sumTree_.find(rands);
       
    m_indices.lock();
    sampledIndices_.push_back(currSampledIndices);    
    m_indices.unlock();

    // convert probs to weights via importance sampling and normalize by max
    // float n = (float)sumTree_.size();
    auto probs = weights / pTotal;
    {
      // sanity check
      auto absdiff = std::abs(1 - probs.sum().item<float>());
      if (absdiff > 1 + 1e-5) {
        std::cout << "probs sum: " << probs.sum().item<float>() << std::endl;
        assert(false);
      }
    }
    auto ws = torch::pow(n * probs, -beta_);
    ws /= ws.max();
    
    m_buffer.lock();
    sample_buffer_.push_back(std::make_tuple(samples,ws));
    m_buffer.unlock();
    cv_buffer_empty.notify_one();
  }

  void wait_to_fill_buffer() {
    std::unique_lock<std::mutex> lg(m_buffer);
    while (sample_buffer_.size() > bufferSize_ || sumTree_.size() < batchsize_ || done_) {
        cv_fill_buffer.wait(lg);        
    }
  }

  void wait_for_buffer_to_fill() {
    if (sumTree_.size() >= batchsize_) cv_fill_buffer.notify_one();

    std::unique_lock<std::mutex> lg(m_buffer);
    while (sample_buffer_.size() == 0) {
        cv_buffer_empty.wait(lg);
    }
  }

  void worker_thread() {
    while(true) {
        wait_to_fill_buffer();
        if (done_) {
            cv_done.notify_one();
            return;
        }
        sample_batch(batchsize_);    
    }
  }

  void wait() {
    std::unique_lock<std::mutex> lg(m_buffer);
    while(!done_) {
        cv_done.wait(lg);
    }
  }

  const float alpha_;
  const float beta_;

  SumTree<DataType> sumTree_;
  std::deque<std::vector<int>> sampledIndices_;
  std::deque<std::tuple<DataType, torch::Tensor>> sample_buffer_;
  std::mt19937 rng_;

  std::atomic<int> numAdd_;

  const int batchsize_;
  size_t bufferSize_;

  bool done_;

  std::thread sample_thr;

  std::mutex m_indices;
  std::mutex m_buffer;

  std::condition_variable cv_buffer_empty;
  std::condition_variable cv_fill_buffer;
  std::condition_variable cv_done;
};

using FFPrioritizedReplay = PrioritizedReplay<FFTransition>;
using RNNPrioritizedReplay = PrioritizedReplay<RNNTransition>;
}  // namespace rela
