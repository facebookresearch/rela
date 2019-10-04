#pragma once

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "rela/prioritized_replay.h"

namespace rela {

template <class PrioritizedReplay>
class Prefetcher {
 public:
  Prefetcher(std::shared_ptr<PrioritizedReplay> replayer){
    replayer_ = replayer;
  }
 private:
  std::shared_ptr<PrioritizedReplay> replayer_;

};

using FFPrefetcher = Prefetcher<FFPrioritizedReplay>;
using RNNPrefetcher = Prefetcher<RNNPrioritizedReplay>;
}
