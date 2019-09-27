#pragma once

#include "rela/types.h"

namespace rela {

class ModelLocker {
 public:
  ModelLocker(TorchJitModel model, const std::string& device)
      : device(torch::Device(device))
      , refModel_(model)
      , updated_(true) {
  }

  void updateModel(TorchJitModel model) {
    // note this model has to be different every time this function is called
    // i.e. model_locker.updateModel(model.clone()) in Python
    std::lock_guard<std::mutex> lk(m_);
    refModel_ = model;
    updated_ = true;
  }

  const TorchJitModel getModel() {
    if (updated_) {
      // to avoid grabbing lock very freq
      std::lock_guard<std::mutex> lk(m_);
      if (updated_) {
        // models_.push_back(model_);
        model_ = refModel_;
        model_.to(device);
        updated_ = false;
      }
    }
    return model_;
  }

  const torch::Device device;

 private:
  std::mutex m_;
  TorchJitModel refModel_;  // model ref used for update
  TorchJitModel model_;     // read-only model
  bool updated_;
};

}  // namespace rela
