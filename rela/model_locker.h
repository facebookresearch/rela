#pragma once

#include <pybind11/pybind11.h>

#include "rela/types.h"

namespace rela {

// class ModelLocker {
//  public:
//   ModelLocker(TorchJitModel model, const std::string& device)
//       : device(torch::Device(device))
//       , refModel_(model)
//       , updated_(true) {
//   }

//   void updateModel(TorchJitModel model) {
//     // note this model has to be different every time this function is called
//     // i.e. model_locker.updateModel(model.clone()) in Python
//     std::lock_guard<std::mutex> lk(m_);
//     refModel_ = model;
//     updated_ = true;
//   }

//   const TorchJitModel getModel() {
//     if (updated_) {
//       // to avoid grabbing lock very freq
//       std::lock_guard<std::mutex> lk(m_);
//       if (updated_) {
//         // models_.push_back(model_);
//         model_ = refModel_;
//         model_.to(device);
//         updated_ = false;
//       }
//     }
//     return model_;
//   }

//   const torch::Device device;

//  private:
//   std::mutex m_;
//   TorchJitModel refModel_;  // model ref used for update
//   TorchJitModel model_;     // read-only model
//   bool updated_;
// };

class ModelLocker {
 public:
  ModelLocker(std::vector<py::object> pyModels, const std::string& device)
      : device(torch::Device(device))
      , pyModels_(pyModels)
      , modelCallCounts_(pyModels.size(), 0)
      , latestModel_(0) {
    // assert(pyModels_.size() > 1);
    for (size_t i = 0; i < pyModels_.size(); ++i) {
      models_.push_back(pyModels_[i].attr("_c").cast<TorchJitModel*>());
      // modelCallCounts_.push_back(0);
    }
  }

  void updateModel(py::object pyModel) {
    std::unique_lock<std::mutex> lk(m_);
    int id = (latestModel_ + 1) % modelCallCounts_.size();
    cv_.wait(lk, [this, id] { return modelCallCounts_[id] == 0; });
    lk.unlock();

    pyModels_[id].attr("load_state_dict")(pyModel.attr("state_dict")());

    lk.lock();
    latestModel_ = id;
    lk.unlock();
  }

  const TorchJitModel getModel(int* id) {
    std::lock_guard<std::mutex> lk(m_);
    *id = latestModel_;
    // std::cout << "using mdoel: " << latestModel_ << std::endl;
    ++modelCallCounts_[latestModel_];
    return *models_[latestModel_];
  }

  void releaseModel(int id) {
    std::unique_lock<std::mutex> lk(m_);
    --modelCallCounts_[id];
    if (modelCallCounts_[id] == 0) {
      cv_.notify_one();
    }
  }

  const torch::Device device;

 private:
  // py::function model_cons_;
  std::vector<py::object> pyModels_;
  std::vector<int> modelCallCounts_;
  int latestModel_;

  std::vector<TorchJitModel*> models_;
  std::mutex m_;
  std::condition_variable cv_;
};

}  // namespace rela
