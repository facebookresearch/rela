#pragma once

#include <torch/extension.h>
#include <unordered_map>

namespace rela {

using TensorDict = std::unordered_map<std::string, torch::Tensor>;
using TensorVecDict =
    std::unordered_map<std::string, std::vector<torch::Tensor>>;

using TorchTensorDict = torch::Dict<std::string, torch::Tensor>;
using TorchJitInput = std::vector<torch::jit::IValue>;
using TorchJitOutput = torch::jit::IValue;
using TorchJitModel = torch::jit::script::Module;

class FFTransition {
 public:
  FFTransition() = default;

  FFTransition(TensorDict& obs,
               TensorDict& action,
               torch::Tensor& reward,
               torch::Tensor& terminal,
               torch::Tensor& bootstrap,
               TensorDict& nextObs,
               torch::Tensor& gameIdx)
      : obs(obs)
      , action(action)
      , reward(reward)
      , terminal(terminal)
      , bootstrap(bootstrap)
      , nextObs(nextObs)
      , gameIdx(gameIdx) {
  }

  static FFTransition makeBatch(std::vector<FFTransition> transitions);

  FFTransition index(int i) const;

  FFTransition padLike() const;

  TorchJitInput toJitInput(const torch::Device& device) const;

  TensorDict obs;
  TensorDict action;
  torch::Tensor reward;
  torch::Tensor terminal;
  torch::Tensor bootstrap;
  TensorDict nextObs;
  torch::Tensor gameIdx;
};

class RNNTransition {
 public:
  RNNTransition() = default;

  RNNTransition(const std::vector<FFTransition>& transitions,
                TensorDict h0,
                torch::Tensor seqLen);

  RNNTransition index(int i) const;

  static RNNTransition makeBatch(std::vector<RNNTransition> transitions);

  TensorDict obs;
  TensorDict h0;
  TensorDict action;
  torch::Tensor reward;
  torch::Tensor terminal;
  torch::Tensor bootstrap;
  torch::Tensor seqLen;
};
}  // namespace rela
