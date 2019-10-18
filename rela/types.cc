#include "rela/types.h"
#include "rela/utils.h"

using namespace rela;

FFTransition FFTransition::makeBatch(std::vector<FFTransition> transitions) {
  TensorVecDict obsVec;
  TensorVecDict actionVec;
  std::vector<torch::Tensor> rewardVec;
  std::vector<torch::Tensor> terminalVec;
  std::vector<torch::Tensor> bootstrapVec;
  TensorVecDict nextObsVec;
  std::vector<torch::Tensor> gameIdxVec;

  for (size_t i = 0; i < transitions.size(); i++) {
    utils::tensorVecDictAppend(obsVec, transitions[i].obs);
    utils::tensorVecDictAppend(actionVec, transitions[i].action);
    rewardVec.push_back(transitions[i].reward);
    terminalVec.push_back(transitions[i].terminal);
    bootstrapVec.push_back(transitions[i].bootstrap);
    utils::tensorVecDictAppend(nextObsVec, transitions[i].nextObs);
    gameIdxVec.push_back(transitions[i].gameIdx);
  }

  FFTransition batch;
  batch.obs = utils::tensorDictJoin(obsVec, 0);
  batch.action = utils::tensorDictJoin(actionVec, 0);
  batch.reward = torch::stack(rewardVec, 0);
  batch.terminal = torch::stack(terminalVec, 0);
  batch.bootstrap = torch::stack(bootstrapVec, 0);
  batch.nextObs = utils::tensorDictJoin(nextObsVec, 0);
  batch.gameIdx = torch::stack(gameIdxVec, 0);
  return batch;
}

FFTransition FFTransition::index(int i) const {
  FFTransition element;

  for (auto& name2tensor : obs) {
    element.obs.insert({name2tensor.first, name2tensor.second[i]});
  }
  for (auto& name2tensor : action) {
    element.action.insert({name2tensor.first, name2tensor.second[i]});
  }

  element.reward = reward[i];
  element.terminal = terminal[i];
  element.bootstrap = bootstrap[i];

  for (auto& name2tensor : nextObs) {
    element.nextObs.insert({name2tensor.first, name2tensor.second[i]});
  }
  element.gameIdx = gameIdx[i];

  return element;
}

FFTransition FFTransition::padLike() const {
  FFTransition pad;

  pad.obs = utils::tensorDictZerosLike(obs);
  pad.action = utils::tensorDictZerosLike(action);
  pad.reward = torch::zeros_like(reward);
  pad.terminal = torch::ones_like(terminal);
  pad.bootstrap = torch::zeros_like(bootstrap);
  pad.nextObs = utils::tensorDictZerosLike(nextObs);

  return pad;
}

TorchJitInput FFTransition::toJitInput(const torch::Device& device) const {
  TorchJitInput input;
  input.push_back(utils::tensorDictToTorchDict(obs, device));
  input.push_back(utils::tensorDictToTorchDict(action, device));
  input.push_back(reward.to(device));
  input.push_back(terminal.to(device));
  input.push_back(bootstrap.to(device));
  input.push_back(utils::tensorDictToTorchDict(nextObs, device));
  input.push_back(gameIdx.to(device));
  return input;
}

RNNTransition::RNNTransition(const std::vector<FFTransition>& transitions,
                             TensorDict h0,
                             torch::Tensor seqLen)
    : h0(h0)
    , seqLen(seqLen) {
  TensorVecDict obsVec;
  TensorVecDict actionVec;
  std::vector<torch::Tensor> rewardVec;
  std::vector<torch::Tensor> terminalVec;
  std::vector<torch::Tensor> bootstrapVec;

  for (size_t i = 0; i < transitions.size(); i++) {
    utils::tensorVecDictAppend(obsVec, transitions[i].obs);
    utils::tensorVecDictAppend(actionVec, transitions[i].action);
    rewardVec.push_back(transitions[i].reward);
    terminalVec.push_back(transitions[i].terminal);
    bootstrapVec.push_back(transitions[i].bootstrap);
  }

  obs = utils::tensorDictJoin(obsVec, 0);
  action = utils::tensorDictJoin(actionVec, 0);
  reward = torch::stack(rewardVec, 0);
  terminal = torch::stack(terminalVec, 0);
  bootstrap = torch::stack(bootstrapVec, 0);
}

RNNTransition RNNTransition::index(int i) const {
  RNNTransition element;

  for (auto& name2tensor : obs) {
    element.obs.insert({name2tensor.first, name2tensor.second[i]});
  }
  for (auto& name2tensor : h0) {
    auto t = name2tensor.second.narrow(1, i, 1).squeeze(1);
    element.h0.insert({name2tensor.first, t});
  }
  for (auto& name2tensor : action) {
    element.action.insert({name2tensor.first, name2tensor.second[i]});
  }

  element.reward = reward[i];
  element.terminal = terminal[i];
  element.bootstrap = bootstrap[i];
  element.seqLen = seqLen[i];
  return element;
}

RNNTransition RNNTransition::makeBatch(std::vector<RNNTransition> transitions) {
  TensorVecDict obsVec;
  TensorVecDict h0Vec;
  TensorVecDict actionVec;
  std::vector<torch::Tensor> rewardVec;
  std::vector<torch::Tensor> terminalVec;
  std::vector<torch::Tensor> bootstrapVec;
  std::vector<torch::Tensor> seqLenVec;

  for (size_t i = 0; i < transitions.size(); i++) {
    utils::tensorVecDictAppend(obsVec, transitions[i].obs);
    utils::tensorVecDictAppend(h0Vec, transitions[i].h0);
    utils::tensorVecDictAppend(actionVec, transitions[i].action);
    rewardVec.push_back(transitions[i].reward);
    terminalVec.push_back(transitions[i].terminal);
    bootstrapVec.push_back(transitions[i].bootstrap);
    seqLenVec.push_back(transitions[i].seqLen);
  }

  RNNTransition batch;
  batch.obs = utils::tensorDictJoin(obsVec, 1);
  batch.h0 = utils::tensorDictJoin(h0Vec, 1);  // 1 is batch for rnn hid
  batch.action = utils::tensorDictJoin(actionVec, 1);
  batch.reward = torch::stack(rewardVec, 1);
  batch.terminal = torch::stack(terminalVec, 1);
  batch.bootstrap = torch::stack(bootstrapVec, 1);
  batch.seqLen = torch::stack(seqLenVec, 0).squeeze(1);
  return batch;
}
