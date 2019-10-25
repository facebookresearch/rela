// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#pragma once

#include "rela/actor.h"
#include "rela/dqn_actor.h"

namespace rela {

class R2D2TransitionBuffer {
 public:
  R2D2TransitionBuffer(int batchsize, int multiStep, int seqLen, int burnin)
      : batchsize_(batchsize)
      , multiStep_(multiStep)
      , seqLen_(seqLen)
      , burnin_(burnin)
      , batchNextIdx_(batchsize, 0)
      , batchH0_(batchsize)
      , batchNextH0_(batchsize)
      , cache_(batchsize)
      , batchSeqTransition_(batchsize, std::vector<FFTransition>(burnin + seqLen + multiStep))
      , batchSeqPriority_(batchsize, std::vector<float>(seqLen + multiStep))
      , batchLen_(batchsize, 0)
      , canPop_(false) {
    assert(burnin_ <= seqLen_);
    assert(multiStep_ <= seqLen_);
  }

  void push(const FFTransition& transition,
            const torch::Tensor& priority,
            const TensorDict& hid) {
    assert(priority.size(0) == batchsize_);

    auto priorityAccessor = priority.accessor<float, 1>();
    for (int i = 0; i < batchsize_; ++i) {
      auto t = transition.index(i);

      if (batchNextIdx_[i] == 0) {
        // it does not matter here, should be reset after burnin
        batchH0_[i] = utils::tensorDictNarrow(hid, 1, i, 1, true);
        for (auto& kv :batchH0_[i]) {
          assert(kv.second.sum().item<float>() == 0);
        }

        while (batchNextIdx_[i] < burnin_) {
          batchSeqTransition_[i][batchNextIdx_[i]] = t.padLike();
          ++batchNextIdx_[i];
        }
      } else {
        // should not append after terminal
        // terminal should be processed when it is pushed
        int nextIdx = batchNextIdx_[i];
        if (batchSeqTransition_[i][nextIdx - 1].terminal.item<bool>()) {
          std::cout << nextIdx << std::endl;
          assert(false);
        }
        assert(batchLen_[i] == 0);
      }

      int nextIdx = batchNextIdx_[i];
      // std::cout << "next idx: " << nextIdx << std::endl;
      assert(nextIdx < burnin_ + seqLen_ + multiStep_ && nextIdx >= burnin_);

      // burnin_ + seqLen_ - burnin_ = seqLen_
      if (nextIdx == seqLen_) {
        // will become stored hidden for next trajectory
        batchNextH0_[i] = utils::tensorDictNarrow(hid, 1, i, 1, true);
      }

      batchSeqTransition_[i][nextIdx] = t;
      batchSeqPriority_[i][nextIdx - burnin_] = priorityAccessor[i];

      ++batchNextIdx_[i];
      if (!t.terminal.item<bool>()
          && batchNextIdx_[i] < burnin_ + seqLen_ + multiStep_) {
        continue;
      }

      // pad the rest of the seq in case of terminal
      batchLen_[i] = batchNextIdx_[i];
      while (batchNextIdx_[i] < burnin_ + seqLen_ + multiStep_) {
        batchSeqTransition_[i][batchNextIdx_[i]] = t.padLike();
        batchSeqPriority_[i][batchNextIdx_[i]  - burnin_] = 0;
        ++batchNextIdx_[i];
      }
      canPop_ = true;
    }
  }

  bool canPop() {
    return canPop_;
  }

  std::tuple<std::vector<RNNTransition>, torch::Tensor, torch::Tensor>
  popTransition() {
    assert(canPop_);

    std::vector<RNNTransition> batchTransition;
    std::vector<torch::Tensor> batchSeqPriority;
    std::vector<float> batchLen;

    for (int i = 0; i < batchsize_; ++i) {
      if (batchLen_[i] == 0) {
        continue;
      }
      assert(batchNextIdx_[i] == burnin_ + seqLen_ + multiStep_);
      auto& seqTransition = batchSeqTransition_[i];
      auto& seqPriority = batchSeqPriority_[i];
      float len = std::min(batchLen_[i],  burnin_ + seqLen_);

      auto t = RNNTransition(seqTransition, batchH0_[i], torch::tensor(len));
      auto p = torch::tensor(seqPriority);
      p = p.narrow(0, 0, seqLen_);

      batchTransition.push_back(t);
      batchSeqPriority.push_back(p);
      batchLen.push_back(len);

      // int checkTerminalIdx = std::min(batchLen_[i] - 1, burnin_ + seqLen_ - 1);
      const auto& terminal = seqTransition[len - 1].terminal;
      if (terminal.item<bool>()) {
        // terminal encountered, treat as fresh start
        // std::cout << "episode ends at " << batchLen_[i] << std::endl;
        batchNextIdx_[i] = 0;
      } else {
        for (int j = 0; j < burnin_; ++j) {
          // burnin_ + seqLen_ - burnin_ = seqLen_
          int k = seqLen_ + j;
          seqTransition[j] = seqTransition[k];
        }
        // this part of the data has not been trained on yet
        // need to reuse for the next trajectory
        float len = -1;
        for (int j = burnin_; j < burnin_ + multiStep_; ++j) {
          int k = seqLen_ + j;
          seqTransition[j] = seqTransition[k];
          seqPriority[j] = seqPriority[k];
          if (seqTransition[j].terminal.item<bool>() && len == -1) {
            len = j + 1;
            assert(len <= burnin_ + seqLen_);
          }
        }

        batchNextIdx_[i] = burnin_ + multiStep_;
        batchH0_[i] = batchNextH0_[i];

        if (len != -1) {
          assert(seqTransition[batchNextIdx_[i] - 1].terminal.item<bool>());
          const auto& refTransition = seqTransition[batchNextIdx_[i] - 1];
          while (batchNextIdx_[i] < burnin_ + seqLen_ + multiStep_) {
            seqTransition[batchNextIdx_[i]] = refTransition.padLike();
            seqPriority[batchNextIdx_[i]  - burnin_] = 0;
            ++batchNextIdx_[i];
          }
          auto t = RNNTransition(seqTransition, batchH0_[i], torch::tensor(len));
          auto p = torch::tensor(seqPriority);
          p = p.narrow(0, 0, seqLen_);

          batchTransition.push_back(t);
          batchSeqPriority.push_back(p);
          batchLen.push_back(len);

          batchNextIdx_[i] = 0;
        }
      }
      batchLen_[i] = 0;
    }
    canPop_ = false;
    assert(batchTransition.size() > 0);

    return std::make_tuple(batchTransition,
                           torch::stack(batchSeqPriority, 0),
                           torch::tensor(batchLen));
  }

 private:
  const int batchsize_;
  const int multiStep_;
  const int seqLen_;
  const int burnin_;

  std::vector<int> batchNextIdx_;
  std::vector<TensorDict> batchH0_;
  std::vector<TensorDict> batchNextH0_;

  std::vector<std::vector<FFTransition>> cache_;
  std::vector<std::vector<FFTransition>> batchSeqTransition_;
  std::vector<std::vector<float>> batchSeqPriority_;
  std::vector<int> batchLen_;

  bool canPop_;
};

class R2D2Actor : public Actor {
 public:
  R2D2Actor(std::shared_ptr<ModelLocker> modelLocker,
            int multiStep,
            int batchsize,
            float gamma,
            int seqLen,
            int burnin,
            std::shared_ptr<RNNPrioritizedReplay> replayBuffer)
      : batchsize_(batchsize)
      , modelLocker_(std::move(modelLocker))
      , r2d2Buffer_(batchsize, multiStep, seqLen, burnin)
      , multiStepBuffer_(multiStep, batchsize, gamma)
      , replayBuffer_(std::move(replayBuffer))
      , hidden_(getH0(batchsize))
      , numAct_(0) {
  }

  R2D2Actor(std::shared_ptr<ModelLocker> modelLocker)
      : batchsize_(1)
      , modelLocker_(std::move(modelLocker))
      , r2d2Buffer_(1, 1, 1, 0)
      , multiStepBuffer_(1, 1, 1)
      , replayBuffer_(nullptr)
      , hidden_(getH0(1))
      , numAct_(0) {
  }

  int numAct() const {
    return numAct_;
  }

  virtual TensorDict act(TensorDict& obs) override {
    torch::NoGradGuard ng;
    assert(!hidden_.empty());

    if (replayBuffer_ != nullptr) {
      historyHidden_.push_back(hidden_);
    }

    TorchJitInput input;
    auto jitObs = utils::tensorDictToTorchDict(obs, modelLocker_->device);
    auto jitHid = utils::tensorDictToTorchDict(hidden_, modelLocker_->device);
    input.push_back(jitObs);
    input.push_back(jitHid);

    int id = -1;
    auto model = modelLocker_->getModel(&id);
    auto output = model.get_method("act")(input).toTuple()->elements();
    modelLocker_->releaseModel(id);

    auto action = utils::iValueToTensorDict(output[0], torch::kCPU, true);
    hidden_ = utils::iValueToTensorDict(output[1], torch::kCPU, true);

    if (replayBuffer_ != nullptr) {
      multiStepBuffer_.pushObsAndAction(obs, action);
    }

    numAct_ += batchsize_;
    return action;
  }

  // r is float32 tensor, t is byte tensor
  virtual void setRewardAndTerminal(torch::Tensor& r,
                                    torch::Tensor& t) override {
    assert(replayBuffer_ != nullptr);
    multiStepBuffer_.pushRewardAndTerminal(r, t);

    // if ith state is terminal, reset hidden states
    // h0: [num_layers * num_directions, batch, hidden_size]
    TensorDict h0 = getH0(1);
    auto terminal = t.accessor<bool, 1>();
    for (int i = 0; i < terminal.size(0); i++) {
      if (terminal[i]) {
        for (auto& name2tensor : hidden_) {
          // batch dim is 1
          name2tensor.second.narrow(1, i, 1) = h0.at(name2tensor.first);
        }
      }
    }
  }

  // should be called after setRewardAndTerminal
  // Pops a batch of transitions and inserts it into the replay buffer
  virtual void postStep() override {
    assert(replayBuffer_ != nullptr);
    assert(multiStepBuffer_.size() == historyHidden_.size());

    if (!multiStepBuffer_.canPop()) {
      assert(!r2d2Buffer_.canPop());
      return;
    }

    {
      FFTransition transition = multiStepBuffer_.popTransition();
      TensorDict hid = historyHidden_.front();
      TensorDict nextHid = historyHidden_.back();
      historyHidden_.pop_front();

      torch::Tensor priority = computePriority(transition, hid, nextHid);
      r2d2Buffer_.push(transition, priority, hid);
    }

    if (!r2d2Buffer_.canPop()) {
      return;
    }

    std::vector<RNNTransition> batch;
    torch::Tensor batchSeqPriority;
    torch::Tensor batchLen;

    std::tie(batch, batchSeqPriority, batchLen) = r2d2Buffer_.popTransition();
    auto priority = aggregatePriority(batchSeqPriority, batchLen);
    replayBuffer_->add(batch, priority);
  }

 private:
  TensorDict getH0(int batchsize) {
    TorchJitInput input;
    input.push_back(batchsize);
    int id = -1;
    auto model = modelLocker_->getModel(&id);
    auto output = model.get_method("get_h0")(input);
    modelLocker_->releaseModel(id);
    return utils::iValueToTensorDict(output, torch::kCPU, true);
  }

  torch::Tensor computePriority(const FFTransition& transition,
                                TensorDict hid,
                                TensorDict nextHid) {
    torch::NoGradGuard ng;
    auto device = modelLocker_->device;
    auto input = transition.toJitInput(device);
    input.push_back(utils::tensorDictToTorchDict(hid, device));
    input.push_back(utils::tensorDictToTorchDict(nextHid, device));

    int id = -1;
    auto model = modelLocker_->getModel(&id);
    auto priority = model.get_method("compute_priority")(input).toTensor();
    modelLocker_->releaseModel(id);
    return priority.detach().to(torch::kCPU);
  }

  torch::Tensor aggregatePriority(torch::Tensor priority, torch::Tensor len) {
    // priority: [batchsize, seqLen]
    TorchJitInput input;
    input.push_back(priority);
    input.push_back(len);
    int id = -1;
    auto model = modelLocker_->getModel(&id);
    auto aggPriority = model.get_method("aggregate_priority")(input).toTensor();
    modelLocker_->releaseModel(id);
    return aggPriority;
  }

  const int batchsize_;
  std::shared_ptr<ModelLocker> modelLocker_;

  std::deque<TensorDict> historyHidden_;
  R2D2TransitionBuffer r2d2Buffer_;
  MultiStepTransitionBuffer multiStepBuffer_;
  std::shared_ptr<RNNPrioritizedReplay> replayBuffer_;

  TensorDict hidden_;
  std::atomic<int> numAct_;
};
}
