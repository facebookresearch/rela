// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#pragma once

#include <random>

#include "rela/env.h"
#include "atari/game_state.h"

// this has to be included at the bottom to avoid namespace corruption
#include <ale_interface.hpp>
namespace {
// work around bug in ALE. see Arcade-Learning-Environment/issues/86
std::mutex ALE_GLOBAL_LOCK;
}

namespace atari {

torch::Tensor getLegalActionMask(ALEInterface& ale, bool useMinAction) {
  auto legalAction = ale.getLegalActionSet();
  if (!useMinAction) {
    auto mask = torch::ones({(int)legalAction.size()}, torch::kFloat32);
    return mask;
  }

  auto mask = torch::zeros({(int)legalAction.size()}, torch::kFloat32);
  auto minimalAction = ale.getMinimalActionSet();
  auto maskAccessor = mask.accessor<float, 1>();
  for (auto action : minimalAction) {
    int index = (int)action;
    assert(index >= 0 && index < maskAccessor.size(0));
    maskAccessor[index] = 1;
  }
  return mask;
}

class AtariEnv : public rela::Env {
 public:
  AtariEnv(std::string romFile,
           float exploreEps,
           int seed,
           int frameStack,
           int frameSkip,
           int noOpStart,
           int sHeight,
           int sWidth,
           int maxNumFrame,
           bool terminalOnLifeLoss,
           bool terminalSignalOnLifeLoss)
      : romFile_(romFile)
      , exploreEps_(torch::tensor(exploreEps))
      , frameSkip_(frameSkip)
      , maxNumFrame_(maxNumFrame)
      , terminalSignalOnLifeLoss_(terminalSignalOnLifeLoss)
      , numSteps_(0)
      , rng_(seed)
      , noOpStartSampler_(0, noOpStart) {
    ale::Logger::setMode(ale::Logger::mode::Error);
    std::lock_guard<std::mutex> lg(ALE_GLOBAL_LOCK);
    ale_ = std::make_unique<ALEInterface>();
    ale_->setInt("random_seed", seed);
    ale_->setFloat("repeat_action_probability", 0.0);
    ale_->setBool("showinfo", false);
    ale_->loadROM(romFile_);

    int height = ale_->getScreen().height();
    int width = ale_->getScreen().width();

    legalAction_ = ale_->getLegalActionSet();
    legalActionMask_ = getLegalActionMask(*ale_, true);
    actionSampler_ = std::make_unique<std::uniform_int_distribution<>>(
        0, legalAction_.size() - 1);

    state_ = std::make_unique<GameState>(
        height, width, terminalOnLifeLoss, frameStack, sHeight, sWidth);
  }

  int numAction() const {
    return (int)legalAction_.size();
  }

  // reset the game and return the first observation {'obs'}
  rela::TensorDict reset() final {
    // reset all attributes
    ale_->reset_game();
    state_->reset();
    numSteps_ = 0;

    // press start key if needed
    pressStartKey();

    int noOp = noOpStartSampler_(rng_);
    for (int i = 0; i < noOp; ++i) {
      int a = (*actionSampler_)(rng_);
      while (legalActionMask_[a].item<float>() != (float)1) {
        a = (*actionSampler_)(rng_);
      }
      aleStep(a);
    }

    // get first observation
    ale_->getScreenRGB(state_->getObservationBuffer());
    torch::Tensor obs = state_->computeFeature();
    rela::TensorDict input = {
      {"s", obs},
      {"eps", exploreEps_},
      {"legal_move", legalActionMask_}
    };
    return input;
  }

  // return {'obs', 'reward', 'terminal'}
  std::tuple<rela::TensorDict, float, bool> step(
      const rela::TensorDict& action) final {
    // take an ale step
    torch::Tensor a = action.at("a");
    float reward = aleStep(a.item<int>());

    // update state
    state_->addReward(reward);
    state_->setLives(ale_->lives());
    if (ale_->game_over() || numSteps_ * frameSkip_ > maxNumFrame_) {
      state_->setTerminal();
    }

    // compute reward and terminal signal
    float clippedReward = clipRewards(reward);
    bool terminalSignal =
        state_->terminal() || (terminalSignalOnLifeLoss_ && state_->lostLife());
    if (state_->terminal()) {
      // state should not matter, but we still need to send it
      torch::Tensor obs = state_->computeFeature();
      rela::TensorDict input = {
        {"s", obs},
        {"eps", exploreEps_},
        {"legal_move", legalActionMask_}
      };
      return std::make_tuple(input, clippedReward, true);
    }

    // if lost life (but game is not over) need to press start key again
    if (state_->lostLife()) {
      pressStartKey();
    }

    // compute obs
    ale_->getScreenRGB(state_->getObservationBuffer());
    torch::Tensor obs = state_->computeFeature();
    rela::TensorDict input = {
      {"s", obs},
      {"eps", exploreEps_},
      {"legal_move", legalActionMask_}
    };
    return std::make_tuple(input, clippedReward, terminalSignal);
  }

  bool terminated() const final {
    return state_->terminal();
  }

  float getEpisodeReward() {
    return state_->getAccReward();
  }

 private:
  float clipRewards(float reward) {
    if (reward > 0) {
      return 1;
    } else if (reward < 0) {
      return -1;
    } else {
      return 0;
    }
  }

  // perform action (frameSkip number of times) and record the rewards
  float aleStep(int actIdx) {
    assert(!ale_->game_over());
    float reward = 0;
    for (int i = 0; i < frameSkip_; i++) {
      // need previous observation buffer for max
      if (i == frameSkip_ - 1) {
        ale_->getScreenRGB(state_->getPrevObservationBuffer());
      }

      assert(actIdx >= 0 && actIdx < (int)legalAction_.size());
      assert(legalActionMask_[actIdx].item<float>() == (float)1);
      reward += ale_->act(legalAction_[actIdx]);

    }
    numSteps_++;
    return reward;
  }

  void pressStartKey() {
    // breakout requires pressing FIRE key after each life loss
    if (romFile_.find("breakout") != std::string::npos) {
      aleStep(1);  // fire
    }
  }


  const std::string romFile_;
  const torch::Tensor exploreEps_;
  const int frameSkip_;
  const int maxNumFrame_;
  const bool terminalSignalOnLifeLoss_;

  std::unique_ptr<ALEInterface> ale_;
  std::unique_ptr<GameState> state_;
  std::vector<::Action> legalAction_;
  torch::Tensor legalActionMask_;
  int numSteps_;

  std::mt19937 rng_;
  std::unique_ptr<std::uniform_int_distribution<>> actionSampler_;
  std::uniform_int_distribution<int> noOpStartSampler_;
};
}
