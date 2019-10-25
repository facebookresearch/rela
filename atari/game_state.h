// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#pragma once

#include <torch/extension.h>

namespace atari {

class GameState {
 public:
  GameState(int height,
            int width,
            bool terminalOnLifeLoss,
            int frameStack,
            int sHeight,
            int sWidth)
      : height(height)
      , width(width)
      , frameStack(frameStack)
      , sHeight(sHeight)
      , sWidth(sWidth)
      , observation_(3 * height * width, 0)
      , prevObservation_(3 * height * width, 0)
      , terminal_(true)
      , terminalOnLifeLoss_(terminalOnLifeLoss) {
  }

  float addReward(float reward) {
    lastReward_ = reward;
    accReward_ += reward;
    // std::cout << "game state, reward: " << lastReward_ << std::endl;
    return accReward_;
  }

  float getAccReward() const {
    return accReward_;
  }

  float lastReward() const {
    return lastReward_;
  }

  // used for filling buffer with new observation
  std::vector<unsigned char>& getObservationBuffer() {
    return observation_;
  }

  // used for filling buffer with new observation
  std::vector<unsigned char>& getPrevObservationBuffer() {
    return prevObservation_;
  }

  torch::Tensor computeFeature() {
    torch::Tensor s = getObservation();
    s = s.view({1, 3, height, width});
    s = torch::upsample_bilinear2d(s, {sHeight, sWidth}, true);
    s = s.view({3, sHeight, sWidth});
    // 0.21 * r + 0.72 * g + 0.07 * b
    s = 0.21 * s[0] + 0.72 * s[1] + 0.07 * s[2];
    s = (s * 255.).to(torch::kUInt8);
    assert(s.dim() == 2);
    assert(s.size(0) == sHeight);
    assert(s.size(1) == sWidth);

    if (stackedS_.size() == 0) {
      for (int i = 0; i < frameStack; ++i) {
        stackedS_.push_back(s);
      }
    } else {
      assert((int)stackedS_.size() == frameStack);
      stackedS_.pop_front();
      stackedS_.push_back(s);
    }
    assert((int)stackedS_.size() == frameStack);

    torch::Tensor obs =
        torch::zeros({frameStack, sHeight, sWidth}, torch::kUInt8);
    for (int i = 0; i < frameStack; ++i) {
      obs[i].copy_(stackedS_[i]);
    }
    return obs;
  }

  void setLives(int lives) {
    prevLives_ = lives_;
    lives_ = lives;
  }

  void setTerminal() {
    terminal_ = true;
  }

  // returns true if we lost a life on this action
  bool lostLife() const {
    return lives_ < prevLives_;
  }

  bool terminal() const {
    return terminal_ || (terminalOnLifeLoss_ && lostLife());
  }

  void reset() {
    // std::cout << "game state reset::prev game acc reward: " << accReward_ <<
    // std::endl;
    stackedS_.clear();
    std::fill(observation_.begin(), observation_.end(), 0);
    std::fill(prevObservation_.begin(), prevObservation_.end(), 0);
    accReward_ = 0;
    lastReward_ = 0;
    terminal_ = false;
    prevLives_ = -1;
    lives_ = -1;
  }

  const int height;
  const int width;
  const int frameStack;
  const int sHeight;
  const int sWidth;

 private:
  torch::Tensor getObservation() const {
    auto obs = torch::zeros({3, height, width}, torch::kFloat32);
    auto accessor = obs.accessor<float, 3>();
    for (int i = 0; i < (int)observation_.size(); ++i) {
      int color = i % 3;
      int c = (i / 3) % width;
      int r = (i / 3) / width;
      float v = (float)std::max(observation_[i], prevObservation_[i]);
      accessor[color][r][c] = v / 255.0;
    }
    return obs;
  }

  std::vector<unsigned char> observation_;
  std::vector<unsigned char> prevObservation_;
  float accReward_;
  float lastReward_;
  int prevLives_;
  int lives_;
  bool terminal_;
  bool terminalOnLifeLoss_;

  std::deque<torch::Tensor> stackedS_;
};

}  // namespace atari
