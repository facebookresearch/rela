// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#include <pybind11/pybind11.h>

#include "atari/atari_env.h"

namespace py = pybind11;

PYBIND11_MODULE(atari, m) {
  using namespace atari;

  py::class_<AtariEnv, rela::Env, std::shared_ptr<AtariEnv>>(m, "AtariEnv")
      .def(py::init<std::string,  // romFile
                    float,        // exploreEps
                    int,          // seed
                    int,          // frameStack
                    int,          // frameSkip
                    int,          // noOpStart
                    int,          // sHeight
                    int,          // sWidth
                    int,          // maxNumFrame
                    bool,         // terminalOnLifeLoss
                    bool          // terminalSignalOnLifeLoss
                    >())
      .def("num_action", &AtariEnv::numAction)
      .def("reset", &AtariEnv::reset)
      .def("step", &AtariEnv::step)
      .def("terminated", &AtariEnv::terminated)
      .def("get_episode_reward", &AtariEnv::getEpisodeReward);
}
