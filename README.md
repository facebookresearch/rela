# Reinforcement Learning Assembly

### Prerequisite

load the following modules
```
module purge
module load cudnn/v7.4-cuda.10.0
module load cuda/10.0

export OMP_NUM_THREADS=1
```

Get a clone of conda env with PyTorch compiled with `gcc7.4.0` from
```
/private/home/hengyuan/miniconda3/envs/pytorch1.2
```
and add the following lines to `.bashrc` after `conda active conda-env`
```
CONDA_PREFIX=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export CPATH=${CONDA_PREFIX}/include:${CPATH}
export LIBRARY_PATH=${CONDA_PREFIX}/lib:${LIBRARY_PATH}
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
```

Or create your own conda env & compile pytorch with compiler of your choice
```
# create a fresh conda environment with python3
conda create --name [your env name] python=3.7

conda activate [your env name] # Or source activate [your env name], depending on conda version.

conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing
conda install -c pytorch magma-cuda100

# clone the repo
# Note: put the repo onto /scratch partition for MUCH FASTER building speed.
git clone -b v1.2.0 --recursive https://github.com/pytorch/pytorch
cd pytorch

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
# set cuda arch list so that the built binary can be run on both pascal and volta
TORCH_CUDA_ARCH_LIST="6.0;7.0" python setup.py install
```

### Clone & Build this repo
```
git clone ...
git submodule sync && git submodule update --init --recursive
```

#### prerequisite for grpc (skip for now)
first install protobuf with

```
conda install -c anaconda protobuf
conda install grpcio
conda install grpcio-tools
sh install_grpc.sh
```
then add the following to your .bashrc/.zshrc
```
CONDA_PREFIX=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export CPATH=${CONDA_PREFIX}/include:${CPATH}
export LIBRARY_PATH=${CONDA_PREFIX}/lib:${LIBRARY_PATH}
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
```

run `install_grpc.sh` under the root folder (i.e torchRL if
not changed).  This will install the grpc lib in your current conda
env. Thanks Heinrich for this script!

#### build
Then to build this repo with atari
```
cd atari

# build atari
cd Arcade-Learning-Environment
mkdir build
cd build
cmake -DUSE_SDL=OFF -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=OFF ..
make -j40

# build rela & atari env wrapper
cd ..
mkdir build
cd build
cmake ..
make
```

### run
Note that we need to set the following before running any multi-threading
program that uses torch::Tensor. Otherwise a simple tensor operation will
use all cores by default. (ignore if you have put this line in your `bashrc`.)
```
export OMP_NUM_THREADS=1
```
`pyrela/scripts` contains some examples scripts for training/debugging
```
cd pyrela
sh scripts/dev.sh  # for fast launching & debugging
sh scripts/run_apex.sh  # launching a job onto cluster
```

### Contribute

#### Python
use ```[black](https://github.com/psf/black)``` to format python code,
run `black *.py` before pushing

#### C++
the root contains a ```.clang-format``` file that define the coding style of
this repo, run the following command before submitting PR or push
```
clang-format -i *.h
clang-format -i *.cc
```

### License
Reinforcement Learning Assembly is licensed under [MIT License](LICENSE).
