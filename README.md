# ESIM: an Open Event Camera Simulator

This repository contains the source code for the improved [ESIM](https://rpg.ifi.uzh.ch/esim.html) event camera simulator used in **Robust *e*-NeRF** <sub>[![Project Page](https://img.shields.io/badge/Project_Page-black
)](https://wengflow.github.io/robust-e-nerf) [![arXiv](https://img.shields.io/badge/arXiv-black)](https://arxiv.org/abs/2309.08596) [![Code](https://img.shields.io/badge/Code-black)](https://github.com/wengflow/robust-e-nerf) [![Dataset](https://img.shields.io/badge/Dataset-black
)](https://huggingface.co/datasets/wengflow/robust-e-nerf)</sub>. In particular, we incorporate the following changes:

1. Event simulation model
   - Improve the overall event simulation accuracy by accounting for additional edge cases 
   - Improve the refractory period model by explicitly resetting the pixel reference intensity and timestamp at the end of the refractory period
   - Desynchronize initial event generation across pixels by randomly initializing pixel reference timestamps
   - Model junction leakage, which increases the rate of ON events and decreases the rate of OFF events
   - Modify the pixel-to-pixel contrast threshold variation model to be time-independent
   - Merge `feature/color` branch to support color event cameras
2. Rendering engine
   - Support [Blender](https://www.blender.org) as a rendering engine
   - Support [Unreal Engine](https://www.unrealengine.com) 4.27.2 with a modified [UnrealCV](https://unrealcv.org/) plugin
3. Camera trajectory
   - Circumvent singularities in interpolating quaternion orientations by supporting rotation vector/angle-axis orientation representation in the trajectory CSV for interpolation
4. Miscellaneous
   - Fix various bugs & installation errors

If you use this improved version of ESIM for your work, please cite:

```bibtex
@inproceedings{low2023_robust-e-nerf,
  title = {Robust e-NeRF: NeRF from Sparse & Noisy Events under Non-Uniform Motion},
  author = {Low, Weng Fei and Lee, Gim Hee},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year = {2023}
}
```

```bibtex
@inproceedings{rebecq18_esim,
  author = {Henri Rebecq and Daniel Gehrig and Davide Scaramuzza},
  title = {{ESIM}: an Open Event Camera Simulator},
  journal = {Conf. on Robotics Learning (CoRL)},
  year = 2018
}
```

## Installation

The following installation steps were tested on Ubuntu 20.04 and 22.04 with GTX 1080 Ti, RTX 3090 and RTX A5000 GPUs.

### ESIM

We recommend creating a new Catkin workspace specifically for ESIM. We name the workspace as `esim_ws` and place it under the home directory, as follows:
```bash
mkdir -p ~/esim_ws/src
```

Clone this repository into the source space of the workspace with:
```bash
cd ~/esim_ws/src
git clone https://github.com/wengflow/rpg_esim.git
```

We also recommend using [Conda](https://docs.conda.io/en/latest/) to set up an environment with the appropriate dependencies for running ESIM, as follows:
1. Install [Mamba](https://mamba.readthedocs.io/en/latest/index.html), an improved re-implementation of Conda, according to the [official instructions](https://mamba.readthedocs.io/en/latest/mamba-installation.html#mamba-install)
2. Create the `esim` environment with:
   ```bash
   mamba env create -f rpg_esim/environment.yml
   ```
3. Activate the environment and initialize `rosdep` with:
   ```bash
   conda activate esim
   rosdep init
   rosdep update
   ```
   
Initialize and configure the workspace with:
```bash
catkin init
catkin config --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
vcs-import < rpg_esim/dependencies.yaml

cd ze_oss
touch imp_3rdparty_cuda_toolkit/CATKIN_IGNORE \
      imp_app_pangolin_example/CATKIN_IGNORE \
      imp_benchmark_aligned_allocator/CATKIN_IGNORE \
      imp_bridge_pangolin/CATKIN_IGNORE \
      imp_cu_core/CATKIN_IGNORE \
      imp_cu_correspondence/CATKIN_IGNORE \
      imp_cu_imgproc/CATKIN_IGNORE \
      imp_ros_rof_denoising/CATKIN_IGNORE \
      imp_tools_cmd/CATKIN_IGNORE \
      ze_data_provider/CATKIN_IGNORE \
      ze_geometry/CATKIN_IGNORE \
      ze_imu/CATKIN_IGNORE \
      ze_trajectory_analysis/CATKIN_IGNORE
```
We set `-DCMAKE_BUILD_TYPE=RelWithDebInfo`, instead of `-DCMAKE_BUILD_TYPE=Release`, to preserve the ease of debugging while still enabling code optimization during compilation.

Build the `esim_ros` package with:
```bash
catkin build esim_ros
```

Lastly, add the following alias to your `.bashrc` file for activating the `esim` Conda environment and sourcing the [ROS](https://www.ros.org/) environment setup files:
```bash
alias cae='conda activate esim; source ~/esim_ws/devel/setup.bash'
```

The installation steps described above were consolidated from the following sources, with some modifications:
1. https://github.com/uzh-rpg/rpg_esim/wiki/Installation
2. https://github.com/uzh-rpg/rpg_esim/wiki/Installation-(ROS-Melodic)
3. https://robostack.github.io/GettingStarted.html

### Blender

To use Blender as the rendering engine, we require [Blender as a Python module](https://docs.blender.org/api/current/info_advanced_blender_as_bpy.html) being installed in a separate `blender` Conda environment with a compatible Python version and [PyZMQ](https://pyzmq.readthedocs.io/en/latest/) also installed.

We provide a Python wheel for Blender 3.4.0 at [this link](https://github.com/wengflow/rpg_esim/releases/download/v1.0/bpy-3.4.0a0-cp310-cp310-manylinux_2_31_x86_64.whl), which is compiled with [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 11.4 and [NVIDIA OptiX](https://developer.nvidia.com/rtx/ray-tracing/optix) 7.3.0 support (requires NVIDIA R465.84 driver or newer), for Linux and Python 3.10. The `blender` environment may be created with all the necessary dependencies, including this wheel, with the following:
   ```bash
   conda env create -f rpg_esim/blender_environment.yml
   ```

Alternatively, build Blender as a Python module from source, according to the [official instructions](https://wiki.blender.org/w/index.php?title=Building_Blender/Other/BlenderAsPyModule), and install it via `pip` in the `blender` environment.

### UnrealEngine and UnrealCV

To use UnrealEngine as the rendering engine, first install UnrealEngine 4.27 ([Windows/Mac](https://docs.unrealengine.com/4.27/en-US/Basics/InstallingUnrealEngine/), [Linux](https://docs.unrealengine.com/4.27/en-US/SharingAndReleasing/Linux/BeginnerLinuxDeveloper/SettingUpAnUnrealWorkflow/)). Then, build a compatible UnrealCV Plugin from [this modified source code](https://github.com/wengflow/unrealcv/tree/esim), according to the [official instructions](https://docs.unrealcv.org/en/master/plugin/install.html#compile-from-source-code).

## Using Blender as the Rendering Engine

### Configuration Options
Configuration options for the Blender rendering engine are detailed at the top of `blender_renderer.cpp` of the `imp_blender_renderer` ROS package. Note that with N GPUs, the valid `blender_render_device_type` and `blender_render_device_id` combinations are as follows:

| `blender_render_device_type` | `blender_render_device_id` |
| :---: | :---: |
| 0 (CPU) | N |
| 1 (CUDA) | 0 to N-1 |
| 2 (OptiX) | N+1 to 2N |

A sample usage is provided in `cfg/blender.conf` of the `esim_ros` package.

### Running ESIM with Blender

First, start the Blender rendering server on a given port (*e.g.* 5555) with:
```bash
cae
roscd imp_blender_renderer
conda activate blender
python scripts/blender_bridge.py --port 5555
```

Then, run ESIM under a given configuration (*e.g.* `cfg/blender.conf`) in another terminal with:
```bash
cae
roscd esim_ros
roslaunch esim_ros esim.launch config:=cfg/blender.conf
```

To visualize the output of the simulator, you can open `rviz` (from a new terminal) as follows:
```bash
cae
roscd esim_visualization
rviz -d cfg/esim.rviz
```

You can also open `rqt` for more visualizations, as follows:
```bash
cae
roscd esim_visualization
rqt --perspective-file cfg/esim.perspective
```

Please refer to the [ESIM wiki](https://github.com/uzh-rpg/rpg_esim/wiki) for further details on the usage of other rendering engines.
