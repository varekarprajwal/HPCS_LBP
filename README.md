# Build OpenCV with CUDA and C++ Support on Ubuntu

This guide explains how to build and install **OpenCV** with **CUDA** and **cuDNN** support for **C++ development** on Ubuntu. The result is a GPU-accelerated OpenCV installation ready for high-performance computer vision tasks.

---

## ✅ System Configuration (Tested With)

- **OS**: Ubuntu
- **GPU**: NVIDIA GeForce GTX 1050 Ti
- **NVIDIA Driver**: 575.64.03
- **CUDA Toolkit**: 12.2
- **cuDNN**: 8.x (for CUDA 12.x)
- **OpenCV Version**: 4.9.0
- **GCC Version**: 12

---

## ⚙️ Step 1: Install Dependencies and Build Tools
First, update your package list and install the essential tools required for the build process, including cmake, git, and libraries for handling various media formats.

```bash
sudo apt update && sudo apt install -y \
  build-essential \
  cmake \
  git \
  wget \
  unzip \
  libgtk2.0-dev \
  libavcodec-dev \
  libavformat-dev \
  libswscale-dev \
  libtbb-dev \
  libjpeg-dev \
  libpng-dev \
  libtiff-dev \
  libdc1394-dev \
  pkg-config
```

## Step 2: Install a CUDA-Compatible C++ Compiler

The NVIDIA CUDA Toolkit has strict requirements for the host C++ compiler version. For CUDA 12.x, the nvcc compiler requires a GCC version of 12 or older. Modern Ubuntu versions often default to a newer, incompatible compiler.

Install gcc-12 and g++-12 to ensure compatibility.

```bash
sudo apt install -y gcc-12 g++-12
```
## Step 3: Download OpenCV and OpenCV-Contrib Sources
You must download both the main OpenCV repository and the opencv_contrib repository, which contains additional modules needed for full CUDA functionality. It is critical that both repositories are of the exact same version.

Create a build directory:
```bash
mkdir -p ~/opencv_build && cd ~/opencv_build
```
Clone the repositories (version 4.9.0):
```bash
git clone https://github.com/opencv/opencv.git --branch 4.9.0
git clone https://github.com/opencv/opencv_contrib.git --branch 4.9.0
```
## Step 4: Configure the Build with CMake
This is the most critical step. We will configure the build using cmake, pointing it to our specific compiler and enabling all necessary CUDA options.

Create a build directory inside the opencv folder:
```bash
cd opencv
mkdir build && cd build
```
Run the cmake command:
This command is long, but each flag is important. We are:

Specifying the compatible C/C++ compiler (gcc-12/g++-12).

Enabling CUDA, cuBLAS, and cuDNN support.

Setting the CUDA architecture for your GPU (e.g., 6.1 for a GTX 1050 Ti). Find your card's "Compute Capability" if you have a different GPU.

Pointing to the opencv_contrib modules.

Forcing the generation of the pkg-config file (opencv4.pc), which is crucial for compiling your own projects later.
```bash
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D CMAKE_C_COMPILER=/usr/bin/gcc-12 \
      -D CMAKE_CXX_COMPILER=/usr/bin/g++-12 \
      -D WITH_CUDA=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D WITH_CUBLAS=1 \
      -D CUDA_ARCH_BIN=6.1 \ #change according to your GPU
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D BUILD_EXAMPLES=ON ..
```
Verify the CMake Output:
After the command finishes, scroll through the output summary. Ensure you see YES for NVIDIA CUDA and that the C++ Compiler is listed as /usr/bin/g++-12.

## Step 5: Compile and Install OpenCV
This process is resource-intensive and can take a long time.

Run the make command:
The -j$(nproc) flag uses all available CPU cores to speed up compilation.
```bash
make -j$(nproc)
```
⚠️ Potential Error: Out of Memory
If the compilation fails with an error like make: *** [Makefile:166: all] Error 2, it's likely your system ran out of RAM. Re-run the command with fewer parallel jobs. Using half your cores or just four is a safe alternative.

# If the first command fails, try this one:
```bash
make -j4
```
Install the compiled libraries:
```bash
sudo make install
```
Update the library cache:
```bash
sudo ldconfig
```
## Step 6: Final Verification
Let's confirm that the installation was successful and that your system can find the new libraries.

Find and Configure pkg-config:
The system needs to know where to find the opencv4.pc file.

First, locate the file:
```bash
sudo find /usr/local -name "opencv4.pc"
```
This will output a path, for example: /usr/local/lib/x86_64-linux-gnu/pkgconfig/opencv4.pc.

Copy the directory part of that path.

Add this directory to your PKG_CONFIG_PATH for your current session. Use the actual path you found.

# Example command - replace the path with your own
export PKG_CONFIG_PATH=/usr/local/lib/x86_64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH

To make this change permanent, add that same export line to the end of your ~/.bashrc file and run source ~/.bashrc.

Check the OpenCV Version:
```bash
pkg-config --modversion opencv4
```
This should correctly output: 4.9.0.

Compile and Run a C++ Test Program:

Create a test file:
```bash
nano cv_test.cpp
```
# Paste the following code inside:
```bash
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core/cuda.hpp"

int main() {
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    int cuda_devices = cv::cuda::getCudaEnabledDeviceCount();
    if (cuda_devices > 0) {
        std::cout << "CUDA is enabled." << std::endl;
        std::cout << "Number of CUDA devices: " << cuda_devices << std::endl;
        cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
    } else {
        std::cout << "CUDA is NOT enabled in this OpenCV build." << std::endl;
    }
    return 0;
}
```
# Save and exit (Ctrl+X, Y, Enter).

# Compile the program using pkg-config to supply the flags:
```bash
g++ cv_test.cpp -o cv_test $(pkg-config --cflags --libs opencv4)
```
# Run the executable:
```bash
./cv_test
```
You should see output confirming your OpenCV version and that CUDA support is enabled, along with details about your GPU.

Congratulations! You are now ready to build GPU-accelerated computer vision projects with OpenCV and C++.
