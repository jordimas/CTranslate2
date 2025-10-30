#!/bin/bash
set -e
set -x

# Ensure CMake and LLVM setup
pip install "cmake==3.18.4"
cmake --version

brew install llvm libomp

# Explicitly point to Homebrew LLVM
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
export CC="/opt/homebrew/opt/llvm/bin/clang"
export CXX="/opt/homebrew/opt/llvm/bin/clang++"

# Add include and lib flags for OpenMP and LLVM
export LDFLAGS="-L/opt/homebrew/opt/llvm/lib -L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/llvm/include -I/opt/homebrew/opt/libomp/include"
export CFLAGS="$CPPFLAGS"
export CXXFLAGS="$CPPFLAGS -fopenmp"
export CMAKE_CXX_FLAGS="-fopenmp"
export CMAKE_EXE_LINKER_FLAGS="-L/opt/homebrew/opt/libomp/lib -lomp"

mkdir build-release && cd build-release

CMAKE_EXTRA_OPTIONS=''

if [ "$CIBW_ARCHS" == "arm64" ]; then
    CMAKE_EXTRA_OPTIONS='-DCMAKE_OSX_ARCHITECTURES=arm64 -DWITH_ACCELERATE=ON -DWITH_MKL=OFF -DOPENMP_RUNTIME=NONE -DWITH_RUY=ON'
else
    # Install Intel oneAPI MKL
    ONEAPI_INSTALLER_URL="https://registrationcenter-download.intel.com/akdlm/IRC_NAS/cd013e6c-49c4-488b-8b86-25df6693a9b7/m_BaseKit_p_2023.2.0.49398.dmg"
    wget -q $ONEAPI_INSTALLER_URL
    hdiutil attach -noverify -noautofsck $(basename $ONEAPI_INSTALLER_URL)
    sudo /Volumes/$(basename $ONEAPI_INSTALLER_URL .dmg)/bootstrapper.app/Contents/MacOS/bootstrapper --silent --eula accept --components intel.oneapi.mac.mkl.devel

    ONEDNN_VERSION=3.1.1
    wget -q https://github.com/oneapi-src/oneDNN/archive/refs/tags/v${ONEDNN_VERSION}.tar.gz
    tar xf *.tar.gz && rm *.tar.gz
    cd oneDNN-*
    cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
          -DCMAKE_BUILD_TYPE=Release \
          -DONEDNN_LIBRARY_TYPE=STATIC \
          -DONEDNN_BUILD_EXAMPLES=OFF \
          -DONEDNN_BUILD_TESTS=OFF \
          -DONEDNN_ENABLE_WORKLOAD=INFERENCE \
          -DONEDNN_ENABLE_PRIMITIVE="CONVOLUTION;REORDER" \
          -DONEDNN_BUILD_GRAPH=OFF .
    make -j$(sysctl -n hw.physicalcpu_max) install
    cd ..
    rm -r oneDNN-*
    CMAKE_EXTRA_OPTIONS='-DWITH_DNNL=ON'
fi

# Re-run cmake and build
cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_CLI=OFF \
      -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
      $CMAKE_EXTRA_OPTIONS \
      ..
VERBOSE=1 make -j$(sysctl -n hw.physicalcpu_max) install

cd ..
rm -r build-release

cp README.md python/

