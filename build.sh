#!/bin/bash
set -e
set -x

#brew install libomp

# Get the actual libomp path
LIBOMP_PREFIX=$(brew --prefix libomp)

# Set environment variables
export LDFLAGS="-L${LIBOMP_PREFIX}/lib"
export CPPFLAGS="-I${LIBOMP_PREFIX}/include"
export CMAKE_PREFIX_PATH="${LIBOMP_PREFIX}"

# Critical: Set OpenMP flags explicitly for CMake
export OpenMP_C_FLAGS="-Xpreprocessor;-fopenmp;-I${LIBOMP_PREFIX}/include"
export OpenMP_C_LIB_NAMES="omp"
export OpenMP_CXX_FLAGS="-Xpreprocessor;-fopenmp;-I${LIBOMP_PREFIX}/include"
export OpenMP_CXX_LIB_NAMES="omp"
export OpenMP_omp_LIBRARY="${LIBOMP_PREFIX}/lib/libomp.dylib"

mkdir build-release && cd build-release

CMAKE_EXTRA_OPTIONS=''

CMAKE_EXTRA_OPTIONS='-DCMAKE_OSX_ARCHITECTURES=arm64  -DWITH_ACCELERATE=ON -DWITH_MKL=OFF -DOPENMP_RUNTIME=NONE -DWITH_RUY=ON'


cd build-release && cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DCMAKE_INSTALL_PREFIX=/Users/masj/dev/CTranslate2/install \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_CLI=OFF \
      -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
      -DOpenMP_C_FLAGS="${OpenMP_C_FLAGS}" \
      -DOpenMP_CXX_FLAGS="${OpenMP_CXX_FLAGS}" \
      -DOpenMP_omp_LIBRARY="${OpenMP_omp_LIBRARY}" \
      $CMAKE_EXTRA_OPTIONS ..

sudo make -j$(sysctl -n hw.physicalcpu_max) install
#cd ..
#rm -r build-release
#cp README.md python/
