#!/usr/bin/env bash
CAFFE_USE_MPI=${1:-OFF}
CAFFE_MPI_PREFIX=${MPI_PREFIX:-""}


# install common dependencies: OpenCV
# adpated from OpenCV.sh
version="2.4.13"

# build caffe
echo "Building Caffe, MPI status: ${CAFFE_USE_MPI}"
cd lib/caffe-action
[[ -d build ]] || mkdir build
cd build
if [ "$CAFFE_USE_MPI" == "MPI_ON" ]; then
OpenCV_DIR=../../../3rd-party/opencv-$version/build/ cmake .. -DUSE_MPI=ON -DMPI_CXX_COMPILER="${CAFFE_MPI_PREFIX}/bin/mpicxx" -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF -DBLAS=mkl 
else
OpenCV_DIR=../../../3rd-party/opencv-$version/build/ cmake .. -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF -DBLAS=mkl
fi
if make -j32 install ; then
    echo "Caffe Built."
    echo "All tools built. Happy experimenting!"
    cd ../../../
else
    echo "Failed to build Caffe. Please check the logs above."
    exit 1
fi
