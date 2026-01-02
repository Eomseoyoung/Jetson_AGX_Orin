> 1. jetpack에서 기본으로 다운 받아져있는 opencv를 삭제
> 
        python3 - <<EOF
        import cv2
        print("version:", cv2.__version__)
        print("path:", cv2.__file__)

        -> 위에 있는 코드를 cmd창에 그대로 실행하면 경로 나타남

> 2. APT opnecv 완전제거
>    
        sudo apt purge -y 'libopencv*' 'python3-opencv'
        sudo apt autoremove -y
        sudo rm -rf /usr/lib/python3/dist-packages/cv2
        sudo rm -rf /usr/lib/aarch64-linux-gnu/libopencv*
        sudo ldconfig
        -------------------------------------------------
        python3 - <<EOF
        import cv2
        EOF
        -> 위에로 삭제하고 아래로 확인해서 NoduleNotfounderror나면 됨
        위에 명령어를 쳤을때 APT opencv 4.5.4가 나옴


> 3. APT Opencv 완전 제거(잔재 포함)
>
> 


      sudo apt purge -y \
      libopencv* \
      python3-opencv \
      opencv-data
    
      sudo apt autoremove -y
      -------------------------------------------------
      sudo rm -f /usr/lib/python3/dist-packages/cv2*.so
      sudo rm -rf /usr/lib/python3/dist-packages/cv2
      sudo rm -f /usr/lib/aarch64-linux-gnu/libopencv*
      sudo ldconfig
      -------------------------------------------------
      Python이 실제로 cv2를 어디서 불러오는지 강제로 추적
      python3 - <<EOF
      import cv2, sys
      print(cv2.__file__)
      print(sys.path)
      EOF
      --------------------------------------------------------
      sudo rm -rf /usr/local/lib/libopencv*
      sudo rm -rf /usr/local/include/opencv4
      sudo rm -rf /usr/local/lib/python3.10/dist-packages/cv2*
      sudo ldconfig

      --------------------------------------------------------
      sudo find /usr -name "cv2*.so" 2>/dev/null
      -> 확인 코드 아무것도 안 나와야 정상


> 4. Opencv 소스 준비 + CMake(Python 바인딩 강제)

      cd ~
      rm -rf opencv opencv_contrib
      git clone -b 4.8.0 https://github.com/opencv/opencv.git
      git clone -b 4.8.0 https://github.com/opencv/opencv_contrib.git
      ---------------------------------------------------------------
      cd ~/opencv
      mkdir build && cd build
      ---------------------------------------------------------------
      cmake .. \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
        -D WITH_CUDA=ON \
        -D WITH_CUDNN=ON \
        -D OPENCV_DNN_CUDA=ON \
        -D CUDA_ARCH_BIN=8.7 \
        -D ENABLE_FAST_MATH=1 \
        -D CUDA_FAST_MATH=1 \
        -D WITH_CUBLAS=1 \
        -D WITH_GSTREAMER=ON \
        -D WITH_V4L=ON \
        -D BUILD_opencv_python3=ON \
        -D PYTHON3_EXECUTABLE=$(which python3) \
        -D PYTHON3_INCLUDE_DIR=/usr/include/python3.10 \
        -D PYTHON3_PACKAGES_PATH=/usr/local/lib/python3.10/dist-packages \
        -D BUILD_TESTS=OFF \
        -D BUILD_EXAMPLES=OFF

      --------------------------------------------------------------------
      cd ~/opencv
      mkdir build
      cd build
      --------------------------------------------------------------------
      cmake .. \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
        -D WITH_CUDA=ON \
        -D WITH_CUDNN=ON \
        -D OPENCV_DNN_CUDA=ON \
        -D CUDA_ARCH_BIN=8.7 \
        -D ENABLE_FAST_MATH=1 \
        -D CUDA_FAST_MATH=1 \
        -D WITH_CUBLAS=1 \
        -D WITH_GSTREAMER=ON \
        -D WITH_V4L=ON \
        -D BUILD_opencv_python3=ON \
        -D PYTHON3_EXECUTABLE=$(which python3) \
        -D PYTHON3_INCLUDE_DIR=/usr/include/python3.10 \
        -D PYTHON3_PACKAGES_PATH=/usr/local/lib/python3.10/dist-packages \
        -D BUILD_TESTS=OFF \
        -D BUILD_EXAMPLES=OFF


      > 중간에 멈춤 오류 발견
      --------------------------------------------------------------------

      cd ~/opencv
      rm -rf build
      mkdir build
      cd build
      -> 빌드 디렉토리 초기화
      ---------------------------------------------------------------------
      cmake .. \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
        -D WITH_CUDA=ON \
        -D WITH_CUDNN=ON \
        -D CUDNN_INCLUDE_DIR=/usr/include \
        -D CUDNN_LIBRARY=/usr/lib/aarch64-linux-gnu/libcudnn.so \
        -D OPENCV_DNN_CUDA=ON \
        -D CUDA_ARCH_BIN=8.7 \
        -D ENABLE_FAST_MATH=1 \
        -D CUDA_FAST_MATH=1 \
        -D WITH_CUBLAS=1 \
        -D WITH_GSTREAMER=ON \
        -D WITH_V4L=ON \
        -D BUILD_opencv_python3=ON \
        -D PYTHON3_EXECUTABLE=/usr/bin/python3 \
        -D PYTHON_DEFAULT_EXECUTABLE=/usr/bin/python3 \
        -D PYTHON3_INCLUDE_DIR=/usr/include/python3.10 \
        -D PYTHON3_PACKAGES_PATH=/usr/local/lib/python3.10/dist-packages \
        -D BUILD_opencv_python2=OFF \
        -D BUILD_TESTS=OFF \
        -D BUILD_EXAMPLES=OFF

        CMAKE 재설치 -> JetPack은 정상인데, cuDNN만 빠진 상태 발견
      ----------------------------------------------------------------------------
      
> 5. CUDNN 설치

      sudo apt update
      sudo apt install -y libcudnn8 libcudnn9-dev
      -> jetpack 6.x 부터는 9버전 사용
      ----------------------------------------------------------------------------
      설치확인

      ls /usr/include/cudnn*.h
      ls /usr/lib/aarch64-linux-gnu/libcudnn*

> 6. CMAKE 반복
>


      cd ~/opencv
      rm -rf build
      mkdir build
      cd build
      -> 빌드 디렉토리 초기화
      ---------------------------------------------------------------------
      cmake .. \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
        -D WITH_CUDA=ON \
        -D WITH_CUDNN=ON \
        -D CUDNN_INCLUDE_DIR=/usr/include \
        -D CUDNN_LIBRARY=/usr/lib/aarch64-linux-gnu/libcudnn.so \
        -D OPENCV_DNN_CUDA=ON \
        -D CUDA_ARCH_BIN=8.7 \
        -D ENABLE_FAST_MATH=1 \
        -D CUDA_FAST_MATH=1 \
        -D WITH_CUBLAS=1 \
        -D WITH_GSTREAMER=ON \
        -D WITH_V4L=ON \
        -D BUILD_opencv_python3=ON \
        -D PYTHON3_EXECUTABLE=/usr/bin/python3 \
        -D PYTHON_DEFAULT_EXECUTABLE=/usr/bin/python3 \
        -D PYTHON3_INCLUDE_DIR=/usr/include/python3.10 \
        -D PYTHON3_PACKAGES_PATH=/usr/local/lib/python3.10/dist-packages \
        -D BUILD_opencv_python2=OFF \
        -D BUILD_TESTS=OFF \
        -D BUILD_EXAMPLES=OFF


    ---------------------------------------------------------------------------------
    make -j$(nproc) 
    -> Jetson AGX orin 기준 30-60분 소요됨

    CUDNN 깨진 상태 계속 발견 -> 진행중

      
현재 계속 시도중이지만 해결되지 않아 초기 세팅부터 다시 진행하여 시작 (이전 세팅을 다른 분이해서 어떤게 문제였고 버전이 호환되지 않는지 자세하게 이해 불가)
-> 안에 mobaxterm이랑 연동도 같이 진행 

