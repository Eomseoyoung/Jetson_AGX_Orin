> 1. jetpack에서 기본으로 다운 받아져있는 opencv를 삭제
> 
        python3 - <<EOF
        import cv2
        print("version:", cv2.__version__)
        print("path:", cv2.__file__)

        -> 위에 있는 코드를 cmd창에 그대로 실행하면 경로 나타남

> 2. APT opnecv 완전제거

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

> 3. Opencv 소스 준비 + CMake(Python 바인딩 강제)
> 4. CUDNN 설치




