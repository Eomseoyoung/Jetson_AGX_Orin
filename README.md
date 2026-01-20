# Jetson_AGX_Orin_64


> ### 초기세팅
>  * jetson os 설치
>     1) Nvidia SDK manager를 통해 설치
>    2) [NVIDIA SDK Manager로 Jetson AGX Orin에 Ubuntu 22.04 설치 ](https://2dudwns.tistory.com/29)
> * jetson python lib 설치
>      1) [휠파일(JetPack 6.2.1+b38)](https://pypi.jetson-ai-lab.io/jp6/cu126)
>         
>        wget https://pypi.jetson-ai-lab.io/jp6/cu126/+f/e1e/9e3dc2f4d5551/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl#sha256=e1e9e3dc2f4d5551c5f0b5554cf490d141cd0d339a5a7f4826ac0e04f20a35fc
> ### Jetson_AGX_Orin 경량화
> jetson orin 내부에서 pth를 1) ONNX로 변환후 2) TensorRT로 변환하는 방법
>
> ### onnx to tensorRT
> onnx를 tensorRT로 변환 커맨드 명령어(jetson 내부에서 실행)
>
>      /usr/src/tensorrt/bin/trtexec \
>        --onnx=rail.onnx \
>        --saveEngine=rail_fp16.plan \
>        --fp16 
>
>      * FP16 → 먼저 검증 → INT8은 나중 단계에서 사용하나 현재 모델은 sementic segmentaion 모델이므로 예민하기에 FP16정도로만 양자화

> ### jetson 가속화 명령
>       sudo nvpmodel -m 0
>       sudo jetson_clocks
>
> ### jetson-stats 설치 (HW engine확인)
>       sudo apt update
>       sudo apt install python3-pip python3-setuptools -y
>       sudo pip3 install -U jetson-stats
>       실행방법 cmd 창에서 jtop
>
> ### 멀티프로세싱
>     카메라용 프로세싱부분 구현 (FAST API)
>
> ### OPENCV
>     CUDA용으로 빌드
>     pip install opencv-python 금지
> 
> ### e2e model 사용(YOLO)
>     .engine으로 변환 후 사용
> 

