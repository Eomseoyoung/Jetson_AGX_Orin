# jetson_orin
> ### jetson orin 경량화
> jetson orin 내부에서 pth를 1) ONNX로 변환후 2) TensorRT로 변환하는 방법에 대해 기술합니다.

> ### folder
>
> yolo_to_tensor.py -> 욜로로 학습한 모델을 tensor 엔진으로 변환하는 코드


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
> ### e2e model 사용(YOLO)
>     export부분만 변경
>     nms, half 둘 다 True로 변경
> ### 검증
> 
> ### CUDA
> 
> 
> 

