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
> ### 검증
> 
> ### CUDA
> 
> 
> 

