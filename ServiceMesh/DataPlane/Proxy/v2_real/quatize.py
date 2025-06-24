import torch
from model import StudentEncoder  # 모델 정의 파일 경로에 맞게 수정
import torch.quantization

# 원래 모델 로드
model_fp32 = StudentEncoder()
model_fp32.load_state_dict(torch.load('./Model/student_encoder_kd_k8s_10_2x8.pth', map_location='cpu'))
model_fp32.eval()

# 동적 양자화 적용 (주로 FC 계층에 적용)
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32, {torch.nn.Linear}, dtype=torch.qint8
)

# TorchScript로 변환
example = torch.randn(1, 1, 34, 44)
scripted = torch.jit.trace(model_int8, example)
scripted.save('./Model/student_encoder_int8_ts.pt')
print("✅ int8 양자화 + TorchScript 저장 완료")
