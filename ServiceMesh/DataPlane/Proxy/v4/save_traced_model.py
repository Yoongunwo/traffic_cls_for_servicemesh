import torch
import torch.nn as nn

FEAT_DIM=128

class StudentEncoder(nn.Module):
    def __init__(self, out_dim=FEAT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 8, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
            nn.Linear(8 * 2 * 2, 32), nn.ReLU(),
            nn.BatchNorm1d(32), nn.Dropout(0.3),
            nn.Linear(32, out_dim)
        )

    def forward(self, x):
        return self.net(x)

model = StudentEncoder()
model.load_state_dict(torch.load('./Model/student_encoder_kd_k8s_10_2x8.pth'))
model.eval()

example_input = torch.randn(1, 1, 34, 44)
traced = torch.jit.trace(model, example_input)
traced.save('./Model/student_encoder_ts.pt')
