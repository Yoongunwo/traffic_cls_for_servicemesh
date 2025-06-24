import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

models = ['CNN-4x128\n(teacher)', 'CNN-2x32', 'CNN-2x16', 'CNN-2x8', 'CNN-1x16', 'CNN-1x8']
# avg_times = [6.023, 2.874, 2.581, 2.369, 2.068, 1.635]  # ms/image
# std_devs = [0.373, 0.246, 0.212, 0.286, 0.226, 0.184]   # 표준편차 예시
avg_times = [3.612, 0.802, 0.592, 0.518, 0.426, 0.406]  # ms/image
std_devs = [0.182, 0.023, 0.015, 0.013, 0.011, 0.014]   # 표준편차 예시

# 색상: 위쪽이 진하고 아래로 갈수록 연한 회색
num_models = len(models)
colors = [str(0.7 - 0.4 * i / (num_models - 1)) for i in range(num_models)][::-1]  # 밝은 → 진한 뒤집기

# plt.figure(figsize=(8, 4))
# bars = plt.barh(models, avg_times, xerr=std_devs, capsize=5, color=colors, height=0.5)
# plt.xlabel("Average Detection Time (ms per image)", fontsize=10)
# plt.title("Model-wise Average Detection Time with Variance", fontsize=12)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.grid(axis='x', linestyle='--', alpha=0.5, linewidth=0.5)
# plt.gca().invert_yaxis()  # 빠른 모델이 아래로
# plt.tight_layout(pad=0.8)

# plt.savefig("./Detection_Overhead/model_detection_time_variance.png", dpi=600, bbox_inches='tight')

fig, ax = plt.subplots(figsize=(8, 4))  # 비율 고정

bars = plt.barh(models, avg_times, xerr=std_devs, capsize=5, color=colors, height=0.5)
plt.xlabel("Average Detection Time (ms per image)", fontsize=10)
plt.title("Model-wise Average Detection Time with Variance", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='x', linestyle='--', alpha=0.5, linewidth=0.5)
plt.gca().invert_yaxis()
plt.tight_layout(pad=0.8)

# ➤ 표준편차 숫자를 오른쪽 끝에 표시
for bar, avg, std in zip(bars, avg_times, std_devs):
    y = bar.get_y() + bar.get_height() / 2
    # 막대 왼쪽에 평균 ± 표준편차 표시
    text = f"{avg:.3f}"
    plt.text(0.05, y, text, va='center', ha='left', fontsize=10)

# ✅ 오른쪽 padding 확보: x축 최대값 → 최대 + 여유값
max_x = max([m + s for m, s in zip(avg_times, std_devs)])
ax.set_xlim(0, max_x + 0.4)  # 오른쪽에 여유 padding 확보

# 평균값, 표준편차 텍스트 표시
for bar, avg, std in zip(bars, avg_times, std_devs):
    y = bar.get_y() + bar.get_height() / 2
    ax.text(bar.get_width() + std + 0.05, y, f"±{std:.3f}", va='center', ha='left', fontsize=9)

# max_val = max([m + s for m, s in zip(avg_times, std_devs)])
# plt.xlim(0, max_val + 0.75)


plt.savefig("./Detection_Overhead/model_detection_time_variance.png", dpi=600, bbox_inches='tight')
