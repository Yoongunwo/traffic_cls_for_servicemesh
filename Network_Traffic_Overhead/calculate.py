import matplotlib.pyplot as plt
import numpy as np

systems = ['Default\n(Without Proxy)', 'Linkerd', 'Istio', 'Proposed Method\n(Detection Disabled)', 'Proposed Method\n(Detection Enabled)']
# latencies = [2.78, 4.45, 5.38, 5.16, 5.18] # 0.0173
latencies = [1.58, 4.20, 6.69, 2.95, 13.23] # 0.0173
rps = [5.18, 1.92, 1.21, 2.73, 0.60]

# 한 그림에 두 개의 바 차트
# plt.figure(figsize=(8, 5))
x = np.arange(len(systems))
width = 0.42

fig, ax1 = plt.subplots(figsize=(10, 5))

# rects1 = ax1.bar(x - width/2, latencies, width, label='Latency', color='#a2d2ff', edgecolor='black')
rects1 = ax1.bar(x - width/2, latencies, width, label='Latency', 
                 color='#a2d2ff', edgecolor='#4a90e2', 
                #  color='#e3f0ff', edgecolor='#89cff0', 
                 hatch='\\\\\\', alpha=0.7)  # \\\를 사용하여 왼쪽 대각선 패턴 생성
ax1.set_xticks(x)
ax1.set_xticklabels(systems, rotation=0, fontsize=3)
ax1.tick_params(axis='x', labelsize=11)  # x축 레이블 크기 조절
ax1.tick_params(axis='y', labelsize=12)  # x축 레이블 크기 조절

ax1.set_ylabel('Latency (ms)', fontsize=15)
ax1.set_ylim(0, 15)
ax1.bar_label(rects1, fmt='%.2f', fontsize=14)

ax2 = ax1.twinx()
rects2 = ax2.bar(x + width/2, rps, width, label='Throughput', 
                 color='#ffb088', edgecolor='#ff7043', 
                 hatch='///', alpha=0.7)
ax2.set_ylabel('Throughput (k req/sec)', fontsize=15)
ax2.set_ylim(0, 7)
ax2.tick_params(axis='y', labelsize=12)  # 두 번째 y축 레이블 크기
ax2.bar_label(rects2, fmt='%.2f', fontsize=14)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=13)

plt.tight_layout()
plt.savefig('./Network_Traffic_Overhead/latency_throughput.png', dpi=300, bbox_inches='tight')
plt.close()