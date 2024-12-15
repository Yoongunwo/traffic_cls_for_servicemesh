import matplotlib.pyplot as plt
import numpy as np

systems = ['Default', 'Linkerd', 'Istio', 'Preposed Method\n(Not Detecting)', 'Preposed Method\n(Detecting)']
latencies = [2.78, 4.45, 5.38, 5.16, 93.53]
rps = [4.07, 1.82, 1.57, 0.98, 0.083]

# 한 그림에 두 개의 바 차트
# plt.figure(figsize=(8, 5))
x = np.arange(len(systems))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 5))

rects1 = ax1.bar(x - width/2, latencies, width, label='Latency', color='#1f77b4')
ax1.set_ylabel('Latency (ms)', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(systems, rotation=0, fontsize=12)
ax1.bar_label(rects1, fmt='%.2f', fontsize=10)

ax2 = ax1.twinx()
rects2 = ax2.bar(x + width/2, rps, width, label='Throughput', color='#ff7f0e')
ax2.set_ylabel('Throughput (k req/sec)', fontsize=12)
ax2.set_ylim(0, 5)
ax2.bar_label(rects2, fmt='%.2f', fontsize=10)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig('latency_throughput.png', dpi=300, bbox_inches='tight')
plt.close()