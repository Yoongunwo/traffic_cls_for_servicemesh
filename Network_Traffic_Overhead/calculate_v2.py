import matplotlib.pyplot as plt
import numpy as np

# === 데이터 정의 ===
systems = ['Default\n(Without Proxy)', 'Linkerd', 'Istio', 
           'Proposed Method\n(Detection Disabled)', 'Proposed Method\n(Detection Enabled)']
latencies = [1.58, 4.20, 6.69, 2.95, 13.23]
rps = [5.18, 1.92, 1.21, 2.73, 0.60]

x = np.arange(len(systems))
width = 0.6

# === Subplot 생성 ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

# === Latency 그래프 ===
rects1 = ax1.bar(x, latencies, width, color='#a2d2ff', edgecolor='#4a90e2', hatch='\\\\\\', alpha=0.8, label='Latency')
ax1.set_ylabel('Latency (ms)', fontsize=13)
ax1.set_ylim(0, max(latencies)*1.3)
ax1.bar_label(rects1, fmt='%.2f', fontsize=12)
ax1.tick_params(axis='y', labelsize=12)
ax1.legend(loc='upper left', fontsize=15)
ax1.set_title('Latency and Throughput per System', fontsize=15, pad=10)

# === Throughput 그래프 ===
rects2 = ax2.bar(x, rps, width, color='#ffb088', edgecolor='#ff7043', hatch='///', alpha=0.8, label='Throughput')
ax2.set_ylabel('Throughput (k req/sec)', fontsize=13)
ax2.set_ylim(0, max(rps)*1.15)
ax2.set_xticks(x)
ax2.set_xticklabels(systems, fontsize=10)
ax2.bar_label(rects2, fmt='%.2f', fontsize=12)
ax2.tick_params(axis='x', labelsize=12)
ax2.tick_params(axis='y', labelsize=12)
ax2.legend(loc='upper right', fontsize=15)

# === 레이아웃 정리 및 저장 ===
plt.tight_layout()
plt.savefig('./Network_Traffic_Overhead/latency_throughput_subplots.png', dpi=300, bbox_inches='tight')
plt.close()
