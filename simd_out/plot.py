import matplotlib.pyplot as plt
import numpy as np

# Data from the LaTeX charts
datasets = ['N200 D4', 'N200 D16', 'N800 D32', 'N800 D64']
x_pos = np.arange(len(datasets))

# Chart 1: Clustering time comparison
baseline_times = [971.592, 2852.172, 29906.050, 63818.289]
simd_w4_times = [833.740, 1747.662, 14702.570, 26330.948]
simd_w8_times = [915.935, 1919.640, 14798.200, 24575.659]
simd_w16_times = [1013.405, 1823.451, 15361.229, 25625.906]

# Chart 2: Speedup comparison
speedup_w4 = [1.165, 1.632, 2.034, 2.424]
speedup_w8 = [1.061, 1.486, 2.021, 2.597]
speedup_w16 = [0.959, 1.564, 1.947, 2.490]

# Chart 3: Vector width impact
vector_widths = [4, 8, 16]
n200_d4_times = [833.740, 915.935, 1013.405]
n200_d16_times = [1747.662, 1919.640, 1823.451]
n800_d32_times = [14702.570, 14798.200, 15361.229]
n800_d64_times = [26330.948, 24575.659, 25625.906]

# Set up the figure with 3 subplots
fig = plt.figure(figsize=(18, 6))

# Chart 1: Clustering Time Comparison
ax1 = plt.subplot(1, 3, 1)
width = 0.2
ax1.bar(x_pos - 1.5*width, baseline_times, width, label='Baseline', alpha=0.8)
ax1.bar(x_pos - 0.5*width, simd_w4_times, width, label='SIMD W=4', alpha=0.8)
ax1.bar(x_pos + 0.5*width, simd_w8_times, width, label='SIMD W=8', alpha=0.8)
ax1.bar(x_pos + 1.5*width, simd_w16_times, width, label='SIMD W=16', alpha=0.8)

ax1.set_xlabel('Dataset', fontsize=11)
ax1.set_ylabel('Time (ms)', fontsize=11)
ax1.set_title('Clustering Time by Vector Width', fontsize=12, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(datasets)
ax1.legend(loc='upper left')
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_yscale('log')  # Use log scale due to large range

# Chart 2: Speedup Comparison
ax2 = plt.subplot(1, 3, 2)
width = 0.25
ax2.bar(x_pos - width, speedup_w4, width, label='SIMD W=4', alpha=0.8)
ax2.bar(x_pos, speedup_w8, width, label='SIMD W=8', alpha=0.8)
ax2.bar(x_pos + width, speedup_w16, width, label='SIMD W=16', alpha=0.8)

ax2.set_xlabel('Dataset', fontsize=11)
ax2.set_ylabel('Speedup (x)', fontsize=11)
ax2.set_title('Speedup Relative to Baseline', fontsize=12, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(datasets)
ax2.legend(loc='upper left')
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, linewidth=1)  # Reference line at 1x

# Chart 3: Vector Width Impact
ax3 = plt.subplot(1, 3, 3)
ax3.plot(vector_widths, n200_d4_times, marker='o', label='N200 D4', linewidth=2, markersize=8)
ax3.plot(vector_widths, n200_d16_times, marker='s', label='N200 D16', linewidth=2, markersize=8)
ax3.plot(vector_widths, n800_d32_times, marker='^', label='N800 D32', linewidth=2, markersize=8)
ax3.plot(vector_widths, n800_d64_times, marker='d', label='N800 D64', linewidth=2, markersize=8)

ax3.set_xlabel('Vector Width', fontsize=11)
ax3.set_ylabel('Time (ms)', fontsize=11)
ax3.set_title('Performance vs Vector Width', fontsize=12, fontweight='bold')
ax3.set_xticks(vector_widths)
ax3.legend(loc='upper right')
ax3.grid(alpha=0.3, linestyle='--')
ax3.set_yscale('log')  # Use log scale due to large range

plt.tight_layout()
plt.savefig('simd_performance_charts.png', dpi=300, bbox_inches='tight')
plt.savefig('simd_performance_charts.pdf', bbox_inches='tight')
print("Charts saved as 'simd_performance_charts.png' and 'simd_performance_charts.pdf'")
plt.show()