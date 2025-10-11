import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 1.
z_real = np.random.normal(500, 50, 1000)
z_pred = z_real + np.random.normal(0, 8, 1000)
errors = z_pred - z_real

plt.figure(figsize=(8, 5))
plt.hist(errors, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(x=np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.1f}mm')
plt.xlabel('Depth Error (mm)')
plt.ylabel('Frequency')
plt.title('ZED Depth Measurement Error Distribution')
plt.legend()
plt.grid(axis='y', alpha=0.5)
plt.savefig('zed_depth_error_hist.png', dpi=300, bbox_inches='tight')
plt.show()

# 2.
classes = ['Unripe', 'Ripe', 'Overripe']
y_true = np.random.choice(classes, size=100, p=[0.3, 0.5, 0.2])
y_pred = np.array([np.random.choice(classes, p=[0.1, 0.8, 0.1]) if t == 'Ripe' else
                   np.random.choice(classes, p=[0.7, 0.2, 0.1]) for t in y_true])

cm = confusion_matrix(y_true, y_pred, labels=classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Tomato Ripeness Classification Confusion Matrix')
plt.savefig('ripeness_cm.png', dpi=300, bbox_inches='tight')
plt.show()

# 3.
ious = np.random.uniform(0.1, 0.8, 50)
count_errors = 10 * (1 - ious) + np.random.normal(0, 0.5, 50)

plt.figure(figsize=(8, 5))
plt.scatter(ious, count_errors, color='orange', alpha=0.6, edgecolors='black')
plt.plot(np.unique(ious), np.poly1d(np.polyfit(ious, count_errors, 1))(np.unique(ious)),
         'r--', label=f'Correlation: {np.corrcoef(ious, count_errors)[0,1]:.2f}')
plt.xlabel('Overlap IoU')
plt.ylabel('Count Error (Number of Tomatoes)')
plt.title('Fruit Count Error vs. Overlap IoU')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('count_error_vs_iou.png', dpi=300, bbox_inches='tight')
plt.show()