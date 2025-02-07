import json
from matplotlib import pyplot as plt
import numpy as np
import os
from sklearn.metrics import roc_auc_score, roc_curve

with open('data/Postloss_2.json') as f:
    Postloss_log = json.load(f)

with open('data/Preloss_2.json') as f:
    Preloss_log = json.load(f)

# Calculate histogram (distribution) of the loss values
Premse_hist, Premse_bins = np.histogram(Preloss_log['mse'], bins=30, density=True)
Postmse_hist, Postmse_bins = np.histogram(Postloss_log['mse'], bins=30, density=True)
bins = 30
density = True
# MSE Loss の分布をプロット
plt.figure(figsize=(10, 6))
plt.hist(Preloss_log['mse'], bins=bins, alpha=0.6, density=density, color='blue', label="Pre Distribution", edgecolor='black')
plt.hist(Postloss_log['mse'], bins=bins, alpha=0.6, density=density, color='red', label="Post Distribution", edgecolor='black')

plt.title("Distribution of MSE Loss Values", fontsize=14)
plt.xlabel("MSE Loss Values", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
# plt.annotate("Lower MSE indicates better performance", xy=(np.median(Preloss_log['mse']), 0.1),
#              xytext=(np.median(Preloss_log['mse']) + 0.5, 0.2),
#              arrowprops=dict(facecolor='black', arrowstyle="->"),
#              fontsize=10)

plt.savefig(os.path.join("data/", "MSE_Loss_Distribution.png"))
plt.close()

# LPIPS Loss の分布をプロット
plt.figure(figsize=(10, 6))
plt.hist(Preloss_log['lpips'], bins=bins, alpha=0.6, density=density, color='blue', label="Pre Distribution", edgecolor='black')
plt.hist(Postloss_log['lpips'], bins=bins, alpha=0.6, density=density, color='red', label="Post Distribution", edgecolor='black')

plt.title("Distribution of Loss Values", fontsize=14)
plt.xlabel("LPIPS Loss Values", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
# plt.annotate("Higher LPIPS indicates more perceptual difference", xy=(np.median(Postloss_log['lpips']), 0.1),
#              xytext=(np.median(Postloss_log['lpips']) - 0.5, 0.2),
#              arrowprops=dict(facecolor='black', arrowstyle="->"),
#              fontsize=10)

plt.savefig(os.path.join("data", "LPIPS_Loss_Distribution.png"))
plt.close()

# AUC スコアの計算
PreLabel = np.array([0] * len(Preloss_log['mse']))
PostLabel = np.array([1] * len(Postloss_log['mse']))
mse_scores = np.concatenate((Preloss_log['mse'], Postloss_log['mse']))
lpips_scores = np.concatenate((Preloss_log['lpips'], Postloss_log['lpips']))
labels = np.concatenate((PreLabel, PostLabel))

mse_auc = roc_auc_score(labels, mse_scores)
lpips_auc = roc_auc_score(labels, lpips_scores)

# FPR95% の計算
def calculate_fpr95(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    idx = np.argmax(tpr >= 0.95)
    return fpr[idx]

mse_fpr95 = calculate_fpr95(labels, mse_scores)
lpips_fpr95 = calculate_fpr95(labels, lpips_scores)

# ROC曲線のプロット
plt.figure(figsize=(10, 6))
fpr_mse, tpr_mse, _ = roc_curve(labels, mse_scores)
fpr_lpips, tpr_lpips, _ = roc_curve(labels, lpips_scores)

plt.plot(fpr_mse, tpr_mse, label=f"MSE AUC = {mse_auc:.4f}", color='blue')
plt.plot(fpr_lpips, tpr_lpips, label=f"LPIPS AUC = {lpips_auc:.4f}", color='red')

plt.xlabel("False Positive Rate (FPR)", fontsize=12)
plt.ylabel("True Positive Rate (TPR)", fontsize=12)
plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=14)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

plt.savefig(os.path.join("data", "ROC_Curve.png"))
plt.close()

# 結果を表示
print(f"MSE AUC: {mse_auc:.4f}, FPR95%: {mse_fpr95:.4f}")
print(f"LPIPS AUC: {lpips_auc:.4f}, FPR95%: {lpips_fpr95:.4f}")

# 生成した画像を表示
# import ace_tools as tools
# tools.display_image("data/MSE_Loss_Distribution.png")
# tools.display_image("data/LPIPS_Loss_Distribution.png")
# tools.display_image("data/ROC_Curve.png")
    # # Plot histogram (distribution)
    # plt.figure(figsize=(10, 6))
    # plt.bar(Premse_bins[:-1], Premse_hist, width=np.diff(Premse_bins), alpha=0.6, label="Pre Distribution")
    # plt.bar(Postmse_bins[:-1], Postmse_hist, width=np.diff(Postmse_bins), alpha=0.6, label="Post Distribution")
    # plt.title("Distribution of MSE Loss Values")
    # plt.xlabel("MSE Loss Values")
    # plt.ylabel("Density")
    # plt.legend()
    # plt.grid()
    # plt.savefig(os.path.join("data/", "MSE_Loss_Distribution.png"))
    # plt.close()

    # Prelpips_hist, Prelpips_bins = np.histogram(Preloss_log['lpips'], bins=30, density=True)
    # Postlpips_hist, Postlpips_bins = np.histogram(Postloss_log['lpips'], bins=30, density=True)

    # # Plot histogram (distribution)
    # plt.figure(figsize=(10, 6))
    # plt.bar(Prelpips_bins[:-1], Prelpips_hist, width=np.diff(Prelpips_bins), alpha=0.6, label="Pre Distribution")
    # plt.bar(Postlpips_bins[:-1], Postlpips_hist, width=np.diff(Postlpips_bins), alpha=0.6, label="Post Distribution")
    # plt.title("Distribution of LPIPS Loss Values")
    # plt.xlabel("LPIPS Loss Values")
    # plt.ylabel("Density")
    # plt.legend()
    # plt.grid()
    # plt.savefig(os.path.join("data", "LPIPS_Loss_Distribution.png"))
    # plt.close()

    # PreLabel = np.array([0] * len(Preloss_log['mse']))
    # PostLabel = np.array([1] * len(Postloss_log['mse']))
    # mse_scores = np.concatenate((Preloss_log['mse'], Postloss_log['mse']))
    # lpips_scores = np.concatenate((Preloss_log['lpips'], Postloss_log['lpips']))
    # labels = np.concatenate((PreLabel, PostLabel))
    # mse_auc = roc_auc_score(labels, mse_scores)
    # lpips_auc = roc_auc_score(labels, lpips_scores)

    # # FPR95% calculation
    # def calculate_fpr95(y_true, y_scores):
    #     fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    #     # Find the threshold where TPR is closest to 0.95
    #     idx = np.argmax(tpr >= 0.95)
    #     return fpr[idx]

    # mse_fpr95 = calculate_fpr95(labels, mse_scores)
    # lpips_fpr95 = calculate_fpr95(labels, lpips_scores)

    # # Print results
    # print(f"MSE AUC: {mse_auc:.4f}, FPR95%: {mse_fpr95:.4f}")
    # print(f"LPIPS AUC: {lpips_auc:.4f}, FPR95%: {lpips_fpr95:.4f}")
