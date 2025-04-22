import matplotlib.pyplot as plt
import pickle

# 加载数据
with open('pkl_data/phm_dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

# 提取Bearing1_1的数据
bearing_data = dataset['Bearing1_1']['data']  # 形状 (2803, 2560, 2)

# 处理数据 ----------------------------------------------------------------
# 提取通道0数据并展平
channel_0 = bearing_data[:, :, 0].flatten()  # 形状 (2803*2560,)

# 可视化设置 -------------------------------------------------------------
plt.figure(figsize=(15, 6), dpi=100)

# 绘制完整生命周期信号
plt.plot(channel_0)

# 网格和样式调整
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
