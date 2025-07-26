import os
import warnings
warnings.filterwarnings("ignore")

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ================== 可配置参数 ==================
input_file = "/Users/shuzongfei/Downloads/2025-06-01T00_00_00_cn_flatted.nc"  # 修改为你的 .nc 路径
output_dir = "./weather_frames"
target_lat, target_lon = 31.23, 121.47
max_frames = 265  # 你想导出的帧数上限
os.makedirs(output_dir, exist_ok=True)

# ================== 一些小工具函数 ==================
def pick_name(ds, candidates):
    """从若干候选名里挑出 ds 里真正存在的坐标名"""
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    raise KeyError(f"在数据集中找不到这些坐标名: {candidates}")

def to_numpy(var):
    """把 DataArray 安全转成 numpy，一律返回 np.ndarray（可能含 NaN）"""
    if var is None:
        return None
    return np.asarray(var.values)

def exists(ds, name):
    return name in ds.data_vars

# ================== 1. 读取数据 ==================
ds = xr.open_dataset(input_file)

# 识别经纬度名字
lat_name = pick_name(ds, ["lat", "latitude", "Latitude"])
lon_name = pick_name(ds, ["lon", "longitude", "Longitude"])
time_name = pick_name(ds, ["time", "Time"])

# 若经度是 0~360，处理下 target_lon
lon_vals = ds[lon_name].values
if np.nanmax(lon_vals) > 180:
    sel_lon = (target_lon + 360) % 360
else:
    sel_lon = target_lon

# ================== 2. 选点（最近格点） ==================
# 直接对整个 Dataset 做 sel，方便统一对齐时间轴
ds_point = ds.sel({lat_name: target_lat, lon_name: sel_lon}, method="nearest")

# ================== 3. 取出变量，做安全转换 ==================
t2m = ds_point.get("t2m")
u10 = ds_point.get("u10")
v10 = ds_point.get("v10")
tp6h = ds_point.get("tp6h")  # 可能不存在

time = pd.to_datetime(to_numpy(ds_point[time_name]))
n_time_ds = len(time)

# 计算每个变量的有效长度（如果没有该变量就不计）
def var_len(da):
    return len(da[time_name]) if da is not None and time_name in da.dims else np.inf

lens = [n_time_ds]
for da in [t2m, u10, v10, tp6h]:
    if da is not None:
        lens.append(var_len(da))

# 所有变量共同能覆盖的最小长度
n_frames = min(int(min(lens)), max_frames)

if n_frames == 0:
    raise RuntimeError("没有可用的时间步长，请检查文件内容。")

# 截断时间
time = time[:n_frames]

# 提取并对齐到 n_frames
def safe_series(da, name):
    if da is None:
        print(f"⚠️ 变量 {name} 不存在，将使用 NaN 填充。")
        return np.full(n_frames, np.nan, dtype=float)
    arr = to_numpy(da)
    if arr.shape[0] < n_frames:
        print(f"⚠️ 变量 {name} 只有 {arr.shape[0]} 步，将用 NaN 填充到 {n_frames}。")
        pad = np.full(n_frames - arr.shape[0], np.nan, dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=0)
    return arr[:n_frames]

t2m_series = safe_series(t2m, "t2m")
u_series   = safe_series(u10, "u10")
v_series   = safe_series(v10, "v10")
tp_series  = safe_series(tp6h, "tp6h")

# 单位处理：t2m -> 摄氏度（假设原本是 K）
if not np.isnan(t2m_series).all():
    # 简单判断：如果平均温度 > 100，基本可以确认是开尔文
    if np.nanmean(t2m_series) > 100:
        t2m_series = t2m_series - 273.15

# 风速
wind_speed = np.sqrt(u_series ** 2 + v_series ** 2)

# ================== 4. 作图 ==================
for i in range(n_frames):
    ts = time[i]
    temp = t2m_series[i]
    wind = wind_speed[i]
    rain = tp_series[i]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title(f"上海 ({target_lat}, {target_lon}) | {ts.strftime('%Y-%m-%d %H:%M')}", fontsize=12)

    # 温度
    if not np.isnan(temp):
        ax.text(0.05, 0.82, f"温度: {temp:.1f}℃", color="red", fontsize=12, transform=ax.transAxes)
    else:
        ax.text(0.05, 0.82, f"温度: NaN", color="red", fontsize=12, transform=ax.transAxes)

    # 风速
    if not np.isnan(wind):
        ax.bar(0.35, wind, width=0.2, label=f"风速: {wind:.1f} m/s")
    else:
        ax.bar(0.35, 0, width=0.2, label=f"风速: NaN")

    # 降水
    if not np.isnan(rain):
        ax.bar(0.65, rain, width=0.2, color="green", label=f"降水: {rain:.1f} mm")
        max_y = np.nanmax([wind, rain, 30]) + 5
    else:
        max_y = np.nanmax([wind, 30]) + 5

    ax.set_xlim(0, 1)
    ax.set_ylim(0, max_y)
    ax.set_xticks([])
    ax.set_ylabel("值")
    ax.legend(loc="upper right")

    out_path = os.path.join(output_dir, f"frame_{i:03d}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

print(f"✅ 成功生成 {n_frames} 张图像，输出目录：{output_dir}")
