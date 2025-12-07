# ======================================================================
# 实时输出：距离 / 速度 / 角度（单目标）
# 使用 Infineon BGT60TR13C + Radar SDK 3.6.x
# 算法：DopplerAlgo + DigitalBeamForming（DBF）
# 终端文本输出，不再用 matplotlib 画图
# ======================================================================

import pprint
import numpy as np

from ifxradarsdk import get_version_full
from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwSequenceChirp

from helpers.DigitalBeamForming import DigitalBeamForming
from helpers.DopplerAlgo import DopplerAlgo

C0 = 3e8  # 光速


def num_rx_antennas_from_rx_mask(rx_mask: int) -> int:
    """根据 rx_mask 计算启用的天线数量"""
    c = 0
    for i in range(32):
        if rx_mask & (1 << i):
            c += 1
    return c


def main():
    # ---------------- 雷达配置（与原 DBF demo 一样） ----------------
    num_beams = 27          # 波束数
    max_angle_degrees = 40  # 角度范围：[-40, 40] 度

    config = FmcwSimpleSequenceConfig(
        frame_repetition_time_s=0.15,      # 帧周期 0.15s（约 6.7 Hz）
        chirp_repetition_time_s=0.0005,    # Chirp 周期 0.5 ms
        num_chirps=128,                    # 每帧 chirp 数
        tdm_mimo=False,                    # 关闭 MIMO
        chirp=FmcwSequenceChirp(
            start_frequency_Hz=60e9,       # 起始频率 60 GHz
            end_frequency_Hz=61.5e9,       # 终止频率 61.5 GHz
            sample_rate_Hz=1e6,            # ADC 采样率 1 MHz
            num_samples=64,                # 每个 chirp 64 点
            rx_mask=5,                     # 启用 RX1 和 RX3
            tx_mask=1,                     # 启用 TX1
            tx_power_level=31,             # 发射功率
            lp_cutoff_Hz=500000,           # 低通 500 kHz
            hp_cutoff_Hz=80000,            # 高通 80 kHz
            if_gain_dB=33,                 # IF 增益 33 dB
        ),
    )

    # ---------------- 打开设备 & 配置采集 ----------------
    with DeviceFmcw() as device:
        print(f"Radar SDK Version: {get_version_full()}")
        print("Sensor:", device.get_sensor_type())

        sequence = device.create_simple_sequence(config)
        device.set_acquisition_sequence(sequence)

        # 获取 metrics（里面有最大距离/速度、分辨率等）
        chirp_loop = sequence.loop.sub_sequence.contents
        metrics = device.metrics_from_sequence(chirp_loop)
        print("=== Metrics from sequence ===")
        pprint.pprint(metrics)

        # ---- 距离相关参数 ----
        max_range_m = metrics.max_range_m

        # 距离分辨率
        try:
            range_res_m = metrics.range_resolution_m
        except AttributeError:
            try:
                range_res_m = metrics.range_bin_width_m
            except AttributeError:
                # 兜底：用带宽自己算
                B = config.chirp.end_frequency_Hz - config.chirp.start_frequency_Hz
                range_res_m = C0 / (2.0 * B)

        # ---- 速度分辨率 ----
        try:
            speed_res_m_s = metrics.speed_resolution_m_s
        except AttributeError:
            speed_res_m_s = metrics.max_speed_m_s / config.num_chirps

        chirp = chirp_loop.loop.sub_sequence.contents.chirp
        num_rx_antennas = num_rx_antennas_from_rx_mask(chirp.rx_mask)

        print("\n=== Derived parameters ===")
        print(f"  max_range_m        = {max_range_m:.3f} m")
        print(f"  range_resolution   = {range_res_m:.4f} m")
        print(f"  speed_resolution   = {speed_res_m_s:.4f} m/s")
        print(f"  num_rx_antennas    = {num_rx_antennas}")
        print("==========================\n")

        # ---------------- 创建 Doppler / DBF 算法实例 ----------------
        doppler = DopplerAlgo(
            config.chirp.num_samples,
            config.num_chirps,
            num_rx_antennas,
        )
        dbf = DigitalBeamForming(
            num_rx_antennas,
            num_beams=num_beams,
            max_angle_degrees=max_angle_degrees,
        )

        # 预先算好角度栅格
        angle_grid_deg = np.linspace(-max_angle_degrees, max_angle_degrees, num_beams)

        print("Start streaming... (Ctrl+C to stop)\n")

        # ---------------- 主循环：不断读取帧并计算 RVA ----------------
        while True:
            # frame: [num_rx_antennas, num_chirps, num_samples]
            frame_contents = device.get_next_frame()
            if not frame_contents:
                print("Warning: empty frame received")
                continue

            frame = frame_contents[0]

            # Range-Doppler 光谱：shape [num_samples, 2*num_chirps, num_rx]
            rd_spectrum = np.zeros(
                (
                    config.chirp.num_samples,
                    2 * config.num_chirps,
                    num_rx_antennas,
                ),
                dtype=complex,
            )

            # Range-Angle 能量图：shape [num_samples, num_beams]
            beam_range_energy = np.zeros(
                (config.chirp.num_samples, num_beams),
                dtype=float,
            )

            # ---- 1) 对每个 Rx 做 Doppler 处理 ----
            for i_ant in range(num_rx_antennas):
                # mat: [num_chirps, num_samples]
                mat = frame[i_ant, :, :]
                dfft_dbfs = doppler.compute_doppler_map(mat, i_ant)
                rd_spectrum[:, :, i_ant] = dfft_dbfs

            # ---- 2) DBF：得到 Range-Angle-Doppler 立方体 ----
            # rd_beam_formed: [num_samples, 2*num_chirps, num_beams]
            rd_beam_formed = dbf.run(rd_spectrum)

            # ---- 3) 对 Doppler 维度取范数，得到 Range-Angle 能量图 ----
            for i_beam in range(num_beams):
                doppler_i = rd_beam_formed[:, :, i_beam]  # [num_samples, 2*num_chirps]
                # 对 Doppler 维度求 L2 范数
                beam_range_energy[:, i_beam] += (
                    np.linalg.norm(doppler_i, axis=1) / np.sqrt(num_beams)
                )

            # ---- 4) 在 Range-Angle 图上找主峰 → 距离 + 角度 ----
            max_energy = np.max(beam_range_energy)
            if max_energy <= 0:
                print("No significant target detected.")
                continue

            range_idx, beam_idx = np.unravel_index(
                beam_range_energy.argmax(), beam_range_energy.shape
            )

            distance_m = range_idx * range_res_m
            angle_deg = angle_grid_deg[beam_idx]

            # ---- 5) 在这个距离 + 角度下，沿 Doppler 维度找速度峰值 ----
            doppler_spectrum = rd_beam_formed[range_idx, :, beam_idx]  # [2*num_chirps]
            doppler_power = np.abs(doppler_spectrum) ** 2

            doppler_len = doppler_power.shape[0]
            doppler_idx = int(np.argmax(doppler_power))
            doppler_center = doppler_len // 2  # 认为 0 速度在中间

            rel_idx = doppler_idx - doppler_center
            speed_m_s = -rel_idx * speed_res_m_s  # 加负号以匹配你之前方向定义

            # ---- 6) 在终端输出结果 ----
            print(
                f"R = {distance_m:5.3f} m, "
                f"V = {speed_m_s:6.3f} m/s, "
                f"A = {angle_deg:6.2f} deg"
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
