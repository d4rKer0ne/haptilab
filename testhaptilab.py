# # =====================================================================
# # 实时输出：距离 / 速度 / 角度（单目标）+ 简单 GUI 显示
# # 适配：Infineon BGT60TR13C + Radar SDK 3.6.x
# # =====================================================================

# import numpy as np
# import tkinter as tk
# from tkinter import ttk

# from ifxradarsdk import get_version
# from ifxradarsdk.fmcw import DeviceFmcw
# from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwSequenceChirp
# from ifxradarsdk.common.exceptions import ErrorTimeout  # <--- 新增：专门捕获超时

# C0 = 3e8  # 光速 (m/s)


# # ========================= 雷达配置 =========================
# config = FmcwSimpleSequenceConfig(
#     frame_repetition_time_s=0.15,    # 每帧 0.15 s，大约 ~6.7 Hz
#     chirp_repetition_time_s=0.001,   # chirp 间隔 1 ms
#     num_chirps=32,                   # 每帧 chirp 数，影响速度分辨率
#     tdm_mimo=False,                  # BGT60TR13C 单 Tx
#     chirp=FmcwSequenceChirp(
#         start_frequency_Hz=59e9,
#         end_frequency_Hz=61e9,       # 带宽 2 GHz
#         sample_rate_Hz=2e6,          # ADC 采样率
#         num_samples=256,             # 每个 chirp 采 256 点
#         rx_mask=0b111,               # 三个 Rx
#         tx_mask=0b1,                 # Tx1
#         tx_power_level=31,
#         lp_cutoff_Hz=500_000,
#         hp_cutoff_Hz=80_000,
#         if_gain_dB=30,
#     ),
# )


# def create_device_and_sequence():
#     """打开雷达 + 下发采集序列 + 计算一些后面用到的参数。"""
#     device = DeviceFmcw()

#     print("Radar SDK Version:", get_version())
#     print("UUID:", device.get_board_uuid())
#     print("Sensor:", device.get_sensor_type())

#     # 创建采集序列并下发
#     sequence = device.create_simple_sequence(config)
#     device.set_acquisition_sequence(sequence)

#     # 显式启动采集（有的版本必须这么做）
#     try:
#         device.start_acquisition()
#     except Exception:
#         # 部分版本没有这个函数，忽略即可
#         pass

#     chirp_cfg = config.chirp
#     B = chirp_cfg.end_frequency_Hz - chirp_cfg.start_frequency_Hz   # 带宽
#     fs = chirp_cfg.sample_rate_Hz
#     num_samples = chirp_cfg.num_samples
#     num_chirps = config.num_chirps
#     center_freq = (chirp_cfg.start_frequency_Hz + chirp_cfg.end_frequency_Hz) / 2.0

#     # 距离分辨率 ≈ c / (2B)
#     range_bin_width = C0 / (2.0 * B)
#     num_range_bins = num_samples // 2

#     # 速度分辨率 Δv = λ / (2 * N_chirps * T_rep)
#     lam = C0 / center_freq
#     T_rep = config.chirp_repetition_time_s
#     velocity_bin_width = lam / (2.0 * num_chirps * T_rep)

#     params = {
#         "B": B,
#         "fs": fs,
#         "num_samples": num_samples,
#         "num_chirps": num_chirps,
#         "range_bin_width": range_bin_width,
#         "num_range_bins": num_range_bins,
#         "velocity_bin_width": velocity_bin_width,
#         "lam": lam,
#     }

#     print("=== 推导参数 ===")
#     print(f"  range_bin_width ≈ {range_bin_width:.4f} m")
#     print(f"  velocity_bin_width ≈ {velocity_bin_width:.4f} m/s")
#     print(f"  λ ≈ {lam*1000:.2f} mm")
#     print("================\n")

#     return device, params


# # ========================= GUI + 实时更新 =========================

# class RadarGUI:
#     def __init__(self, root, device, params):
#         self.root = root
#         self.device = device
#         self.params = params

#         self.root.title("Radar RVA Monitor (Distance / Velocity / Angle)")
#         self.root.geometry("520x260")

#         label_font = ("Segoe UI", 14)
#         value_font = ("Segoe UI", 22, "bold")

#         frame = ttk.Frame(root, padding=20)
#         frame.pack(fill=tk.BOTH, expand=True)

#         # 距离
#         ttk.Label(frame, text="Distance (m):", font=label_font).grid(
#             row=0, column=0, sticky="w", pady=5
#         )
#         self.distance_var = tk.StringVar(value="--")
#         ttk.Label(frame, textvariable=self.distance_var, font=value_font).grid(
#             row=0, column=1, sticky="w", pady=5
#         )

#         # 速度
#         ttk.Label(frame, text="Velocity (m/s):", font=label_font).grid(
#             row=1, column=0, sticky="w", pady=5
#         )
#         self.velocity_var = tk.StringVar(value="--")
#         ttk.Label(frame, textvariable=self.velocity_var, font=value_font).grid(
#             row=1, column=1, sticky="w", pady=5
#         )

#         # 角度
#         ttk.Label(frame, text="Angle (deg):", font=label_font).grid(
#             row=2, column=0, sticky="w", pady=5
#         )
#         self.angle_var = tk.StringVar(value="--")
#         ttk.Label(frame, textvariable=self.angle_var, font=value_font).grid(
#             row=2, column=1, sticky="w", pady=5
#         )

#         # 状态栏
#         self.status_var = tk.StringVar(value="Running...")
#         ttk.Label(frame, textvariable=self.status_var, font=("Segoe UI", 10)).grid(
#             row=3, column=0, columnspan=2, sticky="w", pady=10
#         )

#         frame.columnconfigure(0, weight=1)
#         frame.columnconfigure(1, weight=2)

#         # 根据帧周期设置刷新间隔：稍微比 frame_repetition_time 小一点
#         self.update_interval_ms = int(config.frame_repetition_time_s * 1000 * 0.9)
#         if self.update_interval_ms < 50:
#             self.update_interval_ms = 50

#         self.root.after(self.update_interval_ms, self.update_radar)
#         self.root.protocol("WM_DELETE_WINDOW", self.on_close)

#     def update_radar(self):
#         try:
#             # 取一帧
#             frame_contents = self.device.get_next_frame()
#         except ErrorTimeout:
#             # 采集还没完成，过一会再取
#             self.status_var.set("Waiting for next frame (timeout)...")
#             self.root.after(self.update_interval_ms, self.update_radar)
#             return
#         except Exception as e:
#             # 其它错误才显示出来
#             self.status_var.set(f"Error: {e}")
#             self.root.after(self.update_interval_ms, self.update_radar)
#             return

#         if not frame_contents:
#             self.status_var.set("No frame received.")
#             self.root.after(self.update_interval_ms, self.update_radar)
#             return

#         frame = frame_contents[0]  # [Rx, Chirp, Sample]
#         num_rx, num_chirps, num_samples = frame.shape

#         # ---- 1) 去 DC ----
#         frame = frame - np.mean(frame, axis=2, keepdims=True)

#         # ---- 2) Range FFT ----
#         window_range = np.hanning(num_samples)[np.newaxis, np.newaxis, :]
#         frame_win = frame * window_range
#         range_fft = np.fft.fft(frame_win, axis=2)
#         range_fft = range_fft[:, :, : num_samples // 2]
#         num_range_bins = range_fft.shape[2]

#         # ---- 3) Doppler FFT ----
#         window_dopp = np.hanning(num_chirps)[np.newaxis, :, np.newaxis]
#         range_fft_win = range_fft * window_dopp
#         rd_cube = np.fft.fft(range_fft_win, axis=1)        # [Rx, Doppler, Range]
#         rd_cube = np.fft.fftshift(rd_cube, axes=1)

#         # ---- 4) Range-Doppler 功率图 ----
#         rd_power = np.sum(np.abs(rd_cube) ** 2, axis=0)    # [Doppler, Range]

#         # === 在这里屏蔽近距离杂波 ===
#         range_bin_width = self.params["range_bin_width"]
#         min_range_m = 0.30          # 忽略 0.30 m 以内
#         min_bin = int(min_range_m / range_bin_width)
#         rd_power[:, :min_bin] = 0   # 把近距离 bin 的能量清空

#         # ---- 5) 找峰值 ----
#         doppler_idx, range_idx = np.unravel_index(
#             np.argmax(rd_power), rd_power.shape
#         )

#         # ---- 6) 计算距离 / 速度 ----
#         range_bin_width = self.params["range_bin_width"]
#         velocity_bin_width = self.params["velocity_bin_width"]

#         distance_m = range_idx * range_bin_width

#         doppler_center = num_chirps // 2
#         rel_doppler_idx = doppler_idx - doppler_center
#         speed_m_s = -rel_doppler_idx * velocity_bin_width

#         # ---- 7) 角度估计（2 Rx 相位差）----
#         angle_deg = None
#         if num_rx >= 2:
#             lam = self.params["lam"]
#             steering_vec = rd_cube[:, doppler_idx, range_idx]  # [num_rx]
#             phase0 = np.angle(steering_vec[0])
#             phase1 = np.angle(steering_vec[1])
#             dphi = phase1 - phase0
#             dphi = np.arctan2(np.sin(dphi), np.cos(dphi))

#             # 根据实际天线间距调整；这里先用 λ/2 近似
#             RX_SPACING_M = lam / 2.0
#             sin_theta = dphi * lam / (2.0 * np.pi * RX_SPACING_M)
#             sin_theta = np.clip(sin_theta, -1.0, 1.0)
#             angle_rad = np.arcsin(sin_theta)
#             angle_deg = np.degrees(angle_rad)

#         # ---- 8) 更新 GUI ----
#         self.distance_var.set(f"{distance_m:5.3f}")
#         self.velocity_var.set(f"{speed_m_s:6.3f}")
#         self.angle_var.set("N/A" if angle_deg is None else f"{angle_deg:6.2f}")
#         self.status_var.set("Running...")

#         self.root.after(self.update_interval_ms, self.update_radar)

#     def on_close(self):
#         try:
#             try:
#                 self.device.stop_acquisition()
#             except Exception:
#                 pass
#         finally:
#             self.root.destroy()


# def main():
#     device, params = create_device_and_sequence()
#     root = tk.Tk()
#     gui = RadarGUI(root, device, params)
#     try:
#         root.mainloop()
#     finally:
#         print("GUI closed, program exit.")


# if __name__ == "__main__":
#     main()
# ===========================================================================
# Copyright (C) 2022 Infineon Technologies AG
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ===========================================================================

import pprint
import matplotlib.pyplot as plt
import numpy as np

from ifxradarsdk import get_version_full
from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwSequenceChirp
from helpers.DigitalBeamForming import *
from helpers.DopplerAlgo import *

C0 = 3e8  # 光速


def num_rx_antennas_from_rx_mask(rx_mask):
    # popcount for rx_mask
    c = 0
    for i in range(32):
        if rx_mask & (1 << i):
            c += 1
    return c


class LivePlot:
    def __init__(self, max_angle_degrees: float, max_range_m: float):
        # max_angle_degrees: maximum supported angle
        # max_range_m:   maximum supported range
        self.h = None
        self.max_angle_degrees = max_angle_degrees
        self.max_range_m = max_range_m

        plt.ion()

        self._fig, self._ax = plt.subplots(nrows=1, ncols=1)

        self._fig.canvas.manager.set_window_title("Range-Angle-Map using Digital Beam Forming")
        self._fig.canvas.mpl_connect('close_event', self.close)
        self._is_window_open = True

    def _draw_first_time(self, data: np.ndarray):
        # First time draw

        minmin = -60
        maxmax = 0

        self.h = self._ax.imshow(
            data,
            vmin=minmin, vmax=maxmax,
            cmap='viridis',
            extent=(-self.max_angle_degrees,
                    self.max_angle_degrees,
                    0,
                    self.max_range_m),
            origin='lower')

        self._ax.set_xlabel("angle (degrees)")
        self._ax.set_ylabel("distance (m)")
        self._ax.set_aspect("auto")

        self._fig.subplots_adjust(right=0.8)
        cbar_ax = self._fig.add_axes([0.85, 0.0, 0.03, 1])

        cbar = self._fig.colorbar(self.h, cax=cbar_ax)
        cbar.ax.set_ylabel("magnitude (a.u.)")

    def _draw_next_time(self, data: np.ndarray):
        # Update data for each antenna

        self.h.set_data(data)

    def draw(self, data: np.ndarray, title: str):
        if self._is_window_open:
            if self.h:
                self._draw_next_time(data)
            else:
                self._draw_first_time(data)
            self._ax.set_title(title)

            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()

    def close(self, event=None):
        if not self.is_closed():
            self._is_window_open = False
            plt.close(self._fig)
            plt.close('all')
            print('Application closed!')

    def is_closed(self):
        return not self._is_window_open


# -------------------------------------------------
# Main logic
# -------------------------------------------------
if __name__ == '__main__':
    num_beams = 27  # number of beams
    max_angle_degrees = 40  # maximum angle, angle ranges from -40 to +40 degrees

    config = FmcwSimpleSequenceConfig(
        frame_repetition_time_s=0.15,  # Frame repetition time 0.15s (frame rate of 6.667Hz)
        chirp_repetition_time_s=0.0005,  # Chirp repetition time (or pulse repetition time) of 0.5ms
        num_chirps=128,  # chirps per frame
        tdm_mimo=False,  # MIMO disabled
        chirp=FmcwSequenceChirp(
            start_frequency_Hz=60e9,  # start frequency: 60 GHz
            end_frequency_Hz=61.5e9,  # end frequency: 61.5 GHz
            sample_rate_Hz=1e6,  # ADC sample rate of 1MHz
            num_samples=64,  # 64 samples per chirp
            rx_mask=5,  # RX antennas 1 and 3 activated
            tx_mask=1,  # TX antenna 1 activated
            tx_power_level=31,  # TX power level of 31
            lp_cutoff_Hz=500000,  # Anti-aliasing cutoff frequency of 500kHz
            hp_cutoff_Hz=80000,  # 80kHz cutoff frequency for high-pass filter
            if_gain_dB=33,  # 33dB if gain
        )
    )

    with DeviceFmcw() as device:
        print(f"Radar SDK Version: {get_version_full()}")
        print("Sensor: " + str(device.get_sensor_type()))

        # configure device
        sequence = device.create_simple_sequence(config)
        device.set_acquisition_sequence(sequence)

        # get metrics and print them
        chirp_loop = sequence.loop.sub_sequence.contents
        metrics = device.metrics_from_sequence(chirp_loop)
        pprint.pprint(metrics)

        # ---- 这里提取距离/速度分辨率，用于后续数值计算 ----
        # 最大距离
        max_range_m = metrics.max_range_m

        # 距离分辨率（有的版本字段叫 range_resolution_m，有的叫 range_bin_width_m）
        try:
            range_res_m = metrics.range_resolution_m
        except AttributeError:
            try:
                range_res_m = metrics.range_bin_width_m
            except AttributeError:
                # 兜底：自己算
                B = config.chirp.end_frequency_Hz - config.chirp.start_frequency_Hz
                range_res_m = C0 / (2.0 * B)

        # 速度分辨率
        try:
            speed_res_m_s = metrics.speed_resolution_m_s
        except AttributeError:
            # 兜底：用最大速度除以 chirp 数
            speed_res_m_s = metrics.max_speed_m_s / config.num_chirps

        chirp = chirp_loop.loop.sub_sequence.contents.chirp
        num_rx_antennas = num_rx_antennas_from_rx_mask(chirp.rx_mask)

        # Create objects for Range-Doppler, Digital Beam Forming, and plotting.
        doppler = DopplerAlgo(config.chirp.num_samples, config.num_chirps, num_rx_antennas)
        dbf = DigitalBeamForming(num_rx_antennas, num_beams=num_beams, max_angle_degrees=max_angle_degrees)
        plot = LivePlot(max_angle_degrees, max_range_m)

        # 预先算好角度栅格，后面直接索引
        angle_grid_deg = np.linspace(-max_angle_degrees, max_angle_degrees, num_beams)

        while not plot.is_closed():
            # frame has dimension num_rx_antennas x num_chirps_per_frame x num_samples_per_chirp
            frame_contents = device.get_next_frame()
            frame = frame_contents[0]

            rd_spectrum = np.zeros((config.chirp.num_samples,
                                    2 * config.num_chirps,
                                    num_rx_antennas),
                                   dtype=complex)

            beam_range_energy = np.zeros((config.chirp.num_samples, num_beams))

            for i_ant in range(num_rx_antennas):  # For each antenna
                # Current RX antenna (num_samples_per_chirp x num_chirps_per_frame)
                mat = frame[i_ant, :, :]

                # Compute Doppler spectrum（Range-Doppler for this antenna）
                dfft_dbfs = doppler.compute_doppler_map(mat, i_ant)
                rd_spectrum[:, :, i_ant] = dfft_dbfs

            # Compute Range-Angle map
            rd_beam_formed = dbf.run(rd_spectrum)   # shape: [num_samples, 2*num_chirps, num_beams]

            for i_beam in range(num_beams):
                doppler_i = rd_beam_formed[:, :, i_beam]  # [num_samples, 2*num_chirps]
                beam_range_energy[:, i_beam] += np.linalg.norm(doppler_i, axis=1) / np.sqrt(num_beams)

            # Maximum energy in Range-Angle map
            max_energy = np.max(beam_range_energy)

            # Rescale map to better capture the peak
            scale = 150
            beam_range_energy = scale * (beam_range_energy / max_energy - 1)

            # ---- 1) 用 Range–Angle 图找“主目标”的距离 + 角度 ----
            range_idx, beam_idx = np.unravel_index(
                beam_range_energy.argmax(), beam_range_energy.shape
            )
            angle_degrees = angle_grid_deg[beam_idx]
            distance_m = range_idx * range_res_m

            # ---- 2) 在该距离 + 该角度下，沿 Doppler 维度找最大谱峰 → 速度 ----
            doppler_spectrum = rd_beam_formed[range_idx, :, beam_idx]  # [2*num_chirps]
            doppler_power = np.abs(doppler_spectrum) ** 2

            doppler_idx = int(np.argmax(doppler_power))
            doppler_len = doppler_power.shape[0]
            doppler_center = doppler_len // 2  # 假设 0 速度在中间（DopplerAlgo 一般内部 fftshift）

            rel_idx = doppler_idx - doppler_center
            speed_m_s = -rel_idx * speed_res_m_s  # 带符号速度

            # 控制台输出
            print(
                f"Range = {distance_m:5.3f} m, "
                f"Speed = {speed_m_s:6.3f} m/s, "
                f"Angle = {angle_degrees:6.2f} deg"
            )

            # And plot...
            title = (
                f"Range-Angle map using DBF | "
                f"R={distance_m:4.2f} m, "
                f"V={speed_m_s:5.2f} m/s, "
                f"Az={angle_degrees:+05.1f} deg"
            )
            plot.draw(beam_range_energy, title)

        plot.close()

