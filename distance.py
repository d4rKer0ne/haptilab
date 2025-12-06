from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwMetrics
from helpers.DistanceAlgo import DistanceAlgo  # 在 SDK 的 examples 或 helpers 里

with DeviceFmcw() as device:
    # 1. 配 metrics（最大距离、分辨率、最大速度等）
    metrics = FmcwMetrics(
        range_resolution_m=0.1,
        max_range_m=2.0,
        max_speed_m_s=3.0,
        speed_resolution_m_s=0.2,
        center_frequency_Hz=60_750_000_000,
    )

    # 2. 用 simple sequence + metrics 算出 chirp 参数
    simple_cfg = FmcwSimpleSequenceConfig()
    sequence = device.create_simple_sequence(simple_cfg)
    chirp_loop = sequence.loop.sub_sequence.contents
    device.sequence_from_metrics(metrics, chirp_loop)

    # 3. 设置采样率、增益等参数（按示例来）
    chirp = chirp_loop.loop.sub_sequence.contents.chirp
    chirp.sample_rate_Hz = 1_000_000
    chirp.rx_mask = 0b111    # 3 个 Rx 都开
    chirp.tx_mask = 1
    chirp.tx_power_level = 31
    chirp.if_gain_dB = 33

    device.set_acquisition_sequence(sequence)

    # 4. 初始化距离算法
    algo = DistanceAlgo(chirp, chirp_loop.loop.num_repetitions)

    # 5. 主循环：获取 frame → 算距离峰值
    while True:
        frame_contents = device.get_next_frame()
        frame = frame_contents[0]            # 单帧
        antenna_samples = frame[0, :, :]     # 只用第一个 Rx（示例里常这么搞）
        distance_peak, _ = algo.compute_distance(antenna_samples)
        print(f"distance = {distance_peak:.3f} m")
