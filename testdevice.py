import numpy as np
from ifxradarsdk import get_version
from ifxradarsdk.fmcw import DeviceFmcw

with DeviceFmcw() as device:
    print("Radar SDK Version:", get_version())
    print("Board UUID:", device.get_board_uuid())
    print("Sensor type:", device.get_sensor_type())

    seq = device.get_acquisition_sequence()
    print("Acquisition sequence loaded.")

    for frame_number in range(3):
        frame_contents = device.get_next_frame()
        print(f"=== Frame {frame_number} ===")
        for frame in frame_contents:
            num_rx = frame.shape[0]
            print("num_rx =", num_rx, "frame shape =", frame.shape)
