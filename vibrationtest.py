#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import RPi.GPIO as GPIO
import time
import math

# ===== 用户可配置区域 =====
MOTOR_PIN = 18          # 用 BCM 编号的 GPIO18，记得按自己接线修改
PWM_FREQ = 200          # PWM 频率，单位 Hz，一般 100~300 都可以
PERIOD_SECONDS = 5.0    # 震动强度从弱 -> 强 -> 弱 完整周期，单位秒
# ==========================

def setup():
    # 使用 BCM 编号
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(MOTOR_PIN, GPIO.OUT)
    
    # 初始化 PWM：频率为 PWM_FREQ，初始占空比 0（不震动）
    pwm = GPIO.PWM(MOTOR_PIN, PWM_FREQ)
    pwm.start(0)
    return pwm

def main():
    pwm = setup()
    print("开始震动，按 Ctrl + C 停止。")

    start_time = time.time()
    try:
        while True:
            # 已经过的时间
            t = time.time() - start_time

            # 计算当前相位: 0 ~ 2π
            # 一个 PERIOD_SECONDS 完成一个“由弱到强再回到弱”的循环
            phase = 2 * math.pi * (t % PERIOD_SECONDS) / PERIOD_SECONDS

            # 正弦波：-1 ~ 1 映射到 0 ~ 100 占空比
            # (sin(phase) + 1) / 2  -> 0 ~ 1
            duty = (math.sin(phase) + 1) / 2 * 100.0

            # 为了避免极端值抖动，可以稍微限制一下范围
            duty = max(0, min(100, duty))

            # 更新占空比
            pwm.ChangeDutyCycle(duty)

            # 打印调试信息（需要的话）
            # print(f"duty = {duty:.1f}%")

            # 稍微睡一会儿，降低 CPU 占用
            time.sleep(0.02)  # 20ms，50次/秒更新
    except KeyboardInterrupt:
        print("\n停止震动，清理 GPIO 设置...")
    finally:
        pwm.stop()
        GPIO.cleanup()

if __name__ == "__main__":
    main()
