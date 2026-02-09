# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 16:01:34 2025

@author: user
"""

print("--- 正在測試舊版 Gym ---")
try:
    import gym
    print("正在嘗試 gym.make('HalfCheetah-v4')...")
    env = gym.make("HalfCheetah-v4")
    print("✅ 成功！舊版 Gym 找到了 'HalfCheetah-v4'。")
    env.close()
except Exception as e:
    print(f"❌ 舊版 Gym 失敗！錯誤：{e}")

print("\n" + "="*30 + "\n")

print("--- 正在測試新版 Gymnasium ---")
try:
    import gymnasium
    print("正在嘗試 gymnasium.make('HalfCheetah-v4')...")
    env = gymnasium.make("HalfCheetah-v4")
    print("✅ 成功！新版 Gymnasium 找到了 'HalfCheetah-v4'。")
    env.close()
except Exception as e:
    print(f"❌ 新版 Gymnasium 失敗！錯誤：{e}")