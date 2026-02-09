# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 12:27:46 2025

@author: user
"""

import gymnasium as gym
from gymnasium import spaces
import jpype
import jpype.imports
import numpy as np

class FreeroutingJPypeEnv(gym.Env):
    """
    使用 JPype 直接控制 freerouting.jar 的 Gymnasium 環境。
    """
    def __init__(self, jar_path, dsn_file_path):
        super().__init__()


        if not jpype.isJVMStarted():
            jpype.startJVM(classpath=[jar_path])

        try:
            # 這是我們假設找到的核心控制器類別
            self.RoutingController = jpype.JClass('app.freerouting.logic.RoutingController')
        except Exception as e:
            raise ImportError(f"無法從 JAR 檔案中找到指定的 Java 類別。請檢查類別路徑是否正確。錯誤: {e}")

    
        self.controller = self.RoutingController(dsn_file_path)

        self.action_space = spaces.Discrete(100)
        self.observation_space = spaces.Box(low=0, high=255, 
                                            shape=(64, 64, 1), dtype=np.uint8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
      
        self.controller.reset()
        

        java_observation = self.controller.getObservation()
        observation = np.array(java_observation, dtype=np.uint8)
        
        info = {}
        return observation, info

    def step(self, action):

        reward = self.controller.performAction(action)
        
        # 獲取新的狀態
        java_observation = self.controller.getObservation()
        observation = np.array(java_observation, dtype=np.uint8)

        # 檢查是否結束
        done = self.controller.isFinished()
        
        terminated = bool(done)
        truncated = False
        info = {}
        
        return observation, float(reward), terminated, truncated, info

    def close(self):

        print("環境已關閉。JVM 將在主程式結束時關閉。")
        pass

    def render(self, mode='human'):
        pass

# 主程式入口，用於測試
if __name__ == '__main__':
    # 假設您的 jar 檔和設計檔路徑如下
    JAR_PATH = 'freerouting.jar'
    DSN_FILE = 'path/to/your/design.dsn'

    print("正在初始化環境...")
    env = FreeroutingJPypeEnv(jar_path=JAR_PATH, dsn_file_path=DSN_FILE)
    print("環境初始化成功！")

    obs, info = env.reset()
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"步驟 {i+1}: Action={action}, Reward={reward:.4f}, Done={terminated}")
        if terminated or truncated:
            break
            
    env.close()