from typing import List, Dict, Any, Optional
import json
import time
import threading
import logging

from .predictor import RISCVPredictor


class PredictionServer:
    """吞吐量预测服务器类，可以用于构建API服务"""
    
    def __init__(self, model_path: str, device: Optional[str] = None, 
                cache_size: int = 1000, log_file: Optional[str] = None):
        """
        初始化预测服务器
        
        Args:
            model_path: 模型路径
            device: 推理设备
            cache_size: 预测结果缓存大小
            log_file: 日志文件路径
        """
        # 初始化预测器
        self.predictor = RISCVPredictor(model_path, device)
        
        # 设置缓存
        self.cache = {}
        self.cache_size = cache_size
        self.cache_lock = threading.Lock()
        
        # 设置日志
        self.setup_logger(log_file)
        
        # 状态追踪
        self.request_count = 0
        self.start_time = time.time()
        self.is_running = True
    
    def setup_logger(self, log_file: Optional[str] = None):
        """设置日志"""
        self.logger = logging.getLogger("PredictionServer")
        self.logger.setLevel(logging.INFO)
        
        # 控制台处理程序
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 文件处理程序（如果提供了文件路径）
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def predict(self, instructions: List[str]) -> Dict[str, Any]:
        """
        预测单个样本
        
        Args:
            instructions: 指令列表
            
        Returns:
            包含预测结果的字典
        """
        # 更新计数器
        self.request_count += 1
        
        # 尝试从缓存获取结果
        cache_key = hash(tuple(instructions))
        
        with self.cache_lock:
            if cache_key in self.cache:
                self.logger.info(f"缓存命中: {len(instructions)} 条指令")
                return {
                    "throughput": self.cache[cache_key],
                    "cached": True,
                    "instructions_count": len(instructions)
                }
        
        # 执行预测
        start_time = time.time()
        throughput = self.predictor.predict_single(instructions)
        prediction_time = time.time() - start_time
        
        # 更新缓存
        with self.cache_lock:
            if len(self.cache) >= self.cache_size:
                # 简单的LRU策略：删除任意一个
                self.cache.pop(next(iter(self.cache)))
            
            self.cache[cache_key] = throughput
        
        self.logger.info(f"预测完成: {len(instructions)} 条指令, 耗时: {prediction_time:.4f}秒")
        
        return {
            "throughput": float(throughput),
            "cached": False,
            "instructions_count": len(instructions),
            "prediction_time": prediction_time
        }
    
    def predict_batch(self, batch: List[List[str]]) -> List[Dict[str, Any]]:
        """
        批量预测
        
        Args:
            batch: 指令列表的列表
            
        Returns:
            预测结果列表
        """
        results = []
        
        for instructions in batch:
            result = self.predict(instructions)
            results.append(result)
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取服务器状态
        
        Returns:
            状态信息字典
        """
        uptime = time.time() - self.start_time
        
        return {
            "status": "running" if self.is_running else "stopped",
            "uptime": uptime,
            "uptime_formatted": self.format_time(uptime),
            "request_count": self.request_count,
            "requests_per_second": self.request_count / uptime if uptime > 0 else 0,
            "cache_size": len(self.cache),
            "cache_capacity": self.cache_size,
            "model_type": self.predictor.config_obj.model_type,
            "device": str(self.predictor.device)
        }
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """格式化时间"""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def stop(self):
        """停止服务器"""
        self.is_running = False
        self.logger.info("服务器已停止")
