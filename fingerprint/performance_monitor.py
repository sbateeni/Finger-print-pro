import time
import psutil
import platform
import threading
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.steps = []
        self.current_step = None
        self.estimated_completion_time = None
        self.is_monitoring = False
        self.monitor_thread = None
        self.system_info = self._get_system_info()
        self.total_steps = 0
        
    def _get_system_info(self):
        """الحصول على معلومات النظام"""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'processor': platform.processor()
        }
    
    def start_monitoring(self, total_steps):
        """بدء المراقبة"""
        self.start_time = time.time()
        self.steps = []
        self.is_monitoring = True
        self.total_steps = total_steps
        self.monitor_thread = threading.Thread(target=self._monitor_performance)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info(f"بدء مراقبة الأداء - عدد الخطوات المتوقعة: {total_steps}")
        return self.start_time
    
    def _monitor_performance(self):
        """مراقبة أداء النظام في خيط منفصل"""
        while self.is_monitoring:
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            
            # حساب استخدام CPU والذاكرة
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            # تحديث الوقت المتبقي المقدر
            if self.current_step and len(self.steps) > 0:
                avg_time_per_step = elapsed_time / len(self.steps)
                remaining_steps = self.total_steps - len(self.steps)
                self.estimated_completion_time = current_time + (avg_time_per_step * remaining_steps)
            
            # تسجيل المعلومات
            logger.info(f"CPU: {cpu_percent}%, Memory: {memory_percent}%, "
                       f"Elapsed: {elapsed_time:.2f}s, Steps: {len(self.steps)}")
            
            time.sleep(1)  # تحديث كل ثانية
    
    def add_step(self, step_name, status="in_progress"):
        """إضافة خطوة جديدة"""
        step_time = time.time()
        step_info = {
            'name': step_name,
            'start_time': step_time,
            'status': status,
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent
        }
        self.steps.append(step_info)
        self.current_step = step_name
        logger.info(f"إضافة خطوة: {step_name} - الحالة: {status}")
        return step_info
    
    def update_step(self, step_name, status="completed"):
        """تحديث حالة الخطوة"""
        for step in self.steps:
            if step['name'] == step_name:
                step['status'] = status
                step['end_time'] = time.time()
                step['duration'] = step['end_time'] - step['start_time']
                logger.info(f"تحديث خطوة: {step_name} - الحالة: {status} - "
                          f"المدة: {step['duration']:.2f}s")
                break
    
    def stop_monitoring(self):
        """إيقاف المراقبة"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        end_time = time.time()
        total_duration = end_time - self.start_time
        
        # تجميع إحصائيات نهائية
        stats = {
            'total_duration': total_duration,
            'total_steps': len(self.steps),
            'system_info': self.system_info,
            'steps': self.steps
        }
        
        logger.info(f"انتهت المراقبة - المدة الكلية: {total_duration:.2f}s")
        return stats
    
    def get_current_status(self):
        """الحصول على حالة المراقبة الحالية"""
        if not self.is_monitoring:
            return None
            
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        return {
            'elapsed_time': elapsed_time,
            'estimated_completion_time': self.estimated_completion_time,
            'current_step': self.current_step,
            'completed_steps': len(self.steps),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'steps': self.steps
        } 