import time
import psutil
import platform
import threading
import logging
from datetime import datetime
import os
import json
from typing import Dict, List, Optional
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.steps = []
        self.current_step = None
        self.total_steps = 0
        self.completed_steps = 0
        self.errors = []
        self.warnings = []
        self.resource_usage = {
            'cpu': deque(maxlen=100),
            'memory': deque(maxlen=100),
            'disk_io': deque(maxlen=100),
            'network_io': deque(maxlen=100)
        }
        self.monitoring_thread = None
        self.is_monitoring = False
        self.log_file = 'performance_logs.json'
        self.stats_file = 'performance_stats.json'
        self.log_dir = 'logs'
        
        # إنشاء مجلد السجلات إذا لم يكن موجوداً
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self._setup_logging()
        self.system_info = self._get_system_info()
        
    def _setup_logging(self):
        """
        إعداد نظام التسجيل
        """
        try:
            if not os.path.exists('logs'):
                os.makedirs('logs')
            self.log_file = os.path.join('logs', self.log_file)
        except Exception as e:
            logger.error(f'خطأ في إعداد نظام التسجيل: {str(e)}')

    def _get_system_info(self):
        """الحصول على معلومات النظام"""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'processor': platform.processor()
        }
    
    def start_monitoring(self, total_steps: int = 0) -> None:
        """
        بدء مراقبة الأداء
        """
        try:
            self.start_time = time.time()
            self.total_steps = total_steps
            self.completed_steps = 0
            self.steps = []
            self.errors = []
            self.warnings = []
            self.is_monitoring = True
            
            # بدء مراقبة الموارد في خيط منفصل
            self.monitoring_thread = threading.Thread(target=self._monitor_resources)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
            logger.info('بدأت مراقبة الأداء')
            
        except Exception as e:
            logger.error(f'خطأ في بدء المراقبة: {str(e)}')
            raise

    def stop_monitoring(self) -> Dict:
        """
        إيقاف مراقبة الأداء وإرجاع الإحصائيات
        """
        try:
            self.end_time = time.time()
            self.is_monitoring = False
            
            if self.monitoring_thread:
                self.monitoring_thread.join()
            
            stats = self._calculate_statistics()
            self._save_logs(stats)
            
            logger.info('تم إيقاف مراقبة الأداء')
            return stats
            
        except Exception as e:
            logger.error(f'خطأ في إيقاف المراقبة: {str(e)}')
            raise

    def add_step(self, name: str, status: str = 'in_progress') -> None:
        """
        إضافة خطوة جديدة
        """
        try:
            step = {
                'name': name,
                'start_time': time.time(),
                'status': status,
                'end_time': None,
                'duration': None,
                'resources': {
                    'cpu': psutil.cpu_percent(),
                    'memory': psutil.virtual_memory().percent
                }
            }
            self.steps.append(step)
            self.current_step = step
            logger.info(f'تمت إضافة خطوة: {name}')
            
        except Exception as e:
            logger.error(f'خطأ في إضافة خطوة: {str(e)}')
            raise

    def update_step(self, name: str, status: str = 'completed') -> None:
        """
        تحديث حالة الخطوة
        """
        try:
            for step in self.steps:
                if step['name'] == name:
                    step['end_time'] = time.time()
                    step['duration'] = step['end_time'] - step['start_time']
                    step['status'] = status
                    self.completed_steps += 1
                    logger.info(f'تم تحديث خطوة: {name} - الحالة: {status}')
                    break
            
        except Exception as e:
            logger.error(f'خطأ في تحديث الخطوة: {str(e)}')
            raise

    def add_error(self, error: str):
        """
        إضافة خطأ
        """
        try:
            self.errors.append({
                'message': error,
                'timestamp': datetime.now().isoformat(),
                'step': self.current_step['name'] if self.current_step else None
            })
            logger.error(f'تمت إضافة خطأ: {error}')
            
        except Exception as e:
            logger.error(f'خطأ في إضافة خطأ: {str(e)}')
            raise

    def add_warning(self, warning: str):
        """
        إضافة تحذير
        """
        try:
            self.warnings.append({
                'message': warning,
                'timestamp': datetime.now().isoformat(),
                'step': self.current_step['name'] if self.current_step else None
            })
            logger.warning(f'تمت إضافة تحذير: {warning}')
            
        except Exception as e:
            logger.error(f'خطأ في إضافة تحذير: {str(e)}')
            raise

    def get_current_status(self) -> Optional[Dict]:
        """
        الحصول على الحالة الحالية
        """
        try:
            if not self.is_monitoring:
                return None

            current_time = time.time()
            elapsed_time = current_time - self.start_time

            return {
                'elapsed_time': elapsed_time,
                'completed_steps': self.completed_steps,
                'total_steps': self.total_steps,
                'current_step': self.current_step['name'] if self.current_step else None,
                'progress': (self.completed_steps / self.total_steps) * 100 if self.total_steps > 0 else 0,
                'resource_usage': {
                    'cpu': psutil.cpu_percent(),
                    'memory': psutil.virtual_memory().percent
                }
            }
            
        except Exception as e:
            logger.error(f'خطأ في الحصول على الحالة الحالية: {str(e)}')
            raise

    def _monitor_resources(self) -> None:
        """
        مراقبة استخدام الموارد
        """
        try:
            while self.is_monitoring:
                # مراقبة استخدام CPU
                cpu_percent = psutil.cpu_percent(interval=1)
                self.resource_usage['cpu'].append(cpu_percent)

                # مراقبة استخدام الذاكرة
                memory_info = psutil.virtual_memory()
                self.resource_usage['memory'].append(memory_info.percent)

                # مراقبة استخدام القرص
                disk_io = psutil.disk_io_counters()
                self.resource_usage['disk_io'].append(disk_io.read_bytes + disk_io.write_bytes)

                # مراقبة استخدام الشبكة
                net_io = psutil.net_io_counters()
                self.resource_usage['network_io'].append(net_io.bytes_sent + net_io.bytes_recv)

                time.sleep(1)
                
        except Exception as e:
            logger.error(f'خطأ في مراقبة الموارد: {str(e)}')
            raise

    def _calculate_statistics(self) -> Dict:
        """
        حساب الإحصائيات
        """
        try:
            total_duration = self.end_time - self.start_time
            completed_steps = len([s for s in self.steps if s['status'] == 'completed'])
            failed_steps = len([s for s in self.steps if s['status'] == 'failed'])

            stats = {
                'total_duration': total_duration,
                'total_steps': self.total_steps,
                'completed_steps': completed_steps,
                'failed_steps': failed_steps,
                'success_rate': completed_steps / self.total_steps if self.total_steps > 0 else 0,
                'average_step_duration': sum(s['duration'] for s in self.steps if s['duration']) / len(self.steps) if self.steps else 0,
                'resource_usage': {
                    'cpu': {
                        'mean': np.mean(self.resource_usage['cpu']),
                        'max': np.max(self.resource_usage['cpu']),
                        'min': np.min(self.resource_usage['cpu'])
                    },
                    'memory': {
                        'mean': np.mean(self.resource_usage['memory']),
                        'max': np.max(self.resource_usage['memory']),
                        'min': np.min(self.resource_usage['memory'])
                    },
                    'disk_io': {
                        'total': sum(self.resource_usage['disk_io']),
                        'mean': np.mean(self.resource_usage['disk_io'])
                    },
                    'network_io': {
                        'total': sum(self.resource_usage['network_io']),
                        'mean': np.mean(self.resource_usage['network_io'])
                    }
                },
                'errors': self.errors,
                'warnings': self.warnings,
                'steps': self.steps
            }

            # حفظ الإحصائيات في ملف
            with open(os.path.join(self.log_dir, self.stats_file), 'w') as f:
                json.dump(stats, f, indent=4)

            return stats
            
        except Exception as e:
            logger.error(f'خطأ في حساب الإحصائيات: {str(e)}')
            raise

    def _save_logs(self, stats: Dict) -> None:
        """
        حفظ سجلات الأداء
        """
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'statistics': stats
            }
            
            # إنشاء ملف السجلات إذا لم يكن موجوداً
            if not os.path.exists(self.log_file):
                with open(self.log_file, 'w') as f:
                    json.dump([log_entry], f, indent=4)
            else:
                # إضافة السجل الجديد إلى الملف
                with open(self.log_file, 'r+') as f:
                    logs = json.load(f)
                    logs.append(log_entry)
                    f.seek(0)
                    json.dump(logs, f, indent=4)
            
            logger.info('تم حفظ سجلات الأداء')
            
        except Exception as e:
            logger.error(f'خطأ في حفظ السجلات: {str(e)}')
            raise

    def get_performance_history(self, limit: int = 10) -> List[Dict]:
        """
        الحصول على سجل الأداء
        """
        try:
            if not os.path.exists(self.log_file):
                return []
            
            with open(self.log_file, 'r') as f:
                logs = json.load(f)
                return logs[-limit:]
                
        except Exception as e:
            logger.error(f'خطأ في الحصول على سجل الأداء: {str(e)}')
            raise

    def clear_logs(self) -> None:
        """
        مسح سجلات الأداء
        """
        try:
            if os.path.exists(self.log_file):
                os.remove(self.log_file)
            if os.path.exists(os.path.join(self.log_dir, self.stats_file)):
                os.remove(os.path.join(self.log_dir, self.stats_file))
            logger.info('تم مسح سجلات الأداء')
            
        except Exception as e:
            logger.error(f'خطأ في مسح السجلات: {str(e)}')
            raise 