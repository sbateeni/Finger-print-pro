import multiprocessing

# عدد العمليات
workers = multiprocessing.cpu_count() * 2 + 1

# نوع العمال
worker_class = 'sync'

# المنفذ
bind = '0.0.0.0:10000'

# مهلة الاستجابة
timeout = 120

# إعادة تشغيل العمال عند حدوث خطأ
max_requests = 1000
max_requests_jitter = 50

# تسجيل الأخطاء
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# إعدادات الأمان
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190 