import multiprocessing

# Number of worker processes
workers = multiprocessing.cpu_count() * 2 + 1

# Host and port
bind = "0.0.0.0:10000"

# Timeout
timeout = 120

# Access log file
accesslog = "-"

# Error log file
errorlog = "-"

# Log level
loglevel = "info"

# Worker class
worker_class = "sync"

# Maximum number of requests a worker will process before restarting
max_requests = 1000
max_requests_jitter = 50 