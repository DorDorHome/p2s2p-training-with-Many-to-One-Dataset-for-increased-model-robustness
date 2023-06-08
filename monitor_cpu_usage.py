# created to monitor cpu usage
# to investigate whether slow training was due to 
# limitations in cpu

import psutil
import time

def monitor_cpu_usage(interval):
    while True:
        cpu_usage = psutil.cpu_percent(interval=interval)
        print(f"CPU Usage: {cpu_usage}%")
        time.sleep(interval)

# Usage: Monitor CPU usage every 1 second
monitor_cpu_usage(1)