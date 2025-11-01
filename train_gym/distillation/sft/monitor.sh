#!/bin/bash

# Имя лог-файла. Можете изменить.
LOG_FILE="system_monitor.log"

# Интервал опроса в секундах
INTERVAL=5

echo "Starting monitoring... Logging to $LOG_FILE"
echo "-----------------------------------------" >> $LOG_FILE
echo "Monitoring started at $(date)" >> $LOG_FILE
echo "-----------------------------------------" >> $LOG_FILE

while true; do
    # Добавляем временную метку
    echo -n "$(date '+%Y-%m-%d %H:%M:%S') | " >> $LOG_FILE
    
    # Получаем и форматируем температуру CPU (выбирает самое горячее ядро)
    CPU_TEMP=$(sensors | grep 'Package id 0' | awk '{print $4}')
    echo -n "CPU: $CPU_TEMP | " >> $LOG_FILE
    
    # Получаем температуру и нагрузку GPU от nvidia-smi
    GPU_INFO=$(nvidia-smi --query-gpu=timestamp,temperature.gpu,utilization.gpu,power.draw --format=csv,noheader,nounits)
    echo "GPU Temp: ${GPU_INFO#*,} C | GPU Load: ${GPU_INFO##*,} W" | tr -d ' ' | tr ',' ' ' >> $LOG_FILE
    
    # Ждем указанный интервал
    sleep $INTERVAL
done