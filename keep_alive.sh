
#!/bin/bash
while true; do
    # 模拟鼠标向右移动 1 像素
    xdotool mousemove_relative 1 0
    # 等待 1 秒
    sleep 1
    # 模拟鼠标向左移动 1 像素
    xdotool mousemove_relative -- -1 0
    # 等待 5 分钟
    sleep 3  # 300 秒 = 5 分钟
done
