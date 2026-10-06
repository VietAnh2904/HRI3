# Hướng dẫn chạy Project ur3_llm_control

Package ROS 2 điều khiển tay máy UR3 và Gripper bằng ngôn ngữ tự nhiên thông qua LLM và Camera.

## Yêu cầu hệ thống
- Ubuntu 22.04
- ROS 2 Humble
- Gazebo Classic 11 & MoveIt 2
- Python 3.10

## 1. Hướng dẫn Build

```bash
cd ~/ur3_ws/src
git clone [https://github.com/VietAnh2904/HRI3.git](https://github.com/VietAnh2904/HRI3.git) ur3_llm_control
cd ~/ur3_ws
colcon build --packages-select ur3_llm_control
source install/setup.bash
```

## 2. Cấu hình LLM (Bắt buộc)
Trước khi khởi chạy Node LLM, bạn cần cấp API Key cho 9Router. Hãy điền Key của bạn vào file cấu hình:
- Mở file: `src/ur3_llm_control/config/student_config.yaml`
- Hoặc export trực tiếp vào môi trường Terminal:
```bash
export NINEROUTER_API_KEY="sk-your-api-key-here"
```

## 3. Hướng dẫn Chạy hệ thống

Mở 4 Terminal khác nhau (nhớ chạy `source install/setup.bash` trước mỗi lệnh):

**Terminal 1: Khởi động mô phỏng (Gazebo, Robot, Camera)**
```bash
ros2 launch ur3_llm_control sim.launch.py
```

**Terminal 2: Khởi động MoveIt 2**
```bash
ros2 launch ur3_llm_control moveit.launch.py
```

**Terminal 3: Khởi động Node LLM**
```bash
ros2 launch ur3_llm_control llm_robot.launch.py
```

**Terminal 4: Nhập lệnh điều khiển**
```bash
ros2 run ur3_llm_control send_command
```
*(Ví dụ: `đặt khối cube màu xanh dương vào zone a`)*
