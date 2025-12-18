"""
Surveillance Robot Control System
Raspberry Pi 4 Model B with Camera Module Rev 1.3
GM-25-370 motors with 6-pin encoders (M1, GND, C1, C2, 3.3V, M2)
MPU-9250 IMU for navigation
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import csv
from datetime import datetime
import os
import json
import threading
import time
import sys
import math
from collections import deque

# ========== GPIO PIN DEFINITIONS FOR RASPBERRY PI ==========
# MPU-9250 uses I2C communication - FIXED PINS on Raspberry Pi
MPU_SDA_PIN = 2    # GPIO 2, Physical Pin 3 (I2C1 SDA) - FIXED
MPU_SCL_PIN = 3    # GPIO 3, Physical Pin 5 (I2C1 SCL) - FIXED
# Note: These pins are hardwired for I2C on Raspberry Pi, don't change them!

# Camera module imports
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("⚠ picamera2 not installed. Install with: pip install picamera2")

# GPIO imports for Raspberry Pi
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
    # Use BCM numbering (GPIO numbers, not physical pin numbers)
    GPIO.setmode(GPIO.BCM)
except ImportError:
    GPIO_AVAILABLE = False
    print("⚠ RPi.GPIO not available - running in simulation mode")

# Serial communication imports
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    serial = None
    SERIAL_AVAILABLE = False


class MPU9250:
    """
    MPU-9250 9-Axis IMU Class for Raspberry Pi
    Provides accelerometer, gyroscope, and magnetometer data
    Connects via I2C to GPIO 2 (SDA) and GPIO 3 (SCL)
    """
    
    def __init__(self):
        # I2C addresses
        self.MPU_ADDRESS = 0x68  # MPU-9250 main address
        self.MAG_ADDRESS = 0x0C  # Magnetometer address
        
        # Calibration data
        self.accel_bias = [0, 0, 0]
        self.gyro_bias = [0, 0, 0]
        self.mag_bias = [0, 0, 0]
        self.mag_scale = [1, 1, 1]
        
        # Raw sensor data
        self.accel_raw = [0, 0, 0]
        self.gyro_raw = [0, 0, 0]
        self.mag_raw = [0, 0, 0]
        self.temp_raw = 0
        
        # Scaled data
        self.accel_scaled = [0.0, 0.0, 0.0]  # g
        self.gyro_scaled = [0.0, 0.0, 0.0]   # °/s
        self.mag_scaled = [0.0, 0.0, 0.0]    # µT
        self.temperature = 0.0               # °C
        
        # Orientation
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.yaw_offset = 0.0
        
        # I2C bus
        self.bus = None
        
        # Initialize MPU-9250
        self.init_mpu9250()
        
        # Calibrate sensors
        self.calibrate()
    
    def init_mpu9250(self):
        """
        Initialize MPU-9250 IMU over I2C
        Raspberry Pi I2C pins: GPIO 2 (SDA), GPIO 3 (SCL)
        """
        try:
            import smbus
            # I2C bus 1 is default on Raspberry Pi (pins 3 and 5)
            self.bus = smbus.SMBus(1)
            
            # ========== MPU-9250 INITIALIZATION ==========
            # Wake up MPU-9250 (Power Management 1 register)
            self.bus.write_byte_data(self.MPU_ADDRESS, 0x6B, 0x00)
            time.sleep(0.1)
            
            # Configure accelerometer: ±2g range
            # 0x00 = ±2g, 0x08 = ±4g, 0x10 = ±8g, 0x18 = ±16g
            self.bus.write_byte_data(self.MPU_ADDRESS, 0x1C, 0x00)
            
            # Configure gyroscope: ±250°/s range
            # 0x00 = ±250°/s, 0x08 = ±500°/s, 0x10 = ±1000°/s, 0x18 = ±2000°/s
            self.bus.write_byte_data(self.MPU_ADDRESS, 0x1B, 0x00)
            
            # Configure low pass filter (DLPF)
            self.bus.write_byte_data(self.MPU_ADDRESS, 0x1A, 0x03)
            
            # ========== MAGNETOMETER INITIALIZATION ==========
            # First enable bypass mode to access magnetometer
            self.bus.write_byte_data(self.MPU_ADDRESS, 0x37, 0x02)
            time.sleep(0.1)
            
            # Power down magnetometer
            self.bus.write_byte_data(self.MAG_ADDRESS, 0x0A, 0x00)
            time.sleep(0.1)
            
            # Enter Fuse ROM access mode
            self.bus.write_byte_data(self.MAG_ADDRESS, 0x0A, 0x0F)
            time.sleep(0.1)
            
            # Read magnetometer sensitivity adjustment values
            mag_data = self.bus.read_i2c_block_data(self.MAG_ADDRESS, 0x10, 3)
            self.mag_scale[0] = (mag_data[0] - 128) / 256.0 + 1.0
            self.mag_scale[1] = (mag_data[1] - 128) / 256.0 + 1.0
            self.mag_scale[2] = (mag_data[2] - 128) / 256.0 + 1.0
            
            # Power down magnetometer again
            self.bus.write_byte_data(self.MAG_ADDRESS, 0x0A, 0x00)
            time.sleep(0.1)
            
            # Set magnetometer to continuous measurement mode 2 (100Hz)
            self.bus.write_byte_data(self.MAG_ADDRESS, 0x0A, 0x16)
            time.sleep(0.1)
            
            print("✅ MPU-9250 initialized successfully")
            print(f"   I2C Address: 0x{self.MPU_ADDRESS:02X}")
            print(f"   Magnetometer Address: 0x{self.MAG_ADDRESS:02X}")
            print(f"   I2C Pins: GPIO 2 (SDA), GPIO 3 (SCL)")
            
        except Exception as e:
            print(f"❌ MPU-9250 initialization error: {e}")
            print("   Check: sudo raspi-config -> Interface Options -> I2C -> Enable")
            print("   Check wiring: SDA to Pin 3, SCL to Pin 5, VCC to 3.3V, GND to GND")
            self.bus = None
    
    def read_word(self, address, register):
        """
        Read signed 16-bit word from I2C register
        """
        try:
            high = self.bus.read_byte_data(address, register)
            low = self.bus.read_byte_data(address, register + 1)
            value = (high << 8) + low
            
            # Convert to signed integer
            if value >= 0x8000:
                return -((65535 - value) + 1)
            return value
        except:
            return 0
    
    def read_sensors(self):
        """
        Read all sensors: accelerometer, gyroscope, magnetometer, temperature
        """
        if not self.bus:
            return False
        
        try:
            # Read accelerometer (registers 0x3B-0x40)
            self.accel_raw[0] = self.read_word(self.MPU_ADDRESS, 0x3B)
            self.accel_raw[1] = self.read_word(self.MPU_ADDRESS, 0x3D)
            self.accel_raw[2] = self.read_word(self.MPU_ADDRESS, 0x3F)
            
            # Read temperature (register 0x41)
            self.temp_raw = self.read_word(self.MPU_ADDRESS, 0x41)
            
            # Read gyroscope (registers 0x43-0x48)
            self.gyro_raw[0] = self.read_word(self.MPU_ADDRESS, 0x43)
            self.gyro_raw[1] = self.read_word(self.MPU_ADDRESS, 0x45)
            self.gyro_raw[2] = self.read_word(self.MPU_ADDRESS, 0x47)
            
            # Read magnetometer (registers 0x03-0x08)
            # Check data ready status
            status = self.bus.read_byte_data(self.MAG_ADDRESS, 0x02)
            if status & 0x01:
                self.mag_raw[0] = self.read_word(self.MAG_ADDRESS, 0x04)  # X
                self.mag_raw[1] = self.read_word(self.MAG_ADDRESS, 0x06)  # Y
                self.mag_raw[2] = self.read_word(self.MAG_ADDRESS, 0x08)  # Z
            else:
                # No new magnetometer data
                pass
            
            # Scale raw data to real units
            self.scale_data()
            
            # Calculate orientation
            self.calculate_orientation()
            
            return True
            
        except Exception as e:
            print(f"MPU-9250 read error: {e}")
            return False
    
    def scale_data(self):
        """
        Scale raw sensor data to real units
        """
        # Accelerometer scaling (±2g range: 16384 LSB/g)
        accel_scale = 16384.0
        self.accel_scaled[0] = (self.accel_raw[0] - self.accel_bias[0]) / accel_scale
        self.accel_scaled[1] = (self.accel_raw[1] - self.accel_bias[1]) / accel_scale
        self.accel_scaled[2] = (self.accel_raw[2] - self.accel_bias[2]) / accel_scale
        
        # Gyroscope scaling (±250°/s range: 131 LSB/°/s)
        gyro_scale = 131.0
        self.gyro_scaled[0] = (self.gyro_raw[0] - self.gyro_bias[0]) / gyro_scale
        self.gyro_scaled[1] = (self.gyro_raw[1] - self.gyro_bias[1]) / gyro_scale
        self.gyro_scaled[2] = (self.gyro_raw[2] - self.gyro_bias[2]) / gyro_scale
        
        # Temperature scaling
        self.temperature = (self.temp_raw / 340.0) + 36.53
        
        # Magnetometer scaling (16-bit output)
        mag_scale = 0.15  # µT per LSB (adjust based on datasheet)
        self.mag_scaled[0] = (self.mag_raw[0] - self.mag_bias[0]) * self.mag_scale[0] * mag_scale
        self.mag_scaled[1] = (self.mag_raw[1] - self.mag_bias[1]) * self.mag_scale[1] * mag_scale
        self.mag_scaled[2] = (self.mag_raw[2] - self.mag_bias[2]) * self.mag_scale[2] * mag_scale
    
    def calculate_orientation(self):
        """
        Calculate roll, pitch, and yaw from sensor data
        """
        # Calculate roll and pitch from accelerometer
        self.roll = math.atan2(self.accel_scaled[1], self.accel_scaled[2]) * 180.0 / math.pi
        self.pitch = math.atan2(-self.accel_scaled[0], 
                               math.sqrt(self.accel_scaled[1]**2 + self.accel_scaled[2]**2)) * 180.0 / math.pi
        
        # Calculate yaw from magnetometer (tilt-compensated)
        mag_x = self.mag_scaled[0] * math.cos(self.pitch * math.pi/180.0) + \
                self.mag_scaled[2] * math.sin(self.pitch * math.pi/180.0)
        
        mag_y = self.mag_scaled[0] * math.sin(self.roll * math.pi/180.0) * math.sin(self.pitch * math.pi/180.0) + \
                self.mag_scaled[1] * math.cos(self.roll * math.pi/180.0) - \
                self.mag_scaled[2] * math.sin(self.roll * math.pi/180.0) * math.cos(self.pitch * math.pi/180.0)
        
        self.yaw = math.atan2(-mag_y, mag_x) * 180.0 / math.pi - self.yaw_offset
        
        # Normalize yaw to 0-360 degrees
        if self.yaw < 0:
            self.yaw += 360
        if self.yaw >= 360:
            self.yaw -= 360
    
    def calibrate(self, samples=500):
        """
        Calibrate accelerometer and gyroscope
        Robot must be stationary during calibration
        """
        if not self.bus:
            print("❌ Cannot calibrate - MPU-9250 not connected")
            return
        
        print("🔄 Calibrating MPU-9250... Keep robot stationary!")
        
        # Initialize sums
        accel_sum = [0, 0, 0]
        gyro_sum = [0, 0, 0]
        
        # Collect samples
        for i in range(samples):
            self.read_sensors()
            
            accel_sum[0] += self.accel_raw[0]
            accel_sum[1] += self.accel_raw[1]
            accel_sum[2] += self.accel_raw[2]
            
            gyro_sum[0] += self.gyro_raw[0]
            gyro_sum[1] += self.gyro_raw[1]
            gyro_sum[2] += self.gyro_raw[2]
            
            time.sleep(0.01)
            
            # Show progress
            if i % 50 == 0:
                print(f"   Calibration: {i}/{samples}")
        
        # Calculate averages (biases)
        self.accel_bias[0] = accel_sum[0] / samples
        self.accel_bias[1] = accel_sum[1] / samples
        self.accel_bias[2] = (accel_sum[2] / samples) - 16384  # Subtract 1g for Z-axis
        
        self.gyro_bias[0] = gyro_sum[0] / samples
        self.gyro_bias[1] = gyro_sum[1] / samples
        self.gyro_bias[2] = gyro_sum[2] / samples
        
        print("✅ Calibration complete")
        print(f"   Accel Bias: X={self.accel_bias[0]:.0f}, Y={self.accel_bias[1]:.0f}, Z={self.accel_bias[2]:.0f}")
        print(f"   Gyro Bias: X={self.gyro_bias[0]:.0f}, Y={self.gyro_bias[1]:.0f}, Z={self.gyro_bias[2]:.0f}")
    
    def calibrate_magnetometer(self):
        """
        Calibrate magnetometer (requires rotating robot in figure-8 pattern)
        """
        if not self.bus:
            return
        
        print("🔄 Calibrating magnetometer... Rotate robot in figure-8 pattern!")
        print("   This will take about 30 seconds...")
        
        mag_min = [32767, 32767, 32767]
        mag_max = [-32768, -32768, -32768]
        start_time = time.time()
        
        # Collect magnetometer data while rotating
        while time.time() - start_time < 30:
            self.read_sensors()
            
            # Update min/max values
            for i in range(3):
                if self.mag_raw[i] < mag_min[i]:
                    mag_min[i] = self.mag_raw[i]
                if self.mag_raw[i] > mag_max[i]:
                    mag_max[i] = self.mag_raw[i]
            
            time.sleep(0.1)
        
        # Calculate bias and scale
        for i in range(3):
            self.mag_bias[i] = (mag_max[i] + mag_min[i]) / 2
            self.mag_scale[i] = (mag_max[i] - mag_min[i]) / 2
        
        print("✅ Magnetometer calibration complete")
        print(f"   Mag Bias: X={self.mag_bias[0]:.0f}, Y={self.mag_bias[1]:.0f}, Z={self.mag_bias[2]:.0f}")
    
    def set_yaw_reference(self):
        """
        Set current yaw as reference (0 degrees)
        """
        self.yaw_offset = self.yaw
        print(f"✅ Yaw reference set to {self.yaw_offset:.1f}°")
    
    def get_data(self):
        """
        Get all sensor data in a dictionary
        """
        return {
            'accel': self.accel_scaled.copy(),      # [g]
            'gyro': self.gyro_scaled.copy(),        # [°/s]
            'mag': self.mag_scaled.copy(),          # [µT]
            'temp': self.temperature,               # [°C]
            'roll': self.roll,                      # [°]
            'pitch': self.pitch,                    # [°]
            'yaw': self.yaw,                        # [°]
            'accel_raw': self.accel_raw.copy(),     # Raw values
            'gyro_raw': self.gyro_raw.copy(),       # Raw values
            'mag_raw': self.mag_raw.copy()          # Raw values
        }
    
    def get_heading(self):
        """
        Get compass heading (0° = North, 90° = East, 180° = South, 270° = West)
        """
        return self.yaw
    
    def get_tilt_compensated_heading(self):
        """
        Get tilt-compensated heading
        """
        return self.yaw
    
    def is_connected(self):
        """
        Check if MPU-9250 is connected
        """
        return self.bus is not None


class MotorController:
    """
    Motor Controller for GM-25-370 motors with 6-pin integrated encoders
    Now with MPU-9250 integration for precise navigation
    """
    
    def __init__(self):
        # ========== MOTOR WIRING CONFIGURATION ==========
        # Motor power wires (connected to L298N driver)
        # Left Motor (Motor A)
        self.MOTOR_A_PWM = 18      # GPIO 18, PWM for left motor speed
        self.MOTOR_A_IN1 = 17      # GPIO 17, Left motor direction 1
        self.MOTOR_A_IN2 = 27      # GPIO 27, Left motor direction 2
        
        # Right Motor (Motor B)
        self.MOTOR_B_PWM = 24      # GPIO 24, PWM for right motor speed
        self.MOTOR_B_IN3 = 22      # GPIO 22, Right motor direction 1
        self.MOTOR_B_IN4 = 23      # GPIO 23, Right motor direction 2
        
        # ========== ENCODER WIRING CONFIGURATION ==========
        # Left Motor Encoder (6 pins: M1, GND, C1, C2, 3.3V, M2)
        self.ENCODER_A_CHANNEL_A = 5    # GPIO 5, C1 pin (Phase A)
        self.ENCODER_A_CHANNEL_B = 6    # GPIO 6, C2 pin (Phase B)
        
        # Right Motor Encoder
        self.ENCODER_B_CHANNEL_A = 13   # GPIO 13, C1 pin (Phase A)
        self.ENCODER_B_CHANNEL_B = 19   # GPIO 19, C2 pin (Phase B)
        
        # ========== MPU-9250 INITIALIZATION ==========
        self.mpu = MPU9250()
        
        # ========== MOTOR SPECIFICATIONS ==========
        self.MOTOR_TYPE = "GM-25-370"
        self.NO_LOAD_RPM = 300
        self.GEAR_RATIO = 48
        self.ENCODER_PPR = 11
        self.EFFECTIVE_PPR = 528
        
        # ========== CONTROL VARIABLES ==========
        self.speed = 0
        self.target_speed = 50
        self.direction = "stop"
        self.is_running = False
        
        # ========== ENCODER VARIABLES ==========
        self.encoder_count_a = 0
        self.encoder_count_b = 0
        self.encoder_a_direction = 1
        self.encoder_b_direction = 1
        
        # ========== NAVIGATION VARIABLES ==========
        self.target_yaw = 0
        self.current_yaw = 0
        self.yaw_error_threshold = 2.0  # degrees
        self.distance_traveled = 0
        
        # ========== PID CONTROL ==========
        self.kp = 0.8
        self.ki = 0.01
        self.kd = 0.05
        self.prev_error = 0
        self.integral = 0
        
        # ========== WHEEL SPECIFICATIONS ==========
        self.wheel_diameter_cm = 6.5
        self.wheel_circumference = math.pi * self.wheel_diameter_cm
        self.wheelbase_cm = 15.0  # Distance between wheels
        
        # ========== INITIALIZE HARDWARE ==========
        if GPIO_AVAILABLE:
            self.setup_gpio()
            self.setup_encoders()
        
        print(f"✅ Motor Controller with MPU-9250 initialized")
        print(f"   MPU-9250 connected: {self.mpu.is_connected()}")
    
    def setup_gpio(self):
        """
        Configure GPIO pins for L298N motor driver
        """
        try:
            # Motor control pins
            motor_pins = [
                self.MOTOR_A_PWM, self.MOTOR_A_IN1, self.MOTOR_A_IN2,
                self.MOTOR_B_PWM, self.MOTOR_B_IN3, self.MOTOR_B_IN4
            ]
            
            for pin in motor_pins:
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)
            
            # PWM initialization
            self.pwm_a = GPIO.PWM(self.MOTOR_A_PWM, 1000)
            self.pwm_b = GPIO.PWM(self.MOTOR_B_PWM, 1000)
            self.pwm_a.start(0)
            self.pwm_b.start(0)
            
            print("✅ Motor GPIO pins configured")
            
        except Exception as e:
            print(f"❌ Motor GPIO setup error: {e}")
    
    def setup_encoders(self):
        """
        Configure GPIO pins for quadrature encoders
        """
        try:
            encoder_pins = [
                self.ENCODER_A_CHANNEL_A, self.ENCODER_A_CHANNEL_B,
                self.ENCODER_B_CHANNEL_A, self.ENCODER_B_CHANNEL_B
            ]
            
            for pin in encoder_pins:
                GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            
            # Encoder interrupts
            GPIO.add_event_detect(self.ENCODER_A_CHANNEL_A, GPIO.RISING,
                                callback=lambda x: self.encoder_a_callback())
            GPIO.add_event_detect(self.ENCODER_B_CHANNEL_A, GPIO.RISING,
                                callback=lambda x: self.encoder_b_callback())
            
            print("✅ Encoder GPIO pins configured")
            
        except Exception as e:
            print(f"❌ Encoder GPIO setup error: {e}")
    
    def encoder_a_callback(self):
        """Left encoder callback"""
        self.encoder_count_a += 1
    
    def encoder_b_callback(self):
        """Right encoder callback"""
        self.encoder_count_b += 1
    
    def update_mpu(self):
        """
        Update MPU-9250 data for navigation
        """
        if self.mpu.is_connected():
            self.mpu.read_sensors()
            data = self.mpu.get_data()
            self.current_yaw = data['yaw']
            return data
        return None
    
    def navigate_to_heading(self, target_heading):
        """
        Navigate to specific compass heading using MPU-9250
        """
        if not self.mpu.is_connected():
            print("❌ MPU-9250 not connected for navigation")
            return False
        
        print(f"🧭 Navigating to heading: {target_heading}°")
        
        # Update current heading
        self.update_mpu()
        
        # Calculate shortest turn angle
        angle_diff = target_heading - self.current_yaw
        if angle_diff > 180:
            angle_diff -= 360
        elif angle_diff < -180:
            angle_diff += 360
        
        # Turn in shortest direction
        if abs(angle_diff) > self.yaw_error_threshold:
            if angle_diff > 0:
                self.turn_right(angle=abs(angle_diff))
            else:
                self.turn_left(angle=abs(angle_diff))
        
        print(f"✅ Reached target heading: {self.current_yaw:.1f}°")
        return True
    
    def move_straight_distance(self, distance_cm, speed=None):
        """
        Move straight for specific distance using encoder feedback
        """
        if speed is None:
            speed = self.target_speed
        
        print(f"📏 Moving straight: {distance_cm} cm at {speed}% speed")
        
        # Calculate required encoder counts
        wheel_revs = distance_cm / self.wheel_circumference
        required_counts = wheel_revs * self.EFFECTIVE_PPR
        
        # Reset encoders
        start_count_a = self.encoder_count_a
        start_count_b = self.encoder_count_b
        
        # Start moving
        self.move_forward(speed)
        
        # Move until distance reached
        try:
            while True:
                current_counts_a = self.encoder_count_a - start_count_a
                current_counts_b = self.encoder_count_b - start_count_b
                avg_counts = (current_counts_a + current_counts_b) / 2
                
                # Calculate progress
                progress = avg_counts / required_counts
                distance_done = progress * distance_cm
                
                # Update MPU for straight-line correction
                if self.mpu.is_connected():
                    self.update_mpu()
                    
                    # Simple straight-line correction based on yaw
                    yaw_error = self.current_yaw - self.target_yaw
                    if abs(yaw_error) > 1.0:
                        # Adjust motor speeds to correct yaw
                        correction = yaw_error * 0.5  # Correction factor
                        left_speed = max(0, min(100, speed - correction))
                        right_speed = max(0, min(100, speed + correction))
                        self.set_motor_speed('A', left_speed, 'forward')
                        self.set_motor_speed('B', right_speed, 'forward')
                
                if distance_done >= distance_cm:
                    break
                
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            pass
        
        finally:
            self.stop()
        
        actual_distance = distance_done
        print(f"✅ Distance completed: {actual_distance:.1f} cm")
        return actual_distance
    
    def set_motor_speed(self, motor, speed, direction):
        """
        Control motor speed and direction
        """
        if not GPIO_AVAILABLE:
            return
        
        speed = max(0, min(speed, 100))
        
        if motor == 'A':  # Left motor
            if direction == 'forward':
                GPIO.output(self.MOTOR_A_IN1, GPIO.HIGH)
                GPIO.output(self.MOTOR_A_IN2, GPIO.LOW)
            elif direction == 'backward':
                GPIO.output(self.MOTOR_A_IN1, GPIO.LOW)
                GPIO.output(self.MOTOR_A_IN2, GPIO.HIGH)
            else:  # stop
                GPIO.output(self.MOTOR_A_IN1, GPIO.LOW)
                GPIO.output(self.MOTOR_A_IN2, GPIO.LOW)
            self.pwm_a.ChangeDutyCycle(speed)
            
        elif motor == 'B':  # Right motor
            if direction == 'forward':
                GPIO.output(self.MOTOR_B_IN3, GPIO.HIGH)
                GPIO.output(self.MOTOR_B_IN4, GPIO.LOW)
            elif direction == 'backward':
                GPIO.output(self.MOTOR_B_IN3, GPIO.LOW)
                GPIO.output(self.MOTOR_B_IN4, GPIO.HIGH)
            else:  # stop
                GPIO.output(self.MOTOR_B_IN3, GPIO.LOW)
                GPIO.output(self.MOTOR_B_IN4, GPIO.LOW)
            self.pwm_b.ChangeDutyCycle(speed)
    
    def move_forward(self, speed=None):
        """
        Move robot forward with MPU-9250 heading maintenance
        """
        if speed is None:
            speed = self.target_speed
        
        # Set target heading to current heading
        if self.mpu.is_connected():
            self.update_mpu()
            self.target_yaw = self.current_yaw
        
        self.direction = "forward"
        self.set_motor_speed('A', speed, 'forward')
        self.set_motor_speed('B', speed, 'forward')
        
        print(f"🤖 Forward at {speed}%, Target yaw: {self.target_yaw:.1f}°")
    
    def move_backward(self, speed=None):
        if speed is None:
            speed = self.target_speed
        self.direction = "backward"
        self.set_motor_speed('A', speed, 'backward')
        self.set_motor_speed('B', speed, 'backward')
        print(f"🤖 Backward at {speed}%")
    
    def turn_left(self, speed=None, angle=None):
        """
        Turn left with optional angle using MPU-9250
        """
        if speed is None:
            speed = self.target_speed
        
        if angle and self.mpu.is_connected():
            # Turn by specific angle
            start_yaw = self.current_yaw
            target_yaw = start_yaw - angle
            
            # Normalize target yaw
            if target_yaw < 0:
                target_yaw += 360
            
            print(f"🔄 Turning left {angle}°: {start_yaw:.1f}° → {target_yaw:.1f}°")
            
            # Start turning
            self.set_motor_speed('A', speed, 'backward')
            self.set_motor_speed('B', speed, 'forward')
            
            # Monitor angle until reached
            while True:
                self.update_mpu()
                
                # Calculate shortest angle difference
                angle_diff = target_yaw - self.current_yaw
                if angle_diff > 180:
                    angle_diff -= 360
                elif angle_diff < -180:
                    angle_diff += 360
                
                if abs(angle_diff) < self.yaw_error_threshold:
                    break
                
                time.sleep(0.01)
            
            self.stop()
            print(f"✅ Turn complete: {self.current_yaw:.1f}°")
            
        else:
            # Simple turn without angle measurement
            self.direction = "left"
            self.set_motor_speed('A', speed, 'backward')
            self.set_motor_speed('B', speed, 'forward')
            print(f"🤖 Turning left at {speed}%")
    
    def turn_right(self, speed=None, angle=None):
        """
        Turn right with optional angle using MPU-9250
        """
        if speed is None:
            speed = self.target_speed
        
        if angle and self.mpu.is_connected():
            # Turn by specific angle
            start_yaw = self.current_yaw
            target_yaw = start_yaw + angle
            
            # Normalize target yaw
            if target_yaw >= 360:
                target_yaw -= 360
            
            print(f"🔄 Turning right {angle}°: {start_yaw:.1f}° → {target_yaw:.1f}°")
            
            # Start turning
            self.set_motor_speed('A', speed, 'forward')
            self.set_motor_speed('B', speed, 'backward')
            
            # Monitor angle until reached
            while True:
                self.update_mpu()
                
                # Calculate shortest angle difference
                angle_diff = target_yaw - self.current_yaw
                if angle_diff > 180:
                    angle_diff -= 360
                elif angle_diff < -180:
                    angle_diff += 360
                
                if abs(angle_diff) < self.yaw_error_threshold:
                    break
                
                time.sleep(0.01)
            
            self.stop()
            print(f"✅ Turn complete: {self.current_yaw:.1f}°")
            
        else:
            # Simple turn without angle measurement
            self.direction = "right"
            self.set_motor_speed('A', speed, 'forward')
            self.set_motor_speed('B', speed, 'backward')
            print(f"🤖 Turning right at {speed}%")
    
    def stop(self):
        """Stop all motors"""
        self.direction = "stop"
        self.set_motor_speed('A', 0, 'stop')
        self.set_motor_speed('B', 0, 'stop')
        print("🤖 Stopped")
    
    def set_speed(self, speed):
        """Set target speed"""
        self.target_speed = max(30, min(speed, 100))
        print(f"⚙️ Target speed: {self.target_speed}%")
    
    def get_sensor_data(self):
        """Get all sensor data"""
        mpu_data = self.update_mpu()
        
        return {
            'encoders': {
                'left': self.encoder_count_a,
                'right': self.encoder_count_b
            },
            'speed': self.target_speed,
            'direction': self.direction,
            'mpu': mpu_data,
            'yaw': self.current_yaw if mpu_data else 0,
            'distance_traveled': self.distance_traveled
        }
    
    def calibrate_mpu(self):
        """Calibrate MPU-9250"""
        if self.mpu.is_connected():
            print("Starting MPU-9250 calibration...")
            self.mpu.calibrate()
            return True
        return False
    
    def set_yaw_reference(self):
        """Set current yaw as reference (0°)"""
        if self.mpu.is_connected():
            self.mpu.set_yaw_re