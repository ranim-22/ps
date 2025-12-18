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
from gpiozero import PWMOutputDevice, DigitalOutputDevice

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    serial = None
    SERIAL_AVAILABLE = False

class DualMotorController:
    def __init__(self):
        self.left_motor_pwm = PWMOutputDevice(17)
        self.left_motor_forward = DigitalOutputDevice(22)
        self.left_motor_backward = DigitalOutputDevice(27)
        
        self.right_motor_pwm = PWMOutputDevice(13)
        self.right_motor_forward = DigitalOutputDevice(19)
        self.right_motor_backward = DigitalOutputDevice(26)
        
        self.default_speed = 0.6
        self.turn_speed = 0.4
        
        self.left_encoder_count = 0
        self.right_encoder_count = 0
        
        self.stop_all()
    
    def _set_motor(self, forward_pin, backward_pin, pwm_pin, speed, direction):
        if direction == "forward":
            forward_pin.on()
            backward_pin.off()
        elif direction == "backward":
            forward_pin.off()
            backward_pin.on()
        elif direction == "stop":
            forward_pin.off()
            backward_pin.off()
        
        pwm_pin.value = abs(speed) if direction != "stop" else 0
    
    def move_forward(self, speed=None):
        speed = speed or self.default_speed
        self._set_motor(self.left_motor_forward, self.left_motor_backward,
                       self.left_motor_pwm, speed, "forward")
        self._set_motor(self.right_motor_forward, self.right_motor_backward,
                       self.right_motor_pwm, speed, "forward")
    
    def move_backward(self, speed=None):
        speed = speed or self.default_speed
        self._set_motor(self.left_motor_forward, self.left_motor_backward,
                       self.left_motor_pwm, speed, "backward")
        self._set_motor(self.right_motor_forward, self.right_motor_backward,
                       self.right_motor_pwm, speed, "backward")
    
    def turn_left(self, speed=None):
        speed = speed or self.turn_speed
        self._set_motor(self.left_motor_forward, self.left_motor_backward,
                       self.left_motor_pwm, speed, "backward")
        self._set_motor(self.right_motor_forward, self.right_motor_backward,
                       self.right_motor_pwm, speed, "forward")
    
    def turn_right(self, speed=None):
        speed = speed or self.turn_speed
        self._set_motor(self.left_motor_forward, self.left_motor_backward,
                       self.left_motor_pwm, speed, "forward")
        self._set_motor(self.right_motor_forward, self.right_motor_backward,
                       self.right_motor_pwm, speed, "backward")
    
    def pivot_left(self, speed=None):
        speed = speed or self.turn_speed
        self._set_motor(self.left_motor_forward, self.left_motor_backward,
                       self.left_motor_pwm, speed, "backward")
        self._set_motor(self.right_motor_forward, self.right_motor_backward,
                       self.right_motor_pwm, speed, "forward")
    
    def pivot_right(self, speed=None):
        speed = speed or self.turn_speed
        self._set_motor(self.left_motor_forward, self.left_motor_backward,
                       self.left_motor_pwm, speed, "forward")
        self._set_motor(self.right_motor_forward, self.right_motor_backward,
                       self.right_motor_pwm, speed, "backward")
    
    def stop_all(self):
        self._set_motor(self.left_motor_forward, self.left_motor_backward,
                       self.left_motor_pwm, 0, "stop")
        self._set_motor(self.right_motor_forward, self.right_motor_backward,
                       self.right_motor_pwm, 0, "stop")
    
    def set_speeds(self, left_speed, right_speed):
        if left_speed > 0:
            self._set_motor(self.left_motor_forward, self.left_motor_backward,
                          self.left_motor_pwm, left_speed, "forward")
        elif left_speed < 0:
            self._set_motor(self.left_motor_forward, self.left_motor_backward,
                          self.left_motor_pwm, abs(left_speed), "backward")
        else:
            self._set_motor(self.left_motor_forward, self.left_motor_backward,
                          self.left_motor_pwm, 0, "stop")
        
        if right_speed > 0:
            self._set_motor(self.right_motor_forward, self.right_motor_backward,
                          self.right_motor_pwm, right_speed, "forward")
        elif right_speed < 0:
            self._set_motor(self.right_motor_forward, self.right_motor_backward,
                          self.right_motor_pwm, abs(right_speed), "backward")
        else:
            self._set_motor(self.right_motor_forward, self.right_motor_backward,
                          self.right_motor_pwm, 0, "stop")
    
    def cleanup(self):
        self.stop_all()
        self.left_motor_pwm.close()
        self.left_motor_forward.close()
        self.left_motor_backward.close()
        self.right_motor_pwm.close()
        self.right_motor_forward.close()
        self.right_motor_backward.close()

class MotorBridge:
    def __init__(self, port, baudrate=115200, timeout=0.2):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)

    def send_command(self, payload):
        if not self.serial or not self.serial.is_open:
            return False
        try:
            data = json.dumps(payload).encode("utf-8") + b"\n"
            self.serial.write(data)
            return True
        except Exception as exc:
            return False

    def close(self):
        try:
            if self.serial and self.serial.is_open:
                self.serial.close()
        except Exception:
            pass

class VisionHMI:
    def __init__(self, root):
        self.root = root
        self.root.title("Robot de Surveillance-HMI")
        self.root.geometry("1280x720")
        self.root.minsize(1024, 600)
        self.root.configure(bg="#e6e6e6")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.cap = None
        self.picam2 = None
        self.use_static_image = False
        self.static_image = None
        self.camera_type = None
        self.init_camera()

        self.mode = tk.StringVar(value="Mode Réglage")
        self.mode.trace("w", self.update_mode)

        self.rois = []
        self.selected_roi = None
        self.drawing = False
        self.resizing = False
        self.moving = False
        self.rotating = False
        self.drawing_mask = False
        self.ix, self.iy = -1, -1
        self.roi_id = 0
        self.hovered_roi = None
        self.snap_to_grid = tk.BooleanVar(value=False)
        self.roi_shape = tk.StringVar(value="rectangle")

        self.params = {
            "gpio_trigger_pin": tk.IntVar(value=-1),
            "yolo_confidence": tk.DoubleVar(value=0.5),
            "face_scale_factor": tk.DoubleVar(value=1.1),
            "face_min_neighbors": tk.IntVar(value=5),
        }

        self.surveillance_features = {
            "YOLO Person Detection": tk.BooleanVar(value=True),
            "Face Detection": tk.BooleanVar(value=False),
            "Motion Detection": tk.BooleanVar(value=False),
        }

        self.cycle_state = "Idle"
        self.cycle_results = {}

        self.gpio_trigger_active = False
        self.gpio_thread = None

        self.log_file = "inspection_log.csv"
        self.init_log()

        self.yolo_enabled = tk.BooleanVar(value=False)
        self.yolo_model = None
        self.face_enabled = tk.BooleanVar(value=False)
        self.face_cascade = None
        
        self.surveillance_mode = tk.BooleanVar(value=False)
        self.motion_detection = tk.BooleanVar(value=False)
        self.alert_threshold = tk.DoubleVar(value=0.5)
        self.last_detection_time = None
        self.detection_count = 0
        self.current_detections = 0
        self.behavior_flags = {
            "detect_all_objects": tk.BooleanVar(value=True),
            "smart_obstacle_mode": tk.BooleanVar(value=True),
        }
        self.detect_all_objects = self.behavior_flags["detect_all_objects"]
        self.smart_obstacle_mode = self.behavior_flags["smart_obstacle_mode"]
        self.current_detection_summary = {"human": 0, "obstacle": 0}
        self.human_class_ids = {0}
        self.serial_port = "COM3" if sys.platform.startswith("win") else "/dev/ttyUSB0"
        self.motor_bridge = None
        self.last_turn_direction = "left"
        self.buzzer_enabled = tk.BooleanVar(value=True)
        self.human_avoidance_stop_duration = 3.0
        self.human_avoidance_turn_duration = 1200
        self.is_avoiding_human = False

        self.motor_controller = None
        self.navigation_mode = tk.StringVar(value="auto")
        self.current_speed = tk.DoubleVar(value=0.6)

        self.gpio_simulation = not GPIO_AVAILABLE
        
        if GPIO_AVAILABLE:
            self.setup_real_gpio()
        
        self.init_motor_controller()
        self.setup_gui()
        self.load_settings()
        self.update_video()
        self.start_gpio_simulation()

        self.last_live_blob_results = None
        self.motor_bridge = self.init_motor_bridge()

    def init_motor_controller(self):
        try:
            self.motor_controller = DualMotorController()
            self.show_toast("Contrôleur moteur initialisé", 2000)
        except Exception as e:
            self.show_toast(f"Erreur contrôleur moteur: {e}", 3000)
            self.motor_controller = None

    def init_camera(self):
        if PICAMERA2_AVAILABLE:
            try:
                self.picam2 = Picamera2()
                preview_config = self.picam2.create_preview_configuration(
                    main={"size": (640, 480), "format": "RGB888"},
                    controls={"FrameRate": 30}
                )
                self.picam2.configure(preview_config)
                self.picam2.start()
                time.sleep(0.5)
                test_frame = self.picam2.capture_array()
                if test_frame is not None:
                    self.camera_type = "picamera2"
                    return
                else:
                    self.picam2.stop()
                    self.picam2.close()
                    self.picam2 = None
            except Exception as e:
                if self.picam2:
                    try:
                        self.picam2.close()
                    except:
                        pass
                    self.picam2 = None

    def ensure_yolo_loaded(self):
        try:
            if self.yolo_model is None:
                from ultralytics import YOLO
                self.yolo_model = YOLO("yolov8n.pt")
                self.yolo_model.overrides['imgsz'] = 320
                self.yolo_model.overrides['device'] = 'cpu'
                self.yolo_model.overrides['half'] = False
                self.yolo_model.overrides['verbose'] = False
        except Exception as e:
            messagebox.showerror("YOLO", f"Erreur de chargement modèle: {e}")
            self.yolo_enabled.set(False)

    def ensure_haar_loaded(self):
        try:
            if self.face_cascade is None:
                haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self.face_cascade = cv2.CascadeClassifier(haar_path)
                if self.face_cascade.empty():
                    raise Exception("Haar model not found")
        except Exception as e:
            messagebox.showerror("Face", f"Erreur de chargement Haar: {e}")
            self.face_enabled.set(False)

    def setup_real_gpio(self):
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            self.led_pins = {
                'ok': 18,
                'ng': 19,
                'alert': 20
            }
            self.buzzer_pin = 21

            for pin in self.led_pins.values():
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)
            GPIO.setup(self.buzzer_pin, GPIO.OUT)
            GPIO.output(self.buzzer_pin, GPIO.LOW)
            self.show_toast("GPIO réel configuré pour Raspberry Pi")
        except Exception as e:
            self.show_toast(f"Erreur GPIO: {e}")

    def handle_surveillance_alert(self, confidence):
        current_time = time.time()
        
        if self.last_detection_time and (current_time - self.last_detection_time) < 5:
            return
            
        if self.is_avoiding_human:
            return
            
        self.last_detection_time = current_time
        self.detection_count += 1
        self.is_avoiding_human = True
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert_msg = f"SURVEILLANCE ALERT #{self.detection_count} - Person detected (conf: {confidence:.2f})"
        
        try:
            with open(self.log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, "SURVEILLANCE", "Person Detection", "ALERT", f"Confidence: {confidence:.2f}"])
        except Exception as e:
            print(f"Erreur log: {e}")
        
        self.show_toast(alert_msg, 5000)
        
        if GPIO_AVAILABLE and hasattr(self, 'led_pins'):
            try:
                GPIO.output(self.led_pins['alert'], GPIO.HIGH)
                self.root.after(2000, lambda: GPIO.output(self.led_pins['alert'], GPIO.LOW))
            except Exception as e:
                print(f"Erreur LED: {e}")
        
        self.play_buzzer_pattern("human")
        self.avoid_human(confidence)

    def init_motor_bridge(self):
        if not SERIAL_AVAILABLE:
            self.show_toast("pyserial indisponible - commandes moteur désactivées")
            return None

        try:
            bridge = MotorBridge(self.serial_port)
            self.show_toast(f"Pont moteur connecté ({self.serial_port})")
            return bridge
        except Exception as exc:
            self.show_toast(f"Pont moteur indisponible: {exc}")
            return None

    def play_buzzer_pattern(self, pattern: str = "obstacle"):
        if not GPIO_AVAILABLE or not hasattr(self, "buzzer_pin"):
            return
        if hasattr(self, "buzzer_enabled") and not self.buzzer_enabled.get():
            return

        def beep(duration_ms):
            try:
                GPIO.output(self.buzzer_pin, GPIO.HIGH)
                time.sleep(duration_ms / 1000.0)
                GPIO.output(self.buzzer_pin, GPIO.LOW)
            except Exception:
                pass

        if pattern == "human":
            threading.Thread(target=beep, args=(400,), daemon=True).start()
        else:
            def double_beep():
                beep(150)
                time.sleep(0.1)
                beep(150)
            threading.Thread(target=double_beep, daemon=True).start()

    def send_drive_command(self, action, **kwargs):
        if not self.motor_bridge:
            return False
        payload = {"action": action, **kwargs}
        success = self.motor_bridge.send_command(payload)
        if not success:
            self.show_toast("Commande moteur échouée")
        return success

    def stop_robot(self, reason="safety"):
        if self.motor_controller:
            self.motor_controller.stop_all()
        if self.send_drive_command("stop", reason=reason):
            self.status_var.set(f"Arrêt d'urgence ({reason})")

    def resume_navigation(self):
        if self.motor_controller and self.navigation_mode.get() == "auto":
            self.motor_controller.move_forward(self.current_speed.get())
        if self.send_drive_command("resume"):
            self.status_var.set("Navigation nominale")

    def avoid_human(self, confidence):
        self.stop_robot(reason="human_detected")
        self.status_var.set(f"⚠️ Humain détecté - Arrêt et calcul navigation...")
        
        stop_duration_ms = int(self.human_avoidance_stop_duration * 1000)
        
        def calculate_and_avoid():
            turn_direction = self.last_turn_direction
            self.last_turn_direction = "right" if turn_direction == "left" else "left"
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                with open(self.log_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, "NAVIGATION", "Human Avoidance", "CALC", f"Direction: {turn_direction}"])
            except Exception as e:
                print(f"Erreur log navigation: {e}")
            
            self.status_var.set(f"🗺️ Calcul terminé → Virage {turn_direction} pour éviter l'humain")
            
            if self.send_drive_command(
                "avoid",
                label="human",
                turn=turn_direction,
                duration=self.human_avoidance_turn_duration,
            ):
                self.show_toast(f"Changement de direction: {turn_direction}", 2000)
                
                def resume_after_turn():
                    self.resume_navigation()
                    self.is_avoiding_human = False
                    self.status_var.set("✅ Navigation reprise - Trajectoire mise à jour")
                
                self.root.after(self.human_avoidance_turn_duration + 500, resume_after_turn)
            else:
                self.root.after(2000, lambda: setattr(self, 'is_avoiding_human', False))
        
        self.root.after(stop_duration_ms, calculate_and_avoid)

    def avoid_obstacle(self, label, duration_ms=800):
        turn_direction = self.last_turn_direction
        self.last_turn_direction = "right" if turn_direction == "left" else "left"
        if self.send_drive_command(
            "avoid",
            label=label,
            turn=turn_direction,
            duration=duration_ms,
        ):
            self.status_var.set(f"Avoid {label} → {turn_direction}")
            self.root.after(duration_ms + 500, self.resume_navigation)

    def handle_obstacle_detection(self, label, confidence):
        if not self.smart_obstacle_mode.get():
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        detail = f"Obstacle {label} (conf: {confidence:.2f})"

        try:
            with open(self.log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, "SMART_ROUTING", "Obstacle", "INFO", detail])
        except Exception as e:
            print(f"Erreur log obstacle: {e}")

        self.show_toast(f"⚠️ Obstacle détecté: {label}", 2500)
        self.status_var.set(f"Obstacle détecté → recalcul trajectoire ({label})")
        self.play_buzzer_pattern("obstacle")
        
        if self.motor_controller:
            if self.navigation_mode.get() == "auto":
                self.avoid_obstacle_motor(label)

    def avoid_obstacle_motor(self, label):
        turn_direction = self.last_turn_direction
        self.last_turn_direction = "right" if turn_direction == "left" else "left"
        
        def obstacle_avoidance():
            if turn_direction == "left":
                self.motor_controller.pivot_left(0.4)
            else:
                self.motor_controller.pivot_right(0.4)
                
            time.sleep(0.8)
            if self.navigation_mode.get() == "auto":
                self.motor_controller.move_forward(self.current_speed.get())
            
            self.root.after(0, lambda: self.status_var.set(f"Obstacle évité → {turn_direction}"))
        
        threading.Thread(target=obstacle_avoidance, daemon=True).start()

    def toggle_surveillance_mode(self):
        if self.surveillance_mode.get():
            if not self.yolo_enabled.get():
                self.show_toast("⚠️ Activez 'YOLO Person' d'abord pour la surveillance", 3000)
                self.surveillance_mode.set(False)
                return
            
            self.show_toast("🔍 Mode Surveillance ACTIVÉ", 3000)
            self.detection_count = 0
            self.current_detections = 0
            self.surveillance_status_var.set("🔍 SURVEILLANCE ACTIVE")
            self.surveillance_status_label.config(text="Surveillance: ACTIVE", foreground="green")
            self.current_detection_summary = {"human": 0, "obstacle": 0}
            self.detection_count_label.config(text="Current: H=0 | O=0 | Total Alerts: 0")
            
            if not self.yolo_enabled.get():
                self.yolo_enabled.set(True)
                self.show_toast("YOLO Person activé automatiquement")
        else:
            self.show_toast("🔍 Mode Surveillance DÉSACTIVÉ", 2000)
            self.surveillance_status_var.set("")
            self.surveillance_status_label.config(text="Surveillance: INACTIVE", foreground="red")

    def start_auto_navigation(self):
        if not self.motor_controller:
            return
            
        self.navigation_mode.set("auto")
        self.motor_controller.move_forward(self.current_speed.get())
        self.status_var.set("Navigation autonome activée")

    def stop_auto_navigation(self):
        self.navigation_mode.set("stopped")
        self.stop_robot("manual_stop")
        self.status_var.set("Navigation arrêtée")

    def manual_control(self, command):
        if not self.motor_controller:
            return
            
        commands = {
            "forward": lambda: self.motor_controller.move_forward(self.current_speed.get()),
            "backward": lambda: self.motor_controller.move_backward(self.current_speed.get()),
            "left": lambda: self.motor_controller.turn_left(self.current_speed.get() * 0.7),
            "right": lambda: self.motor_controller.turn_right(self.current_speed.get() * 0.7),
            "stop": lambda: self.motor_controller.stop_all(),
            "pivot_left": lambda: self.motor_controller.pivot_left(0.5),
            "pivot_right": lambda: self.motor_controller.pivot_right(0.5),
        }
        
        if command in commands:
            commands[command]()
            self.status_var.set(f"Commande manuelle: {command}")

    def setup_gui(self):
        self.menubar = tk.Menu(self.root, bg="#f5f5f5", fg="#212121")
        self.root.config(menu=self.menubar)
        
        file_menu = tk.Menu(self.menubar, tearoff=0, bg="#f5f5f5", fg="#212121")
        file_menu.add_command(label="Load Image", command=self.load_image)
        file_menu.add_command(label="Save Image", command=self.save_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        self.menubar.add_cascade(label="File", menu=file_menu)

        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.status_frame = ttk.Frame(self.main_frame, relief=tk.SUNKEN, borderwidth=1)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var = tk.StringVar(value="Ready | Mode Réglage")
        ttk.Label(self.status_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=5)
        self.time_var = tk.StringVar()
        ttk.Label(self.status_frame, textvariable=self.time_var).pack(side=tk.RIGHT, padx=5)
        
        self.mode_selector = ttk.Combobox(self.status_frame, textvariable=self.mode, values=["Mode Réglage", "Run Mode"], state="readonly", width=15)
        self.mode_selector.pack(side=tk.RIGHT, padx=5)
        
        ttk.Checkbutton(self.status_frame, text="YOLO Person", variable=self.yolo_enabled).pack(side=tk.RIGHT, padx=5)
        ttk.Checkbutton(self.status_frame, text="Face (Haar)", variable=self.face_enabled).pack(side=tk.RIGHT, padx=5)
        ttk.Checkbutton(self.status_frame, text="All Objects", variable=self.detect_all_objects).pack(side=tk.RIGHT, padx=5)
        ttk.Checkbutton(self.status_frame, text="Smart Obstacle", variable=self.smart_obstacle_mode).pack(side=tk.RIGHT, padx=5)
        ttk.Checkbutton(self.status_frame, text="🔊 Buzzer", variable=self.buzzer_enabled).pack(side=tk.RIGHT, padx=5)
        
        surveillance_cb = ttk.Checkbutton(self.status_frame, text="🔍 Surveillance", variable=self.surveillance_mode, command=self.toggle_surveillance_mode)
        surveillance_cb.pack(side=tk.RIGHT, padx=5)
        
        self.surveillance_status_var = tk.StringVar(value="")
        ttk.Label(self.status_frame, textvariable=self.surveillance_status_var, foreground="#ff6b35", font=("Segoe UI", 9, "bold")).pack(side=tk.RIGHT, padx=5)
        
        self.update_time()

        self.paned_window = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        self.left_panel = ttk.Frame(self.paned_window)
        self.paned_window.add(self.left_panel, weight=1)

        self.canvas = tk.Canvas(self.left_panel, width=640, height=480, bg="#000000", highlightthickness=1, highlightbackground="#d4d4d4")
        self.canvas.pack(padx=5, pady=5)

        self.right_panel = ttk.Frame(self.paned_window)
        self.paned_window.add(self.right_panel, weight=1)

        self.notebook = ttk.Notebook(self.right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.inspection_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.inspection_frame, text="🔍 Surveillance")
        
        ttk.Label(self.inspection_frame, text="Surveillance Settings", font=("Segoe UI", 12, "bold")).pack(pady=10)
        
        yolo_frame = ttk.LabelFrame(self.inspection_frame, text="YOLO Person Detection", padding=10)
        yolo_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(yolo_frame, text="Confidence Threshold:").pack(side=tk.LEFT)
        ttk.Scale(yolo_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL, variable=self.params["yolo_confidence"], length=150).pack(side=tk.LEFT, padx=5)
        ttk.Label(yolo_frame, textvariable=self.params["yolo_confidence"]).pack(side=tk.LEFT, padx=5)
        
        face_frame = ttk.LabelFrame(self.inspection_frame, text="Face Detection (Haar)", padding=10)
        face_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(face_frame, text="Scale Factor:").pack(side=tk.LEFT)
        ttk.Scale(face_frame, from_=1.05, to=1.5, orient=tk.HORIZONTAL, variable=self.params["face_scale_factor"], length=100).pack(side=tk.LEFT, padx=5)
        ttk.Label(face_frame, textvariable=self.params["face_scale_factor"]).pack(side=tk.LEFT, padx=5)
        ttk.Label(face_frame, text="Min Neighbors:").pack(side=tk.LEFT, padx=(20,0))
        ttk.Scale(face_frame, from_=3, to=10, orient=tk.HORIZONTAL, variable=self.params["face_min_neighbors"], length=100).pack(side=tk.LEFT, padx=5)
        min_neighbors_label = ttk.Label(face_frame, text="5")
        min_neighbors_label.pack(side=tk.LEFT, padx=5)
        
        def update_min_neighbors_label(*args):
            min_neighbors_label.config(text=str(int(self.params["face_min_neighbors"].get())))
        
        self.params["face_min_neighbors"].trace("w", update_min_neighbors_label)
        
        human_avoid_frame = ttk.LabelFrame(self.inspection_frame, text="Human Avoidance", padding=10)
        human_avoid_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(human_avoid_frame, text="Stop Duration (s):").pack(side=tk.LEFT)
        self.human_stop_duration_var = tk.DoubleVar(value=self.human_avoidance_stop_duration)
        ttk.Scale(human_avoid_frame, from_=1.0, to=10.0, orient=tk.HORIZONTAL, variable=self.human_stop_duration_var, length=150).pack(side=tk.LEFT, padx=5)
        ttk.Label(human_avoid_frame, textvariable=self.human_stop_duration_var).pack(side=tk.LEFT, padx=5)
        
        def update_stop_duration(*args):
            self.human_avoidance_stop_duration = self.human_stop_duration_var.get()
        
        self.human_stop_duration_var.trace("w", update_stop_duration)
        
        status_frame = ttk.LabelFrame(self.inspection_frame, text="Status", padding=10)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        self.surveillance_status_label = ttk.Label(status_frame, text="Surveillance: INACTIVE", foreground="red")
        self.surveillance_status_label.pack()
        self.detection_count_label = ttk.Label(status_frame, text="Current: H=0 | O=0 | Total Alerts: 0")
        self.detection_count_label.pack()

        self.control_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.control_frame, text="🚗 Contrôle")
        
        control_inner = ttk.Frame(self.control_frame)
        control_inner.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(control_inner, text="Contrôle du Robot", font=("Segoe UI", 14, "bold")).pack(pady=10)
        
        mode_frame = ttk.Frame(control_inner)
        mode_frame.pack(pady=10)
        
        ttk.Radiobutton(mode_frame, text="Autonome", variable=self.navigation_mode, 
                       value="auto", command=self.start_auto_navigation).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_frame, text="Manuel", variable=self.navigation_mode, 
                       value="manual").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_frame, text="Arrêt", variable=self.navigation_mode,
                       value="stopped", command=self.stop_auto_navigation).pack(side=tk.LEFT, padx=5)
        
        speed_frame = ttk.LabelFrame(control_inner, text="Vitesse", padding=10)
        speed_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(speed_frame, text="Vitesse:").pack(side=tk.LEFT)
        ttk.Scale(speed_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL,
                  variable=self.current_speed, length=200).pack(side=tk.LEFT, padx=10)
        ttk.Label(speed_frame, textvariable=self.current_speed).pack(side=tk.LEFT)
        
        button_frame = ttk.Frame(control_inner)
        button_frame.pack(pady=20)
        
        row1 = ttk.Frame(button_frame)
        row1.pack(pady=5)
        ttk.Button(row1, text="▲ Avancer", width=15,
                   command=lambda: self.manual_control("forward")).pack(side=tk.LEFT, padx=5)
        
        row2 = ttk.Frame(button_frame)
        row2.pack(pady=5)
        ttk.Button(row2, text="◀ Tourner G", width=15,
                   command=lambda: self.manual_control("left")).pack(side=tk.LEFT, padx=5)
        ttk.Button(row2, text="■ Arrêter", width=15,
                   command=lambda: self.manual_control("stop")).pack(side=tk.LEFT, padx=5)
        ttk.Button(row2, text="Tourner D ▶", width=15,
                   command=lambda: self.manual_control("right")).pack(side=tk.LEFT, padx=5)
        
        row3 = ttk.Frame(button_frame)
        row3.pack(pady=5)
        ttk.Button(row3, text="▼ Reculer", width=15,
                   command=lambda: self.manual_control("backward")).pack(side=tk.LEFT, padx=5)
        
        pivot_frame = ttk.Frame(control_inner)
        pivot_frame.pack(pady=10)
        
        ttk.Button(pivot_frame, text="⮜ Pivoter G", 
                   command=lambda: self.manual_control("pivot_left")).pack(side=tk.LEFT)