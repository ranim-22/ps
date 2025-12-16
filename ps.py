#!/usr/bin/env python3
"""
HMI-ROBOT.py - Robot de Surveillance avec Raspberry Pi 4
Version modifiée avec support caméra Raspberry Pi CSI
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
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import threading
import time
import sys

# --- Détection automatique GPIO ---
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("⚠️  RPi.GPIO non disponible - mode simulation activé")

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    serial = None
    SERIAL_AVAILABLE = False
    print("⚠️  pyserial non disponible - commandes moteur désactivées")

class MotorBridge:
    """Simple serial bridge to the Arduino motor controller."""

    def __init__(self, port, baudrate=115200, timeout=0.2):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial = None
        try:
            self.serial = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
            print(f"✅ Pont moteur connecté sur {port}")
        except Exception as exc:
            print(f"❌ Erreur connexion pont moteur: {exc}")

    def send_command(self, payload):
        if not self.serial or not self.serial.is_open:
            return False
        try:
            data = json.dumps(payload).encode("utf-8") + b"\n"
            self.serial.write(data)
            return True
        except Exception as exc:
            print(f"[MotorBridge] Échec envoi: {exc}")
            return False

    def close(self):
        try:
            if self.serial and self.serial.is_open:
                self.serial.close()
                print("✅ Pont moteur fermé")
        except Exception:
            pass

class VisionHMI:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 Robot de Surveillance - Raspberry Pi 4")
        self.root.geometry("1280x720")
        self.root.minsize(1024, 600)
        self.root.configure(bg="#2c3e50")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Variables de caméra
        self.cap = None
        self.picam2 = None
        self.use_static_image = False
        self.static_image = None
        self.camera_type = None  # "picamera2", "usb", ou "static"
        
        # Initialisation de la caméra (MÉTHODE MODIFIÉE POUR RASPBERRY PI)
        self.init_camera()
        
        # Modes
        self.mode = tk.StringVar(value="Mode Réglage")
        self.mode.trace("w", self.update_mode)
        
        # ROI management
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
        
        # Surveillance parameters
        self.params = {
            "gpio_trigger_pin": tk.IntVar(value=-1),
            "yolo_confidence": tk.DoubleVar(value=0.5),
            "face_scale_factor": tk.DoubleVar(value=1.1),
            "face_min_neighbors": tk.IntVar(value=5),
        }
        
        # Surveillance features only
        self.surveillance_features = {
            "YOLO Person Detection": tk.BooleanVar(value=True),
            "Face Detection": tk.BooleanVar(value=False),
            "Motion Detection": tk.BooleanVar(value=False),
        }
        
        # Surveillance cycle
        self.cycle_state = "Idle"
        self.cycle_results = {}
        
        # GPIO simulation
        self.gpio_trigger_active = False
        self.gpio_thread = None
        
        # Logging
        self.log_file = "robot_surveillance_log.csv"
        self.init_log()
        
        # YOLO integration
        self.yolo_enabled = tk.BooleanVar(value=False)
        self.yolo_model = None
        
        # Face detection
        self.face_enabled = tk.BooleanVar(value=False)
        self.face_cascade = None
        
        # Surveillance features
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
        self.human_class_ids = {0}  # YOLO class 0 = person
        
        # Configuration moteur
        self.serial_port = "/dev/ttyACM0"  # Port typique pour Arduino sur Raspberry Pi
        self.motor_bridge = None
        self.last_turn_direction = "left"
        self.buzzer_enabled = tk.BooleanVar(value=True)
        self.human_avoidance_stop_duration = 3.0
        self.human_avoidance_turn_duration = 1200
        self.is_avoiding_human = False
        
        # --- Mode simulation GPIO ---
        self.gpio_simulation = not GPIO_AVAILABLE
        
        # Setup GPIO réel si disponible
        if GPIO_AVAILABLE:
            self.setup_real_gpio()
        
        # Initialisation du pont moteur
        self.motor_bridge = self.init_motor_bridge()
        
        # Setup GUI
        self.setup_gui()
        self.load_settings()
        
        # Démarrer le flux vidéo
        self.update_video()
        
        # Démarrer simulation GPIO (si pin configuré)
        self.start_gpio_simulation()
        
        self.last_live_blob_results = None
        print("✅ Interface Robot de Surveillance initialisée")

    # ==================== MÉTHODE INIT_CAMERA MODIFIÉE POUR RASPBERRY PI ====================
    def init_camera(self):
        """Initialise soit la caméra CSI du Raspberry Pi, soit une webcam USB, soit une image statique."""
        self.cap = None
        self.picam2 = None
        self.use_static_image = False
        self.camera_type = None
        
        print("🔍 Recherche des caméras disponibles...")
        
        # OPTION 1: Essayer la caméra CSI du Raspberry Pi avec picamera2
        try:
            from picamera2 import Picamera2
            print("📷 Tentative d'initialisation de la caméra Raspberry Pi (CSI)...")
            self.picam2 = Picamera2()
            
            # Configuration pour l'aperçu
            preview_config = self.picam2.create_preview_configuration(
                main={"size": (640, 480)},
                controls={"FrameRate": 30}
            )
            self.picam2.configure(preview_config)
            self.picam2.start()
            
            # Test rapide de capture
            test_frame = self.picam2.capture_array()
            if test_frame is not None and test_frame.size > 0:
                self.camera_type = "picamera2"
                print("✅ Caméra Raspberry Pi (CSI) activée avec succès!")
                self.show_toast("✅ Caméra Raspberry Pi activée", duration=3000)
                return
            else:
                self.picam2.stop()
                self.picam2.close()
                self.picam2 = None
                print("⚠️  Capture test échouée avec picamera2")
                
        except ImportError:
            print("❌ picamera2 non installé. Installer avec: sudo apt install python3-picamera2")
        except Exception as e:
            print(f"❌ Erreur caméra Raspberry Pi: {e}")
            if hasattr(self, 'picam2') and self.picam2:
                try:
                    self.picam2.close()
                except:
                    pass
                self.picam2 = None
        
        # OPTION 2: Essayer une webcam USB
        print("🔌 Essai avec webcam USB...")
        for camera_index in range(4):  # Essayer /dev/video0 à /dev/video3
            try:
                cap = cv2.VideoCapture(camera_index)
                if cap.isOpened():
                    # Tester la capture
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        # Configurer la résolution
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        self.cap = cap
                        self.camera_type = f"usb_{camera_index}"
                        print(f"✅ Webcam USB détectée sur /dev/video{camera_index}")
                        self.show_toast(f"✅ Webcam USB détectée (index {camera_index})", duration=3000)
                        return
                    else:
                        cap.release()
                else:
                    cap.release()
            except Exception as e:
                print(f"❌ Erreur webcam index {camera_index}: {e}")
                continue
        
        # OPTION 3: Fallback - Image statique
        print("⚠️  Aucune caméra trouvée. Utilisation du mode image statique.")
        self.use_static_image = True
        
        # Essayer de charger une image d'exemple
        sample_images = ["sample_image.jpg", "test_image.png", "/usr/share/raspberrypi-artwork/raspberry-pi-logo.png"]
        for img_path in sample_images:
            if os.path.exists(img_path):
                self.static_image = cv2.imread(img_path)
                if self.static_image is not None:
                    print(f"✅ Image statique chargée: {img_path}")
                    break
        
        # Si aucune image trouvée, créer une image noire avec message
        if self.static_image is None:
            self.static_image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(self.static_image, "🤖 ROBOT DE SURVEILLANCE", (50, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(self.static_image, "Aucune camera detectee", (100, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(self.static_image, "Mode simulation active", (120, 250), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            print("✅ Image de simulation créée")
        
        self.camera_type = "static"
        self.show_toast("⚠️  Mode image statique (pas de caméra)", duration=5000)

    def ensure_yolo_loaded(self):
        try:
            if self.yolo_model is None:
                print("🔄 Chargement du modèle YOLOv8n...")
                from ultralytics import YOLO
                self.yolo_model = YOLO("yolov8n.pt")  # Modèle nano pour Raspberry Pi
                # Configuration optimisée pour Raspberry Pi
                self.yolo_model.overrides['imgsz'] = 320
                self.yolo_model.overrides['device'] = 'cpu'
                self.yolo_model.overrides['half'] = False
                self.yolo_model.overrides['verbose'] = False
                print("✅ Modèle YOLOv8n chargé avec succès")
        except Exception as e:
            print(f"❌ Erreur chargement YOLO: {e}")
            self.yolo_enabled.set(False)

    def ensure_haar_loaded(self):
        try:
            if self.face_cascade is None:
                print("🔄 Chargement du classifieur Haar...")
                haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self.face_cascade = cv2.CascadeClassifier(haar_path)
                if self.face_cascade.empty():
                    raise Exception("Fichier Haar non trouvé")
                print("✅ Classifieur Haar chargé avec succès")
        except Exception as e:
            print(f"❌ Erreur chargement Haar: {e}")
            self.face_enabled.set(False)

    def setup_real_gpio(self):
        """Setup GPIO réel pour Raspberry Pi"""
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            # Configuration des pins pour robot
            self.led_pins = {
                'ok': 18,    # LED verte
                'ng': 19,    # LED rouge
                'alert': 20  # LED bleue
            }
            # Buzzer
            self.buzzer_pin = 21
            
            # Configuration des pins en sortie
            for pin in self.led_pins.values():
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)
            GPIO.setup(self.buzzer_pin, GPIO.OUT)
            GPIO.output(self.buzzer_pin, GPIO.LOW)
            print("✅ GPIO Raspberry Pi configuré")
            self.show_toast("✅ GPIO réel configuré", duration=2000)
        except Exception as e:
            print(f"❌ Erreur GPIO: {e}")
            self.show_toast(f"❌ Erreur GPIO: {e}")

    def init_motor_bridge(self):
        """Initialise la liaison série avec le contrôleur moteur."""
        if not SERIAL_AVAILABLE:
            print("⚠️  pyserial indisponible - commandes moteur désactivées")
            self.show_toast("⚠️  Commandes moteur désactivées (pyserial manquant)", duration=3000)
            return None
        
        ports_to_try = ["/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyUSB0", "/dev/ttyUSB1"]
        
        for port in ports_to_try:
            try:
                if os.path.exists(port):
                    print(f"🔌 Tentative de connexion sur {port}...")
                    bridge = MotorBridge(port, baudrate=115200, timeout=0.2)
                    if bridge.serial and bridge.serial.is_open:
                        return bridge
            except Exception as e:
                print(f"❌ Échec connexion sur {port}: {e}")
                continue
        
        print("⚠️  Aucun contrôleur moteur détecté")
        self.show_toast("⚠️  Aucun contrôleur moteur détecté", duration=3000)
        return None

    # ==================== MÉTHODE UPDATE_VIDEO MODIFIÉE POUR RASPBERRY PI ====================
    def update_video(self):
        try:
            # === CAPTURE DE LA FRAME ===
            if self.use_static_image:
                frame = self.static_image.copy()
                
            elif self.camera_type == "picamera2" and self.picam2:
                # Capture depuis la caméra Raspberry Pi
                try:
                    frame = self.picam2.capture_array()
                    if frame is not None:
                        # Convertir RGB en BGR pour OpenCV
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    else:
                        print("⚠️  Capture picamera2 retournée None")
                        self.root.after(40, self.update_video)
                        return
                except Exception as e:
                    print(f"❌ Erreur capture picamera2: {e}")
                    self.root.after(40, self.update_video)
                    return
                    
            elif self.cap and self.camera_type and self.camera_type.startswith("usb"):
                # Capture depuis webcam USB
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠️  Échec capture USB, tentative de réouverture...")
                    # Tenter de réinitialiser
                    self.cap.release()
                    time.sleep(0.1)
                    camera_index = int(self.camera_type.split("_")[1])
                    self.cap = cv2.VideoCapture(camera_index)
                    if self.cap.isOpened():
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        ret, frame = self.cap.read()
                    
                    if not ret:
                        self.root.after(100, self.update_video)
                        return
            else:
                # Aucune source valide
                self.root.after(100, self.update_video)
                return
            
            # === TRAITEMENT DE LA FRAME ===
            
            # Détection YOLO
            if self.yolo_enabled.get():
                self.ensure_yolo_loaded()
                try:
                    classes_filter = None if self.detect_all_objects.get() else [0]
                    results = self.yolo_model.predict(
                        frame,
                        classes=classes_filter,
                        conf=self.params["yolo_confidence"].get(),
                        verbose=False,
                        save=False,
                        save_txt=False,
                        save_conf=False,
                    )
                    
                    self.current_detections = 0
                    self.current_detection_summary = {"human": 0, "obstacle": 0}
                    
                    for r in results:
                        if getattr(r, 'boxes', None) is None:
                            continue
                        for b in r.boxes:
                            x1, y1, x2, y2 = map(int, b.xyxy[0])
                            conf = float(b.conf[0]) if getattr(b, 'conf', None) is not None else 0.0
                            cls_id = int(b.cls[0]) if getattr(b, 'cls', None) is not None else None
                            
                            label = "unknown"
                            if self.yolo_model and hasattr(self.yolo_model, "names") and cls_id is not None:
                                label = self.yolo_model.names.get(cls_id, str(cls_id))
                            elif cls_id is not None:
                                label = str(cls_id)
                            
                            is_human = cls_id in self.human_class_ids if cls_id is not None else False
                            category = "human" if is_human else "obstacle"
                            color = (0, 255, 0) if is_human else (0, 165, 255)
                            self.current_detection_summary[category] += 1
                            
                            # Dessiner la boîte
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(
                                frame,
                                f"{label} {conf:.2f}",
                                (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                1,
                            )
                            
                            self.current_detections += 1
                            
                            # Surveillance mode - alerte
                            if is_human and self.surveillance_mode.get() and conf >= self.alert_threshold.get():
                                self.handle_surveillance_alert(conf)
                            elif not is_human:
                                self.handle_obstacle_detection(label, conf)
                    
                    # Mettre à jour l'affichage des compteurs
                    if hasattr(self, 'detection_count_label'):
                        human_count = self.current_detection_summary.get("human", 0)
                        obstacle_count = self.current_detection_summary.get("obstacle", 0)
                        summary_text = f"Détections actuelles: Humains={human_count} | Obstacles={obstacle_count} | Alertes totales={self.detection_count}"
                        self.detection_count_label.config(text=summary_text)
                        
                except Exception as yerr:
                    print(f"❌ Erreur YOLO: {yerr}")
                    self.yolo_enabled.set(False)
            
            # Détection de visages (Haar)
            if self.face_enabled.get():
                self.ensure_haar_loaded()
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=self.params["face_scale_factor"].get(),
                        minNeighbors=self.params["face_min_neighbors"].get(),
                        minSize=(60, 60)
                    )
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cv2.putText(frame, "Visage", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                except Exception as herr:
                    print(f"❌ Erreur détection visage: {herr}")
                    self.face_enabled.set(False)
            
            # === AFFICHAGE DE LA FRAME ===
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            
            # === MISE À JOUR DES INFORMATIONS ===
            fps_text = ""
            if hasattr(self, 'last_frame_time'):
                current_time = time.time()
                fps = 1.0 / (current_time - self.last_frame_time)
                fps_text = f" | FPS: {fps:.1f}"
                self.last_frame_time = current_time
            else:
                self.last_frame_time = time.time()
            
            camera_info = f"Caméra: {self.camera_type if self.camera_type else 'Non détectée'}"
            self.status_var.set(f"Prêt | {camera_info}{fps_text}")
            
            # Planifier la prochaine mise à jour
            self.root.after(40, self.update_video)  # ~25 FPS
            
        except Exception as e:
            print(f"❌ Erreur update_video: {e}")
            self.root.after(100, self.update_video)

    # ==================== AUTRES MÉTHODES (inchangées mais incluses pour complétude) ====================

    def handle_surveillance_alert(self, confidence):
        """Gère les alertes de surveillance"""
        current_time = time.time()
        
        # Éviter les alertes répétitives (cooldown 5s)
        if self.last_detection_time and (current_time - self.last_detection_time) < 5:
            return
        
        # Éviter multiples déclenchements
        if self.is_avoiding_human:
            return
            
        self.last_detection_time = current_time
        self.detection_count += 1
        self.is_avoiding_human = True
        
        # Log
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert_msg = f"🚨 ALERTE #{self.detection_count} - Personne détectée (confiance: {confidence:.2f})"
        
        try:
            with open(self.log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, "SURVEILLANCE", "Détection Personne", "ALERTE", f"Confiance: {confidence:.2f}"])
        except Exception as e:
            print(f"❌ Erreur log: {e}")
        
        # Alerte utilisateur
        self.show_toast(alert_msg, duration=5000)
        
        # Contrôle LED
        if GPIO_AVAILABLE and hasattr(self, 'led_pins'):
            try:
                GPIO.output(self.led_pins['alert'], GPIO.HIGH)
                self.root.after(2000, lambda: GPIO.output(self.led_pins['alert'], GPIO.LOW))
            except Exception as e:
                print(f"❌ Erreur LED: {e}")
        
        # Alarme sonore
        self.play_buzzer_pattern("human")
        
        # Séquence d'évitement
        self.avoid_human(confidence)

    def play_buzzer_pattern(self, pattern: str = "obstacle"):
        """Joue un motif sonore sur le buzzer."""
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
            self.show_toast("❌ Commande moteur échouée")
        return success

    def stop_robot(self, reason="safety"):
        """Arrêt immédiat."""
        if self.send_drive_command("stop", reason=reason):
            self.status_var.set(f"⛔ Arrêt d'urgence ({reason})")

    def resume_navigation(self):
        if self.send_drive_command("resume"):
            self.status_var.set("▶️ Navigation nominale")

    def avoid_human(self, confidence):
        """Gère l'évitement d'un être humain."""
        self.stop_robot(reason="human_detected")
        self.status_var.set(f"🚨 Humain détecté - Arrêt et calcul navigation...")
        
        stop_duration_ms = int(self.human_avoidance_stop_duration * 1000)
        
        def calculate_and_avoid():
            turn_direction = self.last_turn_direction
            self.last_turn_direction = "right" if turn_direction == "left" else "left"
            
            # Log
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                with open(self.log_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, "NAVIGATION", "Évitement Humain", "CALCUL", f"Direction: {turn_direction}"])
            except Exception as e:
                print(f"❌ Erreur log navigation: {e}")
            
            self.status_var.set(f"🔄 Calcul terminé → Virage {turn_direction} pour éviter l'humain")
            
            if self.send_drive_command(
                "avoid",
                label="human",
                turn=turn_direction,
                duration=self.human_avoidance_turn_duration,
            ):
                self.show_toast(f"🔄 Changement de direction: {turn_direction}", duration=2000)
                
                def resume_after_turn():
                    self.resume_navigation()
                    self.is_avoiding_human = False
                    self.status_var.set("✅ Navigation reprise - Trajectoire mise à jour")
                
                self.root.after(self.human_avoidance_turn_duration + 500, resume_after_turn)
            else:
                self.root.after(2000, lambda: setattr(self, 'is_avoiding_human', False))
        
        self.root.after(stop_duration_ms, calculate_and_avoid)

    def avoid_obstacle(self, label, duration_ms=800):
        """Contourne un obstacle."""
        turn_direction = self.last_turn_direction
        self.last_turn_direction = "right" if turn_direction == "left" else "left"
        if self.send_drive_command(
            "avoid",
            label=label,
            turn=turn_direction,
            duration=duration_ms,
        ):
            self.status_var.set(f"🔄 Évitement {label} → {turn_direction}")
            self.root.after(duration_ms + 500, self.resume_navigation)

    def handle_obstacle_detection(self, label, confidence):
        """Gère la détection d'obstacle."""
        if not self.smart_obstacle_mode.get():
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        detail = f"Obstacle {label} (confiance: {confidence:.2f})"

        try:
            with open(self.log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, "NAVIGATION", "Obstacle", "INFO", detail])
        except Exception as e:
            print(f"❌ Erreur log obstacle: {e}")

        self.show_toast(f"⚠️ Obstacle détecté: {label}", duration=2500)
        self.status_var.set(f"⚠️ Obstacle détecté → recalcul trajectoire ({label})")
        self.play_buzzer_pattern("obstacle")
        self.avoid_obstacle(label)

    def toggle_surveillance_mode(self):
        """Active/désactive le mode surveillance."""
        if self.surveillance_mode.get():
            if not self.yolo_enabled.get():
                self.show_toast("⚠️ Activez 'YOLO Person' d'abord pour la surveillance", duration=3000)
                self.surveillance_mode.set(False)
                return
            
            self.show_toast("🔍 Mode Surveillance ACTIVÉ", duration=3000)
            self.detection_count = 0
            self.current_detections = 0
            self.surveillance_status_var.set("🔍 SURVEILLANCE ACTIVE")
            self.surveillance_status_label.config(text="Surveillance: ACTIVE", foreground="green")
            self.current_detection_summary = {"human": 0, "obstacle": 0}
            self.detection_count_label.config(text="Détections actuelles: H=0 | O=0 | Alertes totales: 0")
            
            if not self.yolo_enabled.get():
                self.yolo_enabled.set(True)
                self.show_toast("✅ YOLO Person activé automatiquement")
        else:
            self.show_toast("🔍 Mode Surveillance DÉSACTIVÉ", duration=2000)
            self.surveillance_status_var.set("")
            self.surveillance_status_label.config(text="Surveillance: INACTIVE", foreground="red")

    def setup_gui(self):
        # Menu bar
        self.menubar = tk.Menu(self.root, bg="#34495e", fg="#ecf0f1")
        self.root.config(menu=self.menubar)
        
        file_menu = tk.Menu(self.menubar, tearoff=0, bg="#2c3e50", fg="#ecf0f1")
        file_menu.add_command(label="📁 Charger Image", command=self.load_image)
        file_menu.add_command(label="💾 Sauvegarder Image", command=self.save_image)
        file_menu.add_separator()
        file_menu.add_command(label="🚪 Quitter", command=self.on_closing)
        self.menubar.add_cascade(label="Fichier", menu=file_menu)
        
        # Main layout
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_frame = ttk.Frame(self.main_frame, relief=tk.SUNKEN, borderwidth=1)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var = tk.StringVar(value="🤖 Robot de Surveillance - Prêt")
        ttk.Label(self.status_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=5)
        self.time_var = tk.StringVar()
        ttk.Label(self.status_frame, textvariable=self.time_var).pack(side=tk.RIGHT, padx=5)
        
        # Mode selector
        self.mode_selector = ttk.Combobox(
            self.status_frame,
            textvariable=self.mode,
            values=["Mode Réglage", "Run Mode"],
            state="readonly",
            width=15
        )
        self.mode_selector.pack(side=tk.RIGHT, padx=5)
        
        # Detection toggles
        ttk.Checkbutton(self.status_frame, text="👤 YOLO Person", variable=self.yolo_enabled).pack(side=tk.RIGHT, padx=5)
        ttk.Checkbutton(self.status_frame, text="😀 Visage (Haar)", variable=self.face_enabled).pack(side=tk.RIGHT, padx=5)
        ttk.Checkbutton(self.status_frame, text="📦 Tous Objets", variable=self.detect_all_objects).pack(side=tk.RIGHT, padx=5)
        ttk.Checkbutton(self.status_frame, text="🧠 Évitement Intelligent", variable=self.smart_obstacle_mode).pack(side=tk.RIGHT, padx=5)
        ttk.Checkbutton(self.status_frame, text="🔊 Buzzer", variable=self.buzzer_enabled).pack(side=tk.RIGHT, padx=5)
        
        # Surveillance mode toggle
        surveillance_cb = ttk.Checkbutton(
            self.status_frame,
            text="🔍 Surveillance",
            variable=self.surveillance_mode,
            command=self.toggle_surveillance_mode
        )
        surveillance_cb.pack(side=tk.RIGHT, padx=5)
        
        # Surveillance status indicator
        self.surveillance_status_var = tk.StringVar(value="")
        ttk.Label(
            self.status_frame,
            textvariable=self.surveillance_status_var,
            foreground="#e74c3c",
            font=("Segoe UI", 9, "bold")
        ).pack(side=tk.RIGHT, padx=5)
        
        self.update_time()
        
        # Paned window
        self.paned_window = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Left panel: Video
        self.left_panel = ttk.Frame(self.paned_window)
        self.paned_window.add(self.left_panel, weight=1)
        
        self.canvas = tk.Canvas(
            self.left_panel,
            width=640,
            height=480