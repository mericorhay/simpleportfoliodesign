# -*- coding: utf-8 -*-
from __future__ import annotations
import math
import sys
import time
import serial
import serial.tools.list_ports
from dataclasses import dataclass
from typing import List, Optional
import threading
import queue
import warnings
warnings.filterwarnings("ignore")

# LLM için gerekli kütüphaneler
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("⚠️ LLM kütüphaneleri bulunamadı. 'pip install transformers torch' çalıştırın.")

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QPoint, QRect, QPropertyAnimation, QEasingCurve, QTimer, QSequentialAnimationGroup, \
    QParallelAnimationGroup
from PySide6.QtGui import QColor, QPainter, QBrush, QRadialGradient, QFont, QLinearGradient, QPainterPath, QFontMetrics
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
    QScrollArea, QFrame, QSizePolicy, QTextEdit
)

# Sabit renk paleti - Apple Modern
GLASS_BG = QColor(20, 20, 30, 180)
CARD_BG = QColor(28, 28, 40, 120)
BORDER_GLOW = QColor(64, 156, 255, 100)  # Apple Blue
BORDER_SUBTLE = QColor(255, 255, 255, 30)

TEXT_BRIGHT = QColor(255, 255, 255, 250)
TEXT_NORMAL = QColor(235, 235, 245, 220)
TEXT_MUTED = QColor(174, 174, 178, 180)

ACCENT_BLUE = QColor(64, 156, 255)
BUTTON_GRADIENT_START = QColor(64, 156, 255, 140)
BUTTON_GRADIENT_END = QColor(30, 144, 255, 120)


class LocalLLM(QtCore.QObject):
    """Lokal çalışan LLM - AirDarwin için optimize edilmiş"""
    response_ready = QtCore.Signal(str)
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.is_loading = False
        self.is_ready = False
        self.request_queue = queue.Queue()
        self.worker_thread = None
        
        # AirDarwin specific knowledge base
        self.knowledge_base = {
            'flight_modes': {
                'ARMED': 'Motor çalışıyor ama uçuş başlamadı. Güvenli konum.',
                'TAKEOFF': 'Kalkış modunda. Otomatik yükseliş.',
                'CRUISE': 'Normal uçuş modu. Waypoint navigasyon aktif.',
                'LANDING': 'İniş modu. Otomatik alçalış.',
                'RTL': 'Return to Launch - Ana üsse dönüş.',
                'EMERGENCY': 'Acil durum modu. İmha prosedürü aktif.'
            },
            'safety_states': {
                'SAFE': 'Güvenli durum. Normal operasyon.',
                'CAUTION': 'Dikkat gerektiren durum.',
                'WARNING': 'Uyarı durumu. Pilot müdahalesi öneriliyor.',
                'CRITICAL': 'Kritik durum. Acil müdahale gerekli.',
                'EMERGENCY': 'Acil durum. İmha prosedürü başlatılabilir.'
            },
            'commands': {
                'motor_on': 'Motoru çalıştırır',
                'motor_off': 'Motoru durdurur',
                'takeoff_start': 'Kalkış prosedürünü başlatır',
                'landing_start': 'İniş prosedürünü başlatır',
                'go_around': 'İnişten vazgeç, yeniden tur at',
                'reset': 'Sistemi sıfırla'
            }
        }
        
        if LLM_AVAILABLE:
            self._initialize_llm()
    
    def _initialize_llm(self):
        """LLM'i arka planda yükle"""
        if self.is_loading or self.is_ready:
            return
            
        self.is_loading = True
        self.worker_thread = threading.Thread(target=self._load_model, daemon=True)
        self.worker_thread.start()
    
    def _load_model(self):
        """Küçük ve hızlı model yükle"""
        try:
            # DistilGPT-2 - Hızlı ve küçük model
            model_name = "distilgpt2"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            self.is_ready = True
            self.is_loading = False
            print("✅ LLM hazır - DistilGPT-2 yüklendi")
            
            # Worker thread başlat
            threading.Thread(target=self._process_requests, daemon=True).start()
            
        except Exception as e:
            print(f"❌ LLM yükleme hatası: {e}")
            self.is_loading = False
            self.is_ready = False
    
    def _process_requests(self):
        """Request queue'dan sürekli işle"""
        while True:
            try:
                prompt = self.request_queue.get(timeout=1)
                response = self._generate_response(prompt)
                self.response_ready.emit(response)
                self.request_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"LLM işleme hatası: {e}")
    
    def ask(self, question: str):
        """Soru sor - async"""
        if not self.is_ready:
            if not self.is_loading:
                self._initialize_llm()
            return self._fallback_response(question)
        
        # Queue'ya ekle
        self.request_queue.put(question)
        return "🤖 LLM düşünüyor..."
    
    def _generate_response(self, prompt: str) -> str:
        """LLM ile cevap üret"""
        try:
            # AirDarwin context ekle
            context = """AirDarwin otopilot sistemi için akıllı asistan.
Uçuş güvenliği ve teknik destek sağlar.
Kısa ve net cevaplar verir.

Soru: """
            
            full_prompt = context + prompt + "\nCevap:"
            
            inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=200)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Sadece yeni üretilen kısmı al
            if "Cevap:" in response:
                response = response.split("Cevap:")[-1].strip()
            
            # Temizle ve kısalt
            response = response[:200].strip()
            if not response:
                return self._fallback_response(prompt)
                
            return f"🤖 {response}"
            
        except Exception as e:
            print(f"LLM üretim hatası: {e}")
            return self._fallback_response(prompt)
    
    def _fallback_response(self, question: str) -> str:
        """LLM yokken knowledge base kullan"""
        question_lower = question.lower()
        
        # Flight mode sorguları
        for mode, description in self.knowledge_base['flight_modes'].items():
            if mode.lower() in question_lower:
                return f"🎯 {mode}: {description}"
        
        # Safety state sorguları  
        for state, description in self.knowledge_base['safety_states'].items():
            if state.lower() in question_lower:
                return f"🛡️ {state}: {description}"
        
        # Komut sorguları
        for cmd, description in self.knowledge_base['commands'].items():
            if cmd in question_lower:
                return f"🎮 {cmd}: {description}"
        
        # Genel AirDarwin soruları
        if any(word in question_lower for word in ['battery', 'batarya', 'pil']):
            return "🔋 Batarya durumu: %20 altında acil iniş, %40 altında RTL öneriliyor."
        elif any(word in question_lower for word in ['gps', 'satellit']):
            return "🛰️ GPS: En az 6 uydu güvenli uçuş için gerekli. HDOP <1.8 olmalı."
        elif any(word in question_lower for word in ['speed', 'hız', 'airspeed']):
            return "🏃 Hız sınırları: Min 28 km/h (Vs), Max 110 km/h (VNE). Optimal: 50-70 km/h"
        elif any(word in question_lower for word in ['altitude', 'irtifa', 'yükseklik']):
            return "📏 İrtifa: Yasal limit 350m. Güvenli operasyon 50-200m arası."
        elif any(word in question_lower for word in ['wind', 'rüzgar']):
            return "💨 Rüzgar limitleri: Max 25 km/h genel, 15 km/h yan rüzgar."
        elif any(word in question_lower for word in ['emergency', 'acil', 'kritik']):
            return "🚨 Acil durumda: Motor durdur, RTL aktif et, manuel kontrol al."
        else:
            return "❓ AirDarwin hakkında daha spesifik soru sorabilirsiniz. Örnek: battery, gps, speed, altitude"


class ComPortSelector(QWidget):
    port_changed = QtCore.Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(200, 40)
        
        self._setup_ui()
        self._refresh_ports()
        
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # COM port dropdown
        self.port_combo = QtWidgets.QComboBox()
        self.port_combo.setStyleSheet("""
            QComboBox {
                background: rgba(28, 28, 40, 160);
                border: 1px solid rgba(64, 156, 255, 100);
                border-radius: 8px;
                padding: 5px 10px;
                color: white;
                font: 10pt 'SF Pro Text';
                min-width: 120px;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid rgba(255, 255, 255, 200);
                margin-right: 5px;
            }
            QComboBox QAbstractItemView {
                background: rgba(28, 28, 40, 240);
                border: 1px solid rgba(64, 156, 255, 100);
                border-radius: 8px;
                color: white;
                selection-background-color: rgba(64, 156, 255, 100);
            }
        """)
        
        self.port_combo.currentTextChanged.connect(self.port_changed.emit)
        
        # Refresh button
        self.refresh_btn = QPushButton("🔄")
        self.refresh_btn.setFixedSize(30, 30)
        self.refresh_btn.setStyleSheet("""
            QPushButton {
                background: rgba(64, 156, 255, 100);
                border: 1px solid rgba(64, 156, 255, 150);
                border-radius: 15px;
                color: white;
                font-size: 12pt;
            }
            QPushButton:hover {
                background: rgba(64, 156, 255, 150);
            }
            QPushButton:pressed {
                background: rgba(54, 146, 245, 180);
            }
        """)
        self.refresh_btn.clicked.connect(self._refresh_ports)
        
        layout.addWidget(self.port_combo)
        layout.addWidget(self.refresh_btn)
        
    def _refresh_ports(self):
        self.port_combo.clear()
        self.port_combo.addItem("No Connection", "")
        
        ports = serial.tools.list_ports.comports()
        for port in ports:
            display_name = f"{port.device} - {port.description}"
            self.port_combo.addItem(display_name, port.device)
            
    def get_selected_port(self):
        return self.port_combo.currentData()


class ExitButton(QWidget):
    clicked = QtCore.Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(40, 40)
        self.setCursor(Qt.PointingHandCursor)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Background circle
        painter.setBrush(QBrush(QColor(255, 59, 48, 100)))  # Apple Red
        painter.setPen(QtGui.QPen(QColor(255, 59, 48, 150), 1))
        painter.drawEllipse(5, 5, 30, 30)
        
        # X mark
        painter.setPen(QtGui.QPen(QColor(255, 255, 255, 220), 2))
        painter.drawLine(15, 15, 25, 25)
        painter.drawLine(25, 15, 15, 25)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()


class SerialCommunication(QtCore.QObject):
    data_received = QtCore.Signal(dict)
    connection_status = QtCore.Signal(bool, str)
    
    def __init__(self):
        super().__init__()
        self.serial_port = None
        self.current_port = ""
        
        # Timer for reading data
        self.read_timer = QTimer()
        self.read_timer.timeout.connect(self._read_data)
        
    def connect(self, port_name: str):
        """Connect to specified COM port"""
        if self.is_connected():
            self.disconnect()
            
        if not port_name or port_name == "No Connection":
            return False
            
        try:
            self.serial_port = serial.Serial(
                port=port_name,
                baudrate=9600,  # AirDarwin.ino uses 9600
                timeout=1.0
            )
            self.current_port = port_name
            self.read_timer.start(100)  # Read every 100ms
            self.connection_status.emit(True, f"Connected to {port_name}")
            return True
        except Exception as e:
            self.connection_status.emit(False, f"Failed to connect: {str(e)}")
            return False
            
    def disconnect(self):
        """Disconnect from COM port"""
        if self.serial_port:
            self.read_timer.stop()
            self.serial_port.close()
            self.serial_port = None
            self.current_port = ""
            self.connection_status.emit(False, "Disconnected")
            
    def is_connected(self):
        """Check if currently connected"""
        return self.serial_port is not None and self.serial_port.is_open
            
    def send_command(self, command: str):
        """Send command to AirDarwin"""
        if self.is_connected():
            try:
                self.serial_port.write(f"{command}\n".encode())
                return True
            except Exception as e:
                self.connection_status.emit(False, f"Send error: {str(e)}")
                return False
        return False
        
    def _read_data(self):
        """Read data from serial port"""
        if not self.is_connected():
            return
            
        try:
            if self.serial_port.in_waiting > 0:
                line = self.serial_port.readline().decode('utf-8').strip()
                if line:
                    telemetry_data = self._parse_telemetry(line)
                    if telemetry_data:
                        self.data_received.emit(telemetry_data)
        except Exception as e:
            self.connection_status.emit(False, f"Read error: {str(e)}")
            
    def _parse_telemetry(self, line: str) -> dict:
        """Parse AirDarwin.ino telemetry format"""
        try:
            # Parse summary format: SUM:MODE|AS:50.5|Alt:120|Hdg:180|...
            if line.startswith("SUM:"):
                parts = line[4:].split("|")
                data = {}
                
                for part in parts:
                    if ":" in part:
                        key, value = part.split(":", 1)
                        
                        if key == "AS":  # Airspeed
                            data['airspeed'] = float(value)
                        elif key == "Alt":  # Altitude
                            data['altitude'] = float(value)
                        elif key == "Hdg":  # Heading
                            data['heading'] = float(value)
                        elif key == "Att":  # Attitude (roll,pitch)
                            roll_pitch = value.split(",")
                            if len(roll_pitch) == 2:
                                data['roll'] = float(roll_pitch[0])
                                data['pitch'] = float(roll_pitch[1])
                        elif key == "Thr":  # Throttle
                            data['throttle'] = int(value)
                        elif key == "WP":  # Waypoint
                            data['waypoint'] = int(value)
                        elif key == "Saf":  # Safety
                            data['safety_state'] = value
                        elif key == "Sys":  # System flags
                            data['gps_ok'] = 'G' in value
                            data['imu_ok'] = 'I' in value
                            data['link_ok'] = 'L' in value
                            data['motor_armed'] = 'M' in value
                        elif len(parts) > 0:  # First part is flight mode
                            data['mode'] = parts[0]
                            
                return data
                
        except Exception as e:
            print(f"Parse error: {e}")
            
        return None


class ModernBackground(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Apple style gradient background
        base_gradient = QRadialGradient(self.rect().center(), max(self.width(), self.height()) * 0.8)
        base_gradient.setColorAt(0.0, QColor(8, 12, 18, 200))
        base_gradient.setColorAt(0.6, QColor(12, 16, 22, 220))
        base_gradient.setColorAt(1.0, QColor(18, 25, 30, 240))
        painter.fillRect(self.rect(), base_gradient)


class StreamingMessage(QWidget):
    def __init__(self, text: str, is_user: bool = False, parent=None):
        super().__init__(parent)
        self.full_text = text
        self.is_user = is_user
        self.displayed_text = ""
        self.char_index = 0

        self.setMinimumHeight(60)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self._setup_entrance_animation()

        if not is_user:
            self.typing_timer = QTimer(self)
            self.typing_timer.timeout.connect(self._type_next_char)
            QTimer.singleShot(300, self._start_typing)
        else:
            self.displayed_text = self.full_text

    def _setup_entrance_animation(self):
        self.slide_anim = QPropertyAnimation(self, b"pos")
        self.slide_anim.setDuration(800)
        self.slide_anim.setEasingCurve(QEasingCurve.OutQuad)

        self.opacity_effect = QtWidgets.QGraphicsOpacityEffect()
        self.setGraphicsEffect(self.opacity_effect)

        self.fade_anim = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_anim.setDuration(600)
        self.fade_anim.setStartValue(0.0)
        self.fade_anim.setEndValue(1.0)

    def animate_entrance(self, target_pos: QPoint):
        start_offset = 150 if self.is_user else -150
        start_pos = QPoint(target_pos.x() + start_offset, target_pos.y())

        self.move(start_pos)
        self.slide_anim.setStartValue(start_pos)
        self.slide_anim.setEndValue(target_pos)

        self.slide_anim.start()
        self.fade_anim.start()

    def _start_typing(self):
        self.typing_timer.start(40)

    def _type_next_char(self):
        if self.char_index < len(self.full_text):
            self.displayed_text = self.full_text[:self.char_index + 1]
            self.char_index += 1
            self.update()
        else:
            self.typing_timer.stop()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect().adjusted(20, 10, -20, -10)

        # Apple tarzı modern mesaj kutusu
        bg_gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
        if self.is_user:
            bg_gradient.setColorAt(0.0, QColor(64, 156, 255, 60))
            bg_gradient.setColorAt(1.0, QColor(30, 144, 255, 40))
        else:
            bg_gradient.setColorAt(0.0, QColor(28, 28, 40, 80))
            bg_gradient.setColorAt(1.0, QColor(20, 20, 30, 60))

        path = QPainterPath()
        path.addRoundedRect(rect, 18, 18)  # Daha yuvarlak köşeler
        painter.fillPath(path, bg_gradient)

        # Apple tarzı ince border
        painter.setPen(QtGui.QPen(QColor(255, 255, 255, 20), 0.5))
        painter.drawPath(path)

        # Apple San Francisco font tarzı
        text_rect = rect.adjusted(15, 8, -15, -8)
        painter.setPen(TEXT_BRIGHT)
        painter.setFont(QFont("SF Pro Display", 14, QFont.Medium))

        if self.is_user:
            painter.drawText(text_rect, Qt.TextWordWrap | Qt.AlignRight, self.displayed_text)
        else:
            painter.drawText(text_rect, Qt.TextWordWrap | Qt.AlignLeft, self.displayed_text)


class HyperChatArea(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(40, 40, 40, 40)
        self.layout.setSpacing(20)
        self.layout.addStretch(1)

        self.messages: List[StreamingMessage] = []

    def add_message(self, text: str, is_user: bool = False) -> StreamingMessage:
        message = StreamingMessage(text, is_user, self)

        alignment = Qt.AlignRight if is_user else Qt.AlignLeft
        self.layout.insertWidget(self.layout.count() - 1, message, 0, alignment)

        font_metrics = QFontMetrics(QFont("Inter", 14))
        text_width = min(font_metrics.boundingRect(text).width() + 100, 800)
        message.setFixedSize(text_width, font_metrics.boundingRect(0, 0, text_width - 60, 1000, Qt.TextWordWrap,
                                                                   text).height() + 40)

        QTimer.singleShot(50, lambda: message.animate_entrance(message.pos()))

        self.messages.append(message)
        return message


class ModernComposer(QWidget):
    message_sent = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # Tab completion için komutlar - telemetry kaldırıldı
        self.commands = [
            "motor_on", "motor_off", "takeoff_start", "landing_start", 
            "go_around", "reset", "connect", "disconnect", "status", "help"
        ]
        self.current_completion = ""
        self.completion_index = -1

        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(30, 20, 30, 20)
        layout.setSpacing(20)

        input_container = QWidget()
        input_container.setAttribute(Qt.WA_TranslucentBackground)

        input_layout = QHBoxLayout(input_container)
        input_layout.setContentsMargins(25, 15, 25, 15)
        input_layout.setSpacing(15)

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("🚁 AirDarwin komut girin (Tab: otomatik tamamlama)...")

        # Modern Apple tarzı stil
        self.text_input.setStyleSheet("""
            QLineEdit {
                color: rgba(255, 255, 255, 250);
                background: transparent;
                border: none;
                font: 16pt 'SF Pro Display';
                font-weight: 500;
                padding: 12px;
            }
            QLineEdit::placeholder {
                color: rgba(174, 174, 178, 180);
            }
        """)
        self.text_input.returnPressed.connect(self._send_message)
        self.text_input.installEventFilter(self)

        self.send_btn = QPushButton("Send")
        self.send_btn.setFixedSize(90, 48)
        self.send_btn.setCursor(Qt.PointingHandCursor)
        self.send_btn.clicked.connect(self._send_message)

        # Ultra modern Apple tarzı buton
        self.send_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(64, 156, 255, 160),
                    stop:1 rgba(54, 146, 245, 140));
                border: 0.5px solid rgba(255, 255, 255, 40);
                border-radius: 16px;
                color: white;
                font: 12pt 'SF Pro Text';
                font-weight: 600;
                letter-spacing: 0.5px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(74, 166, 255, 190),
                    stop:1 rgba(64, 156, 255, 170));
                border: 0.5px solid rgba(255, 255, 255, 60);
                transform: translateY(-1px);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(54, 146, 245, 200),
                    stop:1 rgba(44, 136, 235, 180));
                border: 0.5px solid rgba(255, 255, 255, 80);
                transform: translateY(0px);
            }
        """)

        input_layout.addWidget(self.text_input, 1)
        input_layout.addWidget(self.send_btn, 0)

        layout.addWidget(input_container, 1)

    def eventFilter(self, obj, event):
        if obj == self.text_input and event.type() == QtCore.QEvent.KeyPress:
            if event.key() == Qt.Key_Tab:
                self._handle_tab_completion()
                return True
        return super().eventFilter(obj, event)

    def _handle_tab_completion(self):
        current_text = self.text_input.text()

        if not current_text:
            return

        # Tam eşleşme ara
        matches = [cmd for cmd in self.commands if cmd.startswith(current_text.lower())]

        if matches:
            if self.current_completion != current_text:
                # Yeni completion başlat
                self.current_completion = current_text
                self.completion_index = 0
            else:
                # Sonraki eşleşmeye geç
                self.completion_index = (self.completion_index + 1) % len(matches)

            self.text_input.setText(matches[self.completion_index])
        else:
            self.current_completion = ""
            self.completion_index = -1

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Apple tarzı glassmorphism
        bg_path = QPainterPath()
        bg_path.addRoundedRect(self.rect().adjusted(20, 10, -20, -10), 22, 22)

        painter.fillPath(bg_path, GLASS_BG)

        # Subtle border
        painter.setPen(QtGui.QPen(QColor(255, 255, 255, 30), 1))
        painter.drawPath(bg_path)

    def _send_message(self):
        text = self.text_input.text().strip()
        if text:
            self.message_sent.emit(text)
            self.text_input.clear()
            self.current_completion = ""
            self.completion_index = -1


class ModernHUD(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # AirDarwin.ino telemetri verilerine uygun değişkenler - real-time data
        self.airspeed = 0.0
        self.altitude = 0.0
        self.heading = 0.0
        self.battery = 0.0
        self.mode = "NO DATA"
        self.armed = False
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.gps_sats = 0
        self.hdop = 99.0
        self.safety_state = "UNKNOWN"
        self.takeoff_state = "IDLE"
        self.throttle = 1000
        self.wind_speed = 0.0
        
        # Communication status
        self.connection_status = "DISCONNECTED"
        self.last_data_time = 0
        self.data_timeout = 5.0  # seconds
        
        # Akıllı telemetri sistemi için ek değişkenler
        self.rssi = -45  # Signal strength (dBm)
        self.packets_lost = 0
        self.link_quality = 100
        self.cpu_load = 15
        self.memory_usage = 45
        self.temperature = 25.0
        self.vibration_level = 0.2
        self.last_heartbeat = 0
        self.flight_time = 0  # seconds
        self.distance_traveled = 0.0  # km
        self.max_altitude = 0.0
        self.avg_speed = 0.0
        
        # Gelişmiş akıllı sistem - AirDarwin.ino uyumlu
        self.fuel_efficiency = 100.0  # Battery efficiency %
        self.wind_direction = 270.0  # degrees (stable westerly)
        self.crosswind_component = 0.0  # km/h
        self.headwind_component = 0.0  # km/h
        self.ground_speed = 0.0  # km/h (airspeed corrected for wind)
        self.climb_rate = 0.0  # m/s
        self.sink_rate = 0.0  # m/s
        self.g_force = 1.0  # G acceleration
        self.magnetic_declination = 12.5  # degrees (stable)
        self.pressure_altitude = 0.0  # m
        self.density_altitude = 0.0  # m
        self.true_airspeed = 0.0  # km/h
        self.mach_number = 0.0  # Speed ratio
        
        # Performans metrikleri - deterministik
        self.max_climb_rate = 0.0  # m/s
        self.max_descent_rate = 0.0  # m/s
        self.max_speed = 0.0  # km/h
        self.min_speed = 999.0  # km/h
        self.total_energy = 0.0  # Joules
        self.power_consumption = 0.0  # Watts
        self.efficiency_score = 100.0  # %
        
        # Akıllı uyarı sistemi
        self.warnings = []
        self.critical_alerts = []
        self.smart_recommendations = []
        self.performance_tips = []
        self.system_status = []

        self.pulse_phase = 0.0

        self.pulse_timer = QTimer(self)
        self.pulse_timer.timeout.connect(self._update_pulse)
        self.pulse_timer.start(60)
        
        # Remove smart analysis timer - we only display received data
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._check_connection_status)
        self.status_timer.start(1000)  # Check connection every second

    def update_from_telemetry(self, data: dict):
        """Update HUD with real telemetry data from AirDarwin"""
        self.last_data_time = time.time()
        
        # Update values from received data
        self.airspeed = data.get('airspeed', self.airspeed)
        self.altitude = data.get('altitude', self.altitude)
        self.heading = data.get('heading', self.heading)
        self.battery = data.get('battery', self.battery)
        self.mode = data.get('mode', self.mode)
        self.armed = data.get('motor_armed', self.armed)
        self.roll = data.get('roll', self.roll)
        self.pitch = data.get('pitch', self.pitch)
        self.throttle = data.get('throttle', self.throttle)
        self.safety_state = data.get('safety_state', self.safety_state)
        
        # Calculate GPS satellites from flags if available
        if 'gps_ok' in data:
            self.gps_sats = 8 if data['gps_ok'] else 3
            
        # Analyze received data for safety alerts
        self._analyze_real_data()
        
        # Update smart features too
        self._analyze_smart_telemetry()
            
        self.update()
        
    def _check_connection_status(self):
        """Check if we're receiving data"""
        if time.time() - self.last_data_time > self.data_timeout:
            self.connection_status = "NO DATA"
        else:
            self.connection_status = "RECEIVING"

    def _update_pulse(self):
        self.pulse_phase += 0.05
        self.update()
        
    def _analyze_real_data(self):
        """Simple real-time safety analysis based on received telemetry"""
        self.warnings.clear()
        self.critical_alerts.clear()
        
        # Only analyze if we have data
        if self.connection_status == "NO DATA":
            self.critical_alerts.append("🚨 NO TELEMETRY DATA - Check connection")
            return
            
        # AirDarwin.ino safety constants
        CRITICAL_MIN_AIRSPEED = 28.0
        STALL_WARNING_SPEED = 33.0
        CRITICAL_MAX_AIRSPEED = 110.0
        CRITICAL_MAX_ROLL = 30.0
        
        # Critical alerts based on received data only
        if self.battery < 15:
            self.critical_alerts.append(f"🚨 CRITICAL BATTERY: {self.battery:.0f}% - LAND NOW")
        elif self.battery < 25:
            self.critical_alerts.append(f"⚠️ LOW BATTERY: {self.battery:.0f}% - Plan landing")
            
        if self.gps_sats < 4:
            self.critical_alerts.append("🛰️ GPS CRITICAL: <4 sats - Manual control only")
        
        if abs(self.roll) > CRITICAL_MAX_ROLL:
            self.critical_alerts.append("🔄 ATTITUDE CRITICAL: Roll limit exceeded")
            
        # Stall protection
        if self.airspeed <= CRITICAL_MIN_AIRSPEED and self.armed:
            self.critical_alerts.append("🚨 STALL CRITICAL: Speed below Vs - Recovery needed")
            
        # Warnings
        if self.gps_sats < 6 and not any("GPS CRITICAL" in alert for alert in self.critical_alerts):
            self.warnings.append(f"🛰️ GPS LIMITED: {self.gps_sats} sats")
            
        if self.armed and self.airspeed < STALL_WARNING_SPEED:
            self.warnings.append("🏃 SPEED LOW: Approaching stall")
        elif self.airspeed > CRITICAL_MAX_AIRSPEED * 0.9:
            self.warnings.append("⚡ SPEED HIGH: Approaching VNE")
            
        if self.altitude > 350:
            self.warnings.append("📏 HIGH ALTITUDE: Check regulations")
        elif self.armed and self.altitude < 3:
            self.warnings.append("⬇️ VERY LOW: Ground collision risk")

    def _analyze_smart_telemetry(self):
        self.warnings.clear()
        self.critical_alerts.clear()
        self.smart_recommendations.clear()
        self.performance_tips.clear()
        self.system_status.clear()
        
        # Gelişmiş performans hesaplamaları
        if self.armed:
            # Wind component calculation (deterministic)
            wind_angle = abs(self.heading - self.wind_direction)
            self.crosswind_component = self.wind_speed * math.sin(math.radians(wind_angle))
            self.headwind_component = self.wind_speed * math.cos(math.radians(wind_angle))
            self.ground_speed = max(0, self.airspeed - self.headwind_component)
            
            # Energy calculations (correct aircraft mass from AirDarwin.ino)
            aircraft_mass = 5.0  # kg (actual AirDarwin mass)
            kinetic_energy = 0.5 * aircraft_mass * (self.airspeed / 3.6) ** 2
            potential_energy = aircraft_mass * 9.81 * self.altitude
            self.total_energy = kinetic_energy + potential_energy
            
            # Performance tracking
            if self.airspeed > self.max_speed:
                self.max_speed = self.airspeed
            if self.airspeed < self.min_speed and self.airspeed > 5:
                self.min_speed = self.airspeed
                
            # System performance simulation (stable)
            time_factor = time.time() * 0.1
            self.cpu_load = 25 + int(10 * math.sin(time_factor))  # 15-35%
            self.memory_usage = 50 + int(15 * math.sin(time_factor * 0.7))  # 35-65%
            
            # Signal strength based on flight conditions
            base_rssi = -45
            altitude_factor = self.altitude * 0.1  # Better at altitude
            attitude_factor = abs(self.roll) * 0.3  # Worse in banks
            self.rssi = int(base_rssi + altitude_factor - attitude_factor)
            self.rssi = max(-90, min(-30, self.rssi))
        
        # AirDarwin.ino safety constants
        CRITICAL_MIN_AIRSPEED = 28.0
        STALL_WARNING_SPEED = 33.0
        CRITICAL_MAX_AIRSPEED = 110.0
        CRITICAL_MAX_ROLL = 30.0
        GPS_MIN_SATELLITES = 6
        GPS_MAX_HDOP = 1.8
        
        # Kritik uyarılar - AirDarwin.ino safety logic
        if self.battery < 15:
            remaining_time = int(self.battery * 0.3)
            self.critical_alerts.append(f"🚨 CRITICAL BATTERY: {remaining_time}min left - LAND NOW")
        elif self.battery < 25:
            remaining_time = int(self.battery * 0.4)
            self.critical_alerts.append(f"⚠️ LOW BATTERY: {remaining_time}min - Plan landing")
            
        if self.gps_sats < 4:
            self.critical_alerts.append("🛰️ GPS CRITICAL: <4 sats - Manual control only")
        
        if abs(self.roll) > CRITICAL_MAX_ROLL:
            self.critical_alerts.append("🔄 ATTITUDE CRITICAL: Roll limit exceeded")
            
        # Stall protection (exact AirDarwin.ino logic)
        if self.airspeed <= CRITICAL_MIN_AIRSPEED and self.armed:
            self.critical_alerts.append("🚨 STALL CRITICAL: Speed below Vs - Recovery needed")
            
        # Normal uyarılar
        if self.gps_sats < GPS_MIN_SATELLITES and not any("GPS CRITICAL" in alert for alert in self.critical_alerts):
            self.warnings.append(f"🛰️ GPS LIMITED: {self.gps_sats} sats - Navigation degraded")
            
        if self.hdop > GPS_MAX_HDOP:
            self.warnings.append(f"📍 GPS ACCURACY LOW: HDOP {self.hdop:.1f}")
            
        if self.armed and self.airspeed < STALL_WARNING_SPEED:
            self.warnings.append("🏃 SPEED LOW: Approaching stall - Increase throttle")
        elif self.airspeed > CRITICAL_MAX_AIRSPEED * 0.9:
            self.warnings.append("⚡ SPEED HIGH: Approaching VNE - Reduce throttle")
            
        if self.altitude > 350:
            self.warnings.append("📏 HIGH ALTITUDE: Check local regulations")
        elif self.armed and self.altitude < 3:
            self.warnings.append("⬇️ VERY LOW: Ground collision risk")
            
        if abs(self.crosswind_component) > 15:
            self.warnings.append(f"� STRONG CROSSWIND: {self.crosswind_component:.0f} km/h")
            
        # Akıllı öneriler
        if self.mode == "ARMED" and len(self.critical_alerts) == 0:
            self.smart_recommendations.append("✅ Systems ready - Safe for takeoff")
        elif self.mode == "TAKEOFF" and self.altitude > 30:
            self.smart_recommendations.append("🎯 Good climb - Consider waypoint navigation")
        elif self.battery < 40 and self.altitude > 80:
            self.smart_recommendations.append("🏠 Battery <40% - Consider RTL mode")
        elif self.wind_speed > 20:
            self.smart_recommendations.append("💨 High wind - Use auto-land for safer landing")
            
        # Performans ipuçları
        if self.armed:
            efficiency = (self.distance_traveled / max(0.01, self.flight_time / 3600.0)) / max(1, 100 - self.battery) * 100
            if efficiency > 80:
                self.performance_tips.append("🌟 EXCELLENT flight efficiency")
            elif efficiency < 40:
                self.performance_tips.append("💡 TIP: Smoother controls improve efficiency")
                
            if self.avg_speed > 0:
                speed_efficiency = (self.avg_speed / 60.0) * 100  # Optimal speed ~60 km/h
                if speed_efficiency > 90:
                    self.performance_tips.append("⚡ Optimal cruise speed maintained")
                elif speed_efficiency < 50:
                    self.performance_tips.append("🎯 TIP: 50-70 km/h is most efficient")
        
        # Sistem durumu
        signal_strength = "EXCELLENT" if self.rssi > -40 else "GOOD" if self.rssi > -60 else "WEAK"
        self.system_status.append(f"📡 Signal: {signal_strength} ({self.rssi} dBm)")
        
        temp_status = "NORMAL" if 10 < self.temperature < 40 else "EXTREME"
        self.system_status.append(f"🌡️ Temp: {temp_status} ({self.temperature:.0f}°C)")
        
        cpu_status = "NORMAL" if self.cpu_load < 80 else "HIGH"
        self.system_status.append(f"🖥️ CPU: {cpu_status} ({self.cpu_load}%)")
        
        mem_status = "NORMAL" if self.memory_usage < 85 else "HIGH" 
        self.system_status.append(f"💾 Memory: {mem_status} ({self.memory_usage}%)")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect().adjusted(15, 15, -15, -15)

        # Apple tarzı glassmorphism arka plan
        bg_path = QPainterPath()
        bg_path.addRoundedRect(rect, 24, 24)

        # Safety state'e göre arka plan rengi
        if self.safety_state == "CRITICAL":
            bg_color = QColor(60, 20, 20, 180)
        elif self.safety_state == "WARNING":
            bg_color = QColor(60, 50, 20, 170)
        elif self.safety_state == "CAUTION":
            bg_color = QColor(50, 50, 20, 160)
        else:
            bg_color = QColor(20, 20, 30, 180)

        painter.fillPath(bg_path, bg_color)

        # Apple tarzı subtle border
        pulse_intensity = 0.3 + 0.2 * math.sin(self.pulse_phase)
        if self.safety_state == "CRITICAL":
            border_color = QColor(255, 69, 58, int(120 + 60 * pulse_intensity))  # Apple Red
        elif self.safety_state == "WARNING":
            border_color = QColor(255, 214, 10, int(100 + 50 * pulse_intensity))  # Apple Yellow
        else:
            border_color = QColor(64, 156, 255, int(80 + 40 * pulse_intensity))  # Apple Blue

        painter.setPen(QtGui.QPen(border_color, 1.5))
        painter.drawPath(bg_path)

        # Modern başlık - akıllı gösterge
        painter.setPen(QColor(255, 255, 255, 250))
        painter.setFont(QFont("SF Pro Display", 14, QFont.Bold))
        header_text = f"🧠 SMART TELEMETRY"
        if self.critical_alerts:
            header_text = "🚨 CRITICAL ALERTS"
        elif self.warnings:
            header_text = "⚠️ SYSTEM WARNINGS"
        
        painter.drawText(rect.adjusted(20, 18, -20, -18), Qt.AlignTop | Qt.AlignCenter, header_text)

        # Veri alanı
        data_rect = rect.adjusted(20, 55, -20, -20)
        painter.setFont(QFont("SF Pro Text", 10, QFont.Medium))  # Daha küçük font daha fazla bilgi için

        line_height = 22  # Daha sıkışık satırlar
        y_offset = 0

        # Önce kritik uyarıları göster
        if self.critical_alerts:
            painter.setPen(QColor(255, 69, 58))  # Apple Red
            for alert in self.critical_alerts[:2]:  # Max 2 kritik uyarı
                painter.drawText(data_rect.adjusted(0, y_offset, 0, 0), Qt.AlignLeft, alert)
                y_offset += line_height
            y_offset += 8

        # Flight Mode & Status
        mode_color = QColor(255, 69, 58) if self.armed else QColor(52, 199, 89)
        painter.setPen(mode_color)
        status_icon = "🔴" if self.armed else "🟢"
        painter.drawText(data_rect.adjusted(0, y_offset, 0, 0), Qt.AlignLeft,
                         f"{status_icon} {self.mode}")
        y_offset += line_height

        # Akıllı flight data
        painter.setPen(QColor(64, 156, 255))
        painter.drawText(data_rect.adjusted(0, y_offset, 0, 0), Qt.AlignLeft,
                         f"🏃 {self.airspeed:.0f} km/h • 📏 {self.altitude:.0f}m • 🧭 {self.heading:.0f}°")
        y_offset += line_height

        # Akıllı attitude data
        painter.setPen(QColor(100, 210, 255))
        painter.drawText(data_rect.adjusted(0, y_offset, 0, 0), Qt.AlignLeft,
                         f"🔄 R:{self.roll:.0f}° • ⬆️ P:{self.pitch:.0f}° • ↻ Y:{self.yaw:.0f}°")
        y_offset += line_height

        # GPS + Battery compact
        gps_color = QColor(52, 199, 89) if self.gps_sats >= 6 else QColor(255, 149, 0)
        painter.setPen(gps_color)
        battery_icon = "🔋" if self.battery > 20 else "⚠️"
        painter.drawText(data_rect.adjusted(0, y_offset, 0, 0), Qt.AlignLeft,
                         f"🛰️ {self.gps_sats} • {battery_icon} {self.battery:.0f}% • ⚡ {((self.throttle-1000)/10):.0f}%")
        y_offset += line_height + 8

        # Akıllı statistikler
        if self.armed:
            painter.setPen(QColor(174, 174, 178))
            flight_mins = self.flight_time // 60
            painter.drawText(data_rect.adjusted(0, y_offset, 0, 0), Qt.AlignLeft,
                             f"⏱️ {flight_mins}min • � Max: {self.max_altitude:.0f}m • 🛣️ {self.distance_traveled:.1f}km")
            y_offset += line_height

        # Akıllı uyarılar (compact)
        if self.warnings and not self.critical_alerts:
            painter.setPen(QColor(255, 214, 10))  # Apple Yellow
            for warning in self.warnings[:3]:  # Max 3 uyarı
                short_warning = warning[:35] + "..." if len(warning) > 35 else warning
                painter.drawText(data_rect.adjusted(0, y_offset, 0, 0), Qt.AlignLeft, short_warning)
                y_offset += line_height

        # Akıllı öneriler
        if self.smart_recommendations and y_offset < data_rect.height() - 30:
            painter.setPen(QColor(52, 199, 89))  # Apple Green
            for rec in self.smart_recommendations[:2]:  # Max 2 öneri
                short_rec = rec[:40] + "..." if len(rec) > 40 else rec
                painter.drawText(data_rect.adjusted(0, y_offset, 0, 0), Qt.AlignLeft, short_rec)
                y_offset += line_height


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self._drag_pos = None
        self.telemetry_active = False

        # Initialize serial communication
        self.serial_comm = SerialCommunication()
        self.serial_comm.data_received.connect(self._handle_telemetry_data)
        
        # Initialize local LLM
        self.llm = LocalLLM()
        self.llm.response_ready.connect(self._handle_llm_response)

        self._create_hyper_ui()
        self._setup_fullscreen_window()

    def _setup_fullscreen_window(self):
        self.setWindowTitle("AirDarwin Ground Control Station")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground)

        screen = QApplication.primaryScreen()
        self.setGeometry(screen.geometry())
        self.showMaximized()

    def _create_hyper_ui(self):
        self.background = ModernBackground(self)
        self.background.setGeometry(self.rect())

        # Top control bar with COM port selector and exit button
        self.control_bar = QWidget(self)
        self.control_bar.setFixedHeight(60)
        self.control_bar.setAttribute(Qt.WA_TranslucentBackground)
        
        control_layout = QHBoxLayout(self.control_bar)
        control_layout.setContentsMargins(20, 10, 20, 10)
        
        # Exit button (top-left)
        self.exit_button = ExitButton(self)
        self.exit_button.clicked.connect(self.close)
        control_layout.addWidget(self.exit_button, 0, Qt.AlignLeft)
        
        control_layout.addStretch(1)  # Space between buttons
        
        # COM port selector (top-right)
        self.com_selector = ComPortSelector(self)
        self.com_selector.port_changed.connect(self._on_port_changed)
        control_layout.addWidget(self.com_selector, 0, Qt.AlignRight)

        # Ana layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Add control bar at top
        main_layout.addWidget(self.control_bar)
        
        # Content layout
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Sol taraf - Chat alanı
        chat_container = QWidget()
        chat_container.setAttribute(Qt.WA_TranslucentBackground)
        chat_layout = QVBoxLayout(chat_container)
        chat_layout.setContentsMargins(40, 40, 20, 40)

        # Scroll area için chat
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background: rgba(64, 156, 255, 40);
                border: none;
                border-radius: 8px;
                width: 14px;
            }
            QScrollBar::handle:vertical {
                background: rgba(64, 156, 255, 140);
                border-radius: 8px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(64, 156, 255, 200);
            }
        """)

        self.chat_area = HyperChatArea()
        self.scroll_area.setWidget(self.chat_area)

        chat_layout.addWidget(self.scroll_area, 1)

        # Modern Composer chat altına
        self.composer = ModernComposer()
        self.composer.message_sent.connect(self._handle_user_message)
        chat_layout.addWidget(self.composer, 0)

        # Sağ taraf - Telemetri paneli (boydan boya)
        telemetry_container = QWidget()
        telemetry_container.setAttribute(Qt.WA_TranslucentBackground)
        telemetry_container.setFixedWidth(360)

        telemetry_layout = QVBoxLayout(telemetry_container)
        telemetry_layout.setContentsMargins(20, 40, 40, 40)
        telemetry_layout.setSpacing(0)

        # Modern HUD paneli boydan boya
        self.hud = ModernHUD(self)
        self.hud.setMinimumHeight(650)  # Biraz daha yüksek
        telemetry_layout.addWidget(self.hud, 1)  # Stretch ile boydan boya

        # Layout'a ekle
        content_layout.addWidget(chat_container, 1)  # Chat sol tarafta genişleyebilir
        content_layout.addWidget(telemetry_container, 0)  # Telemetri sağda sabit
        
        main_layout.addLayout(content_layout, 1)

        # İlk mesaj
        QTimer.singleShot(500, lambda: self.chat_area.add_message(
            "🚀 AirDarwin Ground Control Station Online\n" +
            "📡 Waiting for telemetry data from AirDarwin autopilot\n" +
            "🤖 Local AI Assistant ready for questions\n" +
            "Select COM port from top-right selector to connect...", False))

    def _on_port_changed(self, port_name):
        """Handle COM port selection change"""
        if port_name == "No Connection":
            self.serial_comm.disconnect()
            self.chat_area.add_message(f"📡 Disconnected from COM port", False)
        else:
            success = self.serial_comm.connect(port_name)
            if success:
                self.chat_area.add_message(f"📡 Connected to {port_name} - Listening for AirDarwin telemetry", False)
            else:
                self.chat_area.add_message(f"❌ Failed to connect to {port_name}", False)

    def _handle_telemetry_data(self, data):
        """Handle received telemetry data"""
        self.hud.update_from_telemetry(data)
        # Optionally log important events to chat
        if 'mode' in data and data['mode'] != getattr(self, '_last_mode', ''):
            self.chat_area.add_message(f"🎯 Flight mode changed to: {data['mode']}", False)
            self._last_mode = data['mode']

    def _handle_llm_response(self, response: str):
        """LLM cevabını chat'e ekle"""
        QTimer.singleShot(100, lambda: self.chat_area.add_message(response, False))
        QTimer.singleShot(300, self._scroll_to_bottom)

    def _handle_user_message(self, message: str):
        self.chat_area.add_message(message, True)
        
        # Simple command processing for ground control station
        cmd = message.strip().lower()
        
        # AirDarwin.ino komutları (gerçek komutlar)
        airdarwin_commands = {
            'motor_on': 'motor_on',
            'motor_off': 'motor_off', 
            'takeoff_start': 'takeoff_start',
            'landing_start': 'landing_start',
            'go_around': 'go_around',
            'reset': 'reset'
        }
        
        if cmd in airdarwin_commands:
            if self.serial_comm.is_connected():
                success = self.serial_comm.send_command(airdarwin_commands[cmd])
                if success:
                    response = f"✅ Command '{cmd}' sent to AirDarwin autopilot"
                else:
                    response = f"❌ Failed to send command '{cmd}'"
            else:
                response = "📡 Not connected - Select COM port to connect to AirDarwin first"
        elif cmd in ['connect', 'bağlan']:
            response = "📡 Select COM port from top-right selector to connect to AirDarwin"
        elif cmd in ['disconnect', 'bağlantıyı kes']:
            self.serial_comm.disconnect()
            response = "📡 Disconnected from AirDarwin"
        elif cmd in ['status', 'durum']:
            if self.serial_comm.is_connected():
                response = f"📡 Connected to {self.serial_comm.current_port} - Receiving telemetry"
            else:
                response = "📡 Not connected - Select COM port to connect"
        elif cmd in ['help', 'yardım']:
            response = """� AirDarwin Ground Control Commands:
� connect - Connect to AirDarwin
� disconnect - Disconnect from AirDarwin  
� status - Show connection status
❓ help - Show this help

🤖 AI Assistant: Herhangi bir teknik soru sorabilirsiniz!
Örnek: "batarya durumu nedir?", "gps nasıl çalışır?", "acil durumda ne yapmalı?"
"""
        elif message.strip().endswith('?') or any(word in cmd for word in ['nedir', 'nasıl', 'ne', 'niye', 'hangi', 'kaç']):
            # AI Assistant sorusu - LLM'e yönlendir
            response = self.llm.ask(message)
        else:
            response = f"❓ Command '{message}' not recognized. Type 'help' for available commands or ask AI questions."
            
        QTimer.singleShot(800, lambda: self.chat_area.add_message(response, False))
        QTimer.singleShot(200, self._scroll_to_bottom)

    def _scroll_to_bottom(self):
        scrollbar = self.scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if self._drag_pos and event.buttons() & Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        super().keyPressEvent(event)

    def resizeEvent(self, event):
        self.background.setGeometry(self.rect())
        super().resizeEvent(event)

    def closeEvent(self, event):
        # Clean up serial connection and LLM
        if hasattr(self, 'serial_comm'):
            self.serial_comm.disconnect()
        if hasattr(self, 'llm') and self.llm.worker_thread:
            # Worker thread is daemon, will close automatically
            pass
        event.accept()


def main():
    QtGui.QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("AirDarwin Control System")
    app.setApplicationVersion("4.0 Modern")
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    print("🚀 AirDarwin Modern UI launched!")
    print("⌨️  ESC: Exit | F11: Fullscreen toggle | Tab: Auto-complete")
    print("💬 Commands: motor_on, takeoff, landing, status, help")
    print("🤖 AI Assistant: Soru işareti ile biten sorular AI'ya yönlendirilir")
    if not LLM_AVAILABLE:
        print("⚠️  LLM desteği için: pip install transformers torch")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
