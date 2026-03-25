import sys
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QLCDNumber,
    QSlider,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QComboBox,
)
from PyQt6.QtCore import QTimer, Qt, QSettings



class MainWindow(QWidget):
    def __init__(self, default_value=7, min_value=1, max_value=90):
        super().__init__()

        self.default_value = default_value
        self.min_value = min_value
        self.max_value = max_value
        self.settings = QSettings("mini-projects", "timer-app")

        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.tick_timer)
        self.remaining_seconds = default_value * 60
        self.is_paused = False

        lcd = QLCDNumber(self)
        lcd.setDigitCount(5)
        lcd.display(self.format_time(self.remaining_seconds))
        self.lcd = lcd

        mode_combo = QComboBox(self)
        mode_combo.addItem("Minutes", "minutes")
        mode_combo.addItem("Seconds", "seconds")
        mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        self.mode_combo = mode_combo

        duration_label = QLabel(self)
        self.duration_label = duration_label

        slider = QSlider(Qt.Orientation.Horizontal, self)
        slider.valueChanged[int].connect(self.on_slider_changed)
        self.slider = slider

        start_button = QPushButton('Start', self)
        start_button.clicked.connect(self.start_button_clicked)
        self.start_button = start_button

        pause_button = QPushButton('Pause', self)
        pause_button.clicked.connect(self.pause_button_clicked)
        pause_button.setEnabled(False)
        self.pause_button = pause_button

        reset_button = QPushButton('Reset', self)
        reset_button.clicked.connect(self.reset_button_clicked)
        reset_button.setEnabled(False)
        self.reset_button = reset_button

        self.preset_buttons = []
        presets_hbox = QHBoxLayout()
        presets_hbox.addWidget(QLabel("Presets:", self))
        for minutes in (1, 5, 10, 25):
            button = QPushButton(f"{minutes} min", self)
            button.clicked.connect(lambda checked=False, m=minutes: self.apply_preset_minutes(m))
            self.preset_buttons.append(button)
            presets_hbox.addWidget(button)
        presets_hbox.addStretch()

        control_hbox = QHBoxLayout()
        control_hbox.addWidget(QLabel("Mode:", self))
        control_hbox.addWidget(mode_combo)
        control_hbox.addStretch()
        control_hbox.addWidget(duration_label)

        hbox = QHBoxLayout()
        hbox.addWidget(start_button)
        hbox.addWidget(pause_button)
        hbox.addWidget(reset_button)
        hbox.addWidget(slider)

        vbox = QVBoxLayout()
        vbox.addWidget(lcd)
        vbox.addLayout(control_hbox)
        vbox.addLayout(presets_hbox)
        vbox.addLayout(hbox)

        self.setLayout(vbox)


        self.setWindowTitle('Timer')
        self.load_settings()


    @staticmethod
    def format_time(seconds):
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes:02}:{secs:02}"

    def current_mode(self):
        return self.mode_combo.currentData()

    def duration_to_seconds(self, value):
        if self.current_mode() == "minutes":
            return value * 60
        return value

    def update_duration_label(self):
        unit = "min" if self.current_mode() == "minutes" else "sec"
        self.duration_label.setText(f"Duration: {self.slider.value()} {unit}")

    def configure_slider_for_mode(self, mode, preferred_value=None):
        if mode == "minutes":
            slider_min, slider_max, slider_default = self.min_value, self.max_value, self.default_value
        else:
            slider_min, slider_max, slider_default = 5, 600, 60

        if preferred_value is None:
            preferred_value = slider_default
        preferred_value = max(slider_min, min(slider_max, preferred_value))

        self.slider.blockSignals(True)
        self.slider.setMinimum(slider_min)
        self.slider.setMaximum(slider_max)
        self.slider.setValue(preferred_value)
        self.slider.blockSignals(False)
        self.on_slider_changed(preferred_value)

    def on_mode_changed(self, index):
        if self.timer.isActive() or self.is_paused:
            return

        mode = self.mode_combo.itemData(index)
        current_value = self.slider.value()
        self.configure_slider_for_mode(mode, current_value)
        self.save_settings()

    def toggle_interface(self, value=True):
        self.slider.setEnabled(value)
        self.mode_combo.setEnabled(value)
        for button in self.preset_buttons:
            button.setEnabled(value)

    def apply_preset_minutes(self, minutes):
        if self.timer.isActive() or self.is_paused:
            return

        mode_index = self.mode_combo.findData("minutes")
        if mode_index != -1 and self.mode_combo.currentIndex() != mode_index:
            self.mode_combo.setCurrentIndex(mode_index)
        self.slider.setValue(minutes)

    def on_slider_changed(self, value):
        if not self.timer.isActive() and not self.is_paused:
            self.remaining_seconds = self.duration_to_seconds(value)
            self.lcd.display(self.format_time(self.remaining_seconds))
            self.update_duration_label()
            self.save_settings()

    def start_button_clicked(self, checked=False):
        if self.timer.isActive() or self.is_paused:
            return

        self.remaining_seconds = self.duration_to_seconds(self.slider.value())
        self.lcd.display(self.format_time(self.remaining_seconds))
        self.toggle_interface(False)
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.pause_button.setText("Pause")
        self.is_paused = False
        self.timer.start()

    def pause_button_clicked(self, checked=False):
        if self.timer.isActive():
            self.timer.stop()
            self.is_paused = True
            self.pause_button.setText("Resume")
            return

        if self.is_paused and self.remaining_seconds > 0:
            self.timer.start()
            self.is_paused = False
            self.pause_button.setText("Pause")

    def reset_button_clicked(self, checked=False):
        self.timer.stop()
        self.is_paused = False
        self.remaining_seconds = self.duration_to_seconds(self.slider.value())
        self.lcd.display(self.format_time(self.remaining_seconds))
        self.toggle_interface(True)
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.pause_button.setText("Pause")

    def tick_timer(self):
        if self.remaining_seconds <= 0:
            self.finish_countdown()
            return

        self.remaining_seconds -= 1
        self.lcd.display(self.format_time(self.remaining_seconds))
        if self.remaining_seconds <= 0:
            self.finish_countdown()

    def finish_countdown(self):
        self.timer.stop()
        self.is_paused = False
        QApplication.beep()
        self.toggle_interface(True)
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.pause_button.setText("Pause")
        self.remaining_seconds = self.duration_to_seconds(self.slider.value())

    def load_settings(self):
        mode = self.settings.value("mode", "minutes", type=str)
        saved_value = self.settings.value("duration_value", self.default_value, type=int)
        width = self.settings.value("window_width", 400, type=int)
        height = self.settings.value("window_height", 300, type=int)
        pos_x = self.settings.value("window_x", -1, type=int)
        pos_y = self.settings.value("window_y", -1, type=int)

        mode_index = self.mode_combo.findData(mode)
        if mode_index == -1:
            mode_index = 0
        self.mode_combo.setCurrentIndex(mode_index)
        self.configure_slider_for_mode(self.current_mode(), saved_value)

        self.resize(width, height)
        if pos_x >= 0 and pos_y >= 0:
            self.move(pos_x, pos_y)

    def save_settings(self):
        self.settings.setValue("mode", self.current_mode())
        self.settings.setValue("duration_value", self.slider.value())
        self.settings.setValue("window_width", self.width())
        self.settings.setValue("window_height", self.height())
        self.settings.setValue("window_x", self.x())
        self.settings.setValue("window_y", self.y())

    def closeEvent(self, event):
        self.save_settings()
        super().closeEvent(event)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exit(app.exec())

if __name__ == '__main__':
    main()
