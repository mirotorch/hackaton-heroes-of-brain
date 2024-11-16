import matplotlib.pyplot as plt
import matplotlib
import time
import numpy as np
from brainaccess.utils import acquisition
from brainaccess.core.eeg_manager import EEGManager
from matplotlib.animation import FuncAnimation
import logging
from threading import Lock
from scipy.signal import butter, filtfilt, sosfilt, iirnotch, sosfreqz, sosfilt_zi, tf2sos, resample

matplotlib.use("TKAgg", force=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeEEG:
    def __init__(self):
        self.eeg = None
        self.mgr = None
        self.device_name = "BA HALO 047"
        
        # Каналы и их цвета
        self.channels = {
            "Fp2": {"color": "#8A2BE2", "position": 0},
            "Fp1": {"color": "#20B2AA", "position": 1},
            "O2": {"color": "#90EE90", "position": 2},
            "O1": {"color": "#FF6B6B", "position": 3},
        }
        
        # Состояние подключения
        self.is_connected = False
        self.connection_lock = Lock()
        
        # Параметры фильтрации
        self.fs = 250  # Частота дискретизации
        self.filter_low = 0.5
        self.filter_high = 45
        self.notch_freq = 50
        self.filter_order = 4
        self.quality_factor = 35
        
        # Создание фильтров
        self.sos_bandpass = self.create_bandpass_filter()
        self.sos_notch = self.create_notch_filter()
        
        # Состояния фильтров
        self.zi_bandpass = {ch: sosfilt_zi(self.sos_bandpass) for ch in self.channels}
        self.zi_notch = {ch: sosfilt_zi(self.sos_notch) for ch in self.channels}
        
        # Параметры отображения
        self.display_time = 10  # секунд
        self.buffer_size = int(self.display_time * self.fs)
        self.time_values = np.linspace(-self.display_time, 0, self.buffer_size)
        
        # Буферы для данных
        self.display_buffers = {ch: np.zeros(self.buffer_size) for ch in self.channels}
        self.temp_buffers = {ch: np.array([]) for ch in self.channels}
        
        # Параметры масштабирования
        self.amplitude_scale = {ch: 100 for ch in self.channels}
        
        # Параметры обработки
        self.chunk_size = int(0.04 * self.fs)
        self.min_samples = 10
        self.update_counter = 0
        self.last_update_time = time.time()
        
        # Создание графика
        self.setup_plot()

    def create_bandpass_filter(self):
        return butter(
            self.filter_order, 
            [self.filter_low, self.filter_high], 
            btype='bandpass', 
            fs=self.fs, 
            output='sos'
        )

    def create_notch_filter(self):
        b, a = iirnotch(self.notch_freq, self.quality_factor, self.fs)
        return tf2sos(b, a)

    def setup_plot(self):
        plt.style.use('dark_background')
        self.fig, self.axes = plt.subplots(len(self.channels), 1, 
                                         figsize=(15, 8),
                                         sharex=True, 
                                         gridspec_kw={'hspace': 0})
        self.fig.patch.set_facecolor('#1C1C1C')
        
        self.lines = {}
        for ax, (ch_name, ch_info) in zip(self.axes, self.channels.items()):
            line, = ax.plot(self.time_values, np.zeros(self.buffer_size), 
                          color=ch_info['color'], linewidth=1)
            self.lines[ch_name] = line
            
            ax.set_facecolor('#1C1C1C')
            ax.grid(True, color='#333333', linestyle='-', linewidth=0.5)
            ax.set_ylabel(f"{ch_name}\n{self.amplitude_scale[ch_name]} μV", 
                         color=ch_info['color'],
                         rotation=0, labelpad=40,
                         fontsize=10)
            
            # Настройка внешнего вида осей
            for spine in ax.spines.values():
                spine.set_color('#333333')
            ax.tick_params(colors='#888888', direction='out')
        
        self.axes[-1].set_xlabel('Time (s)')
        self.axes[-1].xaxis.label.set_color('#888888')
        plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

    def apply_filters(self, data, channel):
        try:
            # Полосовой фильтр
            filtered, self.zi_bandpass[channel] = sosfilt(
                self.sos_bandpass, data, zi=self.zi_bandpass[channel]
            )
            
            # Режекторный фильтр
            filtered, self.zi_notch[channel] = sosfilt(
                self.sos_notch, filtered, zi=self.zi_notch[channel]
            )
            
            return filtered
        except Exception as e:
            logger.error(f"Filtering error: {e}")
            return data

    def process_chunk(self, data, channel):
        # Фильтрация и прореживание при необходимости
        filtered_data = self.apply_filters(data, channel)
        if len(filtered_data) > self.chunk_size:
            return resample(filtered_data, self.chunk_size)
        return filtered_data

    def update_display_buffer(self, channel, new_data):
        try:
            # Добавляем новые данные во временный буфер
            self.temp_buffers[channel] = np.append(self.temp_buffers[channel], new_data)
            
            # Обрабатываем данные, пока их достаточно
            while len(self.temp_buffers[channel]) >= self.min_samples:
                chunk_size = min(self.chunk_size, len(self.temp_buffers[channel]))
                chunk = self.temp_buffers[channel][:chunk_size]
                
                # Обработка чанка
                processed = self.process_chunk(chunk, channel)
                
                # Обновление буфера отображения
                self.display_buffers[channel] = np.roll(
                    self.display_buffers[channel], 
                    -len(processed)
                )
                self.display_buffers[channel][-len(processed):] = processed
                
                # Удаление обработанных данных
                self.temp_buffers[channel] = self.temp_buffers[channel][chunk_size:]
            
            # Ограничение размера временного буфера
            if len(self.temp_buffers[channel]) > self.buffer_size:
                self.temp_buffers[channel] = \
                    self.temp_buffers[channel][-self.buffer_size:]
                    
        except Exception as e:
            logger.error(f"Buffer update error: {e}")

    def connect(self):
        try:
            with self.connection_lock:
                if not self.is_connected:
                    logger.info("Connecting to device...")
                    self.eeg = acquisition.EEG()
                    self.mgr = EEGManager()
                    self.mgr.__enter__()
                    
                    channel_map = {i: ch for i, ch in enumerate(self.channels.keys())}
                    self.eeg.setup(self.mgr, device_name=self.device_name, cap=channel_map)
                    self.eeg.start_acquisition()
                    
                    time.sleep(2)
                    self.is_connected = True
                    logger.info("Connected successfully")
                    return True
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False

    def update_plot(self, frame):
        if not self.is_connected:
            return list(self.lines.values())
        
        current_time = time.time()
        if current_time - self.last_update_time < 0.04:  # 25 Hz
            return list(self.lines.values())
            
        self.last_update_time = current_time
        
        try:
            self.eeg.get_mne()
            if self.eeg.data and hasattr(self.eeg.data, 'mne_raw'):
                new_data = self.eeg.data.mne_raw.get_data()
                
                if new_data is not None and new_data.size > 0:
                    for idx, (ch_name, ch_info) in enumerate(self.channels.items()):
                        if idx < new_data.shape[0]:
                            channel_data = new_data[idx]
                            
                            if len(channel_data) > 0:
                                # Обновление буфера и графика
                                self.update_display_buffer(ch_name, channel_data)
                                self.lines[ch_name].set_ydata(self.display_buffers[ch_name])
                                
                                # Обновление масштаба каждую секунду
                                if self.update_counter % 25 == 0:
                                    data_range = np.ptp(self.display_buffers[ch_name])
                                    self.amplitude_scale[ch_name] = int(data_range * 1.2)
                                    ylim = self.amplitude_scale[ch_name] / 2
                                    self.axes[ch_info['position']].set_ylim(-ylim, ylim)
                                    
                                    # Обновление подписи с амплитудой
                                    self.axes[ch_info['position']].set_ylabel(
                                        f"{ch_name}\n{self.amplitude_scale[ch_name]} μV",
                                        color=ch_info['color'],
                                        rotation=0, labelpad=40,
                                        fontsize=10
                                    )
                    
                    self.update_counter += 1
        except Exception as e:
            logger.error(f"Plot update error: {e}")
            
        return list(self.lines.values())

    def run(self):
        try:
            if self.connect():
                ani = FuncAnimation(
                    self.fig,
                    self.update_plot,
                    interval=40,  # 25 Hz
                    blit=True
                )
                plt.show()
            else:
                logger.error("Failed to connect")
        except KeyboardInterrupt:
            logger.info("Stopping...")
        finally:
            self.cleanup()

    def cleanup(self):
        try:
            if self.eeg:
                self.eeg.stop_acquisition()
                self.eeg.close()
            if self.mgr:
                self.mgr.disconnect()
                self.mgr.__exit__(None, None, None)
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

if __name__ == "__main__":
    rt_eeg = RealTimeEEG()
    rt_eeg.run()