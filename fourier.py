import mne
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_brainaccess_eeg(filename):
    """
    Анализ данных EEG от Brain Access HALO.
    """
    # Загрузка данных
    raw = mne.io.read_raw_fif(filename, preload=True)
    return raw

def plot_raw_data(raw, save_dir=None):
    """
    Отображение сырых данных
    """
    plt.figure(figsize=(15, 8))
    data, times = raw[:, :]
    
    for i, channel in enumerate(raw.ch_names):
        plt.subplot(len(raw.ch_names), 1, i+1)
        plt.plot(times, data[i])
        plt.title(f'Канал {channel}')
        plt.ylabel('мкВ')
    
    plt.xlabel('Время (с)')
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(Path(save_dir) / 'raw_data.png')
    plt.show()

def plot_fft_analysis(raw, save_dir=None):
    """
    FFT анализ для каждого канала
    """
    data, times = raw[:, :]
    sfreq = raw.info['sfreq']
    
    plt.figure(figsize=(15, 8))
    
    for i, channel in enumerate(raw.ch_names):
        # Вычисление FFT
        n_samples = len(times)
        fft_values = np.fft.fft(data[i])
        freqs = np.fft.fftfreq(n_samples, 1/sfreq)
        
        # Берем только положительные частоты
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        magnitude = 2.0 * np.abs(fft_values[pos_mask]) / n_samples
        
        # Построение графика для канала
        plt.subplot(len(raw.ch_names), 1, i+1)
        plt.plot(freqs, magnitude)
        plt.title(f'Спектр канала {channel}')
        plt.ylabel('Амплитуда')
        plt.xlim(0, 50)  # Ограничиваем до 50 Гц
        plt.grid(True)
    
    plt.xlabel('Частота (Гц)')
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(Path(save_dir) / 'fft_analysis.png')
    plt.show()

# Пример использования:
if __name__ == "__main__":
    # Загрузка данных
    filename = 'left_ernest.fif'  # Замените на ваш файл
    raw = analyze_brainaccess_eeg(filename)
    
    # Создание директории для сохранения
    save_dir = Path(filename[:2])
    save_dir.mkdir(exist_ok=True)
    
    # Вывод информации о данных
    print("\nИнформация о записи:")
    print(f"Количество каналов: {len(raw.ch_names)}")
    print(f"Названия каналов: {raw.ch_names}")
    print(f"Частота дискретизации: {raw.info['sfreq']} Гц")
    print(f"Длительность записи: {raw.times[-1]:.2f} сек")
    
    # Построение графиков
    print("\nПостроение графика сырых данных...")
    plot_raw_data(raw, save_dir)
    
    print("\nПостроение FFT анализа...")
    plot_fft_analysis(raw, save_dir)
