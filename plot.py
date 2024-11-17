import mne

# Загрузка данных
raw = mne.io.read_raw_fif('left_ernest2.fif', preload=True)

# Фильтрация данных
raw.filter(1, 40)

# Аннотация событий

# Проверить аннотации в Raw
print(raw.annotations)

# Преобразовать аннотации в события
events, event_id = mne.events_from_annotations(raw)

# Создание Epochs
epochs = mne.Epochs(raw, events, event_id=1, preload=True)

# Построение вызванного потенциала (Evoked)
evoked = epochs.average()

# Визуализация
evoked.plot()
