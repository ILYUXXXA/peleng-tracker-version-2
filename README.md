# Peleng Tracker

Неофициальный прототип команды для Space Hackathon with Пеленг 2.0. Приложение объединяет потоки нескольких камер, распознаёт и сопровождает объекты, связывает локальные треки между камерами и при наличии калибровки восстанавливает координаты `(x, y, z)` в метрах.

## Установка

Требуются Python 3.11+, [uv](https://docs.astral.sh/uv/) и Node.js 20+. Для offline demo также нужны Docker Desktop и `ffmpeg`.

```bash
uv sync --extra dev
cd web && npm ci && npm run build && cd ..
```

## Запуск

```bash
./run.sh
```

Скрипт создаст локальный `config/hackathon.local.yaml`, установит зависимости, соберёт интерфейс и запустит приложение. Другой конфиг можно передать первым аргументом: `./run.sh config/phone.local.yaml`.

Dashboard откроется на `http://127.0.0.1:8000`. При первом запуске Ultralytics может скачать выбранные веса. Для запуска vision нажмите `Start vision` после подключения потоков.

### Windows + Tailscale

Откройте PowerShell в папке проекта. Установите зависимости один раз, затем запустите сервер:

```powershell
cd D:\peleng-tracker-version-2
uv sync
uv run peleng-tracker run --config config/hackathon.yaml
```

Оставьте это окно открытым. В **PowerShell от имени администратора** включите приватный HTTPS-доступ через Tailscale Serve и проверьте его адрес:

```powershell
& "$env:ProgramFiles\Tailscale\tailscale.exe" serve --bg 8000
& "$env:ProgramFiles\Tailscale\tailscale.exe" serve status
```

Откройте выведенный URL вида `https://имя-устройства.<tailnet>.ts.net` на компьютере. На телефоне, подключённом к тому же tailnet, откройте этот же адрес с `/phone` в конце, разрешите доступ к камере и нажмите кнопку запуска. Вернитесь на дашборд, при необходимости нажмите `Start vision`; подключённые камеры появятся в объединённом потоке. Tailscale Serve доступен только устройствам вашего tailnet и требует HTTPS в tailnet. Подробнее: [Tailscale Serve](https://tailscale.com/docs/reference/tailscale-cli/serve).

Чтобы остановить сервер, нажмите `Ctrl+C` в окне с `uv`. Чтобы отключить HTTPS-прокси Tailscale, выполните в PowerShell администратора:

```powershell
& "$env:ProgramFiles\Tailscale\tailscale.exe" serve off
```

## Камеры

Скопируйте `config/hackathon.example.yaml` в локальный `config/hackathon.local.yaml` и добавьте камеры. RTSP URL рекомендуется передавать через environment variables:

```yaml
cameras:
  - id: 0
    name: left
    url: ${PELENG_CAMERA_0_URL}
    enabled: true
```

```bash
export PELENG_CAMERA_0_URL='rtsp://camera-host/stream'
uv run peleng-tracker run --config config/hackathon.local.yaml
```

Локальные конфиги, веса и калибровка исключены из Git. Не помещайте логины или пароли в example-файлы.

Камеру также можно добавить без перезапуска: откройте `Calibration / Settings` → `Добавить камеру`, укажите уникальный Camera ID, название и RTSP/RTSPS/RTMP/HTTP-ссылку. Добавленный через UI источник хранится только до перезапуска сервера; логин и пароль не возвращаются обратно в интерфейс. Для постоянной конфигурации оставьте URL в ignored-конфиге или environment variable.

## Телефон как камера

Встроенный режим не требует мобильного приложения: запустите сервер с доступом из локальной сети, откройте на телефоне `/phone`, разрешите заднюю камеру и нажмите кнопку трансляции. Телефон автоматически появится среди камер и не отключит обычные RTSP/IP-камеры.

Браузерный Camera API работает только в secure context. На другом устройстве это означает HTTPS с сертификатом, которому доверяет телефон. Скопируйте `config/phone.example.yaml`, положите локальный сертификат и ключ в `config/certs/`, затем запустите:

```bash
uv run peleng-tracker run --config config/phone.local.yaml
```

На телефоне откройте `https://IP_НОУТБУКА:8000/phone`. На Dashboard ноутбука нажмите `Start vision`. Для нескольких телефонов задайте разные Camera ID. В режиме без HTTPS можно использовать мобильное приложение, которое публикует RTSP/MJPEG, и подключить его как обычную камеру через `cameras[].url`.

## Смена модели

YOLO26, YOLO11 и custom Ultralytics `.pt` используют один adapter. Измените одну строку и выполните controlled restart из интерфейса или `POST /api/vision/restart`:

```yaml
vision:
  detector:
    weights: yolo11n.pt
```

Новая модель сначала загружается и прогревается. Если проверка завершится ошибкой, активный pipeline останется прежним.

BoT-SORT используется по умолчанию. Чтобы временно вернуться на DeepSORT, выберите его в `Calibration / Settings` или задайте `vision.tracker.backend: deepsort` в конфиге.

## Выбор цели

Vision сначала создаёт локальные треки с ключом `(camera_id, track_id)`, затем связывает наблюдения одного объекта между камерами под общим `global_id`.

На Dashboard доступны два режима:

- **По классу.** Введите один или несколько числовых ID вручную либо добавьте стандартные классы из списка COCO. Выбранные классы отображаются метками и удаляются нажатием на `×`. Например, `0` соответствует людям, а `2, 7` — автомобилям и грузовикам. После нажатия `Искать по классу` целями станут все активные треки выбранных классов. Ручной ввод также поддерживает ID из custom weights, которых нет в стандартном списке.
- **Конкретный трек.** Нажмите bounding box на видео или карточку трека под видео. Запрос содержит пару `(camera_id, track_id)`; после cross-camera association соответствующий `global_id` подсвечивается жёлтым на всех камерах.

Карточка Target показывает жизненный цикл цели:

- `searching` — запрос принят, но подходящий трек ещё не найден;
- `tracking` — цель присутствует в текущем кадре;
- `lost` — цель пропала в первом кадре после сопровождения;
- `reacquiring` — система продолжает искать выбранный класс или `global_id`;
- `idle` — цель не выбрана.

Кнопка `Сбросить цель` возвращает состояние `idle`. При выключении Vision видео продолжает идти, но bounding boxes, список треков и World View очищаются. Выбранная цель сохраняется, чтобы поиск продолжился после повторного запуска Vision.

## Калибровка 3D

Скопируйте `config/calibration.example.yaml` в ignored-файл `config/calibration.yaml`, замените параметры каждой камеры и проверьте его:

```bash
uv run peleng-tracker validate-calibration config/calibration.yaml
```

Для 3D нужны синхронные наблюдения минимум с двух откалиброванных камер. Система использует правую систему координат и метры, проверяет положительную глубину и reprojection error. Без калибровки позиция имеет статус `unavailable`; planar fallback работает только при явно заданной homography.

Для EPFL Laboratory `4p-c0.avi` и `4p-c1.avi` конвертируйте поставляемый `calibration_4p.txt`:

```bash
uv run python scripts/convert-epfl-calibration.py \
  ~/Downloads/calibration_4p.txt \
  config/calibration.yaml
uv run peleng-tracker validate-calibration config/calibration.yaml
```

Конвертер по умолчанию использует камеры `0,1`, размер кадра `360×288`, высоту head plane `1.75 m`, переводит координаты EPFL из сантиметров в метры и учитывает принятую в наборе отрицательную ось глубины камеры. Для другого набора укажите `--camera-ids`, `--image-width`, `--image-height`, `--head-height-m` или `--ground-units-per-meter`. В EPFL Laboratory камеры 2 и 3 не содержат полной head-plane homography, поэтому этот конвертер не использует их для 3D.

## Демонстрация без камер

Папка `demo/` предназначена только для локальных входных файлов и исключена из Git. Положите в неё два синхронных видео. Для EPFL Laboratory используйте `4p-c0.avi` и `4p-c1.avi`.

Перед запуском:

1. Запустите Docker Desktop.
2. В Docker Desktop включите `Settings → Resources → Network → Enable host networking` и примените изменения.
3. Убедитесь, что команда `ffmpeg -version` работает.

Затем выполните:

```bash
./scripts/offline-demo.sh demo/4p-c0.avi demo/4p-c1.avi
```

Скрипт запускает MediaMTX, циклически публикует оба файла как `camera0` и `camera1` по RTSP-over-TCP и запускает Peleng Tracker с `config/offline-demo.yaml`. `Ctrl+C` остановит приложение и процессы `ffmpeg`. Контейнер MediaMTX можно остановить отдельно:

```bash
docker compose stop mediamtx
```

Если RTSP-поток завершается сообщениями `session timed out` или `Broken pipe`, проверьте, что используется текущая версия `scripts/offline-demo.sh` с параметром `-rtsp_transport tcp`.

Для измерения inference latency:

```bash
uv run peleng-tracker benchmark --config config/hackathon.yaml --iterations 20
```

## Обработанная 3D-запись

Replay-режим читает синхронные 2D-наблюдения из локального bundle, один раз восстанавливает траекторию и затем открывает её в редакторе. RTSP, Docker и модель распознавания для этого режима не нужны.

Минимальная структура bundle:

```text
demo/<scenario>/
├── scenario.yaml
├── calibration.yaml
├── observations.csv
├── ground-truth.csv        # необязательно
├── camera-0.mp4
├── camera-1.mp4
└── trajectory.json         # создаётся командой process
```

`observations.csv` содержит столбцы `frame_index,timestamp,object_id,camera_id,u,v,confidence`. Координаты `u,v` задают одну и ту же точку объекта в каждом синхронном изображении. Для дрона из DPJAIT используется центр bounding box.

Проверка, обработка и запуск:

```bash
uv run peleng-tracker demo validate demo/<scenario>
uv run peleng-tracker demo process demo/<scenario>
uv run peleng-tracker demo evaluate demo/<scenario>
uv run peleng-tracker demo run demo/<scenario>
```

После запуска откройте `http://127.0.0.1:8000` и перейдите на вкладку `3D Replay`. Редактор синхронизирует видео и 3D-точку, поддерживает покадровую перемотку, скорости от `0.25×` до `5×`, градиент траектории по давности, проекцию на выбранную камеру и выгрузку CSV/JSON.

Полный DPJAIT и полученные видео не добавляются в Git. Последовательность `R02_D1` из распакованного архива можно собрать одной командой:

```bash
uv run peleng-tracker demo import-dpjait demo/dpjait-source/Real_Data/R02_D1 \
  --sequence R02_D1 --output demo/dpjait-r02
uv run peleng-tracker demo process demo/dpjait-r02 --max-reprojection-error-px 20
uv run peleng-tracker demo evaluate demo/dpjait-r02
uv run peleng-tracker demo run demo/dpjait-r02
```

Импортёр преобразует Rodrigues-векторы и OpenCV `tvec` из DPJAIT, нормализует центры YOLO bounding box, переводит четыре AVI в браузерные MP4 и выбирает Vicon-отсчёты с частоты 100 Гц для видео 25 FPS. Эталонная позиция строится из центра четырёх Vicon-маркеров с указанным авторами датасета вертикальным смещением визуального центра на 100 мм вниз.
