# Инструкция по развёртыванию Clearcam на VPS

Документ описывает установку self-hosted NVR **Clearcam** (`clearcam.py`) на Linux-VPS
(Ubuntu 22.04/24.04 или Debian 12) с systemd-сервисом, nginx (HTTPS) и базовыми
настройками безопасности.

---

## 1. Требования к VPS

| Параметр | Минимум | Рекомендуемо |
|---|---|---|
| vCPU | 2 | 4+ |
| RAM | 4 ГБ (только детекция YOLOv9-t) | 8–16 ГБ (с CLIP/CLIP-поиском), 16+ ГБ (с Qwen3-VL саммари) |
| Диск | 20 ГБ SSD | 50–100+ ГБ (архив записей в `data/cameras/`) |
| ОС | Ubuntu 22.04 / 24.04, Debian 12 | то же |
| Python | 3.11+ | 3.11/3.12 |
| Сеть | доступ к камерам по RTSP из интернета либо белый IP внутри одной сети с камерами | — |

Важные особенности проекта (влияют на выбор VPS):

- Инференс идёт на **tinygrad**. По умолчанию используется CPU; для GPU запускайте
  с переменной окружения `DEV=NV` (NVIDIA CUDA) или `DEV=AMD` (ROCm). На типичном
  CPU-VPS планируйте ~1 ядро под YOLOv9-t при разрешении модели 960.
- При первом запуске скачиваются веса моделей (~несколько ГБ): YOLOv9
  (`huggingface.co/roryclear/yolov9`), опционально CLIP/AdaFace и GGUF-файлы
  Qwen3-VL (`huggingface.co/Qwen/...`). Убедитесь, что домен huggingface.co доступен.
- Веб-UI и HLS-поток отдаются на **порт 8080**, привязка к `0.0.0.0`,
  **без аутентификации и без TLS** (см. раздел 7 «Безопасность»).
- Windows не поддерживается (проблемы с ffmpeg); на VPS это неактуально.

Если камеры доступны только из локальной сети домашнего провайдера, вариант «VPS +
туннель» (см. раздел 8) или размещение на машине внутри той же сети предпочтительнее.

---

## 2. Подготовка сервера

```bash
ssh root@<IP_ВАШЕГО_VPS>

apt update && apt upgrade -y
apt install -y git curl ffmpeg python3 python3-venv python3-pip ufw
# проверьте версию Python: нужна 3.11+
python3 --version
```

Необязательно, но полезно:

```bash
adduser clearcam --disabled-password   # отдельный непривилегированный пользователь
loginctl enable-linger clearcam        # чтобы сервис жил после выхода (альтернатива — root systemd)
```

Откройте firewall, но **пока не открывайте 8080**:

```bash
ufw allow OpenSSH
ufw enable
```

---

## 3. Установка Clearcam

```bash
# от имени пользователя clearcam (su - clearcam) либо в /opt
mkdir -p /opt/clearcam && cd /opt/clearcam
git clone https://github.com/roryclear/clearcam.git .
# Либо скопируйте готовый код репозитория:
#   rsync -av --exclude .git /workspace/ clearcam@<VPS_IP>:/opt/clearcam/

python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt
```

`requirements.txt` тянет tinygrad из GitHub (зафиксированный коммит), numpy 2.0.0
и opencv-python-headless — headless-версия подходит для сервера без X11.

Проверка ручного запуска:

```bash
cd /opt/clearcam
./venv/bin/python clearcam.py
# в логе: "Serving at http://<IP>:8080"
```

Первый запуск скачивает веса YOLOv9 — подождите несколько минут. Откройте в браузере
`http://<IP_VPS>:8080` (временно, только для проверки; затем закройте порт в ufw),
добавьте камеру через UI (RTSP-ссылка сохраняется в БД `data/cc_cache.db`, таблица `links`).

Полезные переменные окружения при старте:

```bash
DEV=NV ./venv/bin/python clearcam.py     # использовать NVIDIA GPU
DEV=AMD ./venv/bin/python clearcam.py    # использовать AMD GPU (ROCm)
BEAM=2 ./venv/bin/python clearcam.py     # ускорение, но долгая первая компиляция
--cam_name=front_door                    # аргумент CLI, имя камеры по умолчанию
```

Функции CLIP-поиска, распознавания лиц (AdaFace) и AI-саммари (Qwen3-VL) включаются
в настройках веб-UI (`use_clip`, `use_face`, `use_qwen`). Для VPS со скромной
видеокартой выбирайте `qwen_size=2` (Qwen3-VL-2B); на CPU саммари работают медленно.

---

## 4. Автозапуск: systemd-сервис

`/etc/systemd/system/clearcam.service`:

```ini
[Unit]
Description=Clearcam NVR
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=clearcam
Group=clearcam
WorkingDirectory=/opt/clearcam
ExecStart=/opt/clearcam/venv/bin/python clearcam.py
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1
# раскомментируйте для GPU:
# Environment=DEV=NV
# для ограничения CPU-инференса (BEAM=2 даёт долгий старт):
# Environment=BEAM=0

[Install]
WantedBy=multi-user.target
```

```bash
chown -R clearcam:clearcam /opt/clearcam
systemctl daemon-reload
systemctl enable --now clearcam
systemctl status clearcam
journalctl -u clearcam -f          # живые логи
```

---

## 5. Данные, диск и ротация

Все данные лежат в `/opt/clearcam/data/`:

- `data/cc_cache.db` — SQLite (камеры-`links`, глобальные настройки, кэш);
- `data/cameras/<имя_камеры>/streams/<дата>/` — HLS-записи;
- `…/event_images`, `…/objects`, `…/faces`, `…/event_clips` — кадры событий, клипы, эмбеддинги (`embeddings.pkl`).

Сервер сам удаляет самые старые записи при заполнении диска (поток cleanup-потока),
но подстрахуйтесь:

```bash
df -h /opt/clearcam/data
# мониторинг места
echo '*/10 * * * * root df -h /opt/clearcam/data | tail -1 >> /var/log/clearcam_disk.log' > /etc/cron.d/clearcam-disk
```

Резервные копии (обязательно выключите сервис на время снятия sqlite-копии или используйте `sqlite3 .backup`):

```bash
/opt/clearcam/venv/bin/pip install restic  # или любой бэкап-инструмент
sqlite3 data/cc_cache.db ".backup '/backup/cc_cache.db'"
rsync -a data/cameras/ /backup/clearcam-data/
```

---

## 6. (Опционально) HTTPS через nginx + Let's Encrypt

Веб-сервер Clearcam не умеет TLS и не имеет авторизации — публиковать 8080 наружу
нельзя. Ставим reverse proxy:

```bash
apt install -y nginx certbot python3-certbot-nginx
```

`/etc/nginx/sites-available/clearcam`:

```nginx
server {
    listen 80;
    server_name cam.example.com;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;            # HLS-стриминг
        chunked_transfer_encoding on;
        client_max_body_size 50M;       # загрузка фото для CLIP-поиска
    }
}
```

```bash
ln -s /etc/nginx/sites-available/clearcam /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx
certbot --nginx -d cam.example.com      # выпуск и автопродление сертификата
```

---

## 7. Безопасность (критично для этой сборки)

В коде отсутствует какая-либо аутентификация HTTP-API, поэтому:

1. **Никогда не открывайте 8080 напрямую в интернет.** Закройте его:
   ```bash
   ufw deny 8080
   ```
   И убедитесь, что nginx слушает снаружи только 80/443.
2. Добавьте basic-auth на уровне nginx (минимальная защита UI):
   ```bash
   apt install -y apache2-utils
   htpasswd -c /etc/nginx/.htpasswd admin
   ```
   В `location /` добавьте:
   ```nginx
   auth_basic "Clearcam";
   auth_basic_user_file /etc/nginx/.htpasswd;
   ```
   Учтите: нативные мобильные клиенты могут не работать поверх basic-auth —
   тогда ограничьте доступ по IP (`allow <ваш_IP>; deny all;`).
3. Уведомления отправляются на `server_url` (по умолчанию `https://clearcam.org`
   для premium, либо свой webhook — Home Assistant / Pushover / n8n, см.
   `utils/sample_server.py`). Если используете свой приёмник уведомлений, держите его
   тоже за закрытым портом.
4. Шифрование E2E (`key`) действует только для загрузки клипов в premium-облако;
   локальные записи на диске **не шифруются** — учитывайте это при выборе VPS-провайдера.
5. Обновления: `cd /opt/clearcam && git pull && ./venv/bin/pip install -r requirements.txt && systemctl restart clearcam`.

---

## 8. Доступ к камерам: варианты топологии

- **Камеры с белым IP / port-forward RTSP** — самый простой вариант для VPS:
  добавляете `rtsp://user:pass@<camera_public_ip>:554/...` в UI.
- **Камеры за NAT (домашняя сеть)** — пробросьте RTSP на VPS туннелем, например
  [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/)
  (`cloudflared access tcp --hostname <CAM_LAN_IP> --port 554` на машине дома)
  или WireGuard между VPS и домашним роутером. Тогда VPS видит камеру как
  `rtsp://10.8.0.x:554/...`.
- **Публичный тестовый фид** (для проверки установки без камеры):
  `https://webcam.elcat.kg/Too-Ashu_Tunnel_North/index.m3u8`

---

## 9. Проверка после развёртывания

```bash
systemctl is-active clearcam                       # active
curl -s http://127.0.0.1:8080/ | head -c 200       # HTML главной страницы
ss -tlnp | grep 8080                               # слушает только localhost/127.0.0.1 (после ufw deny)
journalctl -u clearcam --since "10 min ago" | grep -i error
```

Чек-лист:
- [ ] UI открывается по HTTPS, камеры добавляют и показывают live-поток;
- [ ] события пишутся в `data/cameras/.../streams/<дата>`;
- [ ] уведомления доходят до выбранного приёмника (webhook/Pushover/HA);
- [ ] `data/` на отдельном достаточном разделе, настроен бэкап;
- [ ] 8080 недоступен снаружи, включены basic-auth/IP-ограничение.

---

## 10. Частые проблемы

| Симптом | Причина / решение |
|---|---|
| Долгий первый запуск, «завис» на загрузке | Скачивание весов с HuggingFace; проверьте доступ к `huggingface.co`, повторите запуск |
| Высокая загрузка CPU, низкий FPS | Используйте `model_size="t"` и уменьшите `model_res`; для GPU — `DEV=NV/AMD`; не ставьте `BEAM=2` на слабых CPU (долгая JIT-компиляция) |
| `Port in use, server not started.` | 8080 занят другим процессом: `ss -tlnp \| grep 8080`, освободите порт |
| RTSP-камера не подключается | Включите `-rtsp_transport tcp` (код сам добавляет его для rtsp://-ссылок), проверьте лог ffmpeg-субпроцесса в journalctl; проверьте доступность камеры с VPS (NAT!) |
| Нет уведомлений | Не задан `userID`/`server_url` в настройках; проверьте хост `/send`-эндпоинта (sample: `utils/sample_server.py`) |
| Мало места на диске | Cleanup-поток удаляет старые папки дат; при агрессивной ротации увеличьте диск или уменьшите число камер/разрешение |
| Ошибка tinygrad/GPU | Убедитесь в драйверах CUDA/ROCm; на CPU-VPS уберите `DEV` из сервиса |
