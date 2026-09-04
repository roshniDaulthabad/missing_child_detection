# Authorized RTSP Stream Configuration

## 1. Security Guarantee
RTSP URLs contain sensitive camera credentials (`rtsp://user:password@ip:port/h264`).
Hackwave guarantees:
1. RTSP credentials are encrypted at rest with AES-256-GCM.
2. The browser NEVER receives raw RTSP credentials or stream URLs.
3. The server connects directly to the RTSP feed, runs YOLO11 tracking in a background worker, and streams only the processed MJPEG feed to authorized browser sessions.

## 2. Adding an Authorized Camera
1. Navigate to `/settings`.
2. Click **Add Camera Node**.
3. Enter:
   - Camera ID: `CAM-05`
   - Name: `East Gate Terminal`
   - Location: `Sector 12 Exit`
   - RTSP URL: `rtsp://police_admin:secret@10.0.1.55:554/live`
4. The server validates and encrypts the stream URL.
