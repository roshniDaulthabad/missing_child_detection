# Hackathon Demonstration Walkthrough

Follow this step-by-step guide to demonstrate the complete end-to-end system to judges and evaluators:

### Step 1: Start the Application
In your terminal, execute:
```bash
python run.py
```
Open **`http://127.0.0.1:5000`** in your browser.

### Step 2: Showcase the Police Command Dashboard
- Point out the real-time CPU telemetry banner (FPS, latency in ms, CPU load %, active tracks).
- Emphasize the mandatory warning: `"Potential Match — Human Verification Required"`.

### Step 3: Register a Missing Child (Zero-Retraining Enrollment)
- Go to **Missing Children** (`/cases`).
- Click **Register New Case**.
- Enter child name (e.g. `Aarav Sharma`, age 9), guardian details, and upload a reference photo.
- Click **Enroll & Extract Biometrics**.
- Explain to judges: The system generates a 512-D AdaFace embedding once, encrypts the image with AES-256-GCM, verifies SHA-3 integrity, and caches the embedding in memory. **No model retraining occurs.**

### Step 4: Start Authorized CCTV Stream
- Go to **Live Stream** (`/live`).
- Click **Start Stream Worker**.
- Observe the real-time video feed:
  - YOLO11n person detections
  - BoT-SORT persistent Track IDs (`ID: 1`, `ID: 2`)
  - Motion trajectory trail
  - Live FPS and CPU utilization overlay

### Step 5: Potential Match & Alert Generation
- The system periodically samples face crops from tracked persons.
- AdaFace extracts the embedding and computes cosine similarity against active cases in memory (<0.2ms).
- When a similarity score exceeds the threshold (e.g. 88%), the bounding box turns RED with `"MATCH: MC-2026-001"`.
- A new alert is generated.

### Step 6: Side-by-Side Officer Verification
- Go to **Potential Alerts** (`/alerts`).
- Click **Side-by-Side Review** on the alert.
- Highlight the side-by-side comparison modal:
  - LEFT: Registered child photo (Primary Reference)
  - RIGHT: CCTV capture observation
  - Similarity score, camera location, track ID, and model version
- Select **Confirm Potential Match** and add officer notes.
- Notice: The AI did NOT automatically close or recover the case; the officer verified the match.

### Step 7: Cross-Camera Movement Tracking
- Go to **Movement Tracking** (`/movement`).
- Select the case to view the reconstructed transit route on the Leaflet.js map (`CAM-01 -> CAM-02 -> CAM-03`).

### Step 8: Post-Location Recovery & Welfare Management
- Go to **Recovery & Welfare** (`/welfare`).
- Select the child case.
- Fill in:
  - Recovery location
  - Legal Guardian Identity Verified (Photo ID)
  - Medical Checkup Referral
  - Psychological Trauma Referral
  - Reunification State: `Reunified`
- Click **Submit Official Recovery & Welfare Record**.

### Step 9: Verify Case Timeline & Cryptographic Audit Trail
- Go to **Case Timeline** (`/timeline`).
- Inspect the complete audit trail showing every timestamped event signed with SHA-3 hashes:
  `Case Registered -> Alert Generated -> Officer Review -> Status Changed -> Recovery Documented -> Reunified`.
