# Facial Attendance System

Browser-based attendance tracking using face recognition. Enroll a face once, then recognize and log attendance from any device with a camera and browser — no desktop install required, works locally or deployed to the cloud.

## About

Camera capture runs client-side in the browser (via [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc)) and streams frames to the server for detection and recognition. That's a deliberate design choice: a cloud server has no physical webcam, so this is what makes the app usable from a phone, a lab PC, or a public URL, not just `localhost`.

- **Detection**: OpenCV Haar cascade
- **Recognition**: k-nearest-neighbors over 50×50 RGB face crops (scikit-learn)
- **UI**: Streamlit, 3 pages — Take Attendance, Register New Face, Attendance Records
- **Storage**: local pickle files for face data/model, one CSV per day for attendance

## Features

- **Register New Face** — capture 50 samples per person straight from the browser, retrains the model automatically, no separate script to run.
- **Take Attendance** — live recognition with bounding boxes and names; each person is logged **once per day** (previous version logged a duplicate row every single frame).
- **Attendance Records** — browse by date, download as CSV.
- **Unknown-face rejection** — recognition uses a distance threshold, so an unregistered face is labeled "Unknown" instead of being forced into the closest match.
- Optional offline CLI (`src/add_faces.py`, `src/train_model.py`) for enrolling without a browser.

## Project structure

```text
app.py                  Streamlit app (all 3 pages)
src/face_utils.py        Shared detection/recognition/attendance logic
src/add_faces.py         Optional: offline enrollment via local webcam window
src/train_model.py       Optional: retrain model.pkl from saved samples
data/                     haarcascade + trained model + face samples (gitignored, see below)
Attendance/               Attendance_<date>.csv, one file per day (gitignored)
```

## Setup (local)

Requires Python 3.10+.

```bash
git clone <your-repo-url>
cd Facial-Attendance-System
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints, allow camera access when the browser asks, go to **Register New Face** and enroll yourself, then use **Take Attendance**.

## Deployment

### Option A — Streamlit Community Cloud (recommended, free, always-on URL)

1. **Push this repo to GitHub first** — see the data/privacy note below before making it public.

   ```bash
   git push origin main
   ```

2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"Create app"** → **"Deploy a public app from GitHub"** (or connect a private repo if you kept it private).
4. Fill in the deploy form:
   - **Repository**: `shaikhsameer18/Facial-Attendance-System`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - App URL (optional): pick a custom subdomain, e.g. `facial-attendance` → `https://facial-attendance.streamlit.app`
5. Click **"Advanced settings"** before deploying:
   - Python version: `3.11` (matches `runtime.txt`)
   - You don't need to add secrets/env vars for this app.
6. Click **Deploy**. First build takes a few minutes — it installs `requirements.txt` and the apt packages listed in `packages.txt` (needed for OpenCV on Linux), and reads `runtime.txt` for the Python version.
7. Once it's live, open the URL, allow camera access when your browser prompts, go to **Register New Face** and enroll yourself first — the deployed app starts with no trained model since the face/model pickle files are gitignored.
8. Any later `git push` to `main` auto-redeploys the app (Streamlit Cloud watches the branch).

No extra config needed beyond that — the app already uses browser-side camera capture (streamlit-webrtc), so this works out of the box unlike a plain `cv2.VideoCapture(0)` app, which would fail on a server with no webcam.

**Note on data persistence**: Streamlit Community Cloud's filesystem is ephemeral — anyone you enroll and any attendance logged will be wiped on redeploy/restart. Fine for demos; for real ongoing use, see Option B/C with a mounted volume, or swap the pickle/CSV storage for a real database.

### Option B — Docker (Render, Railway, Fly.io, a home server, anywhere containers run)

```bash
docker build -t facial-attendance .
docker run -p 8501:8501 facial-attendance
```

Push the image to any container host and point it at port `8501`.

### Option C — Hugging Face Spaces

Create a Space with SDK "Docker", push this repo — the included `Dockerfile` runs as-is.

## Data & privacy — read before pushing to a public repo

`data/faces_data.pkl` and `data/model.pkl` contain actual face pixel data for every enrolled person; `Attendance/*.csv` contains attendance records. These are now in `.gitignore` for future commits, but **they were committed in the original version of this repo** — if this repo is or will be public, treat that history as exposed and either:

- keep the repo private, or
- scrub the files from git history (`git filter-repo` or BFG) before making it public, or
- deploy with these files supplied at runtime (mounted volume / secret storage) instead of committed at all.

For any real (non-demo) use, get explicit consent from anyone whose face you enroll.

## Known limitations

- Recognition accuracy is modest — raw-pixel KNN, not a deep embedding model (e.g. FaceNet/ArcFace). Fine for a small class/team roster, not for large populations or adversarial conditions.
- No authentication on the app itself — anyone with the URL can register faces or view attendance. Add `streamlit-authenticator` or put it behind a reverse-proxy login if that matters for your use case.
- Single-process file storage (pickle/CSV) — fine for one deployment instance; move to a real database if you need concurrent writers or multi-instance scaling.

## License

MIT — see [LICENSE](LICENSE).
