# -*- coding: utf-8 -*-
"""
PARALLAXSTACK
Multi-Epoch Astronomical Image Analyzer

ANDROID / PYDROID:
    pip install numpy
    python parallaxstack.py

WINDOWS / STREAMLIT:
    pip install numpy streamlit astropy
    streamlit run parallaxstack.py

Optional FITS support:
    pip install astropy
"""

import os
import io
import csv
import json
import math
import socket
import threading
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np


APP_NAME = "ParallaxStack"
HOST = "127.0.0.1"
PORT = 8765

MAX_UPLOAD = 100 * 1024 * 1024
MAX_VALUES = 25_000_000
MAX_DETECTIONS = 500

LOCK = threading.RLock()

DATA = {
    "cube": None,
    "wcs": None,
    "name": None
}


# ============================================================
# TRANSLATIONS
# ============================================================

LANG = {
    "fa": {
        "title": "پارالاکس‌استک",
        "subtitle": "سامانه تحلیل تصاویر نجومی چنددوره‌ای",
        "upload": "بارگذاری تصویر",
        "analyze": "شروع تحلیل",
        "mode": "حالت نمایش",
        "epoch": "دوره زمانی",
        "sigma": "آستانه تشخیص",
        "radius": "شعاع فوتومتری",
        "distance": "حداکثر فاصله حرکت",
        "mintrack": "حداقل طول مسیر",
        "align": "هم‌ترازی انتقالی",
        "background": "حذف پس‌زمینه",
        "sources": "منابع شناسایی‌شده",
        "tracks": "مسیرهای متحرک",
        "preview": "نمای تصویر",
        "catalogue": "فهرست منابع",
        "success": "تحلیل کامل شد.",
        "waiting": "در حال پردازش...",
        "no_file": "ابتدا یک فایل انتخاب کن.",
        "static": "منبع دوره‌ای",
        "transient": "منبع گذرا",
        "error": "خطا",
        "epoch_mode": "تصویر یک دوره",
        "difference": "تصویر تفاضلی",
        "median": "میانه دوره‌ها",
        "mean": "میانگین دوره‌ها",
        "reference": "تفاضل با مرجع",
        "global": "پس‌زمینه سراسری",
        "rowcol": "اصلاح سطر و ستون",
        "none": "بدون هم‌ترازی",
        "translation": "انتقالی",
        "csv": "دریافت CSV",
        "json": "دریافت JSON",
        "file_loaded": "فایل بارگذاری شد",
        "wcs": "مختصات آسمانی",
        "no_data": "هنوز تصویری بارگذاری نشده است."
    },
    "en": {
        "title": "ParallaxStack",
        "subtitle": "Multi-Epoch Astronomical Image Analyzer",
        "upload": "Upload image",
        "analyze": "Analyze",
        "mode": "Display mode",
        "epoch": "Epoch",
        "sigma": "Detection threshold",
        "radius": "Photometry radius",
        "distance": "Maximum motion distance",
        "mintrack": "Minimum track length",
        "align": "Translation alignment",
        "background": "Background subtraction",
        "sources": "Detected sources",
        "tracks": "Moving tracks",
        "preview": "Image preview",
        "catalogue": "Source catalogue",
        "success": "Analysis completed.",
        "waiting": "Processing...",
        "no_file": "Choose a file first.",
        "static": "Epoch source",
        "transient": "Transient source",
        "error": "Error",
        "epoch_mode": "Single epoch",
        "difference": "Difference image",
        "median": "Temporal median",
        "mean": "Temporal mean",
        "reference": "Reference difference",
        "global": "Global background",
        "rowcol": "Row/column correction",
        "none": "No alignment",
        "translation": "Translation",
        "csv": "Download CSV",
        "json": "Download JSON",
        "file_loaded": "File loaded",
        "wcs": "World coordinates",
        "no_data": "No image loaded yet."
    }
}


# ============================================================
# DATA LOADING
# ============================================================

def load_cube(raw, filename):

    if len(raw) > MAX_UPLOAD:
        raise ValueError("File exceeds upload limit.")

    ext = os.path.splitext(filename.lower())[1]
    wcs = None

    if ext == ".npy":

        arr = np.load(
            io.BytesIO(raw),
            allow_pickle=False
        )

    elif ext == ".npz":

        archive = np.load(
            io.BytesIO(raw),
            allow_pickle=False
        )

        if not archive.files:
            raise ValueError("Empty NPZ file.")

        arr = archive[archive.files[0]]
        archive.close()

    elif ext in (".fits", ".fit", ".fts"):

        try:
            from astropy.io import fits
        except ImportError:
            raise ValueError(
                "FITS requires Astropy. Run: pip install astropy"
            )

        with fits.open(
            io.BytesIO(raw),
            memmap=False
        ) as hdus:

            selected = next(
                (h for h in hdus if h.data is not None),
                None
            )

            if selected is None:
                raise ValueError("No image found in FITS.")

            arr = np.asarray(
                selected.data,
                dtype=np.float32
            )

            header = selected.header.copy()

        try:
            from astropy.wcs import WCS

            candidate = WCS(header)

            if candidate.has_celestial:
                wcs = candidate.celestial

        except Exception:
            wcs = None

    else:
        raise ValueError("Supported files: NPY, NPZ, FITS.")

    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]

    if arr.ndim != 3:
        raise ValueError(
            "Expected 2D image or 3D cube."
        )

    if arr.size > MAX_VALUES:
        raise ValueError(
            "Image cube exceeds memory limit."
        )

    if min(arr.shape) < 3:
        raise ValueError("Image dimensions are too small.")

    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return arr, wcs


# ============================================================
# STATISTICS
# ============================================================

def robust_stats(image):

    image = np.asarray(image, dtype=np.float32)

    median = float(np.median(image))

    mad = float(
        np.median(np.abs(image - median))
    )

    noise = 1.4826 * mad

    if not np.isfinite(noise) or noise < 1e-8:
        noise = float(np.std(image))

    return median, max(noise, 1e-8)


def remove_background(image, mode="global"):

    image = image.astype(np.float32, copy=True)

    if mode == "rowcol":

        row = np.median(image, axis=1)
        col = np.median(image, axis=0)

        image -= (
            row[:, None] - np.median(row)
        )

        image -= (
            col[None, :] - np.median(col)
        )

    else:

        median, _ = robust_stats(image)
        image -= median

    return image


# ============================================================
# ALIGNMENT
# ============================================================

def estimate_shift(reference, moving):

    a = reference.astype(np.float32)
    b = moving.astype(np.float32)

    if a.shape != b.shape:
        raise ValueError("Image dimensions do not match.")

    step = max(1, int(max(a.shape) / 512))

    a = a[::step, ::step]
    b = b[::step, ::step]

    a = a - np.mean(a)
    b = b - np.mean(b)

    f1 = np.fft.fft2(a)
    f2 = np.fft.fft2(b)

    cross = f1 * np.conj(f2)

    cross /= np.maximum(
        np.abs(cross),
        1e-12
    )

    correlation = np.fft.ifft2(cross).real

    py, px = np.unravel_index(
        np.argmax(correlation),
        correlation.shape
    )

    if py > correlation.shape[0] // 2:
        py -= correlation.shape[0]

    if px > correlation.shape[1] // 2:
        px -= correlation.shape[1]

    return int(px * step), int(py * step)


def shift_image(image, dx, dy):

    h, w = image.shape

    output = np.zeros_like(image)

    sx0 = max(0, -dx)
    sx1 = min(w, w - dx)

    sy0 = max(0, -dy)
    sy1 = min(h, h - dy)

    dx0 = max(0, dx)
    dx1 = min(w, w + dx)

    dy0 = max(0, dy)
    dy1 = min(h, h + dy)

    if sx1 > sx0 and sy1 > sy0:

        output[dy0:dy1, dx0:dx1] = (
            image[sy0:sy1, sx0:sx1]
        )

    return output


def align_cube(cube):

    reference = cube[0]

    output = [reference]
    shifts = [(0, 0)]

    for image in cube[1:]:

        dx, dy = estimate_shift(
            reference,
            image
        )

        output.append(
            shift_image(image, dx, dy)
        )

        shifts.append((dx, dy))

    return np.stack(output), shifts


# ============================================================
# SOURCE DETECTION
# ============================================================

def detect_sources(
    image,
    threshold_sigma=5,
    radius=3,
    limit=MAX_DETECTIONS
):

    median, noise = robust_stats(image)

    threshold = median + threshold_sigma * noise

    h, w = image.shape

    padded = np.pad(
        image,
        1,
        mode="edge"
    )

    mask = image > threshold

    for yy in range(3):
        for xx in range(3):

            if yy == 1 and xx == 1:
                continue

            mask &= (
                image >=
                padded[yy:yy+h, xx:xx+w]
            )

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return []

    values = image[ys, xs]

    order = np.argsort(values)[::-1][:limit]

    radius = max(1, int(radius))

    detections = []

    for index in order:

        x = int(xs[index])
        y = int(ys[index])

        y0 = max(0, y-radius)
        y1 = min(h, y+radius+1)

        x0 = max(0, x-radius)
        x1 = min(w, x+radius+1)

        patch = image[y0:y1, x0:x1]

        yy, xx = np.mgrid[y0:y1, x0:x1]

        weights = np.maximum(
            patch - median,
            0
        )

        total = float(np.sum(weights))

        if total > 0:

            cx = float(
                np.sum(xx * weights) / total
            )

            cy = float(
                np.sum(yy * weights) / total
            )

        else:
            cx, cy = float(x), float(y)

        aperture = (
            (xx-cx)**2 + (yy-cy)**2
            <= radius**2
        )

        flux = float(
            np.sum(patch[aperture] - median)
        )

        area = max(1, int(np.sum(aperture)))

        snr = flux / (
            noise * math.sqrt(area)
        )

        detections.append({
            "x": round(cx, 3),
            "y": round(cy, 3),
            "peak": float(image[y, x]),
            "flux": flux,
            "snr": float(snr)
        })

    return detections


# ============================================================
# TRACKING
# ============================================================

def build_tracks(
    detections_by_epoch,
    max_distance=8,
    min_epochs=2
):

    tracks = []

    for epoch, detections in enumerate(
        detections_by_epoch
    ):

        available = set(range(len(detections)))

        active = [
            track for track in tracks
            if track[-1]["epoch"] == epoch - 1
        ]

        for track in active:

            last = track[-1]

            if len(track) >= 2:

                previous = track[-2]

                px = last["x"] + (
                    last["x"] - previous["x"]
                )

                py = last["y"] + (
                    last["y"] - previous["y"]
                )

            else:
                px = last["x"]
                py = last["y"]

            choices = []

            for j in available:

                item = detections[j]

                distance = math.hypot(
                    item["x"] - px,
                    item["y"] - py
                )

                if distance <= max_distance:
                    choices.append((distance, j))

            if choices:

                _, j = min(choices)

                item = dict(detections[j])
                item["epoch"] = epoch

                track.append(item)
                available.remove(j)

        for j in available:

            item = dict(detections[j])
            item["epoch"] = epoch

            tracks.append([item])

    return [
        track for track in tracks
        if len(track) >= min_epochs
    ]


# ============================================================
# WCS
# ============================================================

def add_world_coordinates(sources, wcs):

    if wcs is None:
        return

    for source in sources:

        try:

            sky = wcs.pixel_to_world(
                source["x"],
                source["y"]
            )

            source["ra"] = float(sky.ra.deg)
            source["dec"] = float(sky.dec.deg)

        except Exception:
            source["ra"] = None
            source["dec"] = None


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze(options):

    with LOCK:

        cube = DATA["cube"]
        wcs = DATA["wcs"]

    if cube is None:
        raise ValueError("No image cube loaded.")

    sigma = float(np.clip(
        float(options.get("sigma", 5)),
        2,
        15
    ))

    radius = int(np.clip(
        int(options.get("radius", 3)),
        1,
        12
    ))

    distance = float(np.clip(
        float(options.get("distance", 8)),
        1,
        100
    ))

    mintrack = int(np.clip(
        int(options.get("mintrack", 2)),
        2,
        20
    ))

    background_mode = options.get(
        "background",
        "global"
    )

    processed = np.stack([
        remove_background(image, background_mode)
        for image in cube
    ])

    shifts = [(0, 0)] * len(processed)

    if (
        options.get("align") == "translation"
        and len(processed) > 1
    ):

        processed, shifts = align_cube(processed)

    temporal_median = np.median(
        processed,
        axis=0
    )

    temporal_mean = np.mean(
        processed,
        axis=0
    )

    detections_by_epoch = []

    for image in processed:

        sources = detect_sources(
            image,
            sigma,
            radius
        )

        detections_by_epoch.append(sources)

    tracks = build_tracks(
        detections_by_epoch,
        distance,
        mintrack
    )

    epoch = int(np.clip(
        int(options.get("epoch", 0)),
        0,
        len(processed)-1
    ))

    mode = options.get("mode", "epoch")

    if mode == "epoch":
        display = processed[epoch]

    elif mode == "difference":
        display = processed[epoch] - temporal_median

    elif mode == "median":
        display = temporal_median

    elif mode == "mean":
        display = temporal_mean

    elif mode == "reference":
        display = processed[epoch] - processed[0]

    else:
        raise ValueError("Invalid display mode.")

    residual_detections = []

    for image in processed:

        residual = image - temporal_median

        residual_detections.append(
            detect_sources(
                residual,
                sigma,
                radius
            )
        )

    catalogue = []

    for epoch_index, sources in enumerate(
        detections_by_epoch
    ):

        for source in sources:

            row = dict(source)

            row["epoch"] = epoch_index
            row["source_type"] = "epoch_source"

            catalogue.append(row)

    transients = []

    for epoch_index, sources in enumerate(
        residual_detections
    ):

        for source in sources:

            row = dict(source)

            row["epoch"] = epoch_index
            row["source_type"] = "transient"

            transients.append(row)

    add_world_coordinates(catalogue, wcs)
    add_world_coordinates(transients, wcs)

    track_lines = [
        [
            [point["x"], point["y"]]
            for point in track
        ]
        for track in tracks
    ]

    return {
        "width": int(cube.shape[2]),
        "height": int(cube.shape[1]),
        "epochs": int(cube.shape[0]),
        "pixels": normalize_image(display).tolist(),
        "detections": [
            [s["x"], s["y"]]
            for s in detections_by_epoch[epoch]
        ],
        "track_lines": track_lines,
        "catalogue": catalogue,
        "transients": transients,
        "tracks": tracks,
        "candidate_count": len(catalogue),
        "track_count": len(tracks),
        "shifts": shifts,
        "alignment": str(shifts),
        "note": (
            f"{len(catalogue)} detections, "
            f"{len(tracks)} candidate tracks."
        )
    }


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_image(image):

    image = np.asarray(image, dtype=np.float32)

    low, high = np.percentile(
        image,
        [1, 99]
    )

    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)

    result = np.clip(
        (image - low) / (high - low),
        0,
        1
    )

    return (result * 255).astype(np.uint8)


# ============================================================
# EXPORT
# ============================================================

def export_csv(result):

    output = io.StringIO()

    fields = [
        "epoch",
        "x",
        "y",
        "peak",
        "flux",
        "snr",
        "source_type",
        "ra",
        "dec"
    ]

    writer = csv.DictWriter(
        output,
        fieldnames=fields,
        extrasaction="ignore"
    )

    writer.writeheader()

    for row in (
        result["catalogue"] +
        result["transients"]
    ):
        writer.writerow(row)

    return output.getvalue()


def export_json(result):

    return json.dumps(
        result,
        ensure_ascii=False,
        allow_nan=False
    )


# ============================================================
# ANDROID WEB PAGE
# ============================================================

PAGE = r"""
<!DOCTYPE html>
<html lang="fa">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<title>ParallaxStack</title>

<style>
* { box-sizing: border-box; }

body {
    margin: 0;
    padding: 15px;
    background: #091321;
    color: #edf3ff;
    font-family: sans-serif;
}

main {
    max-width: 1100px;
    margin: auto;
}

header {
    background: #17345a;
    padding: 16px;
    border-radius: 12px;
}

.panel {
    background: #142238;
    border: 1px solid #304563;
    padding: 15px;
    border-radius: 12px;
    margin-top: 14px;
}

.row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    align-items: center;
}

label {
    display: block;
    margin-top: 10px;
}

input, select, button {
    padding: 10px;
    border-radius: 7px;
    border: 1px solid #496181;
    background: #0b1728;
    color: white;
    max-width: 100%;
}

button {
    background: #326bd8;
    border: 0;
    font-weight: bold;
}

button:disabled {
    opacity: .5;
}

canvas {
    width: 100%;
    max-height: 65vh;
    object-fit: contain;
    background: #02050a;
}

table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
}

td, th {
    padding: 7px;
    border-bottom: 1px solid #34445c;
    text-align: center;
}

.scroll {
    overflow-x: auto;
}

#status {
    overflow-wrap: anywhere;
    color: #9ee6ae;
}

.error {
    color: #ff8d8d !important;
}

.stat {
    background: #0a1728;
    padding: 10px;
    border-radius: 8px;
}

.muted {
    color: #aabbd3;
    font-size: 13px;
}
</style>
</head>

<body>
<main>

<header>
<h1 id="title">پارالاکس‌استک</h1>
<p id="subtitle">سامانه تحلیل تصاویر نجومی چنددوره‌ای</p>

<div class="row">
<button onclick="setLanguage('fa')">فارسی</button>
<button onclick="setLanguage('en')">English</button>
</div>
</header>

<section class="panel">
<h2 id="uploadTitle">بارگذاری تصویر</h2>

<input id="file" type="file"
accept=".npy,.npz,.fits,.fit,.fts">

<button id="uploadButton" onclick="uploadFile()">
بارگذاری
</button>

<p id="fileInfo" class="muted"></p>
</section>

<section class="panel">
<h2 id="analysisTitle">تنظیمات تحلیل</h2>

<div class="row">

<div>
<label id="modeLabel">حالت نمایش</label>
<select id="mode">
<option value="epoch">Single epoch</option>
<option value="difference">Difference</option>
<option value="median">Median</option>
<option value="mean">Mean</option>
<option value="reference">Reference</option>
</select>
</div>

<div>
<label id="epochLabel">دوره</label>
<input id="epoch" type="number" value="0" min="0" style="width:80px">
</div>

<div>
<label id="sigmaLabel">آستانه</label>
<input id="sigma" type="number" value="5" min="2" max="15" step=".5" style="width:80px">
</div>

<div>
<label id="radiusLabel">شعاع</label>
<input id="radius" type="number" value="3" min="1" max="12" style="width:80px">
</div>

<div>
<label id="distanceLabel">حداکثر حرکت</label>
<input id="distance" type="number" value="8" min="1" max="100" style="width:80px">
</div>

<div>
<label id="trackLabel">حداقل طول مسیر</label>
<input id="mintrack" type="number" value="2" min="2" max="20" style="width:80px">
</div>

<div>
<label id="alignLabel">هم‌ترازی</label>
<select id="align">
<option value="none">None</option>
<option value="translation">Translation</option>
</select>
</div>

<div>
<label id="backgroundLabel">پس‌زمینه</label>
<select id="background">
<option value="global">Global</option>
<option value="rowcol">Row/Column</option>
</select>
</div>

</div>

<button id="runButton" onclick="analyze()">
شروع تحلیل
</button>

<p id="status"></p>

<div class="row">
<span class="stat" id="shape">Cube: —</span>
<span class="stat" id="sourceCount">Sources: —</span>
<span class="stat" id="trackCount">Tracks: —</span>
</div>
</section>

<section class="panel">
<h2 id="previewTitle">نمای تصویر</h2>
<canvas id="view" width="512" height="512"></canvas>
<p class="muted" id="legend">
دایره زرد: منابع شناسایی‌شده | خط فیروزه‌ای: مسیرها
</p>
</section>

<section class="panel">
<h2 id="catalogueTitle">فهرست منابع</h2>
<div class="scroll">
<table>
<thead>
<tr>
<th>Epoch</th>
<th>X</th>
<th>Y</th>
<th>Peak</th>
<th>Flux</th>
<th>SNR</th>
<th>RA</th>
<th>Dec</th>
</tr>
</thead>
<tbody id="catalogue"></tbody>
</table>
</div>
</section>

<section class="panel">
<h2 id="transientTitle">منابع گذرا</h2>
<div class="scroll">
<table>
<thead>
<tr>
<th>Epoch</th>
<th>X</th>
<th>Y</th>
<th>Flux</th>
<th>SNR</th>
</tr>
</thead>
<tbody id="transients"></tbody>
</table>
</div>
</section>

<section class="panel">
<button onclick="download('csv')" id="csvButton">
Download CSV
</button>
<button onclick="download('json')" id="jsonButton">
Download JSON
</button>
</section>

</main>

<script>

const $ = id => document.getElementById(id);

let language = "fa";
let loaded = false;
let resultData = null;
let currentEpochs = 1;

const translations = {
fa: {
title: "پارالاکس‌استک",
subtitle: "سامانه تحلیل تصاویر نجومی چنددوره‌ای",
upload: "بارگذاری تصویر",
uploadButton: "بارگذاری",
analysis: "تنظیمات تحلیل",
mode: "حالت نمایش",
epoch: "دوره",
sigma: "آستانه تشخیص",
radius: "شعاع فوتومتری",
distance: "حداکثر فاصله حرکت",
track: "حداقل طول مسیر",
align: "هم‌ترازی",
background: "پس‌زمینه",
run: "شروع تحلیل",
preview: "نمای تصویر",
catalogue: "فهرست منابع",
transient: "منابع گذرا",
waiting: "در حال پردازش...",
success: "تحلیل کامل شد.",
noFile: "ابتدا فایل را بارگذاری کن.",
loaded: "فایل بارگذاری شد.",
error: "خطا",
shape: "ابعاد",
sources: "منابع",
tracks: "مسیرها",
legend: "دایره زرد: منابع شناسایی‌شده | خط فیروزه‌ای: مسیرها"
},
en: {
title: "ParallaxStack",
subtitle: "Multi-Epoch Astronomical Image Analyzer",
upload: "Upload image",
uploadButton: "Upload",
analysis: "Analysis settings",
mode: "Display mode",
epoch: "Epoch",
sigma: "Detection threshold",
radius: "Photometry radius",
distance: "Maximum motion distance",
track: "Minimum track length",
align: "Alignment",
background: "Background",
run: "Run analysis",
preview: "Image preview",
catalogue: "Source catalogue",
transient: "Transient sources",
waiting: "Processing...",
success: "Analysis completed.",
noFile: "Upload a file first.",
loaded: "File loaded.",
error: "Error",
shape: "Shape",
sources: "Sources",
tracks: "Tracks",
legend: "Yellow circles: sources | Cyan lines: tracks"
}
};

function setLanguage(lang) {

    language = lang;

    const t = translations[lang];

    document.documentElement.lang = lang;
    document.documentElement.dir =
        lang === "fa" ? "rtl" : "ltr";

    $("title").textContent = t.title;
    $("subtitle").textContent = t.subtitle;
    $("uploadTitle").textContent = t.upload;
    $("uploadButton").textContent = t.uploadButton;
    $("analysisTitle").textContent = t.analysis;
    $("modeLabel").textContent = t.mode;
    $("epochLabel").textContent = t.epoch;
    $("sigmaLabel").textContent = t.sigma;
    $("radiusLabel").textContent = t.radius;
    $("distanceLabel").textContent = t.distance;
    $("trackLabel").textContent = t.track;
    $("alignLabel").textContent = t.align;
    $("backgroundLabel").textContent = t.background;
    $("runButton").textContent = t.run;
    $("previewTitle").textContent = t.preview;
    $("catalogueTitle").textContent = t.catalogue;
    $("transientTitle").textContent = t.transient;
    $("legend").textContent = t.legend;
    $("csvButton").textContent = "CSV";
    $("jsonButton").textContent = "JSON";

    const modes = [
        ["epoch", lang === "fa" ? "تصویر یک دوره" : "Single epoch"],
        ["difference", lang === "fa" ? "تصویر تفاضلی" : "Difference image"],
        ["median", lang === "fa" ? "میانه دوره‌ها" : "Temporal median"],
        ["mean", lang === "fa" ? "میانگین دوره‌ها" : "Temporal mean"],
        ["reference", lang === "fa" ? "تفاضل با مرجع" : "Reference difference"]
    ];

    const old = $("mode").value;
    $("mode").innerHTML = "";

    modes.forEach(([value, label]) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = label;
        $("mode").appendChild(option);
    });

    $("mode").value = old;
}

async function readResponse(response) {

    const text = await response.text();

    let data;

    try {
        data = JSON.parse(text);
    } catch (error) {
        throw new Error(
            "Invalid server response: " + text.slice(0, 300)
        );
    }

    if (!response.ok) {
        throw new Error(data.error || "HTTP " + response.status);
    }

    return data;
}

async function uploadFile() {

    const file = $("file").files[0];

    if (!file) {
        setStatus(translations[language].noFile, true);
        return;
    }

    $("uploadButton").disabled = true;
    setStatus(translations[language].waiting);

    try {

        const response = await fetch(
            "/upload?name=" + encodeURIComponent(file.name),
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/octet-stream"
                },
                body: file
            }
        );

        const data = await readResponse(response);

        loaded = true;
        currentEpochs = data.epochs;

        $("epoch").max = data.epochs - 1;
        $("epoch").value = 0;

        $("shape").textContent =
            data.epochs + " × " +
            data.height + " × " +
            data.width;

        $("fileInfo").textContent =
            data.name + " | WCS: " + data.wcs;

        setStatus(translations[language].loaded);

    } catch (error) {

        setStatus(error.message, true);

    } finally {

        $("uploadButton").disabled = false;
    }
}

async function analyze() {

    if (!loaded) {
        setStatus(translations[language].noFile, true);
        return;
    }

    $("runButton").disabled = true;
    setStatus(translations[language].waiting);

    const options = {
        mode: $("mode").value,
        epoch: Number($("epoch").value),
        sigma: Number($("sigma").value),
        radius: Number($("radius").value),
        distance: Number($("distance").value),
        mintrack: Number($("mintrack").value),
        align: $("align").value,
        background: $("background").value
    };

    try {

        const response = await fetch("/analyze", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(options)
        });

        const result = await readResponse(response);

        resultData = result;

        draw(result);
        renderTable(result.catalogue, "catalogue");
        renderTable(result.transients, "transients");

        $("sourceCount").textContent =
            translations[language].sources +
            ": " + result.candidate_count;

        $("trackCount").textContent =
            translations[language].tracks +
            ": " + result.track_count;

        setStatus(
            translations[language].success +
            " " + result.note
        );

    } catch (error) {

        setStatus(error.message, true);

    } finally {

        $("runButton").disabled = false;
    }
}

function setStatus(message, error=false) {

    $("status").textContent = message;

    $("status").className =
        error ? "error" : "";
}

function draw(result) {

    const canvas = $("view");
    const ctx = canvas.getContext("2d");

    const w = result.width;
    const h = result.height;

    canvas.width = w;
    canvas.height = h;

    const image = ctx.createImageData(w, h);

    for (let i = 0; i < result.pixels.length; i++) {

        const value = result.pixels[i];
        const k = i * 4;

        image.data[k] = value;
        image.data[k+1] = value;
        image.data[k+2] = value;
        image.data[k+3] = 255;
    }

    ctx.putImageData(image, 0, 0);

    ctx.strokeStyle = "#32dfd0";
    ctx.lineWidth = Math.max(1, w/500);

    for (const line of result.track_lines) {

        if (line.length < 2) continue;

        ctx.beginPath();

        line.forEach((point, i) => {

            if (i === 0) {
                ctx.moveTo(point[0], point[1]);
            } else {
                ctx.lineTo(point[0], point[1]);
            }
        });

        ctx.stroke();
    }

    ctx.strokeStyle = "#ffca55";

    for (const point of result.detections) {

        ctx.beginPath();

        ctx.arc(
            point[0],
            point[1],
            Math.max(3, w/170),
            0,
            Math.PI*2
        );

        ctx.stroke();
    }
}

function renderTable(rows, id) {

    const body = $(id);
    body.innerHTML = "";

    for (const row of rows.slice(0, 1000)) {

        const values = id === "catalogue"
            ? [
                row.epoch,
                row.x,
                row.y,
                row.peak,
                row.flux,
                row.snr,
                row.ra ?? "—",
                row.dec ?? "—"
            ]
            : [
                row.epoch,
                row.x,
                row.y,
                row.flux,
                row.snr
            ];

        const tr = document.createElement("tr");

        values.forEach(value => {

            const td = document.createElement("td");

            td.textContent =
                typeof value === "number"
                ? Number(value).toPrecision(5)
                : value;

            tr.appendChild(td);
        });

        body.appendChild(tr);
    }
}

function download(kind) {

    if (!resultData) {
        setStatus(
            language === "fa"
            ? "ابتدا تحلیل را اجرا کن."
            : "Run analysis first.",
            true
        );
        return;
    }

    let content;
    let mime;
    let filename;

    if (kind === "json") {

        content = JSON.stringify(
            resultData,
            null,
            2
        );

        mime = "application/json";
        filename = "parallaxstack.json";

    } else {

        const rows = [
            ["epoch","x","y","peak","flux","snr","ra","dec"]
        ];

        resultData.catalogue.forEach(row => {
            rows.push([
                row.epoch,
                row.x,
                row.y,
                row.peak,
                row.flux,
                row.snr,
                row.ra ?? "",
                row.dec ?? ""
            ]);
        });

        content = rows.map(
            row => row.join(",")
        ).join("\n");

        mime = "text/csv";
        filename = "parallaxstack.csv";
    }

    const blob = new Blob(
        [content],
        {type: mime}
    );

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = url;
    link.download = filename;
    link.click();

    setTimeout(
        () => URL.revokeObjectURL(url),
        2000
    );
}

setLanguage("fa");

</script>
</body>
</html>
"""


# ============================================================
# HTTP SERVER
# ============================================================

class ParallaxHandler(BaseHTTPRequestHandler):

    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print("[ParallaxStack]", fmt % args)

    def reply(self, status, data, content_type="application/json"):

        if isinstance(data, str):
            data = data.encode("utf-8")

        try:

            self.send_response(status)

            self.send_header(
                "Content-Type",
                content_type
            )

            self.send_header(
                "Content-Length",
                str(len(data))
            )

            self.send_header(
                "Cache-Control",
                "no-store"
            )

            self.send_header(
                "Connection",
                "close"
            )

            self.end_headers()

            self.wfile.write(data)
            self.wfile.flush()

        except (BrokenPipeError, ConnectionResetError):
            print("[HTTP] Client disconnected before response.")

    def reply_json(self, status, data):

        raw = json.dumps(
            data,
            ensure_ascii=False,
            allow_nan=False
        )

        self.reply(
            status,
            raw,
            "application/json; charset=utf-8"
        )

    def do_GET(self):

        path = urlparse(self.path).path

        if path in ("/", "/index.html"):

            self.reply(
                200,
                PAGE,
                "text/html; charset=utf-8"
            )

        elif path == "/health":

            self.reply_json(
                200,
                {"status": "ok"}
            )

        else:

            self.reply_json(
                404,
                {"error": "Not found"}
            )

    def do_POST(self):

        path = urlparse(self.path).path

        try:

            length = int(
                self.headers.get("Content-Length", "0")
            )

            if length <= 0:
                raise ValueError("Empty request.")

            if length > MAX_UPLOAD + 1024 * 1024:
                raise ValueError("Request too large.")

            raw = self.rfile.read(length)

            if len(raw) != length:
                raise ValueError("Incomplete request body.")

            if path == "/upload":

                filename = parse_qs(
                    urlparse(self.path).query
                ).get(
                    "name",
                    ["upload.npy"]
                )[0]

                filename = os.path.basename(filename)

                cube, wcs = load_cube(
                    raw,
                    filename
                )

                with LOCK:

                    DATA["cube"] = cube
                    DATA["wcs"] = wcs
                    DATA["name"] = filename

                self.reply_json(
                    200,
                    {
                        "name": filename,
                        "epochs": int(cube.shape[0]),
                        "height": int(cube.shape[1]),
                        "width": int(cube.shape[2]),
                        "wcs": wcs is not None
                    }
                )

            elif path == "/analyze":

                options = json.loads(
                    raw.decode("utf-8")
                )

                result = analyze(options)

                self.reply_json(
                    200,
                    result
                )

            else:

                self.reply_json(
                    404,
                    {"error": "Not found"}
                )

        except Exception as exc:

            print("\n========== PARALLAXSTACK ERROR ==========")
            traceback.print_exc()
            print("ERROR:", repr(exc))
            print("==========================================\n")

            self.reply_json(
                400,
                {"error": str(exc)}
            )


# ============================================================
# ANDROID SERVER START
# ============================================================

def find_port():

    for port in range(PORT, PORT + 30):

        sock = socket.socket()

        try:
            sock.bind((HOST, port))
            return port

        except OSError:
            pass

        finally:
            sock.close()

    raise RuntimeError("No free local port found.")


def run_android():

    port = find_port()

    server = ThreadingHTTPServer(
        (HOST, port),
        ParallaxHandler
    )

    server.daemon_threads = True

    url = f"http://{HOST}:{port}"

    print()
    print("=" * 55)
    print("       PARALLAXSTACK STARTED")
    print("=" * 55)
    print("Open in your phone browser:")
    print(url)
    print()
    print("Keep Pydroid running.")
    print("Press Ctrl+C to stop.")
    print("=" * 55)

    try:

        import webbrowser
        webbrowser.open(url)

    except Exception:
        pass

    try:

        server.serve_forever()

    except KeyboardInterrupt:

        print("Stopping server...")

    finally:

        server.server_close()


# ============================================================
# STREAMLIT MODE
# ============================================================

def run_streamlit():

    import streamlit as st

    st.set_page_config(
        page_title="ParallaxStack",
        page_icon="🔭",
        layout="wide"
    )

    if "language" not in st.session_state:
        st.session_state.language = "fa"

    language = st.sidebar.radio(
        "Language / زبان",
        ["فارسی", "English"]
    )

    st.session_state.language = (
        "fa" if language == "فارسی" else "en"
    )

    t = LANG[st.session_state.language]

    st.title("🔭 " + t["title"])
    st.caption(t["subtitle"])

    uploaded = st.file_uploader(
        t["upload"],
        type=["npy", "npz", "fits", "fit", "fts"]
    )

    if uploaded is None:
        st.info(t["no_data"])
        return

    try:

        cube, wcs = load_cube(
            uploaded.getvalue(),
            uploaded.name
        )

    except Exception as exc:

        st.error(str(exc))
        return

    st.write("Shape:", cube.shape)

    col1, col2, col3 = st.columns(3)

    with col1:
        sigma = st.slider(
            t["sigma"],
            2.0,
            15.0,
            5.0,
            0.5
        )

    with col2:
        radius = st.slider(
            t["radius"],
            1,
            12,
            3
        )

    with col3:
        mintrack = st.slider(
            t["mintrack"],
            2,
            20,
            2
        )

    align = st.checkbox(
        t["align"],
        value=True
    )

    background = st.selectbox(
        t["background"],
        ["global", "rowcol"]
    )

    if st.button(
        t["analyze"],
        type="primary"
    ):

        with st.spinner(t["waiting"]):

            result = analyze_streamlit_data(
                cube,
                wcs,
                sigma,
                radius,
                mintrack,
                align,
                background
            )

        st.success(t["success"])

        mode_labels = {
            t["epoch_mode"]: "epoch",
            t["difference"]: "difference",
            t["median"]: "median",
            t["mean"]: "mean",
            t["reference"]: "reference"
        }

        selected = st.selectbox(
            t["mode"],
            list(mode_labels.keys())
        )

        mode = mode_labels[selected]

        epoch = st.slider(
            t["epoch"],
            0,
            len(cube)-1,
            0
        )

        if mode == "epoch":
            image = result["processed"][epoch]
        elif mode == "difference":
            image = result["processed"][epoch] - result["median"]
        elif mode == "median":
            image = result["median"]
        elif mode == "mean":
            image = result["mean"]
        else:
            image = result["processed"][epoch] - result["processed"][0]

        st.image(
            normalize_image(image),
            caption=t["preview"],
            use_container_width=True
        )

        st.metric(
            t["sources"],
            len(result["catalogue"])
        )

        st.metric(
            t["tracks"],
            len(result["tracks"])
        )

        st.subheader(t["catalogue"])

        st.dataframe(
            result["catalogue"],
            use_container_width=True
        )

        st.subheader(t["transient"])

        st.dataframe(
            result["transients"],
            use_container_width=True
        )

        st.download_button(
            t["json"],
            data=export_json(result),
            file_name="parallaxstack.json",
            mime="application/json"
        )

        st.download_button(
            t["csv"],
            data=export_csv(result),
            file_name="parallaxstack.csv",
            mime="text/csv"
        )


def analyze_streamlit_data(
    cube,
    wcs,
    sigma,
    radius,
    mintrack,
    align,
    background
):

    old_data = DATA.copy()

    try:

        with LOCK:
            DATA["cube"] = cube
            DATA["wcs"] = wcs

        result = analyze({
            "sigma": sigma,
            "radius": radius,
            "mintrack": mintrack,
            "align": "translation" if align else "none",
            "background": background,
            "mode": "epoch",
            "epoch": 0
        })

        processed = result

        return {
            **processed,
            "median": np.median(
                processed["pixels"]
            ) if False else np.median(
                cube,
                axis=0
            ),
            "mean": np.mean(cube, axis=0)
        }

    finally:

        with LOCK:
            DATA.update(old_data)


# ============================================================
# ENTRY POINT
# ============================================================

def is_streamlit():

    try:

        from streamlit.runtime.scriptrunner import (
            get_script_run_ctx
        )

        return (
            get_script_run_ctx(
                suppress_warning=True
            ) is not None
        )

    except Exception:
        return False


def main():

    if is_streamlit():

        run_streamlit()

    else:

        run_android()


if __name__ == "__main__":
    main()