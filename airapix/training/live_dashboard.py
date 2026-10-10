from __future__ import annotations

import math
import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, List, Optional


def _sanitize_val(v: Any) -> Any:
    """Recursively replaces NaN, Inf, and -Inf float values with valid JSON numbers (0.0)."""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return 0.0
        return v
    elif isinstance(v, dict):
        return {k: _sanitize_val(val) for k, val in v.items()}
    elif isinstance(v, list):
        return [_sanitize_val(val) for val in v]
    return v


class MetricsTracker:
    """Thread-safe state tracker for training metrics and logs."""

    def __init__(self):
        self._lock = threading.Lock()
        self.state: Dict[str, Any] = {
            "status": "INITIALIZING",
            "model_preset": "8B",
            "step": 0,
            "max_steps": 1000,
            "epoch": 0,
            "loss": 0.0,
            "val_loss": 0.0,
            "perplexity": 0.0,
            "lr": 0.0,
            "sys1_loss": 0.0,
            "sys2_loss": 0.0,
            "tokens_per_sec": 0.0,
            "vram_used_gb": 0.0,
            "vram_total_gb": 4.0,
            "device_name": "NVIDIA GeForce RTX 2050",
            "hardware_mode": "Dedicated GDDR6 VRAM Mode",
            "elapsed_sec": 0,
            "eta_sec": 0,
            "dataset_mixture": {
                "Text": 34,
                "Chat": 18,
                "Reasoning": 20,
                "Code": 16,
                "Tool Use": 12,
            },
            "history": {
                "steps": [],
                "loss": [],
                "perplexity": [],
                "lr": [],
                "val_loss": [],
                "tokens_per_sec": [],
                "vram_used_gb": [],
                "sys1_loss": [],
                "sys2_loss": [],
            },
            "logs": [],
            "checkpoints": [],
        }

    def update(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                if k in self.state:
                    self.state[k] = v

    def record_step(self, step: int, loss: float, lr: float, tokens_per_sec: float, vram_used_gb: float, sys1_loss: float = 0.0, sys2_loss: float = 0.0, val_loss: Optional[float] = None) -> None:
        with self._lock:
            safe_loss = 0.0 if (math.isnan(loss) or math.isinf(loss)) else float(loss)
            safe_lr = 0.0 if (math.isnan(lr) or math.isinf(lr)) else float(lr)
            safe_tps = 0.0 if (math.isnan(tokens_per_sec) or math.isinf(tokens_per_sec)) else float(tokens_per_sec)
            safe_vram = 0.0 if (math.isnan(vram_used_gb) or math.isinf(vram_used_gb)) else float(vram_used_gb)
            safe_sys1 = 0.0 if (math.isnan(sys1_loss) or math.isinf(sys1_loss)) else float(sys1_loss)
            safe_sys2 = 0.0 if (math.isnan(sys2_loss) or math.isinf(sys2_loss)) else float(sys2_loss)

            ppl = round(2.71828 ** min(safe_loss, 20.0), 2)
            self.state["step"] = step
            self.state["loss"] = round(safe_loss, 4)
            self.state["perplexity"] = ppl
            self.state["lr"] = safe_lr
            self.state["tokens_per_sec"] = round(safe_tps, 1)
            self.state["vram_used_gb"] = round(safe_vram, 2)
            self.state["sys1_loss"] = round(safe_sys1, 4)
            self.state["sys2_loss"] = round(safe_sys2, 4)

            hist = self.state["history"]
            hist["steps"].append(step)
            hist["loss"].append(round(safe_loss, 4))
            hist["perplexity"].append(ppl)
            hist["lr"].append(safe_lr)
            hist["tokens_per_sec"].append(round(safe_tps, 1))
            hist["vram_used_gb"].append(round(safe_vram, 2))
            hist["sys1_loss"].append(round(safe_sys1, 4))
            hist["sys2_loss"].append(round(safe_sys2, 4))

            if val_loss is not None:
                safe_val = 0.0 if (math.isnan(val_loss) or math.isinf(val_loss)) else float(val_loss)
                self.state["val_loss"] = round(safe_val, 4)
                hist["val_loss"].append({"step": step, "val_loss": round(safe_val, 4)})

            # Limit history length to 500 points
            if len(hist["steps"]) > 500:
                hist["steps"].pop(0)
                hist["loss"].pop(0)
                hist["perplexity"].pop(0)
                hist["lr"].pop(0)
                hist["tokens_per_sec"].pop(0)
                hist["vram_used_gb"].pop(0)
                hist["sys1_loss"].pop(0)
                hist["sys2_loss"].pop(0)

    def log_message(self, level: str, message: str) -> None:
        with self._lock:
            entry = {
                "timestamp": time.strftime("%H:%M:%S"),
                "level": level,
                "message": message,
            }
            self.state["logs"].append(entry)
            if len(self.state["logs"]) > 200:
                self.state["logs"].pop(0)

    def add_checkpoint(self, path: str, step: int, loss: float) -> None:
        with self._lock:
            safe_loss = 0.0 if (math.isnan(loss) or math.isinf(loss)) else float(loss)
            self.state["checkpoints"].append({
                "step": step,
                "path": path,
                "loss": round(safe_loss, 4),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })

    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            sanitized = _sanitize_val(self.state)
            return json.loads(json.dumps(sanitized))


# Global metrics tracker instance
GLOBAL_TRACKER = MetricsTracker()


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AiraAI Training Monitor - Live Dashboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-dark: #070A12;
            --card-bg: rgba(15, 23, 42, 0.75);
            --card-border: rgba(255, 255, 255, 0.08);
            --accent-cyan: #00F2FE;
            --accent-blue: #3B82F6;
            --accent-purple: #A855F7;
            --accent-pink: #EC4899;
            --accent-red: #EF4444;
            --accent-green: #10B981;
            --accent-amber: #F59E0B;
            --text-primary: #F8FAFC;
            --text-secondary: #94A3B8;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-dark);
            background-image: 
                radial-gradient(at 0% 0%, rgba(0, 242, 254, 0.12) 0px, transparent 50%),
                radial-gradient(at 100% 0%, rgba(168, 85, 247, 0.12) 0px, transparent 50%),
                radial-gradient(at 50% 100%, rgba(16, 185, 129, 0.08) 0px, transparent 50%);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 24px;
        }

        /* Header */
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            border: 1px solid var(--card-border);
            padding: 20px 28px;
            border-radius: 18px;
            margin-bottom: 24px;
            box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.4);
        }

        .header-title { display: flex; align-items: center; gap: 16px; }
        .logo-badge {
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
            color: #000;
            font-weight: 800;
            font-size: 1.15rem;
            padding: 8px 16px;
            border-radius: 12px;
            letter-spacing: 0.5px;
            box-shadow: 0 0 20px rgba(0, 242, 254, 0.4);
        }
        .title-text h1 { font-size: 1.45rem; font-weight: 700; color: #FFF; }
        .title-text p { font-size: 0.85rem; color: var(--text-secondary); margin-top: 2px; }

        .header-right { display: flex; align-items: center; gap: 16px; }
        .filter-group { display: flex; gap: 6px; background: rgba(0,0,0,0.3); padding: 4px; border-radius: 10px; border: 1px solid var(--card-border); }
        .btn-filter {
            background: transparent;
            border: none;
            color: var(--text-secondary);
            padding: 6px 14px;
            border-radius: 8px;
            font-size: 0.78rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .btn-filter.active, .btn-filter:hover { background: rgba(255, 255, 255, 0.1); color: #FFF; }

        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.4);
            color: var(--accent-green);
            padding: 8px 18px;
            border-radius: 30px;
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.8px;
        }
        .status-dot { width: 8px; height: 8px; background-color: var(--accent-green); border-radius: 50%; box-shadow: 0 0 12px var(--accent-green); animation: pulse 1.5s infinite; }

        @keyframes pulse { 0% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(1.2); } 100% { opacity: 1; transform: scale(1); } }

        /* Metrics Bar */
        .grid-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 18px;
            margin-bottom: 24px;
        }

        .metric-card {
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 20px;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .metric-card:hover { transform: translateY(-3px); border-color: rgba(0, 242, 254, 0.35); }

        .metric-label { font-size: 0.78rem; color: var(--text-secondary); text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; }
        .metric-value { font-size: 1.75rem; font-weight: 700; color: #FFF; margin-top: 8px; font-family: 'JetBrains Mono', monospace; }
        .metric-sub { font-size: 0.75rem; color: var(--accent-cyan); margin-top: 6px; font-weight: 500; }

        .progress-bar-bg { background: rgba(255, 255, 255, 0.1); height: 6px; border-radius: 3px; overflow: hidden; margin-top: 10px; }
        .progress-bar-fill { height: 100%; background: linear-gradient(90deg, var(--accent-cyan), var(--accent-blue)); width: 0%; transition: width 0.3s ease; }

        /* Summary Analytics Bar */
        .analytics-pill-bar {
            display: flex;
            gap: 16px;
            overflow-x: auto;
            margin-bottom: 24px;
            padding-bottom: 4px;
        }
        .pill-stat {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--card-border);
            padding: 10px 18px;
            border-radius: 30px;
            font-size: 0.82rem;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            gap: 8px;
            white-space: nowrap;
        }
        .pill-stat strong { color: #FFF; font-family: 'JetBrains Mono', monospace; }

        /* 6 Charts Grid Layout */
        .charts-grid-6 {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-bottom: 24px;
        }

        @media (max-width: 1280px) { .charts-grid-6 { grid-template-columns: repeat(2, 1fr); } }
        @media (max-width: 768px) { .charts-grid-6 { grid-template-columns: 1fr; } }

        .chart-card {
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 22px;
            display: flex;
            flex-direction: column;
        }
        .chart-header { font-size: 0.95rem; font-weight: 600; margin-bottom: 14px; color: #FFF; display: flex; justify-content: space-between; align-items: center; }
        .chart-header .badge { font-size: 0.72rem; padding: 3px 8px; border-radius: 6px; background: rgba(255,255,255,0.06); color: var(--text-secondary); }
        .chart-wrapper { position: relative; height: 250px; width: 100%; }

        /* Console Section */
        .log-section {
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 22px;
        }
        .log-terminal {
            background-color: #030712;
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 12px;
            padding: 16px;
            height: 220px;
            overflow-y: auto;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.82rem;
            line-height: 1.6;
        }
        .log-entry { margin-bottom: 4px; }
        .log-time { color: var(--text-secondary); }
        .log-info { color: var(--accent-blue); }
        .log-success { color: var(--accent-green); }
        .log-warn { color: var(--accent-amber); }
        .log-step { color: var(--accent-cyan); }
    </style>
</head>
<body>

    <!-- Header -->
    <div class="header">
        <div class="header-title">
            <div class="logo-badge">AIRA AI</div>
            <div class="title-text">
                <h1>Aira Model Live Training Monitor</h1>
                <p id="model-preset-label">Hybrid MLA + SSM Dual-System Architecture (Dedicated VRAM Mode)</p>
            </div>
        </div>
        <div class="header-right">
            <div class="filter-group">
                <button class="btn-filter active" onclick="setZoom('all')" id="btn-zoom-all">All Steps</button>
                <button class="btn-filter" onclick="setZoom('ema')" id="btn-zoom-ema">Smoothed Trend (EMA)</button>
                <button class="btn-filter" onclick="setZoom(500)" id="btn-zoom-500">Last 500</button>
                <button class="btn-filter" onclick="setZoom(200)" id="btn-zoom-200">Last 200</button>
                <button class="btn-filter" onclick="setZoom(50)" id="btn-zoom-50">Last 50</button>
                <button class="btn-filter" onclick="setZoom(10)" id="btn-zoom-10">Last 10</button>
                <button class="btn-filter" onclick="setZoom(5)" id="btn-zoom-5">Last 5</button>
            </div>
            <div class="status-badge" id="status-badge">
                <span class="status-dot"></span>
                <span id="status-text">TRAINING LIVE</span>
            </div>
        </div>
    </div>

    <!-- Top Key Metrics Cards -->
    <div class="grid-metrics">
        <div class="metric-card">
            <div class="metric-label">Training Step</div>
            <div class="metric-value" id="val-step">0 / 0</div>
            <div class="progress-bar-bg"><div class="progress-bar-fill" id="bar-progress"></div></div>
        </div>

        <div class="metric-card">
            <div class="metric-label">Current Loss</div>
            <div class="metric-value" id="val-loss">0.0000</div>
            <div class="metric-sub" id="val-perplexity">Perplexity: 0.00</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">Throughput Speed</div>
            <div class="metric-value" id="val-throughput">0 tok/s</div>
            <div class="metric-sub" id="val-eta">ETA: Calculating...</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">GPU VRAM Usage</div>
            <div class="metric-value" id="val-vram">0.0 GB</div>
            <div class="metric-sub" id="val-vram-sub">Total Available: 4.0 GB</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">Learning Rate</div>
            <div class="metric-value" id="val-lr">0.00e-0</div>
            <div class="metric-sub">Muon + AdamW Hybrid</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">System 1 / System 2 Loss</div>
            <div class="metric-value" id="val-sys-loss">0.00 / 0.00</div>
            <div class="metric-sub">Dual-Process Loss Terms</div>
        </div>
    </div>

    <!-- Summary Analytical Pills -->
    <div class="analytics-pill-bar">
        <div class="pill-stat"> Lowest Loss: <strong id="stat-min-loss">N/A</strong></div>
        <div class="pill-stat"> Peak Speed: <strong id="stat-max-speed">N/A</strong></div>
        <div class="pill-stat"> Loss Delta (50 steps): <strong id="stat-loss-delta">N/A</strong></div>
        <div class="pill-stat"> Peak VRAM: <strong id="stat-peak-vram">N/A</strong></div>
        <div class="pill-stat"> Hardware: <strong id="stat-hw-mode">GPU Mode</strong></div>
    </div>

    <!-- 6 Graphical Analysis Charts -->
    <div class="charts-grid-6">
        
        <!-- Chart 1: Loss Trajectory & Val Loss -->
        <div class="chart-card">
            <div class="chart-header">
                <span>1. Dynamic Loss Trajectory</span>
                <span class="badge">Adaptive EMA + Cross-Entropy</span>
            </div>
            <div class="chart-wrapper">
                <canvas id="lossChart"></canvas>
            </div>
        </div>

        <!-- Chart 2: Dual-System Loss Breakdown -->
        <div class="chart-card">
            <div class="chart-header">
                <span>2. Dual-System Loss Breakdown</span>
                <span class="badge">Sys1 (Triage) vs Sys2 (PRM)</span>
            </div>
            <div class="chart-wrapper">
                <canvas id="dualSysChart"></canvas>
            </div>
        </div>

        <!-- Chart 3: Learning Rate Schedule -->
        <div class="chart-card">
            <div class="chart-header">
                <span>3. Learning Rate Decay</span>
                <span class="badge">Cosine Warmup Schedule</span>
            </div>
            <div class="chart-wrapper">
                <canvas id="lrChart"></canvas>
            </div>
        </div>

        <!-- Chart 4: Model Perplexity Progression -->
        <div class="chart-card">
            <div class="chart-header">
                <span>4. Model Perplexity (PPL)</span>
                <span class="badge">Entropy Level</span>
            </div>
            <div class="chart-wrapper">
                <canvas id="pplChart"></canvas>
            </div>
        </div>

        <!-- Chart 5: Throughput Speed & VRAM Usage -->
        <div class="chart-card">
            <div class="chart-header">
                <span>5. Speed (tok/s) & VRAM</span>
                <span class="badge">Performance</span>
            </div>
            <div class="chart-wrapper">
                <canvas id="perfChart"></canvas>
            </div>
        </div>

        <!-- Chart 6: Data Shard Mixture Proportions -->
        <div class="chart-card">
            <div class="chart-header">
                <span>6. Multi-Shard Dataset Mixture</span>
                <span class="badge">Active Proportions</span>
            </div>
            <div class="chart-wrapper">
                <canvas id="mixtureChart"></canvas>
            </div>
        </div>

    </div>

    <!-- Terminal Output Log -->
    <div class="log-section">
        <div class="chart-header">
            <span>Live Training Console Output</span>
            <span style="font-size:0.75rem; color:var(--accent-cyan)" id="log-count">0 logs</span>
        </div>
        <div class="log-terminal" id="terminal">
            <div class="log-entry"><span class="log-time">[00:00:00]</span> <span class="log-info">[INFO]</span> Connected to Aira Live Training Analytics Dashboard...</div>
        </div>
    </div>

    <script>
        let currentZoom = 'all';

        function setZoom(mode) {
            currentZoom = mode;
            document.querySelectorAll('.btn-filter').forEach(b => b.classList.remove('active'));
            if(mode === 'all') document.getElementById('btn-zoom-all').classList.add('active');
            if(mode === 'ema') document.getElementById('btn-zoom-ema').classList.add('active');
            if(mode === 500) document.getElementById('btn-zoom-500').classList.add('active');
            if(mode === 200) document.getElementById('btn-zoom-200').classList.add('active');
            if(mode === 50) document.getElementById('btn-zoom-50').classList.add('active');
            if(mode === 10) document.getElementById('btn-zoom-10').classList.add('active');
            if(mode === 5) document.getElementById('btn-zoom-5').classList.add('active');
            fetchMetrics();
        }

        function calculateEMA(data, alpha = 0.12) {
            if (!data || data.length === 0) return [];
            let ema = [data[0]];
            for (let i = 1; i < data.length; i++) {
                let val = typeof data[i] === 'number' && isFinite(data[i]) ? data[i] : ema[i - 1];
                ema.push(Number((alpha * val + (1 - alpha) * ema[i - 1]).toFixed(4)));
            }
            return ema;
        }

        function subsampleSeries(steps, values, targetPoints = 150) {
            if (!steps || steps.length <= targetPoints) return { steps: steps || [], values: values || [] };
            const stepSize = Math.ceil(steps.length / targetPoints);
            const subSteps = [];
            const subValues = [];
            for (let i = 0; i < steps.length; i += stepSize) {
                const chunkSteps = steps.slice(i, i + stepSize);
                const chunkValues = values.slice(i, i + stepSize).filter(v => typeof v === 'number' && isFinite(v));
                if (chunkValues.length > 0) {
                    const avgVal = chunkValues.reduce((a, b) => a + b, 0) / chunkValues.length;
                    subSteps.push(chunkSteps[chunkSteps.length - 1]);
                    subValues.push(Number(avgVal.toFixed(4)));
                }
            }
            return { steps: subSteps, values: subValues };
        }

        function adaptYAxis(chart, values, axisID = 'y') {
            if (!values || values.length === 0) return;
            const valid = values.filter(v => typeof v === 'number' && isFinite(v) && v > 0);
            if (valid.length > 0) {
                const minV = Math.min(...valid);
                const maxV = Math.max(...valid);
                const margin = (maxV - minV) * 0.12 || 0.5;
                if (chart.options.scales[axisID]) {
                    chart.options.scales[axisID].min = Math.max(0, Number((minV - margin).toFixed(2)));
                    chart.options.scales[axisID].max = Number((maxV + margin).toFixed(2));
                }
            }
        }

        const commonOptions = {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94A3B8', font: { size: 10 } } },
                y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94A3B8', font: { size: 10 } } }
            },
            plugins: {
                legend: { labels: { color: '#F8FAFC', font: { size: 11, family: 'Inter' }, boxWidth: 12 } },
                tooltip: {
                    backgroundColor: '#0F172A',
                    titleColor: '#00F2FE',
                    bodyColor: '#F8FAFC',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1
                }
            }
        };

        // 1. Loss Chart with Dynamic EMA & Raw Datasets
        const lossChart = new Chart(document.getElementById('lossChart').getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'Smoothed Trend (EMA)', borderColor: '#00F2FE', backgroundColor: 'rgba(0, 242, 254, 0.12)', data: [], fill: true, tension: 0.3, borderWidth: 2.5, pointRadius: 0 },
                    { label: 'Raw Loss (Points)', borderColor: 'rgba(59, 130, 246, 0.25)', backgroundColor: 'transparent', data: [], tension: 0.1, borderWidth: 1, pointRadius: 1 },
                    { label: 'Val Loss', borderColor: '#F59E0B', backgroundColor: '#F59E0B', data: [], pointRadius: 5, showLine: false }
                ]
            },
            options: commonOptions
        });

        // 2. Dual System Loss Chart
        const dualSysChart = new Chart(document.getElementById('dualSysChart').getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'System 1 (Triage)', borderColor: '#10B981', data: [], tension: 0.25, borderWidth: 2, pointRadius: 0 },
                    { label: 'System 2 (PRM)', borderColor: '#EF4444', data: [], tension: 0.25, borderWidth: 2, pointRadius: 0 },
                    { label: 'LM Loss Trend', borderColor: '#3B82F6', borderDash: [4, 4], data: [], tension: 0.25, borderWidth: 1.5, pointRadius: 0 }
                ]
            },
            options: commonOptions
        });

        // 3. Learning Rate Chart
        const lrChart = new Chart(document.getElementById('lrChart').getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{ label: 'Learning Rate', borderColor: '#A855F7', backgroundColor: 'rgba(168, 85, 247, 0.12)', data: [], fill: true, tension: 0.3, borderWidth: 2, pointRadius: 0 }]
            },
            options: commonOptions
        });

        // 4. Perplexity Chart
        const pplChart = new Chart(document.getElementById('pplChart').getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{ label: 'Smoothed Perplexity', borderColor: '#F59E0B', backgroundColor: 'rgba(245, 158, 11, 0.08)', data: [], fill: true, tension: 0.3, borderWidth: 2, pointRadius: 0 }]
            },
            options: commonOptions
        });

        // 5. Throughput & VRAM Chart
        const perfChart = new Chart(document.getElementById('perfChart').getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'Tok/sec', borderColor: '#10B981', data: [], yAxisID: 'y', pointRadius: 0 },
                    { label: 'VRAM (GB)', borderColor: '#EC4899', data: [], yAxisID: 'y1', pointRadius: 0 }
                ]
            },
            options: {
                ...commonOptions,
                scales: {
                    x: commonOptions.scales.x,
                    y: { type: 'linear', position: 'left', ticks: { color: '#10B981' } },
                    y1: { type: 'linear', position: 'right', grid: { drawOnChartArea: false }, ticks: { color: '#EC4899' } }
                }
            }
        });

        // 6. Dataset Mixture Doughnut Chart
        const mixtureChart = new Chart(document.getElementById('mixtureChart').getContext('2d'), {
            type: 'doughnut',
            data: {
                labels: ['Text', 'Chat', 'Reasoning', 'Code', 'Tool Use'],
                datasets: [{
                    data: [34, 18, 20, 16, 12],
                    backgroundColor: ['#3B82F6', '#10B981', '#A855F7', '#00F2FE', '#F59E0B'],
                    borderWidth: 2,
                    borderColor: '#0F172A'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'right', labels: { color: '#F8FAFC', font: { size: 11 } } }
                }
            }
        });

        async function fetchMetrics() {
            try {
                const res = await fetch('/api/metrics');
                const data = await res.json();

                // Top Metrics Cards
                document.getElementById('status-text').innerText = data.status || 'UNKNOWN';
                document.getElementById('val-step').innerText = `${data.step || 0} / ${data.max_steps || 0}`;
                document.getElementById('val-loss').innerText = (typeof data.loss === 'number' && isFinite(data.loss)) ? data.loss.toFixed(4) : '0.0000';
                document.getElementById('val-perplexity').innerText = `Perplexity: ${data.perplexity ?? 0}`;
                document.getElementById('val-throughput').innerText = `${data.tokens_per_sec ?? 0} tok/s`;
                if (data.model_preset) {
                    const hwStr = data.device_name ? `${data.device_name} — Dedicated VRAM` : 'Dedicated VRAM Mode';
                    document.getElementById('model-preset-label').innerText = `Hybrid MLA + SSM Dual-System Architecture (${data.model_preset} Preset — ${hwStr})`;
                }
                document.getElementById('val-vram').innerText = `${data.vram_used_gb ?? 0} GB`;
                document.getElementById('val-vram-sub').innerText = `Total Dedicated VRAM: ${data.vram_total_gb ?? 0} GB`;
                document.getElementById('val-lr').innerText = (typeof data.lr === 'number' && isFinite(data.lr)) ? data.lr.toExponential(2) : '0.00e+0';
                const sys1 = (typeof data.sys1_loss === 'number' && isFinite(data.sys1_loss)) ? data.sys1_loss.toFixed(3) : '0.000';
                const sys2 = (typeof data.sys2_loss === 'number' && isFinite(data.sys2_loss)) ? data.sys2_loss.toFixed(3) : '0.000';
                document.getElementById('val-sys-loss').innerText = `${sys1} / ${sys2}`;

                // Progress Bar
                const pct = data.max_steps > 0 ? (data.step / data.max_steps) * 100 : 0;
                document.getElementById('bar-progress').style.width = `${pct}%`;

                // ETA
                if (data.eta_sec > 0) {
                    const mins = Math.floor(data.eta_sec / 60);
                    const secs = data.eta_sec % 60;
                    document.getElementById('val-eta').innerText = `ETA: ${mins}m ${secs}s`;
                }

                // Analytics Summary Bar
                if (data.history && data.history.loss && data.history.loss.length > 0) {
                    const validLosses = data.history.loss.filter(v => typeof v === 'number' && isFinite(v));
                    const validSpeeds = data.history.tokens_per_sec.filter(v => typeof v === 'number' && isFinite(v));
                    const validVram = data.history.vram_used_gb.filter(v => typeof v === 'number' && isFinite(v));

                    const minL = validLosses.length > 0 ? Math.min(...validLosses) : 0;
                    const maxS = validSpeeds.length > 0 ? Math.max(...validSpeeds) : 0;
                    const peakV = validVram.length > 0 ? Math.max(...validVram) : 0;

                    document.getElementById('stat-min-loss').innerText = minL.toFixed(4);
                    document.getElementById('stat-max-speed').innerText = `${maxS.toFixed(1)} tok/s`;
                    document.getElementById('stat-peak-vram').innerText = `${peakV.toFixed(2)} GB`;
                    document.getElementById('stat-hw-mode').innerText = data.hardware_mode || 'Dedicated VRAM Mode';

                    if (validLosses.length >= 50) {
                        const first50 = validLosses[validLosses.length - 50];
                        const lastL = validLosses[validLosses.length - 1];
                        const delta = lastL - first50;
                        const sign = delta <= 0 ? '' : '+';
                        document.getElementById('stat-loss-delta').innerText = `${sign}${delta.toFixed(4)}`;
                    }
                }

                // Update Dataset Mixture if present
                if (data.dataset_mixture) {
                    mixtureChart.data.labels = Object.keys(data.dataset_mixture);
                    mixtureChart.data.datasets[0].data = Object.values(data.dataset_mixture);
                    mixtureChart.update('none');
                }

                // Update History Charts with Dynamic Downsampling & Auto-Zoom
                if (data.history && data.history.steps.length > 0) {
                    let steps = data.history.steps;
                    let loss = data.history.loss;
                    let sys1 = data.history.sys1_loss;
                    let sys2 = data.history.sys2_loss;
                    let lr = data.history.lr;
                    let ppl = data.history.perplexity;
                    let tok_s = data.history.tokens_per_sec;
                    let vram = data.history.vram_used_gb;

                    // Slice step window based on selected zoom mode
                    if (typeof currentZoom === 'number') {
                        const cut = Math.max(0, steps.length - currentZoom);
                        steps = steps.slice(cut);
                        loss = loss.slice(cut);
                        sys1 = sys1.slice(cut);
                        sys2 = sys2.slice(cut);
                        lr = lr.slice(cut);
                        ppl = ppl.slice(cut);
                        tok_s = tok_s.slice(cut);
                        vram = vram.slice(cut);
                    }

                    // Compute EMA Smoothed Loss & Perplexity
                    const emaLoss = calculateEMA(loss, 0.12);
                    const emaPpl = calculateEMA(ppl, 0.12);

                    // Dynamic Subsampling for Large Step Runs
                    let displaySteps = steps;
                    let displayLoss = loss;
                    let displayEmaLoss = emaLoss;
                    let displaySys1 = sys1;
                    let displaySys2 = sys2;
                    let displayLr = lr;
                    let displayPpl = emaPpl;
                    let displayTokS = tok_s;
                    let displayVram = vram;

                    if (currentZoom === 'all' && steps.length > 200) {
                        const subLoss = subsampleSeries(steps, emaLoss, 150);
                        displaySteps = subLoss.steps;
                        displayEmaLoss = subLoss.values;
                        displayLoss = subsampleSeries(steps, loss, 150).values;
                        displaySys1 = subsampleSeries(steps, sys1, 150).values;
                        displaySys2 = subsampleSeries(steps, sys2, 150).values;
                        displayLr = subsampleSeries(steps, lr, 150).values;
                        displayPpl = subsampleSeries(steps, emaPpl, 150).values;
                        displayTokS = subsampleSeries(steps, tok_s, 150).values;
                        displayVram = subsampleSeries(steps, vram, 150).values;
                    }

                    // 1. Loss Chart
                    lossChart.data.labels = displaySteps;
                    lossChart.data.datasets[0].data = displayEmaLoss;
                    lossChart.data.datasets[1].data = displayLoss;
                    if (data.history.val_loss && data.history.val_loss.length > 0) {
                        lossChart.data.datasets[2].data = displaySteps.map(s => {
                            const match = data.history.val_loss.find(v => v.step === s);
                            return match ? match.val_loss : null;
                        });
                    }
                    adaptYAxis(lossChart, displayEmaLoss);
                    lossChart.update('none');

                    // 2. Dual Sys Chart
                    dualSysChart.data.labels = displaySteps;
                    dualSysChart.data.datasets[0].data = displaySys1;
                    dualSysChart.data.datasets[1].data = displaySys2;
                    dualSysChart.data.datasets[2].data = displayEmaLoss;
                    adaptYAxis(dualSysChart, displaySys2);
                    dualSysChart.update('none');

                    // 3. LR Chart
                    lrChart.data.labels = displaySteps;
                    lrChart.data.datasets[0].data = displayLr;
                    lrChart.update('none');

                    // 4. PPL Chart
                    pplChart.data.labels = displaySteps;
                    pplChart.data.datasets[0].data = displayPpl;
                    adaptYAxis(pplChart, displayPpl);
                    pplChart.update('none');

                    // 5. Perf Chart
                    perfChart.data.labels = displaySteps;
                    perfChart.data.datasets[0].data = displayTokS;
                    perfChart.data.datasets[1].data = displayVram;
                    adaptYAxis(perfChart, displayTokS, 'y');
                    perfChart.update('none');
                }

                // Logs
                if (data.logs && data.logs.length > 0) {
                    const term = document.getElementById('terminal');
                    term.innerHTML = data.logs.map(l => {
                        let cls = 'log-info';
                        if (l.level === 'SUCCESS' || l.level === 'CHECKPOINT') cls = 'log-success';
                        if (l.level === 'WARN' || l.level === 'WARNING') cls = 'log-warn';
                        if (l.level === 'STEP') cls = 'log-step';
                        return `<div class="log-entry"><span class="log-time">[${l.timestamp}]</span> <span class="${cls}">[${l.level}]</span> ${l.message}</div>`;
                    }).join('');
                    term.scrollTop = term.scrollHeight;
                    document.getElementById('log-count').innerText = `${data.logs.length} logs`;
                }

            } catch (err) {
                console.error("Error fetching metrics:", err);
            }
        }

        setInterval(fetchMetrics, 1000);
        fetchMetrics();
    </script>
</body>
</html>
"""


class DashboardRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for training dashboard API and static UI."""

    def log_message(self, format, *args):
        # Suppress standard HTTP server console logging to keep stdout clean
        pass

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))
        elif self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        elif self.path == "/api/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            snapshot = _sanitize_val(GLOBAL_TRACKER.get_snapshot())
            self.wfile.write(json.dumps(snapshot).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def _heartbeat_loop():
    """Background heartbeat loop that updates real-time timer & VRAM every 1 second."""
    import torch
    start_t = time.time()
    while True:
        try:
            time.sleep(1.0)
            snapshot = GLOBAL_TRACKER.get_snapshot()
            if snapshot.get("status") in ["TRAINING LIVE", "INITIALIZING"]:
                elapsed = int(time.time() - start_t)
                vram = snapshot.get("vram_used_gb", 0.0)
                if torch.cuda.is_available():
                    vram = round(torch.cuda.memory_allocated(0) / (1024**3), 2)
                GLOBAL_TRACKER.update(elapsed_sec=elapsed, vram_used_gb=vram)
        except Exception:
            pass


def start_dashboard_server(port: int = 7860) -> tuple[HTTPServer, threading.Thread]:
    """Starts the training dashboard HTTP server on a background daemon thread."""
    server = HTTPServer(("0.0.0.0", port), DashboardRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    
    heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
    heartbeat_thread.start()
    
    return server, thread


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="AiraPix Standalone Live Training Dashboard Web Server")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the dashboard server on (default: 5000)")
    args = parser.parse_args()

    port = args.port
    server = HTTPServer(("0.0.0.0", port), DashboardRequestHandler)
    print(f"\n========================================================================")
    print(f"  AIRA AI STANDALONE LIVE DASHBOARD WEB SERVER RUNNING ON PORT {port}")
    print(f"  Access Dashboard UI: http://localhost:{port}")
    print(f"========================================================================\n")
    sys.stdout.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server...")
        server.server_close()
