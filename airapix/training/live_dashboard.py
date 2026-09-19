from __future__ import annotations

import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, List, Optional


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
            "vram_total_gb": 15.0,
            "elapsed_sec": 0,
            "eta_sec": 0,
            "history": {
                "steps": [],
                "loss": [],
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
            self.state["step"] = step
            self.state["loss"] = round(loss, 4)
            self.state["perplexity"] = round(2.71828 ** min(loss, 20.0), 2)
            self.state["lr"] = lr
            self.state["tokens_per_sec"] = round(tokens_per_sec, 1)
            self.state["vram_used_gb"] = round(vram_used_gb, 2)
            self.state["sys1_loss"] = round(sys1_loss, 4)
            self.state["sys2_loss"] = round(sys2_loss, 4)
            
            hist = self.state["history"]
            hist["steps"].append(step)
            hist["loss"].append(round(loss, 4))
            hist["tokens_per_sec"].append(round(tokens_per_sec, 1))
            hist["vram_used_gb"].append(round(vram_used_gb, 2))
            hist["sys1_loss"].append(round(sys1_loss, 4))
            hist["sys2_loss"].append(round(sys2_loss, 4))
            
            if val_loss is not None:
                self.state["val_loss"] = round(val_loss, 4)
                hist["val_loss"].append({"step": step, "val_loss": round(val_loss, 4)})

            # Limit history length to 500 points
            if len(hist["steps"]) > 500:
                hist["steps"].pop(0)
                hist["loss"].pop(0)
                hist["tokens_per_sec"].pop(0)
                hist["vram_used_gb"].pop(0)

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
            self.state["checkpoints"].append({
                "step": step,
                "path": path,
                "loss": round(loss, 4),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })

    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self.state))


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
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-dark: #090D16;
            --card-bg: rgba(19, 27, 46, 0.75);
            --card-border: rgba(255, 255, 255, 0.08);
            --accent-cyan: #00F2FE;
            --accent-blue: #4FACFE;
            --accent-pink: #FF0844;
            --accent-green: #00E676;
            --text-primary: #F1F5F9;
            --text-secondary: #94A3B8;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-dark);
            background-image: 
                radial-gradient(at 0% 0%, rgba(0, 242, 254, 0.12) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(79, 172, 254, 0.12) 0px, transparent 50%);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 24px;
        }

        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            border: 1px solid var(--card-border);
            padding: 18px 28px;
            border-radius: 16px;
            margin-bottom: 24px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }

        .header-title { display: flex; align-items: center; gap: 14px; }
        .logo-badge {
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
            color: #000;
            font-weight: 800;
            font-size: 1.1rem;
            padding: 6px 14px;
            border-radius: 10px;
            letter-spacing: 0.5px;
        }
        .title-text h1 { font-size: 1.4rem; font-weight: 700; color: #FFF; }
        .title-text p { font-size: 0.85rem; color: var(--text-secondary); margin-top: 2px; }

        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(0, 230, 118, 0.15);
            border: 1px solid rgba(0, 230, 118, 0.4);
            color: var(--accent-green);
            padding: 8px 16px;
            border-radius: 30px;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .status-dot { width: 8px; height: 8px; background-color: var(--accent-green); border-radius: 50%; box-shadow: 0 0 10px var(--accent-green); animation: pulse 1.5s infinite; }

        @keyframes pulse { 0% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(1.2); } 100% { opacity: 1; transform: scale(1); } }

        .grid-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 18px;
            margin-bottom: 24px;
        }

        .metric-card {
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 20px;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .metric-card:hover { transform: translateY(-3px); border-color: rgba(0, 242, 254, 0.3); }

        .metric-label { font-size: 0.8rem; color: var(--text-secondary); text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; }
        .metric-value { font-size: 1.8rem; font-weight: 700; color: #FFF; margin-top: 8px; font-family: 'JetBrains Mono', monospace; }
        .metric-sub { font-size: 0.75rem; color: var(--accent-cyan); margin-top: 6px; }

        .charts-container {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            margin-bottom: 24px;
        }

        @media (max-width: 1024px) { .charts-container { grid-template-columns: 1fr; } }

        .chart-card {
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 22px;
        }
        .chart-header { font-size: 1rem; font-weight: 600; margin-bottom: 16px; color: #FFF; display: flex; justify-content: space-between; align-items: center; }
        .chart-wrapper { position: relative; height: 280px; width: 100%; }

        .log-section {
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 22px;
        }
        .log-terminal {
            background-color: #050811;
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 16px;
            height: 200px;
            overflow-y: auto;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.82rem;
            line-height: 1.5;
        }
        .log-entry { margin-bottom: 4px; }
        .log-time { color: var(--text-secondary); }
        .log-info { color: var(--accent-blue); }
        .log-success { color: var(--accent-green); }
        .log-warn { color: #FFB74D; }

        .progress-bar-bg { background: rgba(255, 255, 255, 0.1); height: 8px; border-radius: 4px; overflow: hidden; margin-top: 10px; }
        .progress-bar-fill { height: 100%; background: linear-gradient(90deg, var(--accent-cyan), var(--accent-blue)); width: 0%; transition: width 0.3s ease; }
    </style>
</head>
<body>

    <!-- Header -->
    <div class="header">
        <div class="header-title">
            <div class="logo-badge">AIRA AI</div>
            <div class="title-text">
                <h1>Aira Model Live Training Monitor</h1>
                <p id="model-preset-label">Hybrid MLA + SSM Dual-System Architecture (Colab GPU Mode)</p>
            </div>
        </div>
        <div class="status-badge" id="status-badge">
            <span class="status-dot"></span>
            <span id="status-text">TRAINING LIVE</span>
        </div>
    </div>

    <!-- Metrics Cards -->
    <div class="grid-metrics">
        <div class="metric-card">
            <div class="metric-label">Training Step</div>
            <div class="metric-value" id="val-step">0 / 0</div>
            <div class="progress-bar-bg"><div class="progress-bar-fill" id="bar-progress"></div></div>
        </div>

        <div class="metric-card">
            <div class="metric-label">Current Training Loss</div>
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
            <div class="metric-sub" id="val-vram-sub">Total Available: 15.0 GB</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">Learning Rate</div>
            <div class="metric-value" id="val-lr">0.00e-0</div>
            <div class="metric-sub">Muon + AdamW Hybrid</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">System 1 vs System 2 Loss</div>
            <div class="metric-value" id="val-sys-loss">0.00 / 0.00</div>
            <div class="metric-sub">Dual-Process Loss Terms</div>
        </div>
    </div>

    <!-- Charts Row -->
    <div class="charts-container">
        <div class="chart-card">
            <div class="chart-header">
                <span>Loss Trajectory Curve</span>
                <span style="font-size:0.75rem; color:var(--text-secondary)">Real-time Step Updates</span>
            </div>
            <div class="chart-wrapper">
                <canvas id="lossChart"></canvas>
            </div>
        </div>

        <div class="chart-card">
            <div class="chart-header">
                <span>System Throughput & VRAM</span>
            </div>
            <div class="chart-wrapper">
                <canvas id="perfChart"></canvas>
            </div>
        </div>
    </div>

    <!-- Terminal Log Section -->
    <div class="log-section">
        <div class="chart-header">
            <span>Live Training Console Output</span>
            <span style="font-size:0.75rem; color:var(--accent-cyan)" id="log-count">0 logs</span>
        </div>
        <div class="log-terminal" id="terminal">
            <div class="log-entry"><span class="log-time">[00:00:00]</span> <span class="log-info">[INFO]</span> Connected to Aira Training Dashboard server...</div>
        </div>
    </div>

    <script>
        // Setup Chart.js
        const lossCtx = document.getElementById('lossChart').getContext('2d');
        const perfCtx = document.getElementById('perfChart').getContext('2d');

        const lossChart = new Chart(lossCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Training Loss',
                    borderColor: '#00F2FE',
                    backgroundColor: 'rgba(0, 242, 254, 0.08)',
                    data: [],
                    fill: true,
                    tension: 0.3,
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94A3B8' } },
                    y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94A3B8' } }
                },
                plugins: { legend: { labels: { color: '#FFF' } } }
            }
        });

        const perfChart = new Chart(perfCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'Tok/sec', borderColor: '#00E676', data: [], yAxisID: 'y' },
                    { label: 'VRAM (GB)', borderColor: '#FF0844', data: [], yAxisID: 'y1' }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94A3B8' } },
                    y: { type: 'linear', position: 'left', ticks: { color: '#00E676' } },
                    y1: { type: 'linear', position: 'right', grid: { drawOnChartArea: false }, ticks: { color: '#FF0844' } }
                },
                plugins: { legend: { labels: { color: '#FFF' } } }
            }
        });

        async function fetchMetrics() {
            try {
                const res = await fetch('/api/metrics');
                const data = await res.json();

                // Update UI elements
                document.getElementById('status-text').innerText = data.status;
                document.getElementById('val-step').innerText = `${data.step} / ${data.max_steps}`;
                document.getElementById('val-loss').innerText = data.loss.toFixed(4);
                document.getElementById('val-perplexity').innerText = `Perplexity: ${data.perplexity}`;
                document.getElementById('val-throughput').innerText = `${data.tokens_per_sec} tok/s`;
                document.getElementById('val-vram').innerText = `${data.vram_used_gb} GB`;
                document.getElementById('val-vram-sub').innerText = `Total Available: ${data.vram_total_gb} GB`;
                document.getElementById('val-lr').innerText = data.lr.toExponential(2);
                document.getElementById('val-sys-loss').innerText = `${data.sys1_loss.toFixed(3)} / ${data.sys2_loss.toFixed(3)}`;

                // Progress Bar
                const pct = data.max_steps > 0 ? (data.step / data.max_steps) * 100 : 0;
                document.getElementById('bar-progress').style.width = `${pct}%`;

                // ETA Calculation
                if (data.eta_sec > 0) {
                    const mins = Math.floor(data.eta_sec / 60);
                    const secs = data.eta_sec % 60;
                    document.getElementById('val-eta').innerText = `ETA: ${mins}m ${secs}s`;
                }

                // Update Charts
                if (data.history && data.history.steps.length > 0) {
                    lossChart.data.labels = data.history.steps;
                    lossChart.data.datasets[0].data = data.history.loss;
                    lossChart.update('none');

                    perfChart.data.labels = data.history.steps;
                    perfChart.data.datasets[0].data = data.history.tokens_per_sec;
                    perfChart.data.datasets[1].data = data.history.vram_used_gb;
                    perfChart.update('none');
                }

                // Update Terminal Logs
                if (data.logs && data.logs.length > 0) {
                    const term = document.getElementById('terminal');
                    term.innerHTML = data.logs.map(l => {
                        let cls = 'log-info';
                        if (l.level === 'SUCCESS' || l.level === 'CHECKPOINT') cls = 'log-success';
                        if (l.level === 'WARN' || l.level === 'WARNING') cls = 'log-warn';
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
        elif self.path == "/api/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            snapshot = GLOBAL_TRACKER.get_snapshot()
            self.wfile.write(json.dumps(snapshot).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def start_dashboard_server(port: int = 7860) -> tuple[HTTPServer, threading.Thread]:
    """Starts the training dashboard HTTP server on a background daemon thread."""
    server = HTTPServer(("0.0.0.0", port), DashboardRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread
