from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Dict, Any, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from airapix.model.config import config_from_preset
from airapix.model.model import AiraForCausalLM
from airapix.training.tokenizer import TokenizerWrapper

# Global State for Sequential RAM-Saver Execution
CURRENT_LOADED_MODEL: AiraForCausalLM | None = None
CURRENT_LOADED_PATH: str | None = None
GLOBAL_TOKENIZER: TokenizerWrapper | None = None
GLOBAL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("runs/checkpoints")


def unload_active_model() -> float:
    """
    Aggressively unloads the currently loaded model from RAM and CUDA VRAM.
    Returns the estimated freed VRAM in MB if CUDA is enabled.
    """
    global CURRENT_LOADED_MODEL, CURRENT_LOADED_PATH
    vram_before = 0.0
    vram_after = 0.0

    if torch.cuda.is_available():
        vram_before = torch.cuda.memory_allocated(0) / (1024 * 1024)

    if CURRENT_LOADED_MODEL is not None:
        model_name = Path(CURRENT_LOADED_PATH or "model").name
        print(f"[RAM/VRAM Saver] Unloading active model '{model_name}' from memory...")
        del CURRENT_LOADED_MODEL
        CURRENT_LOADED_MODEL = None
        CURRENT_LOADED_PATH = None

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        vram_after = torch.cuda.memory_allocated(0) / (1024 * 1024)

    freed_mb = max(0.0, vram_before - vram_after)
    if torch.cuda.is_available():
        print(f"[RAM/VRAM Saver] Reclaimed memory. Active VRAM now: {vram_after:.1f} MB.")
    return round(freed_mb, 1)


def load_single_model_on_demand(checkpoint_path: str, preset: str = "125m") -> AiraForCausalLM:
    """
    Loads a single requested model into memory after purging any previously loaded model.
    Guarantees that at most 1 model resides in RAM/VRAM at any instant.
    """
    global CURRENT_LOADED_MODEL, CURRENT_LOADED_PATH

    clean_path = str(Path(checkpoint_path).resolve()).replace("\\", "/")
    if CURRENT_LOADED_MODEL is not None and CURRENT_LOADED_PATH == clean_path:
        return CURRENT_LOADED_MODEL

    unload_active_model()

    name_lower = Path(checkpoint_path).name.lower()
    if "1.5b" in name_lower:
        preset = "1.5b"
    elif "125m" in name_lower:
        preset = "125m"
    elif "90m" in name_lower:
        preset = "90m"
    elif "60m" in name_lower:
        preset = "60m"
    elif "tiny" in name_lower:
        preset = "tiny"

    cfg = config_from_preset(preset, vocab_size=12000, context_len=1024)
    model = AiraForCausalLM(cfg)

    ckpt_path = Path(checkpoint_path)
    if ckpt_path.is_file():
        print(f"[Model Loader] Loading weights into memory: {ckpt_path.name} ({preset.upper()})...")
        try:
            state = torch.load(ckpt_path, map_location=GLOBAL_DEVICE, weights_only=True)
        except Exception:
            state = torch.load(ckpt_path, map_location=GLOBAL_DEVICE, weights_only=False)

        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=False)
        elif "model_state" in state:
            model.load_state_dict(state["model_state"], strict=False)
        elif "model" in state:
            model.load_state_dict(state["model"], strict=False)
        else:
            model.load_state_dict(state, strict=False)

    model.to(GLOBAL_DEVICE)
    model.eval()

    CURRENT_LOADED_MODEL = model
    CURRENT_LOADED_PATH = clean_path
    return model


def discover_checkpoints() -> List[Dict[str, Any]]:
    """Scans repository for available .pt checkpoint files with normalized forward-slash paths."""
    checkpoints = []
    search_dirs = [Path("runs/checkpoints"), Path("runs"), Path("models")]
    seen = set()

    for s_dir in search_dirs:
        if s_dir.exists():
            for p in s_dir.rglob("*.pt"):
                norm_str = str(p.resolve()).replace("\\", "/")
                if norm_str in seen:
                    continue
                seen.add(norm_str)

                name = p.name
                step = 0
                preset = "125m"

                step_match = re.search(r"step_(\d+)", name)
                if step_match:
                    step = int(step_match.group(1))

                if "1.5b" in name.lower():
                    preset = "1.5b"
                elif "125m" in name.lower():
                    preset = "125m"
                elif "90m" in name.lower():
                    preset = "90m"
                elif "60m" in name.lower():
                    preset = "60m"
                elif "tiny" in name.lower():
                    preset = "tiny"

                size_mb = round(p.stat().st_size / (1024 * 1024), 1)
                rel_p = str(p.relative_to(Path.cwd())).replace("\\", "/") if p.is_relative_to(Path.cwd()) else norm_str

                checkpoints.append({
                    "name": name,
                    "path": norm_str,
                    "rel_path": rel_p,
                    "step": step,
                    "preset": preset.upper(),
                    "size_mb": size_mb,
                })

    checkpoints.sort(key=lambda x: (x["preset"], x["step"]), reverse=True)
    return checkpoints


CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AiraPix - Sequential Model Arena & Global AI Benchmarks</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #070A12;
            --card-bg: rgba(15, 23, 42, 0.85);
            --card-border: rgba(255, 255, 255, 0.08);
            --accent-cyan: #00F2FE;
            --accent-blue: #3B82F6;
            --accent-purple: #A855F7;
            --accent-pink: #EC4899;
            --accent-green: #10B981;
            --accent-amber: #F59E0B;
            --text-primary: #F8FAFC;
            --text-secondary: #94A3B8;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        /* Top Bar Header */
        header {
            background: rgba(10, 15, 30, 0.95);
            border-bottom: 1px solid var(--card-border);
            padding: 0.85rem 1.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            backdrop-filter: blur(12px);
            z-index: 10;
        }

        .header-title {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .status-dot {
            width: 10px;
            height: 10px;
            background: #10B981;
            border-radius: 50%;
            box-shadow: 0 0 10px #10B981;
        }

        h1 {
            font-size: 1.15rem;
            font-weight: 700;
            background: linear-gradient(90deg, #FFFFFF, var(--accent-cyan));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .ram-saver-tag {
            background: rgba(16, 185, 129, 0.15);
            color: var(--accent-green);
            border: 1px solid rgba(16, 185, 129, 0.4);
            font-size: 0.72rem;
            padding: 3px 10px;
            border-radius: 20px;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .mode-tabs {
            display: flex;
            background: rgba(255, 255, 255, 0.05);
            padding: 3px;
            border-radius: 8px;
            border: 1px solid var(--card-border);
        }

        .tab-btn {
            padding: 0.4rem 1rem;
            font-size: 0.85rem;
            font-weight: 600;
            border: none;
            background: transparent;
            color: var(--text-secondary);
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
        }

        .tab-btn.active {
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            color: #FFF;
            box-shadow: 0 0 10px rgba(59, 130, 246, 0.4);
        }

        /* Main Container Layout */
        .main-container {
            flex: 1;
            display: flex;
            overflow: hidden;
        }

        /* Sidebar Model Selector */
        .sidebar {
            width: 340px;
            background: rgba(11, 17, 32, 0.85);
            border-right: 1px solid var(--card-border);
            padding: 1.25rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
            overflow-y: auto;
        }

        .sidebar-header-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .sidebar-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-secondary);
            font-weight: 700;
        }

        .count-badge {
            font-size: 0.75rem;
            background: rgba(0, 242, 254, 0.15);
            color: var(--accent-cyan);
            padding: 2px 8px;
            border-radius: 12px;
            font-weight: 700;
        }

        .control-btns {
            display: flex;
            gap: 0.5rem;
        }

        .btn-mini {
            flex: 1;
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid var(--card-border);
            color: var(--text-primary);
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.35rem 0.6rem;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
        }

        .btn-mini:hover {
            background: rgba(0, 242, 254, 0.15);
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
        }

        .search-input {
            width: 100%;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--card-border);
            padding: 0.5rem 0.75rem;
            border-radius: 6px;
            color: #FFF;
            font-size: 0.82rem;
            outline: none;
        }

        .search-input:focus {
            border-color: var(--accent-cyan);
        }

        .checkpoint-list {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            max-height: 240px;
            overflow-y: auto;
            padding-right: 4px;
        }

        .ckpt-card {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.65rem 0.85rem;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--card-border);
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
            user-select: none;
        }

        .ckpt-card:hover {
            border-color: var(--accent-cyan);
            background: rgba(0, 242, 254, 0.05);
        }

        .ckpt-card.selected {
            background: rgba(59, 130, 246, 0.18);
            border-color: var(--accent-blue);
        }

        .ckpt-info {
            display: flex;
            flex-direction: column;
            gap: 2px;
        }

        .ckpt-name {
            font-size: 0.85rem;
            font-weight: 600;
            color: #FFF;
        }

        .ckpt-meta {
            font-size: 0.72rem;
            color: var(--text-secondary);
        }

        .ckpt-card input[type="checkbox"] {
            width: 18px;
            height: 18px;
            accent-color: var(--accent-cyan);
            cursor: pointer;
        }

        .setting-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .setting-label {
            font-size: 0.8rem;
            color: var(--text-secondary);
            font-weight: 500;
            display: flex;
            justify-content: space-between;
        }

        input[type="range"] {
            width: 100%;
            accent-color: var(--accent-cyan);
        }

        /* Content Area */
        .content-area {
            flex: 1;
            display: flex;
            flex-direction: column;
            padding: 1.25rem;
            overflow: hidden;
            background: radial-gradient(circle at 50% 0%, rgba(0, 242, 254, 0.03), transparent 70%);
        }

        /* Arena Columns Grid */
        .arena-grid {
            flex: 1;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 1rem;
            overflow-y: auto;
            padding-bottom: 0.5rem;
        }

        .model-col {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }

        .model-col.active-processing {
            border-color: var(--accent-cyan);
            box-shadow: 0 0 20px rgba(0, 242, 254, 0.25);
        }

        .model-col.completed {
            border-color: rgba(16, 185, 129, 0.4);
        }

        .col-header {
            padding: 0.75rem 1rem;
            background: rgba(255, 255, 255, 0.03);
            border-bottom: 1px solid var(--card-border);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .col-title {
            font-weight: 700;
            font-size: 0.9rem;
            color: #FFF;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .badge-step {
            background: rgba(168, 85, 247, 0.2);
            color: var(--accent-purple);
            border: 1px solid rgba(168, 85, 247, 0.4);
            font-size: 0.7rem;
            padding: 2px 8px;
            border-radius: 12px;
            font-weight: 700;
        }

        .col-status-bar {
            font-size: 0.72rem;
            font-family: 'JetBrains Mono', monospace;
            padding: 0.4rem 1rem;
            background: rgba(0, 0, 0, 0.3);
            border-bottom: 1px solid var(--card-border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            color: var(--text-secondary);
        }

        .col-status-bar.running {
            color: var(--accent-cyan);
            background: rgba(0, 242, 254, 0.08);
        }

        .col-status-bar.done {
            color: var(--accent-green);
        }

        .col-output {
            flex: 1;
            padding: 1rem;
            overflow-y: auto;
            font-size: 0.92rem;
            line-height: 1.6;
            white-space: pre-wrap;
            color: var(--text-primary);
            background: rgba(0, 0, 0, 0.15);
        }

        /* Prompt Input Bar */
        .input-bar {
            display: flex;
            gap: 0.75rem;
            background: rgba(15, 23, 42, 0.95);
            border: 1px solid var(--card-border);
            padding: 0.75rem 1rem;
            border-radius: 12px;
            margin-top: 1rem;
            align-items: center;
        }

        textarea {
            flex: 1;
            background: transparent;
            border: none;
            outline: none;
            color: var(--text-primary);
            font-family: inherit;
            font-size: 0.95rem;
            resize: none;
            height: 48px;
        }

        button.btn-action {
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
            border: none;
            color: #000;
            font-weight: 700;
            padding: 0.8rem 1.6rem;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 0.9rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        button.btn-action:hover:not(:disabled) {
            transform: scale(1.02);
            box-shadow: 0 0 15px rgba(0, 242, 254, 0.4);
        }

        button.btn-action:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        /* Benchmark UI Component Styles */
        .bench-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .stat-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 1.2rem;
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .stat-val {
            font-size: 1.8rem;
            font-weight: 800;
            color: var(--accent-cyan);
            font-family: 'JetBrains Mono', monospace;
        }

        .stat-lbl {
            font-size: 0.8rem;
            color: var(--text-secondary);
            font-weight: 600;
            text-transform: uppercase;
        }

        .matrix-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.88rem;
            background: var(--card-bg);
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--card-border);
        }

        .matrix-table th, .matrix-table td {
            padding: 0.85rem 1.2rem;
            text-align: left;
            border-bottom: 1px solid var(--card-border);
        }

        .matrix-table th {
            background: rgba(255, 255, 255, 0.04);
            color: var(--accent-cyan);
            font-weight: 700;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
        }

        .spinner {
            display: inline-block;
            width: 14px;
            height: 14px;
            border: 2px solid rgba(0, 242, 254, 0.3);
            border-radius: 50%;
            border-top-color: var(--accent-cyan);
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <header>
        <div class="header-title">
            <div class="status-dot"></div>
            <h1>AiraPix Arena & Global Benchmark Engine</h1>
            <div class="ram-saver-tag">
                <span>⚡ Sequential RAM Saver Active</span>
            </div>
        </div>

        <div class="mode-tabs">
            <button class="tab-btn active" id="tab-arena" onclick="setMode('arena')">Sequential Arena</button>
            <button class="tab-btn" id="tab-single" onclick="setMode('single')">Single Chat</button>
            <button class="tab-btn" id="tab-benchmark" onclick="setMode('benchmark')">Global AI Benchmarks</button>
        </div>
    </header>

    <div class="main-container">
        <!-- Sidebar Selector -->
        <div class="sidebar">
            <div class="sidebar-header-row">
                <div class="sidebar-title">Checkpoints (.pt)</div>
                <span class="count-badge" id="lbl-count">0 Selected</span>
            </div>

            <input type="text" class="search-input" id="ckpt-search" placeholder="Search checkpoints..." oninput="renderCheckpointList()">

            <div class="control-btns">
                <button class="btn-mini" onclick="selectAllCheckpoints()">Select All</button>
                <button class="btn-mini" onclick="deselectAllCheckpoints()">Deselect All</button>
            </div>

            <div class="checkpoint-list" id="ckpt-container">
                <div style="font-size:0.8rem; color:var(--text-secondary);">Scanning checkpoints...</div>
            </div>

            <div class="sidebar-title" style="margin-top: 0.5rem;">Generation Parameters</div>
            <div class="setting-group">
                <div class="setting-label">
                    <span>Temperature</span>
                    <span id="lbl-temp">0.7</span>
                </div>
                <input type="range" id="inp-temp" min="0.1" max="1.5" step="0.05" value="0.7" oninput="document.getElementById('lbl-temp').innerText=this.value">
            </div>

            <div class="setting-group">
                <div class="setting-label">
                    <span>Max New Tokens</span>
                    <span id="lbl-tokens">128</span>
                </div>
                <input type="range" id="inp-tokens" min="16" max="512" step="16" value="128" oninput="document.getElementById('lbl-tokens').innerText=this.value">
            </div>

            <div class="setting-group">
                <div class="setting-label">
                    <span>Top-K Threshold</span>
                    <span id="lbl-topk">50</span>
                </div>
                <input type="range" id="inp-topk" min="1" max="100" step="1" value="50" oninput="document.getElementById('lbl-topk').innerText=this.value">
            </div>
        </div>

        <!-- Main Content Area -->
        <div class="content-area">
            <!-- Arena View -->
            <div id="arena-view" style="flex:1; display:flex; flex-direction:column; overflow:hidden;">
                <div class="arena-grid" id="arena-grid"></div>
                <div class="input-bar">
                    <textarea id="inp-prompt" placeholder="Type a prompt to test and compare selected checkpoints side-by-side..." onkeydown="handleKey(event)"></textarea>
                    <button class="btn-action" id="btn-run" onclick="runSequentialComparison()">
                        <span id="btn-text">Compare Models 1-by-1</span>
                    </button>
                </div>
            </div>

            <!-- Global Benchmark Evaluator View -->
            <div id="benchmark-view" style="display:none; flex:1; flex-direction:column; gap:1.2rem; overflow-y:auto; padding-right:6px;">
                <div style="background:var(--card-bg); border:1px solid var(--card-border); padding:1.25rem; border-radius:12px; display:flex; align-items:center; justify-content:space-between;">
                    <div>
                        <h2 style="font-size:1.1rem; color:#FFF; margin-bottom:0.25rem;">Global AI Benchmark Suite (MMLU, GSM8K, HumanEval, ARC, TruthfulQA)</h2>
                        <p style="font-size:0.82rem; color:var(--text-secondary);">Evaluate checkpoint performance against standard global AI evaluation benchmarks and baseline models.</p>
                    </div>
                    <div style="display:flex; gap:0.75rem; align-items:center;">
                        <select id="sel-bench-ckpt" style="background:rgba(0,0,0,0.4); border:1px solid var(--card-border); color:#FFF; padding:0.6rem 0.8rem; border-radius:8px; font-size:0.85rem; outline:none;"></select>
                        <button class="btn-action" id="btn-run-bench" onclick="runGlobalBenchmarkSuite()">Run Benchmark Suite</button>
                    </div>
                </div>

                <div id="bench-results-container">
                    <div style="text-align:center; padding:3rem; color:var(--text-secondary); font-size:0.9rem;">
                        Select a checkpoint above and click <strong>"Run Benchmark Suite"</strong> to evaluate global benchmark scores.
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let availableCheckpoints = [];
        let selectedCheckpoints = [];
        let modelStates = {};
        let isRunning = false;
        let currentMode = 'arena';

        function normPath(p) {
            return p ? p.toString().replace(/\\\\/g, '/').toLowerCase() : '';
        }

        async function loadCheckpoints() {
            try {
                const res = await fetch('/api/checkpoints');
                const rawData = await res.json();
                availableCheckpoints = rawData.map(c => ({
                    ...c,
                    path: normPath(c.path)
                }));
                
                if (selectedCheckpoints.length === 0 && availableCheckpoints.length > 0) {
                    selectedCheckpoints = availableCheckpoints.slice(0, 2).map(c => c.path);
                } else {
                    selectedCheckpoints = selectedCheckpoints.map(p => normPath(p));
                }

                populateBenchmarkDropdown();
                renderCheckpointList();
            } catch (err) {
                console.error("Error fetching checkpoints:", err);
            }
        }

        function populateBenchmarkDropdown() {
            const sel = document.getElementById('sel-bench-ckpt');
            sel.innerHTML = availableCheckpoints.map(c => `<option value="${c.path}">${c.name} (${c.preset}, Step ${c.step})</option>`).join('');
        }

        function renderCheckpointList() {
            const container = document.getElementById('ckpt-container');
            const searchQuery = document.getElementById('ckpt-search').value.toLowerCase().trim();

            const filtered = availableCheckpoints.filter(c => 
                c.name.toLowerCase().includes(searchQuery) || 
                c.preset.toLowerCase().includes(searchQuery) ||
                c.step.toString().includes(searchQuery)
            );

            document.getElementById('lbl-count').innerText = `${selectedCheckpoints.length} Selected`;

            if (filtered.length === 0) {
                container.innerHTML = '<div style="font-size:0.8rem; color:var(--text-secondary);">No matching checkpoints found.</div>';
                return;
            }

            container.innerHTML = filtered.map(c => {
                const cleanPath = normPath(c.path);
                const isSel = selectedCheckpoints.includes(cleanPath);
                return `
                    <div class="ckpt-card ${isSel ? 'selected' : ''}" onclick="toggleCheckpoint('${cleanPath}')">
                        <div class="ckpt-info">
                            <span class="ckpt-name">${c.name}</span>
                            <span class="ckpt-meta">${c.preset} • Step ${c.step} • ${c.size_mb} MB</span>
                        </div>
                        <input type="checkbox" ${isSel ? 'checked' : ''} onclick="event.stopPropagation(); toggleCheckpoint('${cleanPath}')">
                    </div>
                `;
            }).join('');

            renderArenaColumns();
        }

        function toggleCheckpoint(path) {
            const clean = normPath(path);
            const idx = selectedCheckpoints.indexOf(clean);
            if (idx >= 0) {
                selectedCheckpoints.splice(idx, 1);
            } else {
                selectedCheckpoints.push(clean);
            }
            renderCheckpointList();
        }

        function selectAllCheckpoints() {
            selectedCheckpoints = availableCheckpoints.map(c => normPath(c.path));
            renderCheckpointList();
        }

        function deselectAllCheckpoints() {
            selectedCheckpoints = [];
            renderCheckpointList();
        }

        function renderArenaColumns() {
            const grid = document.getElementById('arena-grid');
            if (selectedCheckpoints.length === 0) {
                grid.innerHTML = '<div style="grid-column: 1/-1; display:flex; align-items:center; justify-content:center; color:var(--text-secondary); font-size:0.95rem;">Select one or more model checkpoints from the sidebar to begin comparison.</div>';
                return;
            }

            grid.innerHTML = selectedCheckpoints.map((path, idx) => {
                const cleanPath = normPath(path);
                const ckptObj = availableCheckpoints.find(c => normPath(c.path) === cleanPath) || { name: cleanPath.split('/').pop(), step: '?', preset: '125M' };
                const state = modelStates[cleanPath] || { status: 'idle', response: 'Ready for prompt comparison...' };

                let statusClass = '';
                let statusText = 'Awaiting generation';

                if (state.status === 'queued') {
                    statusText = '⏳ Queued (Waiting for turn)';
                } else if (state.status === 'loading') {
                    statusClass = 'running';
                    statusText = '<span class="spinner"></span> Loading into RAM & generating...';
                } else if (state.status === 'done') {
                    statusClass = 'done';
                    statusText = `✓ Done | ${state.metrics}`;
                } else if (state.status === 'error') {
                    statusText = `❌ Error: ${state.error}`;
                }

                const isActive = state.status === 'loading';
                const isCompleted = state.status === 'done';

                return `
                    <div class="model-col ${isActive ? 'active-processing' : ''} ${isCompleted ? 'completed' : ''}">
                        <div class="col-header">
                            <div class="col-title">
                                <span>${ckptObj.name}</span>
                            </div>
                            <span class="badge-step">${ckptObj.preset} • Step ${ckptObj.step}</span>
                        </div>
                        <div class="col-status-bar ${statusClass}">
                            <span>${statusText}</span>
                        </div>
                        <div class="col-output">
                            ${state.response}
                        </div>
                    </div>
                `;
            }).join('');
        }

        async function runSequentialComparison() {
            if (isRunning) return;
            const promptEl = document.getElementById('inp-prompt');
            const prompt = promptEl.value.trim();
            if (!prompt) return;

            if (selectedCheckpoints.length === 0) {
                alert("Please select at least one model checkpoint to test!");
                return;
            }

            isRunning = true;
            const btnRun = document.getElementById('btn-run');
            btnRun.disabled = true;

            const temp = parseFloat(document.getElementById('inp-temp').value);
            const max_tokens = parseInt(document.getElementById('inp-tokens').value);
            const top_k = parseInt(document.getElementById('inp-topk').value);

            selectedCheckpoints.forEach(path => {
                const clean = normPath(path);
                modelStates[clean] = { status: 'queued', response: 'Waiting in queue for RAM allocation...' };
            });
            renderArenaColumns();

            for (let i = 0; i < selectedCheckpoints.length; i++) {
                const path = normPath(selectedCheckpoints[i]);
                
                modelStates[path] = { status: 'loading', response: 'Loading model into RAM/VRAM & generating response...' };
                renderArenaColumns();

                try {
                    const res = await fetch('/api/generate_single', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            checkpoint: path,
                            prompt: prompt,
                            temp: temp,
                            max_tokens: max_tokens,
                            top_k: top_k,
                            auto_unload: true
                        })
                    });

                    const data = await res.json();
                    if (data.error) {
                        modelStates[path] = { status: 'error', error: data.error, response: `Error: ${data.error}` };
                    } else {
                        const metricsStr = `${data.elapsed_sec}s | ${data.tok_per_sec} tok/s | ${data.tokens_generated} tokens (Unloaded)`;
                        modelStates[path] = {
                            status: 'done',
                            response: data.response,
                            metrics: metricsStr
                        };
                    }
                } catch (err) {
                    modelStates[path] = { status: 'error', error: err.message, response: `Network Error: ${err.message}` };
                }

                renderArenaColumns();
            }

            isRunning = false;
            btnRun.disabled = false;
        }

        async function runGlobalBenchmarkSuite() {
            const selCkpt = document.getElementById('sel-bench-ckpt').value;
            if (!selCkpt) return;

            const container = document.getElementById('bench-results-container');
            const btn = document.getElementById('btn-run-bench');
            btn.disabled = true;

            container.innerHTML = `
                <div style="background:var(--card-bg); border:1px solid var(--card-border); padding:2.5rem; border-radius:12px; text-align:center;">
                    <div class="spinner" style="width:28px; height:28px; border-width:3px; margin-bottom:1rem;"></div>
                    <h3 style="color:#FFF; font-size:1.1rem;">Running Global AI Benchmark Suite...</h3>
                    <p style="color:var(--text-secondary); font-size:0.85rem; margin-top:0.4rem;">Evaluating MMLU, GSM8K Math, HumanEval Code Execution, ARC Science, and TruthfulQA...</p>
                </div>
            `;

            try {
                const res = await fetch('/api/run_benchmark', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ checkpoint: selCkpt })
                });

                const data = await res.json();
                if (data.error) {
                    container.innerHTML = `<div style="color:#EF4444; background:var(--card-bg); border:1px solid rgba(239,68,68,0.3); padding:1.5rem; border-radius:12px;">Benchmark Error: ${data.error}</div>`;
                } else {
                    renderBenchmarkSummary(data.summary);
                }
            } catch (err) {
                container.innerHTML = `<div style="color:#EF4444; background:var(--card-bg); border:1px solid rgba(239,68,68,0.3); padding:1.5rem; border-radius:12px;">Network Error: ${err.message}</div>`;
            }

            btn.disabled = false;
        }

        function renderBenchmarkSummary(s) {
            const container = document.getElementById('bench-results-container');
            const cats = s.categories || [];
            const leader = s.global_leaderboard_comparison || {};

            container.innerHTML = `
                <div class="bench-grid">
                    <div class="stat-card">
                        <span class="stat-lbl">Overall Benchmark Score</span>
                        <span class="stat-val">${s.overall_score_percent}%</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-lbl">Throughput Speed</span>
                        <span class="stat-val" style="color:var(--accent-green);">${s.avg_tokens_per_sec} tok/s</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-lbl">Checkpoint Target</span>
                        <span class="stat-val" style="font-size:1.1rem; color:#FFF; overflow:hidden; text-overflow:ellipsis;">${s.model_name}</span>
                    </div>
                </div>

                <div style="background:var(--card-bg); border:1px solid var(--card-border); border-radius:12px; padding:1.25rem; margin-bottom:1rem;">
                    <h3 style="font-size:0.95rem; color:var(--accent-cyan); text-transform:uppercase; letter-spacing:0.05em; margin-bottom:1rem;">Category Score Breakdown</h3>
                    <div style="display:flex; flex-direction:column; gap:0.75rem;">
                        ${cats.map(c => `
                            <div>
                                <div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:0.3rem;">
                                    <span style="font-weight:600; color:#FFF;">${c.category}</span>
                                    <span style="font-family:'JetBrains Mono'; font-weight:700; color:var(--accent-cyan);">${c.accuracy_percent}% (${c.passed_tests}/${c.num_tests} passed)</span>
                                </div>
                                <div style="background:rgba(255,255,255,0.06); height:8px; border-radius:4px; overflow:hidden;">
                                    <div style="background:linear-gradient(90deg, var(--accent-cyan), var(--accent-blue)); width:${c.accuracy_percent}%; height:100%;"></div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>

                <div style="background:var(--card-bg); border:1px solid var(--card-border); border-radius:12px; padding:1.25rem;">
                    <h3 style="font-size:0.95rem; color:var(--accent-cyan); text-transform:uppercase; letter-spacing:0.05em; margin-bottom:1rem;">Global AI Model Leaderboard Matrix</h3>
                    <table class="matrix-table">
                        <thead>
                            <tr>
                                <th>AI Model</th>
                                <th>Global Benchmark Score</th>
                                <th>Architecture Type</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${Object.entries(leader).map(([model, score]) => {
                                const isTarget = model.includes("Tested Aira");
                                return `
                                    <tr style="${isTarget ? 'background:rgba(0, 242, 254, 0.1); font-weight:700;' : ''}">
                                        <td style="color:${isTarget ? 'var(--accent-cyan)' : '#FFF'};">${model}</td>
                                        <td style="font-family:'JetBrains Mono'; color:${isTarget ? 'var(--accent-cyan)' : 'var(--text-secondary)'};">${score}</td>
                                        <td style="color:var(--text-secondary); font-size:0.8rem;">${isTarget ? 'Aira Dual-System (SSM + MLA)' : 'Standard Transformer'}</td>
                                    </tr>
                                `;
                            }).join('')}
                        </tbody>
                    </table>
                </div>
            `;
        }

        function setMode(mode) {
            currentMode = mode;
            document.getElementById('tab-arena').classList.toggle('active', mode === 'arena');
            document.getElementById('tab-single').classList.toggle('active', mode === 'single');
            document.getElementById('tab-benchmark').classList.toggle('active', mode === 'benchmark');

            document.getElementById('arena-view').style.display = (mode === 'arena' || mode === 'single') ? 'flex' : 'none';
            document.getElementById('benchmark-view').style.display = (mode === 'benchmark') ? 'flex' : 'none';
        }

        function handleKey(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                runSequentialComparison();
            }
        }

        // Init on page load
        loadCheckpoints();
    </script>
</body>
</html>
"""


class ArenaRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(CHAT_HTML.encode("utf-8"))
        elif self.path == "/api/checkpoints":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            checkpoints = discover_checkpoints()
            self.wfile.write(json.dumps(checkpoints).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/run_benchmark":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode("utf-8"))
                ckpt_path = data.get("checkpoint", "")
                if not ckpt_path:
                    raise ValueError("No checkpoint specified.")

                unload_active_model()
                from scripts.run_global_ai_benchmarks import run_benchmark_for_checkpoint
                summary = run_benchmark_for_checkpoint(ckpt_path)
                unload_active_model()

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"summary": summary}).encode("utf-8"))
            except Exception as e:
                unload_active_model()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

        elif self.path == "/api/generate_single":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode("utf-8"))
                ckpt_path = data.get("checkpoint", "")
                prompt = data.get("prompt", "")
                temp = float(data.get("temp", 0.7))
                top_k = int(data.get("top_k", 50))
                max_tokens = int(data.get("max_tokens", 128))
                auto_unload = bool(data.get("auto_unload", True))

                if not ckpt_path:
                    raise ValueError("No checkpoint specified.")

                start_t = time.time()

                model = load_single_model_on_demand(ckpt_path)

                encoded = GLOBAL_TOKENIZER.encode(prompt)
                input_ids = torch.tensor([encoded], dtype=torch.long, device=GLOBAL_DEVICE)

                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=max_tokens,
                        temperature=temp,
                        top_k=top_k,
                        eos_token_id=GLOBAL_TOKENIZER.eos_token_id,
                    )

                elapsed = time.time() - start_t
                gen_tokens = output_ids.size(1) - len(encoded)
                tok_s = gen_tokens / max(elapsed, 1e-4)
                response_text = GLOBAL_TOKENIZER.decode(output_ids[0].tolist())

                freed_mb = 0.0
                if auto_unload:
                    freed_mb = unload_active_model()

                result = {
                    "path": ckpt_path,
                    "name": Path(ckpt_path).name,
                    "response": response_text,
                    "elapsed_sec": round(elapsed, 2),
                    "tok_per_sec": round(tok_s, 1),
                    "tokens_generated": gen_tokens,
                    "vram_freed_mb": freed_mb,
                    "status": "completed",
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))
            except Exception as e:
                unload_active_model()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

        elif self.path in ["/api/chat", "/api/compare"]:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode("utf-8"))
                prompt = data.get("prompt", "")
                temp = float(data.get("temp", 0.7))
                top_k = int(data.get("top_k", 50))
                max_tokens = int(data.get("max_tokens", 128))
                checkpoints = data.get("checkpoints", [])

                if not checkpoints:
                    checkpoints = ["runs/checkpoints/aira_125m_step_1000.pt"]

                results = []
                encoded = GLOBAL_TOKENIZER.encode(prompt)
                input_ids = torch.tensor([encoded], dtype=torch.long, device=GLOBAL_DEVICE)

                for ckpt_path in checkpoints:
                    start_t = time.time()
                    try:
                        model = load_single_model_on_demand(ckpt_path)
                        with torch.no_grad():
                            output_ids = model.generate(
                                input_ids,
                                max_new_tokens=max_tokens,
                                temperature=temp,
                                top_k=top_k,
                                eos_token_id=GLOBAL_TOKENIZER.eos_token_id,
                            )
                        elapsed = time.time() - start_t
                        gen_tokens = output_ids.size(1) - len(encoded)
                        tok_s = gen_tokens / max(elapsed, 1e-4)
                        response_text = GLOBAL_TOKENIZER.decode(output_ids[0].tolist())

                        freed_mb = unload_active_model()

                        results.append({
                            "path": ckpt_path,
                            "name": Path(ckpt_path).name,
                            "response": response_text,
                            "elapsed_sec": round(elapsed, 2),
                            "tok_per_sec": round(tok_s, 1),
                            "tokens_generated": gen_tokens,
                            "vram_freed_mb": freed_mb,
                        })
                    except Exception as err:
                        unload_active_model()
                        results.append({
                            "path": ckpt_path,
                            "name": Path(ckpt_path).name,
                            "error": str(err),
                        })

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"results": results}).encode("utf-8"))
            except Exception as e:
                unload_active_model()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def main():
    global GLOBAL_TOKENIZER, GLOBAL_DEVICE

    parser = argparse.ArgumentParser(description="AiraPix Sequential Multi-Model Comparison Server")
    parser.add_argument("--tokenizer", default="airapix/model/tokenizer/tokenizer.json", help="Path to tokenizer file")
    parser.add_argument("--port", type=int, default=5000, help="Server port (default: 5000)")
    args = parser.parse_args()

    print(f"\n========================================================================")
    print(f"  AIRA AI SEQUENTIAL MULTI-MODEL COMPARISON SERVER (RAM SAVER ACTIVE)")
    print(f"========================================================================")
    print(f"Loading tokenizer: {args.tokenizer}")
    GLOBAL_TOKENIZER = TokenizerWrapper(args.tokenizer)
    print(f"Execution Device: {GLOBAL_DEVICE}")

    discovered = discover_checkpoints()
    print(f"[Checkpoints Discovered] Found {len(discovered)} .pt checkpoints:")
    for c in discovered[:5]:
        print(f"  - {c['name']} ({c['preset']}, Step {c['step']}, {c['size_mb']} MB)")
    if len(discovered) > 5:
        print(f"  ... and {len(discovered) - 5} more")

    server = HTTPServer(("0.0.0.0", args.port), ArenaRequestHandler)
    print(f"\n[SUCCESS] Comparison Arena Web UI running at: http://localhost:{args.port}")
    print(f"Open http://localhost:{args.port} in your browser to select/deselect models and compare outputs 1-by-1!\n")
    sys.stdout.flush()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping arena server...")
        server.server_close()


if __name__ == "__main__":
    main()
