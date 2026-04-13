#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║     Gaussian Phase Field Crack Simulation — CLI              ║
║     MPM + Gaussian Splatting                                 ║
╚══════════════════════════════════════════════════════════════╝

Usage:
    python cli.py run [OPTIONS]
    python cli.py batch [OPTIONS]
    python cli.py info
    python cli.py validate --config CONFIG
"""

import argparse
import os
import sys
import time
import json
import yaml
import subprocess
from pathlib import Path
from datetime import datetime


# ── Helpers ──────────────────────────────────────────────────

def detect_device():
    """Auto-detect GPU or fall back to CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            return "cuda", f"{name} ({mem:.1f} GB VRAM)"
        else:
            return "cpu", "No CUDA GPU detected"
    except ImportError:
        return "cpu", "PyTorch not found"


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║       Gaussian Phase Field Crack Simulation                  ║
║       MPM + Gaussian Splatting                               ║
╚══════════════════════════════════════════════════════════════╝""")


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def load_config(config_path):
    """Load and return a YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_config(config, path):
    """Save a config dict to YAML."""
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def override_config(config, overrides: dict):
    """Apply CLI overrides to a config dict."""
    for key, value in overrides.items():
        if value is None:
            continue
        keys = key.split(".")
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return config


# ── Commands ─────────────────────────────────────────────────

def cmd_info(args):
    """Print system and environment info."""
    print_banner()
    print_section("System Info")

    device, device_info = detect_device()
    print(f"  Device:       {device.upper()} — {device_info}")

    try:
        import torch
        print(f"  PyTorch:      {torch.__version__}")
        print(f"  CUDA avail:   {torch.cuda.is_available()}")
    except ImportError:
        print("  PyTorch:      NOT INSTALLED")

    try:
        import numpy as np
        print(f"  NumPy:        {np.__version__}")
    except ImportError:
        print("  NumPy:        NOT INSTALLED")

    try:
        import cv2
        print(f"  OpenCV:       {cv2.__version__}")
    except ImportError:
        print("  OpenCV:       NOT INSTALLED")

    try:
        import open3d as o3d
        print(f"  Open3D:       {o3d.__version__}")
    except ImportError:
        print("  Open3D:       NOT INSTALLED")

    try:
        from diff_gaussian_rasterization import GaussianRasterizer
        print(f"  DiffGauss:    OK")
    except ImportError:
        print(f"  DiffGauss:    NOT INSTALLED")

    try:
        import simple_knn
        print(f"  SimpleKNN:    OK")
    except ImportError:
        print(f"  SimpleKNN:    NOT INSTALLED")

    try:
        import faiss
        print(f"  FAISS:        OK")
    except ImportError:
        print(f"  FAISS:        NOT INSTALLED")

    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    if result.returncode == 0:
        ver = result.stdout.split("\n")[0]
        print(f"  FFmpeg:       {ver}")
    else:
        print(f"  FFmpeg:       NOT FOUND")

    print_section("Project")
    print(f"  Config dir:   configs/")
    configs = list(Path("configs").glob("*.yaml")) if Path("configs").exists() else []
    for c in configs:
        print(f"    - {c.name}")
    print(f"  Assets dir:   assets/")
    meshes = list(Path("assets/meshes").glob("*.obj")) if Path("assets/meshes").exists() else []
    for m in meshes:
        print(f"    - {m.name}")
    print()


def cmd_validate(args):
    """Validate a config file."""
    print_banner()
    print_section(f"Validating: {args.config}")

    if not Path(args.config).exists():
        print(f"  ✗ Config file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)

    required_keys = ["simulation", "mesh", "particles", "frames"]
    all_ok = True
    for key in required_keys:
        if key in config:
            print(f"  ✓ {key}: {config[key]}")
        else:
            print(f"  ✗ Missing required key: {key}")
            all_ok = False

    mesh_path = config.get("mesh", "")
    if mesh_path and Path(mesh_path).exists():
        print(f"  ✓ Mesh file exists: {mesh_path}")
    elif mesh_path:
        print(f"  ✗ Mesh file not found: {mesh_path}")
        all_ok = False

    if all_ok:
        print(f"\n  ✓ Config is valid!\n")
    else:
        print(f"\n  ✗ Config has issues. Please fix before running.\n")
        sys.exit(1)


def cmd_run(args):
    """Run a single simulation."""
    print_banner()

    # Auto-detect device
    device = args.device
    if device == "auto":
        device, device_info = detect_device()
        print(f"\n  [Auto-detect] Using: {device.upper()} — {device_info}")
    else:
        print(f"\n  [Device] Using: {device.upper()}")

    # Load and override config
    config = load_config(args.config)
    overrides = {
        "particles": args.particles,
        "frames": args.frames,
        "mesh": args.mesh,
        "output_dir": args.output,
        "device": device,
        "render_every": args.render_every,
    }
    config = override_config(config, overrides)

    # Save modified config to temp file
    tmp_config = f"/tmp/crack_run_{int(time.time())}.yaml"
    save_config(config, tmp_config)

    print_section("Simulation Parameters")
    print(f"  Config:       {args.config}")
    print(f"  Mesh:         {config.get('mesh', 'default')}")
    print(f"  Particles:    {config.get('particles', 'default')}")
    print(f"  Frames:       {config.get('frames', 'default')}")
    print(f"  Device:       {device.upper()}")
    print(f"  Output:       {config.get('output_dir', 'output/')}")
    print(f"  Render every: {config.get('render_every', 1)} frame(s)")

    # Create output dir
    out_dir = Path(config.get("output_dir", "output"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run
    print_section("Starting Simulation")
    start = time.time()
    cmd = [sys.executable, "run.py", "--config", tmp_config]
    if args.no_video:
        cmd.append("--no-video")
    if args.verbose:
        cmd.append("--verbose")

    result = subprocess.run(cmd)

    elapsed = time.time() - start
    print_section("Done")
    if result.returncode == 0:
        print(f"  ✓ Simulation completed in {elapsed/60:.1f} minutes")
        print(f"  ✓ Output saved to: {out_dir}/")
    else:
        print(f"  ✗ Simulation failed after {elapsed/60:.1f} minutes")
        sys.exit(result.returncode)

    # Cleanup temp config
    Path(tmp_config).unlink(missing_ok=True)


def cmd_batch(args):
    """Run multiple simulations from a batch config JSON."""
    print_banner()
    print_section(f"Batch Run: {args.batch_config}")

    if not Path(args.batch_config).exists():
        print(f"  ✗ Batch config not found: {args.batch_config}")
        sys.exit(1)

    with open(args.batch_config) as f:
        batch = json.load(f)

    runs = batch.get("runs", [])
    print(f"  Total runs: {len(runs)}")

    results = []
    for i, run in enumerate(runs, 1):
        print_section(f"Run {i}/{len(runs)}: {run.get('name', f'run_{i}')}")

        # Build args for cmd_run
        run_args = argparse.Namespace(
            config=run.get("config", args.base_config),
            device=run.get("device", "auto"),
            particles=run.get("particles", None),
            frames=run.get("frames", None),
            mesh=run.get("mesh", None),
            output=run.get("output", f"output/batch_{run.get('name', i)}"),
            render_every=run.get("render_every", None),
            no_video=run.get("no_video", False),
            verbose=args.verbose,
        )

        start = time.time()
        try:
            cmd_run(run_args)
            elapsed = time.time() - start
            results.append({"name": run.get("name", i), "status": "success", "time_min": elapsed/60})
        except SystemExit:
            elapsed = time.time() - start
            results.append({"name": run.get("name", i), "status": "failed", "time_min": elapsed/60})
            if not args.continue_on_error:
                print("\n  ✗ Stopping batch due to failure. Use --continue-on-error to skip failures.")
                break

    # Summary
    print_section("Batch Summary")
    for r in results:
        icon = "✓" if r["status"] == "success" else "✗"
        print(f"  {icon} {r['name']}: {r['status']} ({r['time_min']:.1f} min)")

    failed = sum(1 for r in results if r["status"] == "failed")
    print(f"\n  Completed: {len(results) - failed}/{len(runs)} runs succeeded\n")

    if failed:
        sys.exit(1)


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Gaussian Phase Field Crack Simulation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show system info
  python cli.py info

  # Run with default config (auto GPU/CPU)
  python cli.py run

  # Run with custom params
  python cli.py run --mesh assets/meshes/bunny.obj --particles 50000 --frames 100

  # Force CPU mode
  python cli.py run --device cpu

  # Batch run
  python cli.py batch --batch-config configs/batch_runs.json

  # Docker GPU
  docker compose --profile gpu run crack-gpu run --frames 50

  # Docker CPU
  docker compose --profile cpu run crack-cpu run --device cpu --frames 50
        """
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ── info ──────────────────────────────────────────────────
    sub.add_parser("info", help="Show system and environment info")

    # ── validate ──────────────────────────────────────────────
    val = sub.add_parser("validate", help="Validate a config file")
    val.add_argument("--config", default="configs/simulation_config.yaml",
                     help="Path to config YAML")

    # ── run ───────────────────────────────────────────────────
    run = sub.add_parser("run", help="Run a single simulation")
    run.add_argument("--config", default="configs/simulation_config.yaml",
                     help="Path to config YAML (default: configs/simulation_config.yaml)")
    run.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                     help="Device to use (default: auto-detect)")
    run.add_argument("--mesh", default=None,
                     help="Path to mesh .obj file (overrides config)")
    run.add_argument("--particles", type=int, default=None,
                     help="Number of particles (overrides config)")
    run.add_argument("--frames", type=int, default=None,
                     help="Number of frames to simulate (overrides config)")
    run.add_argument("--output", default=None,
                     help="Output directory (overrides config)")
    run.add_argument("--render-every", type=int, default=None,
                     help="Render every N frames (default: 1)")
    run.add_argument("--no-video", action="store_true",
                     help="Skip video generation")
    run.add_argument("--verbose", action="store_true",
                     help="Verbose output")

    # ── batch ─────────────────────────────────────────────────
    batch = sub.add_parser("batch", help="Run multiple simulations from a batch config")
    batch.add_argument("--batch-config", required=True,
                       help="Path to batch JSON config")
    batch.add_argument("--base-config", default="configs/simulation_config.yaml",
                       help="Base config to use if not specified per run")
    batch.add_argument("--continue-on-error", action="store_true",
                       help="Continue batch even if a run fails")
    batch.add_argument("--verbose", action="store_true",
                       help="Verbose output")

    args = parser.parse_args()

    dispatch = {
        "info": cmd_info,
        "validate": cmd_validate,
        "run": cmd_run,
        "batch": cmd_batch,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
