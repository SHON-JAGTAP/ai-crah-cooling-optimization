"""
main.py — Entry point for the AI-Based CRAH Cooling Optimization System.

Usage
-----
  # Run the live web dashboard (recommended)
  python main.py

  # Run headless simulation (no browser required)
  python main.py --headless --steps 50

  # Set your Anthropic API key via env or .env file
  ANTHROPIC_API_KEY=sk-ant-... python main.py
"""

import argparse
import os
import sys

# ── Load .env if present ──────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional


def run_dashboard() -> None:
    """Launch the Flask dashboard (imports dashboard/app.py)."""
    # Add the project root to sys.path so dashboard/app.py can find modules/
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard"))

    from dashboard.app import app, get_loop
    from config import DASHBOARD_HOST, DASHBOARD_PORT

    sep = "=" * 60
    print(f"\n{sep}")
    print("  AI-Based Group CRAH Control & Cooling Optimization")
    print(f"  Dashboard -> http://localhost:{DASHBOARD_PORT}")
    print(f"  Claude AI -> {'ENABLED' if os.getenv('ANTHROPIC_API_KEY') else 'FALLBACK (set ANTHROPIC_API_KEY)'}")
    print(f"{sep}\n")

    get_loop()   # pre-warm (train + start background thread)
    app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=False, use_reloader=False)


def run_headless(steps: int = 50) -> None:
    """Run a headless simulation and print results to console."""
    from modules.feedback_loop import FeedbackLoop

    loop = FeedbackLoop(api_key=os.getenv("ANTHROPIC_API_KEY"))
    loop.train()

    sep = "-" * 60
    print(f"\n{sep}")
    print(f" Running {steps}-step headless simulation ...")
    print(sep)

    for i in range(steps):
        snap = loop.step()
        m    = snap["metrics"]
        print(
            f" Step {i+1:>3} | "
            f"Avg: {m['avg_rack_temp']:.1f}°C  "
            f"Max: {m['max_rack_temp']:.1f}°C  "
            f"PUE: {m['pue']:.3f}  "
            f"Hotspots: {m['n_hotspots']}  "
            f"| {m['alert_summary']}"
        )

    sep = "-" * 60
    print(f"\n{sep}")
    print(" Simulation complete.")

    # Print final AI recommendation
    print("\n-- Final AI Recommendation --")
    print(snap["ai_recommendation"])
    print(f"{sep}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AI-Based CRAH Cooling Optimization System"
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without web dashboard"
    )
    parser.add_argument(
        "--steps", type=int, default=50,
        help="Number of simulation steps (headless mode only)"
    )
    args = parser.parse_args()

    if args.headless:
        run_headless(steps=args.steps)
    else:
        run_dashboard()
