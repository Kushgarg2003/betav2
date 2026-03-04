#!/usr/bin/env python3
"""
Beta Relevering Web App
────────────────────────
Full-stack Flask app that chains:
  company_finder → stock_price_tracker → debt_equity_analyzer → unlevered beta
Then accepts a balance sheet upload via balance_sheet_extractor to get
the target company's D/E, and computes the Relevered Beta.

Requirements:
    pip install flask google-genai yfinance numpy pandas openpyxl

Usage:
    python app.py
    Open http://localhost:5000 in your browser
"""

import io
import json
import os
import queue
import sys
import tempfile
import threading

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

TAX_RATE     = 0.22
_pipe_lock   = threading.Lock()   # only one pipeline runs at a time


# ── Stdout capture ─────────────────────────────────────────────────────────────

class _QueueWriter:
    """Captures stdout line-by-line and puts each line into a queue."""
    def __init__(self, q: queue.Queue):
        self._q   = q
        self._buf = ""

    def write(self, text: str):
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._q.put(("log", line.rstrip()))

    def flush(self):
        if self._buf.strip():
            self._q.put(("log", self._buf.strip()))
            self._buf = ""


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Streams the full pipeline as Server-Sent Events.
    Sends a final 'result' event with JSON data when done.
    """
    body         = request.get_json(silent=True) or {}
    company_name = body.get("company_name", "").strip()
    if not company_name:
        return jsonify({"error": "Company name is required"}), 400

    def generate():
        q              = queue.Queue()
        result_holder  = {}

        def run():
            with _pipe_lock:
                writer = _QueueWriter(q)
                orig   = sys.stdout
                sys.stdout = writer
                try:
                    from debt_equity_analyzer import run_full_pipeline
                    pipeline_results = run_full_pipeline(company_name)

                    if not pipeline_results:
                        result_holder["error"] = "No results returned from pipeline."
                        return

                    eligible = [r for r in pipeline_results
                                if r.get("beta") is not None and r.get("de_ratio") is not None]
                    skipped  = [r for r in pipeline_results
                                if r.get("beta") is None or r.get("de_ratio") is None]

                    if not eligible:
                        result_holder["error"] = "No companies have both Beta and D/E."
                        return

                    results = []
                    for r in eligible:
                        beta_u = round(r["beta"] / (1 + (1 - TAX_RATE) * r["de_ratio"]), 6)
                        results.append({
                            "name"    : r["name"],
                            "ticker"  : r["ticker"],
                            "beta_l"  : r["beta"],
                            "de_ratio": r["de_ratio"],
                            "beta_u"  : beta_u,
                        })

                    avg_beta_u = round(sum(r["beta_u"] for r in results) / len(results), 6)
                    result_holder.update({
                        "success"   : True,
                        "results"   : results,
                        "skipped"   : [r["name"] for r in skipped],
                        "avg_beta_u": avg_beta_u,
                        "tax_rate"  : TAX_RATE,
                    })

                except Exception as exc:
                    result_holder["error"] = str(exc)
                finally:
                    sys.stdout = orig
                    q.put(("done", None))

        threading.Thread(target=run, daemon=True).start()

        while True:
            kind, payload = q.get()
            if kind == "done":
                break
            yield f"data: {json.dumps({'type': 'log', 'text': payload})}\n\n"

        if "error" in result_holder:
            yield f"data: {json.dumps({'type': 'error', 'message': result_holder['error']})}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'result', **result_holder})}\n\n"

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return Response(stream_with_context(generate()), mimetype="text/event-stream", headers=headers)


@app.route("/api/extract-de", methods=["POST"])
def extract_de():
    """Accepts a balance sheet file and returns the extracted D/E ratio as JSON."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    ext       = os.path.splitext(file.filename)[1].lower()
    supported = {".pdf", ".png", ".jpg", ".jpeg", ".xlsx", ".xls", ".csv"}
    if ext not in supported:
        return jsonify({"error": f"Unsupported file type '{ext}'. Use: {', '.join(sorted(supported))}"}), 400

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        # Suppress prints from extractor
        orig = sys.stdout
        sys.stdout = io.StringIO()
        try:
            from balance_sheet_extractor import extract_from_file
            result = extract_from_file(tmp_path)
        finally:
            sys.stdout = orig

        if "error" in result:
            return jsonify({"error": result["error"]}), 400

        # Compute D/E if not returned directly
        de = result.get("de_ratio")
        if de is None:
            td = result.get("total_debt")
            te = result.get("total_equity")
            if td is not None and te is not None and float(te) != 0:
                de = round(float(td) / float(te), 4)

        return jsonify({
            "success"         : True,
            "company"         : result.get("company"),
            "financial_period": result.get("financial_period"),
            "currency_unit"   : result.get("currency_unit"),
            "total_debt"      : result.get("total_debt"),
            "total_equity"    : result.get("total_equity"),
            "de_ratio"        : de,
            "notes"           : result.get("notes"),
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    print("\n  Beta Relevering Web App")
    print("  → http://localhost:5000\n")
    app.run(debug=False, port=5000, threaded=True)
