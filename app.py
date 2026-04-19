import logging
import sys

from flask import Flask, jsonify, request, send_from_directory

from core.index_store import IndexStore
from services.crawler_service import CrawlerService
from services.search_service import SearchService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

app = Flask(__name__, static_folder="static")

index_store = IndexStore()
crawler_service = CrawlerService(index_store)
search_service = SearchService(index_store)


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/")
def serve_ui():
    return send_from_directory("static", "index.html")


@app.route("/index", methods=["POST"])
def start_index():
    body = request.get_json(silent=True) or {}
    origin = body.get("origin")
    k = body.get("k")

    if not origin or not isinstance(origin, str):
        return jsonify({"error": "origin is required and must be a URL string"}), 400
    if k is None or not isinstance(k, int) or k < 1:
        return jsonify({"error": "k is required and must be a positive integer"}), 400

    crawler_id = crawler_service.start_crawl(origin, k)
    return jsonify({"crawler_id": crawler_id, "status": "started"}), 201


@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("query", "").strip()
    if not query:
        return jsonify({"error": "query parameter is required"}), 400

    limit = request.args.get("limit", 50, type=int)
    sort_by = request.args.get("sortBy", "relevance")
    results = search_service.search(query, limit=limit, sort_by=sort_by)
    return jsonify({"query": query, "count": len(results), "results": results})


@app.route("/status", methods=["GET"])
def status_all():
    return jsonify(crawler_service.summary())


@app.route("/status/<crawler_id>", methods=["GET"])
def status_one(crawler_id: str):
    info = crawler_service.get_status(crawler_id)
    if info is None:
        return jsonify({"error": "crawler not found"}), 404
    return jsonify(info)


@app.route("/stop/<crawler_id>", methods=["POST"])
def stop_crawler(crawler_id: str):
    if crawler_service.stop_crawl(crawler_id):
        return jsonify({"status": "stopping", "crawler_id": crawler_id})
    return jsonify({"error": "crawler not found"}), 404


@app.route("/pause/<crawler_id>", methods=["POST"])
def pause_crawler(crawler_id: str):
    if crawler_service.pause_crawl(crawler_id):
        return jsonify({"status": "paused", "crawler_id": crawler_id})
    return jsonify({"error": "crawler not found or already stopped"}), 404


@app.route("/resume/<crawler_id>", methods=["POST"])
def resume_crawler(crawler_id: str):
    if crawler_service.resume_crawl(crawler_id):
        return jsonify({"status": "resumed", "crawler_id": crawler_id})
    return jsonify({"error": "crawler not found or already stopped"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3600, debug=False)
