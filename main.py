import logging
import json
from flask import Flask, request, jsonify
from google.cloud import vision
from google.cloud import storage

import image_processor
import image_merger

BUCKET_NAME = "drm-hack-the-north.appspot.com"

app = Flask(__name__)
vision_client = vision.ImageAnnotatorClient()
storage_client = storage.Client()
bucket = storage_client.get_bucket(BUCKET_NAME)
im = image_merger.ImageMerger()
ip = image_processor.ImagesProcessor(vision_client, bucket, BUCKET_NAME, im)


@app.route('/')
def hello():
    """Return a friendly HTTP greeting."""
    return 'Hello World!'


@app.route("/process_images", methods=["POST"])
def process_images():
    data = request.get_json()
    if "image_uris" not in data:
        raise Exception("missing image_uris")
    image_gs_uris = data["image_uris"]
    result_uri = ip.process_images(image_gs_uris)
    return jsonify({
        "image_uri": result_uri,
    })


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
