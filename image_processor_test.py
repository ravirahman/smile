import image_processor
import json

from google.cloud import vision
from google.cloud import storage

import image_merger


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


sample_face_1 = AttrDict({
    "bounding_poly": AttrDict({
        "vertices": [
            AttrDict({
                "x": 293,
                "y": 474
            }),
            AttrDict({
                "x": 823,
                "y": 474
            }),
            AttrDict({
                "x": 823,
                "y": 1090,
            }),
            AttrDict({
                "x": 293,
                "y": 1090,
            }),
        ],
    }),
    "roll_angle": -5.8156538009643555,
    "pan_angle": -21.831066131591797,
    "tilt_angle": 3.9900543689727783,
    "detection_confidence": 0.9998596906661987,
    "landmarking_confidence": 0.6554989814758301,
    "joy_likelihood": 5,
    "sorrow_likelihood": 1,
    "anger_likelihood": 1,
    "surprise_likelihood": 1,
    "under_exposed_likelihood": 1,
    "blurred_likelihood": 1,
    "headwear_likelihood": 1,
})

sample_face_2 = AttrDict({
    "bounding_poly": AttrDict({
        "vertices": [
            AttrDict({
                "x": 406,
                "y": 398
            }),
            AttrDict({
                "x": 901,
                "y": 398
            }),
            AttrDict({
                "x": 901,
                "y": 1042,
            }),
            AttrDict({
                "x": 406,
                "y": 1042,
            }),
        ],
    }),
    "roll_angle": -2.8156538009643555,
    "pan_angle": -2.831066131591797,
    "tilt_angle": 2.9900543689727783,
    "detection_confidence": 0.9998596906661987,
    "joy_likelihood": 4,
    "sorrow_likelihood": 1,
    "anger_likelihood": 1,
    "surprise_likelihood": 1,
    "under_exposed_likelihood": 1,
    "blurred_likelihood": 1,
    "headwear_likelihood": 1,
})


def test_scale():
    assert image_processor.ImagesProcessor.scale(11, 10, 20) == 0.1
    assert image_processor.ImagesProcessor.scale(11, 20, 10) == 0.9


def test_score_face():
    with open("config.json") as configFile:
        config = json.load(configFile)
        scored_face = image_processor.ImagesProcessor.score_face(config, sample_face_1)
        assert scored_face == 0.9764099833435482


def test_get_face_centers():
    face_centers = image_processor.ImagesProcessor.get_face_centers([sample_face_1])
    assert len(face_centers) == 1
    assert face_centers[0][0] == 558
    assert face_centers[0][1] == 782


def test_get_face_vectors():
    face_centers = image_processor.ImagesProcessor.get_face_centers([sample_face_1])
    face_vectors = image_processor.ImagesProcessor.get_face_vectors(face_centers)
    assert len(face_vectors) == 1
    assert face_vectors[0][0] == 0
    assert face_vectors[0][1] == 0

    face_centers_2 = image_processor.ImagesProcessor.get_face_centers([sample_face_1, sample_face_2])
    face_vectors_2 = image_processor.ImagesProcessor.get_face_vectors(face_centers_2)
    assert len(face_vectors_2) == 2
    assert face_vectors_2[0][0] == -0.8374351868224612
    assert face_vectors_2[0][1] == 0.5465366482420273

    assert face_vectors_2[1][0] == 0.8374351868224612
    assert face_vectors_2[1][1] == -0.5465366482420273


def test_process_images():
    img_gs_uris = ["gs://drm-hack-the-north.appspot.com/process/F0F8A625-07FF-4869-AD3B-821666C27777"
                   "/livePhoto_extracted_frame_" + str(i) + ".jpg"
                   for i in range(0, 12)]
    bucket_name = "drm-hack-the-north.appspot.com"
    vision_client = vision.ImageAnnotatorClient()
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    im = image_merger.ImageMerger()
    ip = image_processor.ImagesProcessor(vision_client, bucket, bucket_name, im)

    image_uri = ip.process_images(img_gs_uris)

    image_uri_parts = image_uri.split("/")
    assert len(image_uri_parts) == 5
    assert image_uri_parts[0] == "gs:"
    assert image_uri_parts[1] == ""
    assert image_uri_parts[2] == bucket_name
    assert image_uri_parts[3] == "results"
    assert image_uri_parts[4][-4:] == ".jpg"
