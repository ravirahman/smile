import tempfile
import os
import threading
import numpy as np
import uuid
import json

import image_merger


class ImagesProcessor:
    def __init__(self, vision_client, bucket, bucket_name, image_merger):
        self.vision_client = vision_client
        self.image_merger = image_merger
        self.bucket = bucket
        self.bucket_name = bucket_name

    @staticmethod
    def likelihood_to_score(likelihood: int):
        if likelihood == 0:
            return 0.5
        if likelihood == 1:
            return 0.01
        if likelihood == 2:
            return 0.25
        if likelihood == 3:
            return 0.5
        if likelihood == 4:
            return 0.75
        if likelihood == 5:
            return 0.99

    @staticmethod
    def scale(value: float, zero_value=0, one_value=1):
        return (value - zero_value) / (one_value - zero_value)

    @staticmethod
    def score_face(config, face):
        # all features are scaled to [0,1] where 0 is bad and 1 is good
        detection_confidence = face.detection_confidence
        joy_likelihood = ImagesProcessor.likelihood_to_score(face.joy_likelihood)
        sorrow_likelihood = ImagesProcessor.scale(ImagesProcessor.likelihood_to_score(face.sorrow_likelihood), 1, 0)
        anger_likelihood = ImagesProcessor.scale(ImagesProcessor.likelihood_to_score(face.anger_likelihood), 1, 0)
        surprise_likelihood = ImagesProcessor.scale(ImagesProcessor.likelihood_to_score(face.surprise_likelihood), 1, 0)
        under_exposed_likelihood = ImagesProcessor.scale(ImagesProcessor.likelihood_to_score(
            face.under_exposed_likelihood), 1, 0)
        blurred_likelihood = ImagesProcessor.scale(ImagesProcessor.likelihood_to_score(face.blurred_likelihood), 1, 0)
        headwear_likelihood = ImagesProcessor.scale(ImagesProcessor.likelihood_to_score(face.headwear_likelihood), 1, 0)
        roll_angle = ImagesProcessor.scale(abs(face.roll_angle), 180, 0)
        tilt_angle = ImagesProcessor.scale(abs(face.tilt_angle), 180, 0)
        pan_angle = ImagesProcessor.scale(abs(face.pan_angle), 180, 0)

        # weight the different features to score the face.
        weights = config["weights"]
        return weights["detection_confidence"] * detection_confidence + weights["joy_likelihood"] * joy_likelihood + \
            weights["sorrow_likelihood"] * sorrow_likelihood + weights["anger_likelihood"] * anger_likelihood + \
            weights["surprise_likelihood"] * surprise_likelihood + \
            weights["under_exposed_likelihood"] * under_exposed_likelihood + \
            weights["blurred_likelihood"] * blurred_likelihood + weights["headwear_likelihood"] * headwear_likelihood +\
            weights["roll_angle"] * roll_angle + weights["tilt_angle"] * tilt_angle + weights["pan_angle"] * pan_angle

    @staticmethod
    def get_face_centers(face_annotations):
        face_centers = []
        for face_annotation in face_annotations:
            average_x = 0
            average_y = 0
            for vertex in face_annotation.bounding_poly.vertices:
                average_x += vertex.x
                average_y += vertex.y
            average_x //= len(face_annotation.bounding_poly.vertices)
            average_y //= len(face_annotation.bounding_poly.vertices)
            face_centers.append(np.array([average_x, average_y]))
        return np.array(face_centers)

    @staticmethod
    def get_face_vectors(face_centers: np.array):
        if len(face_centers) == 0:
            return np.array([])
        # determine the bounding box for the centers of the face images
        average = np.average(face_centers, axis=0)
        min_x = np.min(face_centers[:, 0])
        max_x = np.max(face_centers[:, 0])
        min_y = np.min(face_centers[:, 1])
        max_y = np.max(face_centers[:, 1])

        # map the face centers to vectors from the (average_x, average_y) above.
        # Scale them such that half the distance of the diagonal is 1
        half_diag = np.linalg.norm([max_x - min_x, max_y - min_y]) / 2
        if half_diag == 0:
            return np.array([[0, 0]])
        return (face_centers - average)/half_diag

    def download_image(self, gs_uri: str, out_filename: str, index, results, exceptions):
        try:
            path_parts = gs_uri.split("/")
            if len(path_parts) < 4:
                raise Exception("incorrect gs_uri - len is less than 3")
            if path_parts[0] != "gs:" and path_parts[1] != "":
                raise Exception("incorrect gs_uri - invalid protocol")
            if path_parts[2] != self.bucket_name:
                raise Exception("invalid bucket name")
            path = "/".join(gs_uri.split("/")[3:])
            blob = self.bucket.blob(path)
            blob.download_to_filename(out_filename)
        except Exception as e:
            exceptions[index] = e
        finally:
            return True

    def download_config(self, index, results, exceptions):
        try:
            blob = self.bucket.blob("config.json")
            config_string = blob.download_as_string()
            config_json = json.loads(config_string)
            results[index] = config_json
        except Exception as e:
            exceptions[index] = e
        finally:
            return True

    def annotate_images(self, requests, index, results, exceptions):
        try:
            results[index] = self.vision_client.batch_annotate_images(requests)
        except Exception as e:
            exceptions[index] = e
        finally:
            return True

    def process_images(self, image_gs_uris):
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_filenames = []
            requests = []
            threads = []
            results = [None for _ in range(len(image_gs_uris) + 2)]
            exceptions = [None for _ in range(len(image_gs_uris) + 2)]
            for i, uri in enumerate(image_gs_uris):
                filename = os.path.join(tmp_dir, os.path.basename(uri))
                image_filenames.append(filename)
                process = threading.Thread(target=self.download_image, args=(uri, filename, i, results, exceptions))
                process.start()
                threads.append(process)
                requests.append({
                    "image": {
                        "source": {
                            "gcs_image_uri": uri
                        },
                    },
                    "features": [
                        {"type": "FACE_DETECTION"},
                    ],
                })
            config_process = threading.Thread(target=self.download_config, args=(-2, results, exceptions))
            config_process.start()
            threads.append(config_process)

            vision_process = threading.Thread(target=self.annotate_images, args=(requests, -1, results, exceptions))
            vision_process.start()
            threads.append(vision_process)
            for process in threads:
                process.join()
            for e in exceptions:
                if e is not None:
                    raise e
            config = results[-2]
            image_responses = results[-1].responses
            new_filename = str(uuid.uuid4()) + ".jpg"
            new_filepath = os.path.join(tmp_dir, new_filename)
            self.process_faces(config, image_filenames, new_filepath, image_responses)
            destination_blob_name = os.path.join("results", new_filename)
            blob = self.bucket.blob(destination_blob_name)
            blob.upload_from_filename(new_filepath)
        return os.path.join("gs://" + self.bucket_name, destination_blob_name)

    def process_faces(self, config, image_filenames, out_filename, image_responses):
        # find the best base image
        best_image_index = None
        best_score = 0
        scores = []
        for i, image_response in enumerate(image_responses):
            scores.append([])
            for face_annotation in image_response.face_annotations:
                score = ImagesProcessor.score_face(config, face_annotation)
                scores[-1].append(score)
            if len(scores[-1]) == 0:
                continue
            average_score = np.average(scores[-1])
            if average_score > best_score:
                best_score = average_score
                best_image_index = i
        if best_image_index is None:
            raise Exception("no images provided")
        # determine the centers of the faces
        best_image = image_responses[best_image_index]
        face_centers = self.get_face_centers(best_image.face_annotations)
        face_vectors = self.get_face_vectors(face_centers)

        if len(face_vectors) == 0:
            raise Exception("no faces detected")

        # for each face in base image, create a list of all other faces in the other supplied images
        # keys corresponds to a face in the same order as the base image;
        # value is an array of alternative faceAnnotations (one per image)
        alternative_faces = {}
        for alt_image_i, alt_image_response in enumerate(image_responses):
            if alt_image_i == best_image_index:
                continue
            alt_face_centers = self.get_face_centers(alt_image_response.face_annotations)
            alt_face_vectors = self.get_face_vectors(alt_face_centers)
            if len(alt_face_vectors) == 0:
                continue  # skip images without faces
            if len(alt_face_vectors) != len(face_vectors):
                continue  # different # of faces; skip
            for alt_face_i, alt_face_vector in enumerate(alt_face_vectors):
                best_score = 0
                best_face_i = 0
                for i, face_vector in enumerate(face_vectors):
                    # a score of 1 represents a perfect angle match
                    # a score of 0.5 is orthogonal, a score of 0 is opposite
                    score = 0
                    face_norm = np.linalg.norm(face_vector)
                    alt_face_norm = np.linalg.norm(alt_face_vector)
                    if face_norm > 0 and alt_face_norm > 0:
                        angle_score = ImagesProcessor.scale(
                            (face_vector[0] * alt_face_vector[0] + face_vector[1] * alt_face_vector[1]) /
                            (face_norm * alt_face_norm))
                        # a score of 1 represents the same distance; a score of 0 represents
                        distance_score = ImagesProcessor.scale(np.linalg.norm(face_vector - alt_face_vector), 2, 0)
                        score = angle_score * distance_score
                    if score > best_score:
                        best_score = score
                        best_face_i = i
                if best_face_i not in alternative_faces:
                    alternative_faces[best_face_i] = []
                alternative_faces[best_face_i].append((alt_image_i, alt_image_response.face_annotations[alt_face_i]))
        face_replacements = []
        for face_i, alternatives in alternative_faces.items():
            initial_score = ImagesProcessor.score_face(config, best_image.face_annotations[face_i])
            face_replacement = None
            for alt_image_i, face_annotation in alternatives:
                alternative_face_score = ImagesProcessor.score_face(config, face_annotation)
                if alternative_face_score > initial_score:
                    initial_score = alternative_face_score
                    face_replacement = image_merger.Replacement(
                        original_face=best_image.face_annotations[face_i],
                        new_image_path=image_filenames[alt_image_i],
                        new_face=face_annotation,
                    )
            if face_replacement is not None:
                face_replacements.append(face_replacement)
        self.image_merger.merge_image(config, image_filenames[best_image_index], out_filename, face_replacements)
