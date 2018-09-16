import main

import json


def test_index():
    main.app.testing = True
    client = main.app.test_client()

    r = client.get('/')
    assert r.status_code == 200
    assert 'Hello World' in r.data.decode('utf-8')


def test_process_images():
    main.app.testing = True
    client = main.app.test_client()
    r = client.post("/process_images", data=json.dumps({
        "image_uris": ["gs://drm-hack-the-north.appspot.com/image.jpg",
                       "gs://drm-hack-the-north.appspot.com/20180818_155551.jpg"],
    }), content_type='application/json')
    assert r.status_code == 200
    data = json.loads(r.data)
    assert "image_uri" in data
    image_uri_parts = data["image_uri"].split("/")
    assert len(image_uri_parts) == 5
    assert image_uri_parts[0] == "gs:"
    assert image_uri_parts[1] == ""
    assert image_uri_parts[2] == main.BUCKET_NAME
    assert image_uri_parts[3] == "results"
    assert image_uri_parts[4][-4:] == ".jpg"
