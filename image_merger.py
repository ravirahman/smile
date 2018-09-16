import dlib
import numpy as np
import matplotlib.pyplot as plt
import cv2

FACE_MODEL_PATH = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()


class Replacement:
    def __init__(self, original_face, new_image_path, new_face):
        self.original_face = original_face
        self.new_image_path = new_image_path
        self.new_face = new_face


class ImageMerger:
    def __init__(self):
        pass

    @staticmethod
    def merge_image(config, base_image_path: str, new_image_path: str, replacements: [Replacement]):
        base_image = cv2.imread(base_image_path)
        for replacement in replacements:
            plt.imshow(base_image)
            plt.show()
            base_crop_dims = ImageMerger.get_crop_dims(replacement.original_face.bounding_poly.vertices)
            base_cropped_img = ImageMerger.crop_image(base_image, base_crop_dims)
            plt.imshow(base_cropped_img)
            plt.show()
            base_image_points = ImageMerger.recognize_points(base_cropped_img) + base_crop_dims[0]  # add back the crop

            new_img = cv2.imread(replacement.new_image_path)
            new_crop_dims = ImageMerger.get_crop_dims(replacement.new_face.bounding_poly.vertices)
            new_cropped_img = ImageMerger.crop_image(new_img, new_crop_dims)
            plt.imshow(new_cropped_img)
            plt.show()
            new_image_points = ImageMerger.recognize_points(new_cropped_img) + new_crop_dims[0]

            alpha = config["alpha"]

            # Convert Mat to float data type
            img1 = np.float64(base_image)
            img2 = np.float64(new_img)

            # Compute weighted average point coordinates
            points = (1-alpha) * base_image_points + alpha * new_image_points
            points = np.int64(points)

            # Rectangle to be used with Subdiv2D
            size = base_image.shape
            rect = (0, 0, size[1], size[0])

            # Create an instance of Subdiv2D
            subdiv = cv2.Subdiv2D(rect)

            # Insert points into subdiv
            for p in base_image_points:
                subdiv.insert((p[0], p[1]))

            triangle_list = subdiv.getTriangleList()
            for i, t in enumerate(triangle_list):
                pt1 = np.array([int(t[0]), int(t[1])])
                pt2 = np.array([int(t[2]), int(t[3])])
                pt3 = np.array([int(t[4]), int(t[5])])
                if ImageMerger.rect_contains(rect, pt1) and ImageMerger.rect_contains(rect, pt2) and ImageMerger.rect_contains(rect, pt3):
                    x = np.where(np.all(base_image_points == pt1, axis=1))
                    y = np.where(np.all(base_image_points == pt2, axis=1))
                    z = np.where(np.all(base_image_points == pt3, axis=1))
                    if len(x) != 1 or len(y) != 1 or len(z) != 1:
                        raise Exception("non-defined points")
                    x, y, z = int(x[0]), int(y[0]), int(z[0])
                    t1 = np.array([base_image_points[x], base_image_points[y], base_image_points[z]])
                    t2 = np.array([new_image_points[x], new_image_points[y], new_image_points[z]])
                    t = np.array([points[x], points[y], points[z]])
                    # Morph one triangle at a time.
                    ImageMerger.morph_triangle(img1, img2, t1, t2, t, alpha)
            base_image = np.uint8(img1)  # set the base image to the morphed image
        # Display Result
        new_image = np.uint8(base_image)
        cv2.imwrite(new_image_path, new_image)
        plt.imshow(new_image)
        plt.show()

    @staticmethod
    def get_crop_dims(vertices):
        min_x = float('inf')
        min_y = float('inf')
        max_x = 0
        max_y = 0
        for vertex in vertices:
            if vertex.x < min_x:
                min_x = vertex.x
            if vertex.y < min_y:
                min_y = vertex.y
            if vertex.x > max_x:
                max_x = vertex.x
            if vertex.y > max_y:
                max_y = vertex.y
        return np.array([[min_x, min_y], [max_x, max_y]])

    @staticmethod
    def crop_image(img, crop_dims):
        min_x = crop_dims[0][0]
        min_y = crop_dims[0][1]
        max_x = crop_dims[1][0]
        max_y = crop_dims[1][1]

        return img[min_y:max_y, min_x:max_x]

    # Apply affine transform calculated using srcTri and dstTri to src and
    # output an image of size.
    @staticmethod
    def apply_affine_transform(src, src_tri, dst_tri, size):
        # Given a pair of triangles, find the affine transform.
        warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))

        # Apply the Affine Transform just found to the src image
        dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return dst

    # Warps and alpha blends triangular regions from img1 and img2 to img
    @staticmethod
    def morph_triangle(img1, img2, t1, t2, t, alpha):
        # Find bounding rectangle for each triangle
        r1 = cv2.boundingRect(np.float32([t1]))
        r2 = cv2.boundingRect(np.float32([t2]))
        r = cv2.boundingRect(np.float32([t]))

        # Offset points by left top corner of the respective rectangles
        t1_rect = []
        t2_rect = []
        t_rect = []

        for i in range(0, 3):
            t_rect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
            t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
            t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

        # Get mask by filling triangle
        mask = np.zeros((r[3], r[2], 3), dtype=np.float64)
        cv2.fillConvexPoly(mask, np.int32(t_rect), (1.0, 1.0, 1.0), 16, 0)

        # Apply warpImage to small rectangular patches
        img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
        img2_rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

        size = (r[2], r[3])
        warp_img1 = ImageMerger.apply_affine_transform(img1_rect, t1_rect, t_rect, size)
        warp_img2 = ImageMerger.apply_affine_transform(img2_rect, t2_rect, t_rect, size)

        # Alpha blend rectangular patches
        img_rect = (1.0 - alpha) * warp_img1 + alpha * warp_img2

        # Copy triangular region of the rectangular patch to the output image
        img1[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img1[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * (1 - mask) + img_rect * mask

    @staticmethod
    def recognize_points(img):
        predictor = dlib.shape_predictor(FACE_MODEL_PATH)
        # img = dlib.load_rgb_image(face_path)

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = detector(img, 1)
        if not len(dets) == 1:
            raise Exception("not exactly one face found")
        d = dets[0]
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        points = np.zeros((shape.num_parts, 2), dtype=np.int64)
        for i in range(shape.num_parts):
            point = shape.part(i)
            points[i] = np.array([int(point.x), int(point.y)])
        return points

    # Check if a point is inside a rectangle
    @staticmethod
    def rect_contains(rect, point):
        if point[0] < rect[0]:
            return False
        elif point[1] < rect[1]:
            return False
        elif point[0] > rect[2]:
            return False
        elif point[1] > rect[3]:
            return False
        return True



    # Draw a point
    # @staticmethod
    # def draw_point(img, p, color ) :
    #     cv2.circle( img, p, 2, color, cv2.cv.CV_FILLED, cv2.CV_AA, 0 )
    #
    # # Draw delaunay triangles
    # def draw_delaunay(img, subdiv, delaunay_color ) :
    #
    #     triangleList = subdiv.getTriangleList();
    #     size = img.shape
    #     r = (0, 0, size[1], size[0])
    #
    #     for t in triangleList :
    #
    #         pt1 = (t[0], t[1])
    #         pt2 = (t[2], t[3])
    #         pt3 = (t[4], t[5])
    #
    #         if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
    #
    #             cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.CV_AA, 0)
    #             cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.CV_AA, 0)
    #             cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.CV_AA, 0)
    #
    #
    # # Draw voronoi diagram
    # def draw_voronoi(img, subdiv) :
    #
    #     ( facets, centers) = subdiv.getVoronoiFacetList([])
    #
    #     for i in xrange(0,len(facets)) :
    #         ifacet_arr = []
    #         for f in facets[i] :
    #             ifacet_arr.append(f)
    #
    #         ifacet = np.array(ifacet_arr, np.int)
    #         color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    #
    #         cv2.fillConvexPoly(img, ifacet, color, cv2.CV_AA, 0);
    #         ifacets = np.array([ifacet])
    #         cv2.polylines(img, ifacets, True, (0, 0, 0), 1, cv2.CV_AA, 0)
    #         cv2.circle(img, (centers[i][0], centers[i][1]), 3, (0, 0, 0), cv2.cv.CV_FILLED, cv2.CV_AA, 0)
