Smile: A demonstration on facial merging by Daniel McCormick, Matteo Sandrin, and Ravi Rahman

The premise of the project is to take a selection of similar photos (where each person in the group looks the best), and merge the photos together. 

The front end was written in Objective C, where a "live photo" is taken, which is then split up into multiple individual frames. These frames are then sent over to the back end (via firebase?). 

Since people "looking the best" is subjective, we arbitrarily decided the primary factors of what makes people look best by using the google sentiment analyzer to identify in which images people look the most "Joyful" and which they are facing the camera most. 

The best images are then passed to the OpenCV (and Python) code where the images are, and are merged following a fairly simple process:

Using dlib's frontal face detector, we identify common points on each face in the photos.
We then used Delaunay triangulation to break the face up into smaller pieces, which are easier to work with (essentially making triangles in between the points that we got from dlib's frontal face detector).

For each triangle, we then calculated the affine transforms of each pair of triangles to preserve the facial structure (effectively finding a mapping to preserve the structure of each triangle). 
We then used OpenCV's warpaffine to warp each triangle - to transform it appropriately. To handle the fact that OpenCV only works on images and not triangles, we put a mask over everything else. 

Finally, we used alpha blending to have a weighted blending in favour of the the better looking photos.

