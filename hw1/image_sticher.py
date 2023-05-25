import numpy as np
import cv2
import random
import argparse

MY_MATCHES=np.array([[[ 968.,  268.],
        [ 824.,  254.]],

       [[ 984.,  248.],
        [ 838.,  235.]],

       [[1001.,  269.],
        [ 854.,  256.]],

       [[1007.,  283.],
        [ 860.,  269.]],

       [[1019.,  601.],
        [ 867.,  575.]],

       [[1024.,  281.],
        [ 875.,  267.]],

       [[1033.,  228.],
        [ 885.,  216.]],

       [[1034.,  250.],
        [ 886.,  238.]],

       [[1043.,  234.],
        [ 895.,  222.]],

       [[1049.,  685.],
        [ 894.,  655.]],

       [[1051.,  255.],
        [ 902.,  243.]],

       [[1058.,  605.],
        [ 903.,  578.]],

       [[1065.,  666.],
        [ 909.,  637.]],

       [[1070.,  348.],
        [ 918.,  332.]],

       [[1076.,  651.],
        [ 920.,  622.]],

       [[1092.,  366.],
        [ 938.,  350.]],

       [[1095.,  645.],
        [ 937.,  617.]],

       [[1097.,  614.],
        [ 940.,  587.]],

       [[1124.,  468.],
        [ 966.,  447.]],

       [[1134.,  664.],
        [ 973.,  634.]]])

def undistort(image):
    size = image.shape[1::-1]
    assert size == (1280, 720) # Values calculated for quality 11.
    camera_matrix = np.array([[1.21875562e+05, 0.00000000e+00, 6.65079519e+02],
        [0.00000000e+00, 3.86689259e+04, 2.52346547e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    dist_coeffs = np.array([[-1.64370734e+02, -8.00393104e+01,  2.05539224e+00,  1.08168391e-01,
        1.92432149e+01]])

    alpha = 0.
    rect_camera_matrix = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, size, alpha)[0]

    # Calculate undistortion maps
    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, np.eye(3), rect_camera_matrix, size, cv2.CV_32FC1)
    undistorted_img = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
    return undistorted_img


def get_matches(filename1, filename2, visualize=True, lowe_ratio=0.6):
    # Read images from files, convert to greyscale
    img1 = cv2.imread(filename1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(filename2, cv2.IMREAD_GRAYSCALE)

    # Find the keypoints and descriptors with SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    # Ratio test as per Lowe's paper
    good_matches = []
    for i, (m, n) in enumerate(matches):
        if m.distance < lowe_ratio * n.distance:
            matchesMask[i] = [1, 0]
            good_matches.append((kp1[m.queryIdx].pt, kp2[m.trainIdx].pt))

    if visualize:
        draw_params = dict(
            matchColor=(0, 255, 0),
            singlePointColor=(0, 0, 255),
            matchesMask=matchesMask,
            flags=cv2.DrawMatchesFlags_DEFAULT,
        )
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
        cv2.imshow("get_matches", img3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return good_matches

def apply_homography(homography, vector):
    result = homography @ vector
    return result / result[-1]

def ransac(values, num_of_data_points_to_fit=4, k=100, eps=10):
    values = np.concatenate([values, np.ones((*values.shape[:-1], 1))], axis=2)
    best_model = None
    best_point_count = 0
    for _ in range(k):
        sample = random.sample(list(values), num_of_data_points_to_fit)

        x = np.hstack([v[0].reshape(3, 1) for v in sample])
        y = np.hstack([v[1].reshape(3, 1) for v in sample])
        maybe_model = find_projective_transformation(x, y)

        inliers = []
        for v in values:
            res = apply_homography(maybe_model, v[0])
            if np.linalg.norm(res - v[1]) < eps:
                inliers.append(v)
        
        if len(inliers) > best_point_count:
            x = np.hstack([v[0].reshape(3, 1) for v in inliers])
            y = np.hstack([v[1].reshape(3, 1) for v in inliers])
            maybe_model = find_projective_transformation(x, y)
            best_model = maybe_model
            best_point_count = len(inliers)
        
    return best_model


def get_from_image_or_black(image, vectors, dtype=np.uint8):
    size = (image.shape[1], image.shape[0])
    result = np.zeros(vectors.shape, dtype)
    for i, row in enumerate(vectors):
        for j, el in enumerate(row):
            x, y, _ = el
            if x < 0 or x >= size[0] or y < 0 or y >= size[1]:
                result[i, j, :] = np.zeros(3, dtype)
            else:
                result[i, j, :] = image[y, x, :]
    return result

def transform_homography(image, homography_matrix, translation, wanted_shape=None, dtype=np.uint8):
    # Translation is being applied before homography transformation.
    # Calculating inverse homography. TODO Note: it may fail.
    inverse_homography = np.linalg.inv(homography_matrix)
    # OpenCV and maths dimensions are different than numpy.
    if not wanted_shape:
        size = (image.shape[1], image.shape[0])
    else:
        size = (wanted_shape[1], wanted_shape[0])
    
    ys, xs = np.mgrid[0:size[1], 0:size[0]]
    
    homogenous_vectors = np.stack([xs - translation[1], ys - translation[0], np.ones(xs.shape)], axis=-1)
    source_vectors = (homogenous_vectors @ inverse_homography.T)
    
    source_vectors /= np.expand_dims(source_vectors[:, :, -1], 2)
    
    source_vectors = np.rint(source_vectors).astype(int)

    return get_from_image_or_black(image, source_vectors, dtype=dtype)


def find_projective_transformation(src_vectors, dst_vectors):
    # Matching pairs should consist of normalized vectors.
    
    single_pair_matrices = []
    for s, d in zip(src_vectors.T, dst_vectors.T):
        xs, ys, _ = s
        xd, yd, _ = d
        single_pair_matrices.append(np.array([
            [xs, ys, 1, 0, 0, 0, -xd*xs, -xd*ys, -xd],
            [0, 0, 0, xs, ys, 1, -yd*xs, -yd*ys, -yd]
        ]))
    
    A = np.vstack(single_pair_matrices)

    _, _, V = np.linalg.svd(A)
    eigenvector = V[-1, :]

    return eigenvector.reshape((3,3))


def test_find_projective_transformation(test_cases=10):
    # Not 100% exact due to random values and inaccuracy of matrix operations.
    for i in range(1, test_cases + 1):
        homography_matrix = np.random.random((3,3))
        homography_matrix /= np.linalg.norm(homography_matrix, ord=2)
        src_vectors = np.vstack([np.array([[0, 0, 1, 1],[0, 1, 0, 1]], dtype=np.float64), np.ones((1,4), dtype=np.float64)])
        dst_vectors = homography_matrix @ src_vectors
        dst_vectors /= dst_vectors[-1, :]
        calculated_homography_matrix = find_projective_transformation(src_vectors, dst_vectors)
        if np.all(np.isclose(homography_matrix, calculated_homography_matrix, atol=0.1)):
            print(f"Test {i} passed.")
        else:
            print(f"""Test {i} failed.
    Expected:   {homography_matrix}
    Got:        {calculated_homography_matrix}
    Source vectors:         {src_vectors}
    Destination vectors:    {dst_vectors}
    -------------------------------------------"""
            )


def find_optimal_size_and_translation(base_image_shape, second_image_shape, transformation, margin=0): 
    y, x, _ = second_image_shape 
    transformed_corners = transformation @ np.vstack([np.array([[0, 0, x, x], [0, y, 0, y]]), np.ones((1, 4))])
    transformed_corners /= transformed_corners[-1, :]
    
    base_height, base_width, _ = base_image_shape
    possible_corners = np.hstack([np.array([[0, 0, base_width, base_width], [0, base_height, 0, base_height]]), transformed_corners[:-1, :]])
    
    left_top_corner = np.min(possible_corners, axis=1) - np.full((2,), margin)
    right_bottom_corner = np.max(possible_corners, axis=1) + np.full((2,), margin)

    translation_of_origin = tuple(np.rint(-left_top_corner).astype(int))
    new_size = tuple(np.ceil(right_bottom_corner - left_top_corner).astype(int))

    return translation_of_origin[::-1], new_size[::-1]


def distance_from_edge(image):
    y, x, _ = image.shape
    ys, xs = np.mgrid[0:y, 0:x]
    return np.min(np.stack([xs, x - xs, ys, y - ys], axis=-1), axis=2)


def stich_images(image1, image2, transformation):
    translation, new_shape = find_optimal_size_and_translation(
        image1.shape, 
        image2.shape, 
        transformation,
        margin=0
    )
    new_shape += tuple([3])

    image1_resized = np.zeros(new_shape, np.uint8)
    image1_resized[translation[0]:translation[0]+image1.shape[0], translation[1]:translation[1]+image1.shape[1]] = image1

    image1_weights = np.zeros(new_shape[:2], np.float64)
    image1_weights[translation[0]:translation[0]+image1.shape[0], \
        translation[1]:translation[1]+image1.shape[1]] = distance_from_edge(image1)

    image2_transformed = transform_homography(image2, transformation, translation, new_shape)

    image2_weights_before_transformation = np.stack([distance_from_edge(image2)] * 3, axis=-1)
    image2_weights = transform_homography(
        image2_weights_before_transformation, 
        transformation, 
        translation, 
        new_shape,
        np.float64
    )[:, :, 0]
    
    image1_weights = np.expand_dims(image1_weights, 2)
    image2_weights = np.expand_dims(image2_weights, 2)

    # 0/0 warning, but nonzero / 0 will never happen
    result = np.floor(
        ((image1_resized * image1_weights) + (image2_transformed * image2_weights)) / (image1_weights + image2_weights)
    ).astype(np.uint8)

    return result


def pipeline(path1, path2, visualize=False, save=True, use_handpicked_matches=False):
    # Load images.
    image1 = cv2.imread(path1)
    image2 = cv2.imread(path2)

    if visualize:
        cv2.namedWindow("pipeline")
        cv2.imshow("pipeline", cv2.hconcat([image1, image2]))
        cv2.waitKey()

    # Undistort. TODO
    image1 = undistort(image1)
    image2 = undistort(image2)

    if save:
        cv2.imwrite('./image1_undistorted.jpg', image1)
        cv2.imwrite('./image2_undistorted.jpg', image2)

    if visualize:
        cv2.imshow("pipeline", cv2.hconcat([image1, image2]))
        cv2.waitKey()
        

    # Get matching points.
    if not use_handpicked_matches:
        matches = np.array(get_matches(path1, path2, visualize=visualize))
    else:
        matches = MY_MATCHES
    
    # Get best homography with RANSAC.
    homography = ransac(matches, k=50, eps=0.5)

    # Stich images.
    stiched_image = stich_images(image2, image1, homography)
    if visualize:
        cv2.imshow("pipeline", stiched_image)
        cv2.waitKey()
    
    if save:
        cv2.imwrite('./stiched.jpg', stiched_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = 'Image stiching')
    parser.add_argument('path1')
    parser.add_argument('path2')

    parser.add_argument('-v', '--visualize', action='store_true')
    parser.add_argument('--nosave', action='store_true')
    parser.add_argument('--handpickedmatches', action='store_true')

    args = parser.parse_args()

    pipeline(
        args.path1, 
        args.path2, 
        visualize=args.visualize, 
        save=not args.nosave, 
        use_handpicked_matches=args.handpickedmatches
    )
