import numpy as np
import matplotlib.pyplot as plt


def plot_frames_and_points(transformations, points):
    fig = plt.figure(figsize=[10, 8])
    ax  = fig.add_subplot(1, 1, 1, projection='3d')

    frames = [np.eye(4)]

    for t in transformations:
        frames.append(frames[-1] @ t)

    points_homogenous = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)

    overall_transformation = frames[-1]

    points = (overall_transformation @ points_homogenous.T).T

    def plot_matrix(transformation_matrix, label):
        for v, c in zip([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]], ['r', 'g', 'b']):
            zero = [0, 0, 0, 1]
            x, y, z, _ = zip(np.dot(transformation_matrix, zero), np.dot(transformation_matrix, v))
            ax.plot(x, y, z, f'-{c}', linewidth=1)
            ax.text(x[0], y[0], z[0], label, None)

    for i, f in enumerate(frames):
        plot_matrix(f, f"{i}")

    for p in points:
        x, y, z, _ = p
        ax.scatter([x], [y], [z], color="r", s=50)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)

    plt.show()


if __name__ == '__main__':
    matrix_1 = np.array([
        [1, 0, 0, -2],
        [0, 1, 0, 2],
        [0, 0, 1, 1],
        [0, 0, 0, 1]
    ])

    matrix_2 = np.array([
        [ 1,  0,  0, -2],
        [ 0,  0,  1,  1],
        [ 0, -1,  0, -1],
        [ 0,  0,  0,  1]
    ])

    points = np.array([[ 0.5,  0.,   0. ],
        [-0.5,  0.,   0. ],
        [ 0. ,  0.5 , 0. ],
        [ 0. , -0.5 , 0. ],
        [ 0. ,  0. ,  0.5],
        [ 0. ,  0. , -0.5]])

    plot_frames_and_points([matrix_1, matrix_2], points)