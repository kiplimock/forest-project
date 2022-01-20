import numpy as np
from scene3D import SceneReconstruction3D

def main():
    # camera coefficients
    K = np.array([[2759.48/4, 0, 1520.69/4, 0, 2764.16/4, 1006.81/4, 0, 0, 1]]).reshape(3, 3)
    # images are already distortion free
    d = np.array([0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, 5)

    # load images
    scene = SceneReconstruction3D(K, d)
    scene.load_image_pair("images/fountain-p11/0004.jpg", "images/fountain-p11/0005.jpg")

    # perform computations
    scene.plot_optic_flow()
    # scene._plot_rectified_images()

if __name__ == '__main__':
    main()