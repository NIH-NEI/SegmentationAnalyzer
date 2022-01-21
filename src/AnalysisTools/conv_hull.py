from tifffile import imread
from pathlib import Path
import matplotlib.pyplot as plt
import numpy
from scipy.spatial import ConvexHull

def pseudo_hull(mask: numpy.ndarray) -> numpy.ndarray:
    """Get RLE coordinates, or a pseudo hull

    The purpose of this function is to get a list of points associated with a
    pseudo-hull, where the pseudo-hull are the first and last columns in a row
    for a run of pixels. This reduces the number of points required to cycle
    through for many algorithms such as convex hull or feret diameter.

    Args:
        mask (numpy.ndarray): A 2d mask

    Returns:
        numpy.ndarray: An array of points related to the pseudo-hull
    """
    
    # Initialize the pseudo-hull placeholder
    phull = numpy.zeros((mask.shape[0],mask.shape[1] + 1), dtype=numpy.uint8)
    
    # Set pixels along the borders of the image
    phull[:,[0,image.shape[1]]] = image[:,[0,image.shape[1]-1]]
    phull[[0,image.shape[0]-1],:-1] = image[[0,image.shape[0]-1],:]
    
    # Get pseudo-hull points for the rest of the image
    phull[:,1:-1] = numpy.logical_xor(image[:,:-1],image[:,1:])
    
    # Get the coordinates of points in the pseudo-hull
    phull_points = numpy.argwhere(phull)
    
    # The points are suited for RLE, so subtract one from every other point
    phull_points[1::2,1] -= 1
    
    return phull_points

def feret_diam(hull: numpy.ndarray) -> numpy.ndarray:
    """Get feret diameter of all rotations of an object

    This function takes in a hull as an input and returns the feret diameter
    along the x-axis (columns) for each 1 degree rotation from 0 to 179.

    Args:
        hull (numpy.ndarray): A 2d array of points to calculate the feret
            diameter from

    Returns:
        numpy.ndarray: A 1-d array of feret diameters from 0:179
    """
    
    # Generate an array of rotation matrices
    angles = numpy.radians(numpy.arange(180))
    rotation_matrix = numpy.asarray(
        [
            [numpy.cos(angles), -numpy.sin(angles)],
            [numpy.sin(angles), numpy.cos(angles)],
        ]
    ).transpose(2,0,1)
    
    # Shift points so the object is centered around the origin
    shifted = phull - phull.mean(axis=0)
    
    # Generate all rotations of the points
    rotations = shifted @ rotation_matrix
    
    # Calculate the feret diameter along the x-axis
    feret_diam = rotations[...,1].max(axis=1) - rotations[...,1].min(axis=1)
    
    return feret_diam

def remove_noisy_points(hull: numpy.ndarray, threshold: float) -> numpy.ndarray:
    
    current_len = hull.shape[0]
    previous_len = 0
    
    while current_len != previous_len:
        
        previous_len = current_len
        
        vectors = numpy.zeros((2,) + hull.shape,dtype=numpy.float32)
    
        vectors[0] = hull - numpy.roll(hull,-1,axis=0)
        vectors[1] = numpy.roll(hull,1,axis=0) - hull
        
        norm = numpy.sqrt(numpy.sum(vectors ** 2,axis=-1))
        
        vectors /= norm[...,numpy.newaxis]
        
        dot = numpy.prod(vectors,axis=0).sum(axis=-1).squeeze()
        amax = numpy.argmax(dot)
        
        if dot[amax] > threshold:
            vectors = numpy.delete(vectors,amax,axis=1)
            hull = numpy.delete(hull,amax,axis=0)
        
        current_len = hull.shape[0]
    
    return hull

if __name__ == "__main__":
    
    # Demo script to show the results of the above functions
    
    inp_dir = Path("data")

    for img_path in inp_dir.iterdir():
        if img_path.suffix != ".tiff":
            continue

        # Read the image and threshold
        image = imread(img_path) > 0
        
        # Project it along the z-axis
        if image.ndim == 3:
            image = numpy.any(image, axis=0)
            
        # Generate a pseudo-hull
        phull = pseudo_hull(image)
        
        # Get the convex hull
        chull = phull[ConvexHull(phull).vertices]
        
        rhull = remove_noisy_points(chull,0.9)
        
        # Get the feret diameter
        diam = feret_diam(chull)
        
        fig,ax = plt.subplots(1,3,figsize=(12,4))
        
        # Plot the feret diameter
        ax[0].bar(numpy.arange(180),diam)
        ax[0].title.set_text("Feret Diameter")
        
        # Show the convex hull
        ax[1].matshow(image.astype(numpy.uint8))
        ax[1].plot(chull[:,1],chull[:,0],"-or")
        ax[1].title.set_text("Convex Hull")
        
        ax[2].matshow(image.astype(numpy.uint8))
        ax[2].plot(rhull[:,1],rhull[:,0],"-or")
        ax[2].title.set_text("Convex Hull")
        plt.show()