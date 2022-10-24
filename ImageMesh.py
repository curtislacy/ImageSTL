from stl import mesh
import numpy
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import matplotlib.colors as mcolors


def _create_surface(heights):
    width = len(heights[0])
    height = len(heights)
    vertices = numpy.empty([height * width, 3])
    for row in range(0, len(heights)):
        for column in range(0, len(heights[0])):
            index = row * width + column
            vertices[index][0] = row
            vertices[index][1] = column
            vertices[index][2] = heights[row][column]
    quad_rows = height - 1
    quad_columns = width - 1
    faces = numpy.empty([2 * quad_rows * quad_columns, 3], dtype=numpy.uintc)
    for face in range(0, len(faces), 2):
        quad = face / 2
        upper_left = quad + (quad / quad_columns)
        upper_right = upper_left + 1
        lower_left = upper_left + width
        lower_right = lower_left + 1

        # lower triangle
        faces[face][0] = upper_left
        faces[face][1] = lower_right
        faces[face][2] = lower_left

        # upper angle
        faces[face + 1][0] = upper_left
        faces[face + 1][1] = upper_right
        faces[face + 1][2] = lower_right

    return {
        "faces": faces,
        "vertices": vertices
    }


def _create_footprint(heights, thickness=1):

    altitude = numpy.amin(heights) - thickness
    bottom_heights = altitude * numpy.ones(numpy.shape(heights))
    return ImageShell(bottom_heights)


class ImageShell(mesh.Mesh):

    def __init__(self, heights):

        self.height = len(heights)
        self.width = len(heights[0])
        self.heights = heights

        surface = _create_surface(heights)
        faces = surface["faces"]
        vertices = surface["vertices"]
        super().__init__(numpy.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                self.vectors[i][j] = vertices[f[j], :]


class ImageSolid(mesh.Mesh):

    def __init__(self, image_shell, thickness=1):
        self.height = image_shell.height
        self.width = image_shell.width
        self.heights = image_shell.heights

        footprint = _create_footprint(self.heights, thickness)
        super().__init__(numpy.concatenate([image_shell.data, footprint.data]))


def display_as_3d(mesh, xlim=None, ylim=None, zlim=None, elevation=45., azimuth=45.):
    # Make sure we're always working with a list for simplicity.
    if type(mesh) != list:
        shape = [mesh]

    # Create a new plot
    figure = pyplot.figure()
    axes = mplot3d.Axes3D(figure, auto_add_to_figure=False)
    figure.add_axes(axes)

    all_points = numpy.array(0)
    for subshape in shape:
        poly_collection = mplot3d.art3d.Poly3DCollection(subshape.vectors)
        poly_collection.set_edgecolor(mcolors.CSS4_COLORS["black"])
        axes.add_collection3d(poly_collection)
        all_points = numpy.append(all_points, subshape.points)

    # Auto scale to the mesh size
    scale = all_points.flatten()
    axes.auto_scale_xyz(scale, scale, scale)
    if xlim is not None:
        pyplot.xlim(xlim)
    if ylim is not None:
        pyplot.ylim(ylim)
    if zlim is not None:
        axes.set_zlim(zlim)

    axes.view_init(elev=elevation, azim=azimuth)

    # Show the plot to the screen
    pyplot.show()

