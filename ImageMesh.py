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
        for column in range(0, width):
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


def _create_closing_surface(top, bottom, x_coordinates, y_coordinates):
    width = len(x_coordinates)
    height = 2
    vertices = numpy.empty([width * height, 3])
    for column in range(0, width):
        vertices[column*2][0] = x_coordinates[column]
        vertices[column*2][1] = y_coordinates[column]
        vertices[column*2][2] = top[column]

        vertices[column*2+1][0] = x_coordinates[column]
        vertices[column*2+1][1] = y_coordinates[column]
        vertices[column*2+1][2] = bottom[column]

    quad_columns = width - 1
    faces = numpy.empty([2 * quad_columns, 3], dtype=numpy.uintc)
    for face in range(0, len(faces), 2):
        quad = int(face / 2)
        upper_left = quad*2
        lower_left = (quad*2)+1
        upper_right = (quad*2)+2
        lower_right = (quad*2)+3

        # lower triangle
        faces[face][0] = upper_left
        faces[face][1] = lower_right
        faces[face][2] = lower_left

        # upper angle
        faces[face + 1][0] = upper_left
        faces[face + 1][1] = upper_right
        faces[face + 1][2] = lower_right

    return _create_mesh({
        "faces": faces,
        "vertices": vertices
    })


def _create_mesh(shape):
    faces = shape["faces"]
    vertices = shape["vertices"]
    surface = mesh.Mesh(numpy.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            surface.vectors[i][j] = vertices[f[j], :]
    return surface


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

    def __init__(self, heights, thickness=1):
        # Store the dimensions and original heights to simplify editing in the future.
        self.height = len(heights)
        self.width = len(heights[0])
        self.heights = heights

        # Make the shell for the image surface
        self.image_shell = ImageShell(heights)
        # Also make the shell for the footprint
        self.footprint_heights = self._create_footprint(thickness)
        self.footprint_shell = ImageShell(self.footprint_heights)

        # Now make the four walls
        self.top_edge = _create_closing_surface(
            self.heights[0], self.footprint_heights[0],
            numpy.zeros(len(heights[0])),
            numpy.arange(0,len(heights[0]))
        )
        bottom_edge_index = len(self.heights)-1
        self.bottom_edge = _create_closing_surface(
            self.heights[bottom_edge_index], self.footprint_heights[bottom_edge_index],
            bottom_edge_index * numpy.ones(len(self.heights[0])),
            numpy.arange(0, len(heights[0]))
        )
        self.left_edge = _create_closing_surface(
            self.heights[:, 0], self.footprint_heights[:, 0],
            numpy.arange(0, len(heights)),
            numpy.zeros(len(heights))
        )
        right_edge_index = len(self.heights[0])-1
        self.right_edge = _create_closing_surface(
            self.heights[:, right_edge_index], self.footprint_heights[:, right_edge_index],
            numpy.arange(0, len(heights)),
            right_edge_index * numpy.ones(len(heights))
        )

        # Now combine all the surfaces to make the actual solid mesh
        super().__init__(numpy.concatenate([self.image_shell.data,
                                            self.footprint_shell.data,
                                            self.top_edge.data,
                                            self.bottom_edge.data,
                                            self.left_edge.data,
                                            self.right_edge.data]))

    def _create_footprint(self, thickness=1):
        altitude = numpy.amin(self.heights) - thickness
        return altitude * numpy.ones(numpy.shape(self.heights))


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

