from stl import mesh
import numpy
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import matplotlib.colors as mcolors


def _create_surface(heights, scale_factor_xy, counterclockwise=False):
    width = len(heights[0])
    height = len(heights)
    vertices = numpy.empty([height * width, 3])
    for row in range(0, len(heights)):
        for column in range(0, width):
            index = row * width + column
            vertices[index][0] = row * scale_factor_xy[1]
            vertices[index][1] = column * scale_factor_xy[0]
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

        if counterclockwise:
            # lower triangle
            faces[face][0] = upper_left
            faces[face][1] = lower_left
            faces[face][2] = lower_right

            # upper triangle
            faces[face + 1][0] = upper_right
            faces[face + 1][1] = upper_left
            faces[face + 1][2] = lower_right

        else:
            # lower triangle
            faces[face][0] = upper_left
            faces[face][1] = lower_right
            faces[face][2] = lower_left

            # upper triangle
            faces[face + 1][0] = upper_left
            faces[face + 1][1] = upper_right
            faces[face + 1][2] = lower_right

    return {
        "faces": faces,
        "vertices": vertices
    }


def _create_closing_surface(scale_factor_xy, top, bottom, x_coordinates, y_coordinates, counterclockwise=False):
    width = len(x_coordinates)
    height = 2
    vertices = numpy.empty([width * height, 3])
    for column in range(0, width):
        # These look weird because vertices is still in row/column space at this time?
        vertices[column*2][0] = y_coordinates[column] * scale_factor_xy[1]
        vertices[column*2][1] = x_coordinates[column] * scale_factor_xy[0]
        vertices[column*2][2] = top[column]

        vertices[column*2+1][0] = y_coordinates[column] * scale_factor_xy[1]
        vertices[column*2+1][1] = x_coordinates[column] * scale_factor_xy[0]
        vertices[column*2+1][2] = bottom[column]

    quad_columns = width - 1
    faces = numpy.empty([2 * quad_columns, 3], dtype=numpy.uintc)
    for face in range(0, len(faces), 2):
        quad = int(face / 2)
        upper_left = quad*2
        lower_left = (quad*2)+1
        upper_right = (quad*2)+2
        lower_right = (quad*2)+3

        if counterclockwise:
            # lower triangle
            faces[face][0] = upper_left
            faces[face][1] = lower_left
            faces[face][2] = lower_right

            # upper angle
            faces[face + 1][0] = upper_right
            faces[face + 1][1] = upper_left
            faces[face + 1][2] = lower_right
        else:
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


def create_height_from_image(image):
    heights = numpy.zeros((image.shape[0], image.shape[1]))
    for row in range(0, len(heights)):
        for column in range(0, len(image[row])):
            heights[row][column] = max(
                image[row][column][0],
                image[row][column][1],
                image[row][column][2]
            )
    return heights


def calculate_scale_factor(dimension_in_datapoints_xy, x_mm, y_mm):
    if x_mm is None:
        if y_mm is None:
            # If we didn't specify target dimensions, then just bail and use a scale of 1.
            return 1.0, 1.0
        else:
            scale = float(y_mm) / (dimension_in_datapoints_xy[1] - 1)
            return scale, scale
    else:
        if y_mm is None:
            scale = float(x_mm) / (dimension_in_datapoints_xy[0] - 1)
            return scale, scale
        else:
            # Both are specified, so we have to do separate x & y factors.
            return float(x_mm) / (dimension_in_datapoints_xy[0] - 1), float(y_mm) / (dimension_in_datapoints_xy[1] - 1)


class ImageShell(mesh.Mesh):

    def __init__(self, heights, scale_factor_xy, counterclockwise=False):

        self.height = len(heights)
        self.width = len(heights[0])
        self.heights = heights

        surface = _create_surface(heights, scale_factor_xy, counterclockwise)
        faces = surface["faces"]
        vertices = surface["vertices"]
        super().__init__(numpy.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                self.vectors[i][j] = vertices[f[j], :]


class ImageSolid(mesh.Mesh):

    def __init__(self, heights=None, image=None,
                 base_thickness_mm=1, texture_height_mm=255.0, image_height_range=None,
                 x_mm=None, y_mm=None):

        if heights is None and image is not None:
            heights = create_height_from_image(image)

        if image_height_range is None:
            image_height_range = numpy.max(heights)

        # Store the dimensions and original heights to simplify editing in the future.
        self.dimension_in_datapoints_xy = (len(heights[0]), len(heights))
        self.heights = texture_height_mm / image_height_range * heights

        self.scale_factor_xy = calculate_scale_factor(self.dimension_in_datapoints_xy, x_mm, y_mm)

        # Make the shell for the image surface
        self.image_shell = ImageShell(self.heights, self.scale_factor_xy, counterclockwise=True)
        # Also make the shell for the footprint
        self.footprint_heights = self._create_footprint(base_thickness_mm)
        self.footprint_shell = ImageShell(self.footprint_heights, self.scale_factor_xy)

        # Now make the four walls
        self.top_edge = _create_closing_surface(
            self.scale_factor_xy,
            self.heights[0], self.footprint_heights[0],
            numpy.arange(0, len(self.heights[0])),
            numpy.zeros(len(self.heights[0]))
        )
        bottom_edge_index = len(self.heights)-1
        self.bottom_edge = _create_closing_surface(
            self.scale_factor_xy,
            self.heights[bottom_edge_index], self.footprint_heights[bottom_edge_index],
            numpy.arange(0, len(heights[0])),
            bottom_edge_index * numpy.ones(len(self.heights[0])),
            counterclockwise=True
        )
        self.left_edge = _create_closing_surface(
            self.scale_factor_xy,
            self.heights[:, 0], self.footprint_heights[:, 0],
            numpy.zeros(len(self.heights)),
            numpy.arange(0, len(self.heights)),
            counterclockwise=True
        )
        right_edge_index = len(self.heights[0])-1
        self.right_edge = _create_closing_surface(
            self.scale_factor_xy,
            self.heights[:, right_edge_index], self.footprint_heights[:, right_edge_index],
            right_edge_index * numpy.ones(len(self.heights)),
            numpy.arange(0, len(self.heights))
        )

        # Now combine all the surfaces to make the actual solid mesh
        super().__init__(numpy.concatenate([self.image_shell.data,
                                            self.footprint_shell.data,
                                            self.top_edge.data,
                                            self.bottom_edge.data,
                                            self.left_edge.data,
                                            self.right_edge.data
                                            ]))

    def _create_footprint(self, base_thickness_mm=1):
        altitude = numpy.amin(self.heights) - base_thickness_mm
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

