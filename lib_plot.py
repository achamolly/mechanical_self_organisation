import matplotlib.pyplot as plt
import numpy as np
import pyvista
from IPython.display import display, update_display
from dolfinx.plot import vtk_mesh


def plot_margin(N, t, cumt, c, T, gd, margvel, fig=None, ax=None, figureSize=8):
    """Plot Myosin, tension and contraction rate profiles"""

    # Rescale x axis:
    cumt = cumt / cumt[-1] * 360 - 180  # in degrees

    if fig is None:
        fig, ax = plt.subplots(figsize=(figureSize, figureSize))
        newfig = True
    else:
        ax.cla()
        newfig = False

    ax.set_title('t={:.2f}'.format(t))
    ax.plot(cumt, c * 0, 'k')  # x-axis
    ax.plot(cumt, c, label='Myosin')
    ax.plot(cumt, T, label='Tension')
    ax.plot(cumt[0:N], gd, label='Ext. rate')
    ax.plot(cumt[0:N], margvel, label='Margin velocity')
    ax.set_ylim(-1, 1.5)
    ax.set_xlim(-180, 180)
    ax.legend()

    if newfig:
        display(fig, display_id='marginplot')
    else:
        update_display(fig, display_id='marginplot')

    return fig, ax


def plot_mesh_pyvista(W, mesh):
    # pyvista.start_xvfb()
    pyvista.set_jupyter_backend("static")
    _, _, _ = vtk_mesh(W.sub(0).collapse()[0])

    # Create a pyvista-grid for the mesh
    grid = pyvista.UnstructuredGrid(*vtk_mesh(mesh, mesh.topology.dim))

    # Create plotter
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, style="wireframe", color="k")
    plotter.view_xy()
    plotter.set_background('white')
    # plotter.screenshot('foo.png', window_size=[2000, 2000])

    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        pass
        # pyvista.start_xvfb()
        # fig_as_array = plotter.screenshot("mesh.png")
    return


def plot_flow_pyvista(mesh, uh, V, scale, newfig=0, backend="static"):
    """Plot flow as arrows on grid"""
    # pyvista.start_xvfb()
    pyvista.set_jupyter_backend(backend)  # deactivate interactivity
    topology, cell_types, geometry = vtk_mesh(V)
    values = np.zeros((geometry.shape[0], 3), dtype=np.float64)
    values[:, :len(uh)] = uh.x.array.real.reshape((geometry.shape[0], len(uh)))

    # Create a point cloud of glyphs
    function_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    function_grid["u"] = values
    glyphs = function_grid.glyph(orient="u", factor=scale)

    # Create a pyvista-grid for the mesh
    grid = pyvista.UnstructuredGrid(*vtk_mesh(mesh, mesh.topology.dim))

    # Create plotter
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, style="wireframe", color="k")
    plotter.add_mesh(glyphs)
    plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        if newfig:
            display(plotter.show(), display_id='flowplot')
        else:
            update_display(plotter.show(), display_id='flowplot')
    else:
        pass
        # fig_as_array = plotter.screenshot("glyphs.png")

    return
