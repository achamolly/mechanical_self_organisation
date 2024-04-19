# For data I/O:
import inspect
import os
import typing
from tempfile import NamedTemporaryFile

import gmsh
import h5py
import numpy as np
import pandas as pd
import scipy.interpolate as interp
# for handling the margin
import shapely.geometry as geom
# for ModifiedLinearProblem
import skimage.io as skio
import ufl
from dolfinx import la
from dolfinx.cpp.mesh import to_type, cell_entity_type
from dolfinx.fem import Function, FunctionSpace, form, \
    assemble_scalar, Expression, Constant
from dolfinx.fem.bcs import DirichletBC
from dolfinx.fem.forms import Form
from dolfinx.fem.forms import form as _create_form
from dolfinx.fem.function import Function as _Function
from dolfinx.fem.petsc import *
# for vel_on_margin
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from dolfinx.graph import adjacencylist
from dolfinx.io import XDMFFile
# for save_fem
from dolfinx.io import distribute_entity_data
from dolfinx.io.gmshio import extract_geometry, extract_topology_and_markers, ufl_mesh, cell_perm_array
from dolfinx.mesh import create_mesh, meshtags_from_entities
# For meshing:
from mpi4py import MPI
# For femsolve:
from petsc4py import PETSc


# Define active stress as a class, regulation implementation
class ActiveStress_reg:
    def __init__(self, margin, thinterp, Tinterp, N, l, cumt, d, t_dep, **kwargs):
        self.margin = margin
        self.thinterp = thinterp
        self.Tinterp = Tinterp
        self.N = N
        self.l = l
        self.cumt = cumt
        self.d = d
        self.t_dep = t_dep

    def __call__(self, x):
        gdim = 4
        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)  # (4,n) matrix with the rows sxx sxy syx syy
        s = tisstens_reg(x[0], x[1], self.margin, self.thinterp, self.Tinterp, self.N, self.l, self.cumt, self.d,
                         self.t_dep)
        for i in range(4):
            values[i] = s[i]  # define active stress at point x
        return values


# Define Epiboly interpolation as a class (workaround for missing dolfinx feature of direct interpolation on different meshes)
class EpibolyInterpolator:
    def __init__(self, M, mesh, **kwargs):
        self.M = M
        self.mesh = mesh
        self.bb_tree = bb_tree(self.mesh, self.mesh.topology.dim)

    def __call__(self, x):
        # Initialise cell search
        points = x
        cells = []
        points_on_proc = []
        # Find cells whose bounding-box collide with the the points
        cell_candidates = compute_collisions_points(self.bb_tree, points.T)
        # Choose one of the cells that contains the point
        colliding_cells = compute_colliding_cells(self.mesh, cell_candidates, points.T)
        for i, point in enumerate(points.T):
            if len(colliding_cells.links(i)) > 0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])
            else:
                print(
                    'Error: no colliding cell! New mesh does not lie entirely within old mesh. Increase trim parameter to fix.')
        points_on_proc = np.array(points_on_proc, dtype=np.float64)
        # Evaluate M
        epib_values = self.M.eval(points_on_proc, cells)
        return epib_values.T


class FrictionCoef():
    def __init__(self, xmin, d, angle, offset, **kwargs):
        self.xmin = xmin
        self.d = d
        self.angle = angle
        self.offset = offset

    def __call__(self, x):
        values = np.zeros((1, x.shape[1]), dtype=PETSc.ScalarType)
        rot = np.array([[np.cos(self.angle), np.sin(self.angle)], [-np.sin(self.angle), np.cos(self.angle)]])
        x[1] += self.offset
        xrot = np.tensordot(rot, x[:2], axes=1)
        xs = xrot[0]
        ys = xrot[1]
        for i, xx in enumerate(xs):
            if self.d / 2 >= ys[i] >= -self.d / 2:
                if xx >= self.xmin:
                    values[0, i] = 1  # non-zero only in hair domain
        return values


# Define active stress as a class
class ModifiedLinearProblem:
    """Class for solving a linear variational problem of the form :math:`a(u, v) = L(v) \\,  \\forall v \\in V`
    using PETSc as a linear algebra backend.

    Modified to allow for single initialisation outside of the loop (workaround for buggy options destruction in PETSc)

    """

    def __init__(self,
                 u: _Function = None, petsc_options: typing.Optional[dict] = None):  # , u: _Function = None):
        """Initialize solver for a linear variational problem.

        Example::

            problem = ModifiedLinearProblem(petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
            problem.solve(a, L, [bc0, bc1])

        """

        if u is None:
            self._solver = PETSc.KSP().create(MPI.COMM_SELF)
        else:
            self._solver = PETSc.KSP().create(self.u.function_space.mesh.comm)

        # Give PETSc solver options a unique prefix
        problem_prefix = f"dolfinx_solve_{id(self)}"
        self._solver.setOptionsPrefix(problem_prefix)

        # Set PETSc options
        opts = PETSc.Options()
        opts.prefixPush(problem_prefix)
        for k, v in petsc_options.items():
            opts[k] = v
        opts.prefixPop()
        # print(opts.getAll())
        self._solver.setFromOptions()

    def solve(self, a: ufl.Form, L: ufl.Form, bcs: typing.List[DirichletBC] = [],
              u: typing.Optional[_Function] = None, form_compiler_options: typing.Optional[dict] = None,
              jit_options: typing.Optional[dict] = None) -> _Function:
        """Solve the problem.

         Args:
            a: A bilinear UFL form, the left hand side of the variational problem.
            L: A linear UFL form, the right hand side of the variational problem.
            bcs: A list of Dirichlet boundary conditions.
            u: The solution function. It will be created if not provided.
            petsc_options: Parameters that is passed to the linear
                algebra backend PETSc. For available choices for the
                'petsc_options' kwarg, see the `PETSc documentation
                <https://petsc4py.readthedocs.io/en/stable/manual/ksp/>`_.
            form_compiler_params: Parameters used in FFCx compilation of
                this form. Run ``ffcx --help`` at the commandline to see
                all available options.
            jit_params: Parameters used in CFFI JIT compilation of C
                code generated by FFCx. See `python/dolfinx/jit.py` for
                all available parameters. Takes priority over all other
                parameter values.
            """

        self._solver.reset()  # make sure everything is clean

        '''original __init__() from here'''

        self._a = _create_form(a, form_compiler_options=form_compiler_options, jit_options=jit_options)
        self._A = create_matrix(self._a)

        self._L = _create_form(L, form_compiler_options=form_compiler_options, jit_options=jit_options)
        self._b = create_vector(self._L)

        if u is None:
            # Extract function space from TrialFunction (which is at the
            # end of the argument list as it is numbered as 1, while the
            # Test function is numbered as 0)
            self.u = _Function(a.arguments()[-1].ufl_function_space())
        else:
            self.u = u

        self._x = la.create_petsc_vector_wrap(self.u.x)
        self.bcs = bcs

        self._solver.setOperators(self._A)

        # Give PETSc solver options a unique prefix
        problem_prefix = f"dolfinx_solve_{id(self)}"
        self._solver.setOptionsPrefix(problem_prefix)
        # print(problem_prefix)

        # pass options
        opts = PETSc.Options()
        self._solver.setFromOptions()
        # print(opts.getAll())

        # Set matrix and vector PETSc options
        self._A.setOptionsPrefix(problem_prefix)
        self._A.setFromOptions()
        self._b.setOptionsPrefix(problem_prefix)
        self._b.setFromOptions()

        '''Original solve() from here'''

        # Assemble lhs
        self._A.zeroEntries()
        assemble_matrix(self._A, self._a, bcs=self.bcs)
        self._A.assemble()

        # Assemble rhs
        with self._b.localForm() as b_loc:
            b_loc.set(0)
        assemble_vector(self._b, self._L)

        # Apply boundary conditions to the rhs
        apply_lifting(self._b, [self._a], bcs=[self.bcs])
        self._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self._b, self.bcs)

        # Solve linear system and update ghost values in the solution
        self._solver.solve(self._b, self._x)
        self.u.x.scatter_forward()

        return self.u

    @property
    def L(self) -> Form:
        """The compiled linear form"""
        return self._L

    @property
    def a(self) -> Form:
        """The compiled bilinear form"""
        return self._a

    @property
    def A(self) -> PETSc.Mat:
        """Matrix operator"""
        return self._A

    @property
    def b(self) -> PETSc.Vec:
        """Right-hand side vector"""
        return self._b

    @property
    def solver(self) -> PETSc.KSP:
        """Linear solver object"""
        return self._solver


def calc_exrate(u_values, ta, l, N):
    """Calculate margin extension rate from FEM solution"""

    margvel = np.zeros(N)
    gd = np.zeros(N)
    for i in range(N):
        # margin velocity (for information only)
        margvel[i] = np.linalg.norm(u_values[(i + 1) % N, :])
        # contraction rates
        gd[i] = np.dot(u_values[(i + 1) % N, :] - u_values[i, :], ta[i, :]) / l[i]

    return margvel, gd


def create_solver(petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}):
    """Create solver object"""
    solver = ModifiedLinearProblem(petsc_options=petsc_options)
    return solver


def det_newradius(mesh, currentradius, rtol=1e-2, trim=5e-4):
    """Trim currentradius to make sure new mesh lies entirely within the old one"""
    meshnorms = np.linalg.norm(mesh.geometry.x, axis=1)  # L2 norm of mesh vertex positions
    newcurrentradius = np.min(meshnorms[np.isclose(meshnorms, currentradius, rtol)]) * (
            1 - trim)  # new radius is just within old mesh
    return newcurrentradius


def epiboly_init(mesh, geometry_parameters, THorder, t):
    """Return UFL objects for Epiboly forcing"""

    # unwrap
    Er, r, d, offset = geometry_parameters

    # Epiboly parameters
    x = ufl.SpatialCoordinate(mesh)
    ue = Constant(mesh, PETSc.ScalarType(1.0488))
    we = Constant(mesh, PETSc.ScalarType(0.3956))
    o_s = Constant(mesh, PETSc.ScalarType(-offset))
    er = Constant(mesh, PETSc.ScalarType(Er))
    wb = Constant(mesh, PETSc.ScalarType(0.19))
    rs = Constant(mesh, PETSc.ScalarType(0.8694))
    ts = Constant(mesh, PETSc.ScalarType(0.75))
    ws = Constant(mesh, PETSc.ScalarType(0.1196))

    # Epiboly functions (at inital position)
    m1 = 0.5 * (1 - ufl.mathfunctions.Tanh(
        2 * ((x[0] ** 2 + (x[1] - o_s) ** 2) ** 0.5 - ue) / we))  # slight EP expansion at late times
    m2 = (-x[1]) * 0.5 * (1 - ufl.mathfunctions.Tanh(
        2 * ((x[0] ** 2 + (x[1] - o_s) ** 2) ** 0.5 - ue) / we))  # slight A-P anisotropy
    m3 = (1 - 0.5 * (
            1 - ufl.mathfunctions.Tanh(2 * ((x[0] ** 2 + (x[1] - o_s) ** 2) ** 0.5 - ue) / we)))  # EE expansion
    m4 = -ufl.mathfunctions.Exp(
        -0.5 * ((x[0] ** 2 + x[1] ** 2) ** 0.5 - er) ** 2 / (wb ** 2))  # less EE expansion at edge
    m5 = -ufl.mathfunctions.Exp(
        -0.5 * ((x[0] ** 2 + (x[1] - o_s) ** 2) ** 0.5 - rs) ** 2 / (ws ** 2) - 0.5 * ufl.mathfunctions.Atan2(x[0], -(
                x[1] - o_s)) ** 2 / (ts ** 2))  # PS contraction

    # project onto mesh (piecewise linear)
    V = FunctionSpace(mesh, ufl.FiniteElement("Lagrange", mesh.ufl_cell(), THorder + 1))
    # quadratic elements to mitigate resolution loss during contraction

    ms = [m1, m2, m3, m4, m5]
    Ms = []
    for m in ms:
        m_expr = Expression(m, V.element.interpolation_points())
        M = Function(V)
        M.interpolate(m_expr)
        Ms.append(M)

    # Coefficients for epiboly contributions (matched to equal the values in Saadaoui et al. Table S1 at t=8)
    coefs = update_coefs(mesh, t)

    return Ms, coefs


def epiboly_remesh(newmesh, oldmesh, Ms, THorder):
    """interpolate area changes from old mesh onto new mesh"""
    # Include custom EpibolyInterpolator class for missing dolfinx feature of direct interpolation on different meshes.
    # quadratic elements to mitigate resolution loss during contraction
    newV = FunctionSpace(newmesh, ufl.FiniteElement("Lagrange", newmesh.ufl_cell(), THorder + 1))
    newMs = []
    for M in Ms:
        newM = Function(newV)
        M_expr = EpibolyInterpolator(M, oldmesh)
        newM.interpolate(M_expr)  # interpolate onto new mesh
        newMs.append(newM)
    return newMs


def fig_to_array(fig):
    """Convert figure to numpy array"""
    with NamedTemporaryFile(suffix='.png') as f:
        fig.savefig(f.name, bbox_inches='tight')
        height, width, _channels = skio.imread(f.name).shape
        im = skio.imread(f.name)
    return im


def geom_init(N, r, offset):
    # Create margin segments
    angles = 2 * np.pi / N * (np.arange(N) - 0.5) - 3 * np.pi / 2  # set so that 0th segment is at the anterior end
    # [was posterior end]
    ps = np.array([r * np.cos(angles), r * np.sin(angles) - offset]).T  # define array of initial positions
    return ps


def get_fricobj(xmin, d, angle=0, offset=0):
    """Create fricobj to be interpolated onto appropriate FE space"""
    fricobj = FrictionCoef(xmin, d, angle, offset)
    return fricobj


def get_stressobj_reg(margin_geom, interpolators, N, l, cumt, d, t_dep):
    """Create stressobj to be interpolated onto appropriate FE space"""
    '''Regulation implementation'''
    cinterp, Tinterp, thinterp = interpolators
    stressobj = ActiveStress_reg(margin_geom, thinterp, Tinterp, N, l, cumt, d, t_dep)
    return stressobj


def initialise_mso(N, t, initialisation_parameters, geometry_parameters, regulate, mu=None, ten_profile=None,
                   baseline_tension=None):
    """Initialise margin shapely object, margin_variables, interpolators and currentradius"""

    # unwrap settings
    c0, Camp, initialT, zeta, alpha = initialisation_parameters
    Er, r, d, offset = geometry_parameters

    ps = geom_init(N, r, offset)  # find margin coordinates
    margin_geom, l, lt, cumt, th, ta = updatelengths(ps, N)  # derive quantities from margin point cooordinates
    c = myos_init(N, Camp, c0, th, zeta, alpha)  # initialise myosin profile
    T = tens_init(N, initialT, regulate, ps, t, mu, ten_profile, baseline_tension)  # initialise tension profile

    interpolators = interp_init(cumt, c, T, th)  # initialise interpolators

    return margin_geom, ps, l, lt, cumt, th, ta, c, T, interpolators


def interp_init(cumt, c, T, th):
    cinterp = interp.interp1d(cumt, c, kind='linear')  # Myosin interpolator
    Tinterp = interp.interp1d(cumt, T, kind='linear')  # Tension interpolator
    thinterp = interp.interp1d(cumt, th, kind='linear')  # tangent angle interpolator
    interpolators = [cinterp, Tinterp, thinterp]
    return interpolators


def move_mesh(mesh, u, dt):
    """Move 2D mesh using finite element velocity vector field."""

    # This might be more efficient if I get it to work
    # Vex = mesh.ufl_domain().ufl_coordinate_element()
    # mesh.geometry.x[:,:mesh.geometry.dim] += u.x.array.reshape((-1, mesh.geometry.dim))*dt

    x = mesh.geometry.x
    gdim = mesh.geometry.dim

    points = x.T

    # Initialise cell search
    bb_treex = bb_tree(mesh, mesh.topology.dim)
    cells = []
    points_on_proc = []
    # Find cells whose bounding-box collide with the the points
    cell_candidates = compute_collisions_points(bb_treex, points.T)
    # Choose one of the cells that contains the point
    colliding_cells = compute_colliding_cells(mesh, cell_candidates, points.T)
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    # Evaluate u
    u_values = u.eval(points_on_proc, cells)

    # Add zero values for 3rd dimension
    if gdim == 2:
        z_values = np.zeros(u_values.shape[0])
        u_values = np.vstack((u_values.T, z_values)).T

    # Move mesh coordinates
    x += dt * u_values

    return mesh


def mso_femsolve(N, mesh, markers, Ms, coefs, stressobj, fricobj, tissue_parameters, fem_settings, solver=None):
    """Solve variational problem, MSO implementation"""
    # see monolithic direct solver at https://docs.fenicsproject.org/dolfinx/v0.4.1/python/demos/demo_stokes.html

    # unpack parameters
    mu, fr = tissue_parameters
    meshres, pen, THorder = fem_settings
    EP_marker, EE_marker, edge_marker = markers

    tdim = mesh.topology.dim
    fdim = tdim - 1

    # Create the function spaces for velocity and pressure
    P2 = ufl.VectorElement("Lagrange", mesh.ufl_cell(), THorder + 1)
    P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), THorder)
    TH = P2 * P1
    V = FunctionSpace(mesh, P2)
    Q = FunctionSpace(mesh, P1)
    W = FunctionSpace(mesh, TH)
    W1, _ = W.sub(1).collapse()

    # Boundary Velocity
    areaexp = assemble_scalar(
        form((coefs[0] * Ms[0] + coefs[1] * Ms[1] + coefs[2] * Ms[2] + coefs[3] * Ms[3] + coefs[4] * Ms[4]) * ufl.dx))
    circumf = assemble_scalar(form(Constant(mesh, PETSc.ScalarType(1.0)) * ufl.ds))
    un = areaexp / circumf
    UN = Constant(mesh, PETSc.ScalarType(areaexp / circumf))
    n = ufl.FacetNormal(mesh)

    # boundary_facets = ft.indices[(ft.values == edge_marker) ] #identify elements where the BC is applied
    # bc = dirichletbc(u_boundary, locate_dofs_topological((W.sub(0), V), fdim, boundary_facets), W.sub(0)) #cast bc in form ready to use in FEM method
    bcs = []

    # Active Stress
    Vstress = FunctionSpace(mesh, ufl.TensorElement("Lagrange", mesh.ufl_cell(), THorder))
    s = Function(Vstress)  # scalar function
    s.interpolate(stressobj)  # interpolate active stress onto function space

    # Friction (hair)
    fric = Function(Q)
    fric.interpolate(fricobj)

    # Define variational problem
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    stab = Constant(mesh, PETSc.ScalarType(1e-10))
    Mu = Constant(mesh, PETSc.ScalarType(mu))
    Pen = Constant(mesh, PETSc.ScalarType(pen))
    Fr = Constant(mesh, PETSc.ScalarType(fr))
    a = (Mu * ufl.inner(ufl.grad(u), ufl.grad(v)) - p * ufl.div(v) - ufl.div(
        u) * q - stab * p * q + Fr * fric * ufl.dot(u, v)) * ufl.dx + Pen * ufl.dot(u,
                                                                                    v) * ufl.ds  # This line throws a PETSc error in complex mode
    L = - ufl.inner(s, ufl.grad(v)) * ufl.dx - (
            coefs[0] * Ms[0] + coefs[1] * Ms[1] + coefs[2] * Ms[2] + coefs[3] * Ms[3] + coefs[4] * Ms[
        4]) * q * ufl.dx + Pen * UN * ufl.dot(n, v) * ufl.ds

    # Compute solution
    if solver is None:
        solver = ModifiedLinearProblem(
            petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
        print("Info: creating a new solver")
        sol = solver.solve(a, L, bcs)
    else:
        sol = solver.solve(a, L, bcs)

    '''
    # Output info of the solver
    lu_solver = solver.solver
    viewer = PETSc.Viewer().createASCII("lu_output.txt")
    lu_solver.view(viewer)
    solver_output = open("lu_output.txt", "r")
    for line in solver_output.readlines():
        print(line)
    '''

    # Split the mixed solution and collapse
    u = sol.sub(0).collapse()
    p = sol.sub(1).collapse()

    return u, p, W, un


def mso_meshing(N, meshres, currentradius, ps):
    """Mesh is circle with radius _currentradius, containing a polygon with vertices given by
    _meshmarginpoints, and optionally a rectangular subdomain _hair. Need 3 markers for EP, EE and hair region."""

    '''Create mesh using gmsh and return mesh, along with cell tags for EE (1), EP (2) and facet tags for 
    outerboundary (3)'''

    # markers
    EP_marker, EE_marker = 1, 2
    edge_marker = 3

    gdim = 2  # 2d mesh
    res_min = meshres  # resolution near cutout
    res_max = res_min  # resolution far away, pick uniform for now

    # handle correctly on parallel computer
    rank = MPI.COMM_SELF.rank

    # make sure gmsh is blank and initialized
    if not gmsh.isInitialized():
        gmsh.initialize()
    else:
        gmsh.finalize()
        gmsh.initialize()

    gmsh.option.setNumber("General.Verbosity",
                          2)  # output warnings and errors only, see https://gmsh.info/doc/texinfo/gmsh.html#Gmsh-API

    if rank == 0:
        '''Use GMSH to generate the mesh'''
        # (API at https://gitlab.onelab.info/gmsh/gmsh/blob/gmsh_4_10_2/api/gmsh.py)

        # Define margin as mesh contour
        meshmarginpoints = [0] * N
        meshmarginsegments = [0] * N
        for i in range(N):
            meshmarginpoints[i] = gmsh.model.occ.addPoint(ps[i, 0], ps[i, 1], 0, meshSize=meshres)
        for i in range(N):
            meshmarginsegments[i] = gmsh.model.occ.addLine(meshmarginpoints[i], meshmarginpoints[np.mod(i + 1, N)])
        marginmesh = gmsh.model.occ.addCurveLoop(meshmarginsegments)  # 1

        # Define embryo outer boundary
        outermesh_1D = gmsh.model.occ.addCircle(0, 0, 0, currentradius)  # N+1
        outermesh_2D = gmsh.model.occ.addCurveLoop([outermesh_1D])  # 2

        # Define EP and EE mesh
        epmesh = gmsh.model.occ.addPlaneSurface([marginmesh])  # 1
        eemesh = gmsh.model.occ.addPlaneSurface([outermesh_2D, marginmesh])  # 2

        gmsh.model.occ.synchronize()

        # to get GMSH to mesh the tissue, we add physical markers
        volumes = gmsh.model.getEntities(
            dim=gdim)  # entities are returned as a vector of (dim, tag) integer pairs. Here: [(2=gdim,1=tag)]
        boundaries = gmsh.model.getBoundary(volumes, oriented=False)  # [(1,N+1)]
        boundarytags = []
        for b in boundaries:
            boundarytags.append(b[1])

        # Define and tag physical groups
        gmsh.model.addPhysicalGroup(gdim, [epmesh], EP_marker, name="EP")  # (dim,[tags],tag of the physical group)
        gmsh.model.addPhysicalGroup(gdim, [eemesh], EE_marker, name="EE")
        gmsh.model.addPhysicalGroup(gdim - 1, boundarytags, edge_marker, name="OuterEdge")

        # Increase resolution
        #
        # Create distance field from obstacle.
        # Add threshold of mesh sizes based on the distance field
        # LcMax -                  /--------
        #                      /
        # LcMin -o---------/
        #        |         |       |
        #       Point    DistMin DistMax
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "EdgesList", boundaries[0])
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "IField", 1)
        gmsh.model.mesh.field.setNumber(2, "LcMin", res_min)
        gmsh.model.mesh.field.setNumber(2, "LcMax", res_max)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0.1 * currentradius)  # min res near outer boundary
        gmsh.model.mesh.field.setNumber(2, "DistMax", currentradius)  # max res near centre
        # We take the minimum of the two fields as the mesh size
        gmsh.model.mesh.field.add("Min", 3)
        gmsh.model.mesh.field.setNumbers(3, "FieldsList", [2])
        gmsh.model.mesh.field.setAsBackgroundMesh(3)

        # generate the mesh
        gmsh.model.mesh.generate(gdim)
        gmsh.model.mesh.optimize("Netgen")

    if rank == 0:
        '''import the mesh into dolfinx'''
        # Get mesh geometry
        x = extract_geometry(gmsh.model)  # xyz values of the nodes
        # Get mesh topology for each element
        topologies = extract_topology_and_markers(gmsh.model)
        # dictionary containing cell_data (markers) and topology for each element (1d boundary, 2d face) contained in
        # the mesh

        # Get information about each cell type from the msh files
        # and determine which of the cells has to highest topological dimension
        num_cell_types = len(topologies.keys())
        cell_information = {}
        cell_dimensions = np.zeros(num_cell_types, dtype=np.int32)
        for i, element in enumerate(topologies.keys()):
            properties = gmsh.model.mesh.getElementProperties(element)
            name, dim, order, num_nodes, local_coords, _ = properties
            cell_information[i] = {"id": element, "dim": dim, "num_nodes": num_nodes}
            cell_dimensions[i] = dim
        # Sort elements by ascending dimension
        perm_sort = np.argsort(cell_dimensions)

        # Broadcast cell type data and geometric dimension
        cell_id = cell_information[perm_sort[-1]]["id"]  # cells = units with the highest topological dimension
        tdim = cell_information[perm_sort[-1]]["dim"]  # tdim is there dimension (2)
        num_nodes = cell_information[perm_sort[-1]]["num_nodes"]  # 3 nodes (one cell has 3 vertices)
        cell_id, num_nodes = MPI.COMM_SELF.bcast([cell_id, num_nodes], root=0)  # communicate to other CPUs
        if tdim - 1 in cell_dimensions:  # for edges
            num_facet_nodes = MPI.COMM_SELF.bcast(cell_information[perm_sort[-2]]["num_nodes"],
                                                  root=0)  # 2 nodes per edge (end points)
            gmsh_facet_id = cell_information[perm_sort[-2]]["id"]  # facets are edges
            marked_facets = np.asarray(topologies[gmsh_facet_id]["topology"],
                                       dtype=np.int64)  # topology of the boundary
            facet_values = np.asarray(topologies[gmsh_facet_id]["cell_data"], dtype=np.int32)  # boundary marker values

        # extract the topology of the cell with the highest topological dimension from topologies
        cells = np.asarray(topologies[cell_id]["topology"], dtype=np.int64)
        cell_values = np.asarray(topologies[cell_id]["cell_data"], dtype=np.int32)

    else:
        cell_id, num_nodes = MPI.COMM_SELF.bcast([None, None], root=0)
        cells, x = np.empty([0, num_nodes], np.int64), np.empty([0, gdim])
        cell_values = np.empty((0,), dtype=np.int32)
        num_facet_nodes = MPI.COMM_SELF.bcast(None, root=0)
        marked_facets = np.empty((0, num_facet_nodes), dtype=np.int64)
        facet_values = np.empty((0,), dtype=np.int32)

    # Create distributed mesh
    ufl_domain = ufl_mesh(cell_id, gdim)  # generate 2D mesh in fenics
    # Permute cells from MSH to DOLFINx ordering
    gmsh_cell_perm = cell_perm_array(to_type(str(ufl_domain.ufl_cell())), num_nodes)
    cells = cells[:, gmsh_cell_perm]
    # distribute mesh to CPUs
    mesh = create_mesh(MPI.COMM_SELF, cells, x[:, :gdim], ufl_domain)
    # extract topological and facets dimension
    tdim = mesh.topology.dim
    fdim = tdim - 1
    # Permute facets from MSH to DOLFINx ordering
    facet_type = cell_entity_type(to_type(str(ufl_domain.ufl_cell())), fdim,
                                  0)  # Last argument is 0 as all facets are the same for triangles
    gmsh_facet_perm = cell_perm_array(facet_type, num_facet_nodes)
    marked_facets = np.asarray(marked_facets[:, gmsh_facet_perm], dtype=np.int64)

    # Create MeshTags for cell data
    local_entities, local_values = distribute_entity_data(mesh, tdim, cells, cell_values)
    mesh.topology.create_connectivity(tdim, 0)
    adj = adjacencylist(local_entities)
    ct = meshtags_from_entities(mesh, tdim, adj, np.int32(local_values))
    ct.name = "Cell tags"

    # Create MeshTags for facets (boundaries)
    local_entities, local_values = distribute_entity_data(mesh, fdim, marked_facets, facet_values)
    mesh.topology.create_connectivity(fdim, tdim)
    adj = adjacencylist(local_entities)
    ft = meshtags_from_entities(mesh, fdim, adj, np.int32(local_values))
    ft.name = "Facet tags"

    # close gmsh
    gmsh.finalize()

    # export markers
    markers = [EP_marker, EE_marker, edge_marker]
    meshtags = [ct, ft]

    print('Meshing successful')

    return mesh, meshtags, markers


def myos_init(N, Camp, c0, th, zeta, alpha):
    # Initial myosin profile
    c = 1.0 + zeta * np.tanh(alpha) + Camp * np.cos(th)
    # c = (1.0 + Camp)*np.cos(th)
    c[c < Camp] = Camp
    c[N] = c[0]
    c *= c0
    return c


def prepare_output(dataDir):
    # copy code to data folder
    if os.path.exists(dataDir):
        os.system('rm -rf ' + dataDir + '/*')
    else:
        os.makedirs(dataDir)
    # os.system('jupyter nbconvert --to script Whole_ve.ipynb')
    # os.system('mv Whole_ve.py '+dataDir)
    # prepare export
    dataout = pd.DataFrame(columns=['t', 'T', 'gd', 'c', 'l', 'ps', 'mvel'])
    return dataout


def redef_margin(N, margin_geom, l, cumt, interpolators):
    """Redefine margin points and derived quantities and interpolators"""

    # unwrap interpolators
    cinterp, Tinterp, thinterp = interpolators

    # create newps by interpolating margin at regular intervals
    newps = np.zeros((N, 2))
    for i in range(N):
        newps[i, :] = np.squeeze(np.array(margin_geom.interpolate(i * np.sum(l) / N).xy))
    newmargin_geom, newl, newlt, newcumt, newth, newta = updatelengths(newps, N)

    # interpolate myosin and tension to new edge midpoints
    newc = cinterp((newcumt + (newl[0] - l[0]) / 2) % cumt[N])
    newT = Tinterp((newcumt + (newl[0] - l[0]) / 2) % cumt[N])
    # the new polygonal contour may differ in length from the original one, so need to ensure periodicity manually
    newc[N] = newc[0]
    newT[N] = newT[0]

    # define new interpolators
    newinterp = interp_init(newcumt, newc, newT, newth)

    return newmargin_geom, newps, newl, newlt, newcumt, newth, newta, newc, newT, newinterp


def div_on_margin(Ms, coefs, mesh, ps, N):
    """See https://jorgensd.github.io/dolfinx-tutorial/chapter1/membrane_code.html"""

    # Reformat margin points coordinates, taking midpoints
    points = np.zeros((3, N))
    points[0, :] = (np.roll(ps[:, 0], -1) + ps[:, 0]) / 2
    points[1, :] = (np.roll(ps[:, 1], -1) + ps[:, 1]) / 2

    # Initialise cell search
    bb_treex = bb_tree(mesh, mesh.topology.dim)
    cells = []
    points_on_proc = []
    # Find cells whose bounding-box collide with the points
    cell_candidates = compute_collisions_points(bb_treex, points.T)
    # Choose one of the cells that contains the point
    colliding_cells = compute_colliding_cells(mesh, cell_candidates, points.T)
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    # Evaluate M
    M_values = np.asarray([M.eval(points_on_proc, cells) for M in Ms])
    cs = np.asarray([coef.value for coef in coefs])
    area_changes_on_margin = np.squeeze(np.tensordot(cs, M_values, axes=1))
    return area_changes_on_margin


def regulate_margin(N, ps, u_values, gd, c, lt, T, Ms, coefs, mesh, regulation_parameters, regulation_settings, dt,
                    mu=None,
                    ten_profile=None,
                    baseline_tension=None, t=None):
    """update margin locus, myosin and tension"""

    # unpack variables
    alpha, beta, zeta, lam, emax, tau, l_diff, ve_time, c0, Ts = regulation_parameters
    _, subd, regulate, remeshinterval = regulation_settings

    # update margin position (but don't use it downstream, only return it)
    newps = ps + dt * u_values

    # calculate correction to extension rate in tension regulation in response to area changes
    area_changes_on_margin = div_on_margin(Ms, coefs, mesh, ps, N)
    extension_rate_correction = area_changes_on_margin / 2  # divide by 2 since div(u) = exx + eyy roughly
    # gd = gd - extension_rate_correction

    # update myosin and tension
    dc = np.zeros(N + 1)
    dT = np.zeros(N + 1)
    if regulate:
        for it in range(subd):
            for i in range(N):
                dc[i] = 1 / tau * (
                        c0 + c0 * zeta * np.tanh(alpha - beta * gd[i] / (lam * emax)) - c[i] + l_diff ** 2 * (
                        lt[i % N] * c[(i - 1) % N] + lt[(i - 1) % N] * c[(i + 1) % N] - (
                        lt[i % N] + lt[(i - 1) % N]) * c[i]) / (
                                1 / 2 * (lt[i % N] + lt[(i - 1) % N]) * lt[i % N] * lt[(i - 1) % N]))
                dT[i] = c0 * Ts / (lam * emax * ve_time) * (
                            gd[i] - extension_rate_correction[i] - emax * np.tanh(lam * (T[i] / c[i] / Ts - 1)))
            dc[N] = dc[0]
            dT[N] = dT[0]
            c = c + dt / subd * dc
            T = T + dt / subd * dT
    else:
        T = tensions(newps, t, N, mu, ten_profile, baseline_tension)

    return newps, c, T


def report_status(t, un, margvel, gd, T, c):
    print("Time: %.3g" % t)
    print("Boundary velocity: %.4g" % un)
    print("Max margin velocity: %.4g" % np.amax(margvel))
    print("Max margin extension rate: %.4g" % np.amax(gd))
    print("Min margin extension rate: %.4g" % np.amin(gd))
    print("Max Tension: %.4g" % np.amax(T))
    print("Min Tension: %.4g" % np.amin(T))
    print("Max Myosin: %.4g" % np.amax(c))
    print("Min Myosin: %.4g" % np.amin(c))
    return


def save_fem(mesh, fn, dataDir, *args):
    """ Write the solution to XDMF files
    fn is the id of the simulation"""

    frame = inspect.currentframe()
    arg_names = frame.f_back.f_locals.keys()  # Get local variable names in the calling function

    for arg in args:
        for arg_name in arg_names:
            if frame.f_back.f_locals[arg_name] is arg:
                filename = f"{fn}_{arg_name}.xdmf"

                with XDMFFile(MPI.COMM_SELF, f"{dataDir}/FEMdata/{filename}", "w") as file_xdmf:
                    arg.x.scatter_forward()
                    file_xdmf.write_mesh(mesh)
                    try:
                        file_xdmf.write_function(arg)
                    except RuntimeError:  # Interpolate onto FE space on the same order as the mesh
                        dims = arg.ufl_shape
                        if len(dims) > 0:  # possible bug for tensor values
                            newV = FunctionSpace(mesh, ufl.VectorElement("Lagrange", mesh.ufl_cell(), 1))
                        else:
                            newV = FunctionSpace(mesh, ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1))
                        newarg = Function(newV)
                        arg_expr = EpibolyInterpolator(arg, mesh)
                        newarg.interpolate(arg_expr)
                        file_xdmf.write_function(newarg)

    return


def save_imagestack(dataDir, imagestack, fn):
    """ Save TIF stack """
    # WARNING ! Removes tif file before creating new one
    outfn = os.path.join(dataDir, fn + '.tif')
    if os.path.exists(outfn):
        os.system('rm ' + outfn)
    imagout = skio.concatenate_images(imagestack)
    skio.imsave(outfn, imagout)
    return


def save_marg(dataout, t, T, c, gd, l, ps, margvel):
    """Add row to dataout with data of this timestep"""
    data = pd.DataFrame(data={'t': t, 'T': [T], 'c': [c], 'gd': [gd], 'l': [l], 'ps': [ps], 'mvel': [margvel]})
    dataout = pd.concat([dataout, data], ignore_index=True)
    return dataout


def save_output(dataDir, dataout, fn):
    """Save DataFrame with margin variables to h5 file, uses h5py"""
    h5f = h5py.File(dataDir + '/' + fn + '.h5', 'w')
    for i, row in dataout.iterrows():
        grp = h5f.create_group(str(i))
        for label in dataout.columns:
            grp.create_dataset(label, data=row[label])
    h5f.close()
    print('Margin saved successfully to ' + dataDir + '/' + fn + '.h5')
    return


def save_settings(dataDir, general_parameters, regulation_parameters, initialisation_parameters, tissue_parameters,
                  geometry_parameters, regulation_settings, fem_settings, flags):
    """Save all settings to .txt file"""
    with open(dataDir + '/settings.txt', 'w') as file:
        file.write('general_parameters: N, t0 \n')
        file.write(str(general_parameters) + '\n\n')
        file.write('regulation_parameters: alpha, beta, zeta, lam, emax, tau, l_diff, ve_time, c0, Ts \n')
        file.write(str(regulation_parameters) + '\n\n')
        file.write('initialisation_parameters: c0, Camp, initialT, zeta, alpha \n')
        file.write(str(initialisation_parameters) + '\n\n')
        file.write('tissue_parameters: mu, fr*hair \n')
        file.write(str(tissue_parameters) + '\n\n')
        file.write('geometry_parameters: Er, r, d, offset \n')
        file.write(str(geometry_parameters) + '\n\n')
        file.write('#\n\n')
        file.write('regulation_settings: dt, subd, nrep, remeshinterval*remesh \n')
        file.write(str(regulation_settings) + '\n\n')
        file.write('fem_settings: meshres, pen, THorder \n')
        file.write(str(fem_settings) + '\n\n')
        file.write('#\n\n')
        file.write('flags: regulate, remesh, hair, echo, output \n')
        file.write(str(flags) + '\n\n')
    return


def tens_init(N, initialT, regulate, ps, t, mu, ten_profile, baseline_tension):
    # Scalar Tension
    if regulate:
        T = initialT * np.ones(N + 1)
    else:
        T = tensions(ps, t, N, mu, ten_profile, baseline_tension)
    return T


def tensions(ps, t, N, mu, ten_profile, baseline_tension):
    assert (ten_profile in ["unif", "syn"])
    maxTension = 2.7 * mu * (1 + np.tanh(2 * (t - .95) / 2.9)) / 2 * np.exp(-t / 8.3)  # peak tension, time in hours
    match ten_profile:
        case "syn":
            # Gaussian tension profile
            c = geom.Polygon(ps).centroid  # margin center of mass, can be biased posteriorly if not remeshed
            mp = (np.roll(ps, -2) + ps) / 2  # midpoints
            T = np.zeros(N + 1)
            for i in range(N):
                a = np.arctan2(mp[i, 0] - c.x, c.y - mp[i, 1])
                T[i] = np.exp(-np.square(a / .698) / 2)
            T[N] = T[0]  # make periodic
            T = T * maxTension
            T = T + np.ones(N + 1) * baseline_tension
        case "unif":
            T = np.ones(N + 1) * maxTension
            T = T + np.ones(N + 1) * baseline_tension
    return T


def tisstens_reg(vx, vy, margin, thinterp, Tinterp, N, l, cumt, d, t_dep):
    """Calculate tension at point x according to Gaussian profile with interpolation. Return active stress tensor"""
    "Implementation for regulation"
    s = np.zeros([4, vx.size])
    for i, x in enumerate(vx):
        xp = geom.Point(x, vy[i])  # cast as shapely object
        dist = margin.distance(xp)  # distance from margin
        almarg = margin.project(xp)  # distance along margin (absolute)
        the = thinterp((almarg - l[0] / 2) % cumt[N])
        tax = (np.cos(the), np.sin(the))
        # Scale with scalar tension
        ten = 1.0 / np.sqrt(2.0 * np.pi * pow(d, 2)) * np.exp(-pow(dist, 2) / (2.0 * d * d)) * Tinterp(
            (almarg - l[0] / 2) % cumt[N])
        # Define stress tensor
        s[0][i] = tax[0] * tax[0] * ten
        s[1][i] = tax[0] * tax[1] * ten  # off diagonal
        s[2][i] = tax[0] * tax[1] * ten  # off diagonal
        s[3][i] = tax[1] * tax[1] * ten
    return s


def update_coefs(mesh, t):
    """Update margin contraction coefficients (all of them, in case of remeshing)"""
    # Coefficients for epiboly contributions (matched to equal the values in Saadaoui et al. Table S1 at t=8)
    coef1 = Constant(mesh, PETSc.ScalarType(0.27 / 8))
    coef2 = Constant(mesh, PETSc.ScalarType(0))
    coef3 = Constant(mesh, PETSc.ScalarType(1.58 / 8))
    coef4 = Constant(mesh, PETSc.ScalarType(1.06 / 8))
    coef5 = Constant(mesh, PETSc.ScalarType(2.26 / 2.901 * (1 + np.tanh(2 * (t - 5.1) / 2.0)) / 2))
    coefs = [coef1, coef2, coef3, coef4, coef5]
    return coefs


def updatelengths(ps, N):
    # segment lengths l, cumulative lengths cumt, tangent vectors ta, and tangent angles th
    l = np.zeros(N)
    cumt = np.zeros(N + 1)  # cum length of segments measured from midpoint of segments
    ta = np.zeros((N, 2))  # tangent vectors
    th = np.zeros(N + 1)  # tangent angles
    margin_geom = geom.LinearRing(ps)  # define linear ring geometric object from array
    lvec = np.roll(ps, -2) - ps
    for i in range(N):
        l[i] = np.linalg.norm(lvec[i, :])
    lt = (l + np.roll(l, -1)) / 2
    for i in range(N):
        cumt[i + 1] = cumt[i] + lt[i]
        ta[i, :] = lvec[i, :] / l[i]
        th[i] = np.arctan2(ta[i, 1], ta[i, 0])
        if i > 0 and th[i] < th[i - 1] - np.pi:  # arbitrary threshold, may cause bug
            th[i] += 2 * np.pi
    th[N] = th[0] + 2 * np.pi  # make periodic
    return margin_geom, l, lt, cumt, th, ta


def vel_on_margin(u, ps, mesh, N):
    """See https://jorgensd.github.io/dolfinx-tutorial/chapter1/membrane_code.html"""

    # Reformat margin points coordinates
    points = np.zeros((3, N))
    points[0, :] = ps[:, 0]
    points[1, :] = ps[:, 1]

    # Initialise cell search
    bb_treex = bb_tree(mesh, mesh.topology.dim)
    cells = []
    points_on_proc = []
    # Find cells whose bounding-box collide with the points
    cell_candidates = compute_collisions_points(bb_treex, points.T)
    # Choose one of the cells that contains the point
    colliding_cells = compute_colliding_cells(mesh, cell_candidates, points.T)
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    # Evaluate u
    u_values = u.eval(points_on_proc, cells)
    return u_values
