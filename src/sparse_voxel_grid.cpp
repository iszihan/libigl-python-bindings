#include "default_types.h"
#include <igl/sparse_voxel_grid.h>
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/function.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace pyigl
{
  // Wrapper for the first overload of adjacency_list for triangle meshes
  auto sparse_voxel_grid(
    Eigen::MatrixXN p0,
    const std::function<double(const nb::DRef<Eigen::Matrix<double, 1, 3>>)>& scalar_func,
    double eps,
    int expected_number_of_cubes)
  {
    // assert_size_equals(p0, 3, "p0");
    Eigen::MatrixXN cs;
    Eigen::MatrixXN cv;
    Eigen::MatrixXI ci;
    Eigen::Matrix<double, 1, 3> p0_copy;
    if(p0.cols() == 1)
      p0_copy = p0.transpose();
    else
      p0_copy = p0;

    igl::sparse_voxel_grid(p0_copy, scalar_func, eps, expected_number_of_cubes, cs, cv, ci);
    return std::make_tuple(cs, cv, ci);
  }
}

// Bind the wrappers to the Python module
void bind_sparse_voxel_grid(nb::module_ &m)
{
  // Binding for triangle mesh adjacency_list
  m.def(
    "sparse_voxel_grid",
    &pyigl::sparse_voxel_grid,
    "p0"_a,
    "scalar_func"_a,
    "eps"_a,
    "expected_number_of_cubes"_a=0,
    R"(Given a point, p0, on an isosurface, construct a shell of epsilon sized cubes surrounding the surface.
These cubes can be used as the input to marching cubes.

  @param[in] p0  A 3D point on the isosurface surface defined by scalarFunc(x) = 0
  @param[in] scalarFunc  A scalar function from R^3 to R -- points which map to 0 lie
              on the surface, points which are negative lie inside the surface,
              and points which are positive lie outside the surface
  @param[in] eps  The edge length of the cubes surrounding the surface
  @param[in] expected_number_of_cubes  This pre-allocates internal data structures to speed things up
  @param[out] CS  #cube-vertices by 1 list of scalar values at the cube vertices
  @param[out] CV  #cube-vertices by 3 list of cube vertex positions
  @param[out] CI  #number of cubes by 8 list of indexes into CS and CV. Each row represents a cube.)");

}
