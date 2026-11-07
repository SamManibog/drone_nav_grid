#[pyo3::pymodule(name = "drone_nav_grid")]
pub mod drone_nav_grid {
    use pyo3::{PyAny, exceptions::PyException, prelude::*};

    use std::{path::PathBuf, str::FromStr};

    use crate::{
        math::{Aabb, FVec3, IVec2, IVec3},
        nav_grid::OccupiedExploredNavGrid,
        path_planner::{PathPlanningProblem, ScoreGrid},
    };

    fn py_parse_fvec3(py_any: &Bound<'_, PyAny>) -> PyResult<FVec3> {
        Ok(FVec3::new(
            py_any.get_item(0)?.extract::<f32>()?,
            py_any.get_item(1)?.extract::<f32>()?,
            py_any.get_item(2)?.extract::<f32>()?,
        ))
    }

    fn py_parse_ivec2(py_any: &Bound<'_, PyAny>) -> PyResult<IVec2> {
        Ok(IVec2::new(
            py_any.get_item(0)?.extract::<isize>()?,
            py_any.get_item(1)?.extract::<isize>()?,
        ))
    }

    fn py_emit_fvec3(v: FVec3) -> Vec<f32> {
        vec![v.x, v.y, v.z]
    }

    fn py_emit_ivec3(v: IVec3) -> Vec<isize> {
        vec![v.x, v.y, v.z]
    }

    fn py_emit_ivec2(v: IVec2) -> Vec<isize> {
        vec![v.x, v.y]
    }

    fn py_emit_faabb(aabb: Aabb<FVec3>) -> (Vec<f32>, Vec<f32>) {
        (py_emit_fvec3(aabb.min), py_emit_fvec3(aabb.max))
    }

    #[pyclass(name = "NavGrid")]
    pub struct PyNavGrid {
        inner: OccupiedExploredNavGrid,
    }

    #[pymethods]
    impl PyNavGrid {
        #[classattr]
        const OCCUPIED: i32 = 1;

        #[classattr]
        const EMPTY_EXPLORED: i32 = 0;

        #[classattr]
        const EMPTY_UNEXPLORED: i32 = -1;

        /// constructs a new navigation grid using the given aabb
        #[new]
        fn __new__(
            min: &Bound<'_, PyAny>,
            max: &Bound<'_, PyAny>,
            unit_size: f32,
        ) -> PyResult<Self> {
            let min = py_parse_fvec3(min)?;
            let max = py_parse_fvec3(max)?;

            if Aabb::<FVec3>::new(min, max).is_empty() {
                return PyResult::Err(PyErr::new::<PyException, _>("min must be less than max."));
            }
            if unit_size <= 0.0 || !unit_size.is_finite() {
                return PyResult::Err(PyErr::new::<PyException, _>(
                    "unit_size must be positive and finite.",
                ));
            }

            Ok(Self {
                inner: OccupiedExploredNavGrid::new(Aabb::new(min, max), unit_size),
            })
        }

        pub fn unit_size(&self) -> f32 {
            self.inner.unit_size()
        }

        pub fn dimensions(&self) -> Vec<f32> {
            py_emit_fvec3(self.inner.dimensions())
        }

        pub fn cell_dimensions(&self) -> Vec<isize> {
            py_emit_ivec3(self.inner.cell_dimensions())
        }

        pub fn aabb(&self) -> (Vec<f32>, Vec<f32>) {
            py_emit_faabb(self.inner.aabb())
        }

        pub fn is_in_bounds(&self, point: &Bound<'_, PyAny>) -> PyResult<bool> {
            let p = py_parse_fvec3(point)?;
            Ok(self.inner.is_in_bounds(p))
        }

        pub fn cell_at(&self, position: &Bound<'_, PyAny>) -> PyResult<i32> {
            let p = py_parse_fvec3(position)?;
            let cell = self.inner.cell_at(p);

            Ok(cell.into())
        }

        pub fn nearest_safe_position(
            &self,
            position: &Bound<'_, PyAny>,
        ) -> PyResult<Option<Vec<f32>>> {
            let pos = py_parse_fvec3(position)?;

            Ok(self.inner.nearest_safe_position(pos).map(py_emit_fvec3))
        }

        pub fn handle_scan(
            &mut self,
            start: &Bound<'_, PyAny>,
            endpoints: &Bound<'_, PyAny>,
            range: f32,
            safety_margin: f32,
        ) -> PyResult<Vec<Vec<f32>>> {
            let start = py_parse_fvec3(start)?;

            let mut ep_vec: Vec<FVec3> = Vec::new();
            for ep in endpoints.try_iter()? {
                let point = ep?;
                ep_vec.push(py_parse_fvec3(&point)?);
            }

            Ok(self
                .inner
                .handle_scan(start, ep_vec.into_iter(), range, safety_margin)
                .into_iter()
                .map(py_emit_fvec3)
                .collect())
        }

        pub fn pathfind(
            &self,
            start: &Bound<'_, PyAny>,
            end: &Bound<'_, PyAny>,
            search_size: usize,
        ) -> PyResult<Vec<Vec<f32>>> {
            let start = py_parse_fvec3(start)?;
            let end = py_parse_fvec3(end)?;

            Ok(self
                .inner
                .pathfind(start, end, search_size)
                .into_iter()
                .map(|p| py_emit_fvec3(p))
                .collect())
        }

        pub fn find_floor(&self, x: f32, z: f32) -> Option<Vec<f32>> {
            self.inner.find_floor(x, z).map(py_emit_fvec3)
        }

        pub fn export_obstacle_obj(&self, path: &str) -> PyResult<()> {
            self.inner.export_obstacle_obj(PathBuf::from_str(path)?)?;
            Ok(())
        }

        pub fn export_occupied_obj(&self, path: &str) -> PyResult<()> {
            self.inner.export_occupied_obj(PathBuf::from_str(path)?)?;
            Ok(())
        }

        pub fn export_explored_obj(&self, path: &str) -> PyResult<()> {
            self.inner.export_explored_obj(PathBuf::from_str(path)?)?;
            Ok(())
        }
    }

    #[pyclass(name = "PathPlanningProblem")]
    pub struct PyPathPlanningProblem {
        inner: PathPlanningProblem,
    }

    #[pymethods]
    impl PyPathPlanningProblem {
        #[new]
        #[pyo3(signature = (*, start_position, scores, node_cost, distance_multiplier, max_cost))]
        fn __new__(
            start_position: &Bound<'_, PyAny>,
            scores: Vec<Vec<f32>>,
            node_cost: f32,
            distance_multiplier: f32,
            max_cost: f32,
        ) -> PyResult<Self> {
            // Extract start_position from a Python tuple/list (x, y)
            let start_position = py_parse_ivec2(start_position)?;

            // Build the ScoreGrid, handling any validation errors as Python ValueErrors
            let scores_grid = ScoreGrid::new(scores)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

            let inner = PathPlanningProblem {
                start_position,
                scores: scores_grid,
                node_cost,
                distance_multiplier,
                max_cost,
            };

            inner.validate().map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

            Ok(Self { inner })
        }

        pub fn solve(&self) -> Vec<Vec<isize>> {
            let soln = self.inner.greedy_prob_distance_ratio_soln();
            self.inner
                .scores
                .map_indices(soln.into_iter())
                .map(py_emit_ivec2)
                .collect()
        }
    }
}
