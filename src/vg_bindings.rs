#[pyo3::pymodule(name = "drone_nav_grid")]
pub mod drone_nav_grid {
    use pyo3::{
        PyAny,
        exceptions::{PyException, PyValueError},
        prelude::*,
    };

    use std::{path::PathBuf, str::FromStr, time::Duration};

    use crate::{
        math::{Aabb, FVec3, IVec2, IVec3},
        nav_grid::{ExplOccCell, OccupiedExploredNavGrid},
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
        const SAFETY_OCCUPIED: i32 = ExplOccCell::SafetyOccupied.into_i32();

        #[classattr]
        const EMPTY: i32 = ExplOccCell::Empty.into_i32();

        #[classattr]
        const UNEXPLORED: i32 = ExplOccCell::Unexplored.into_i32();

        #[classattr]
        const OBSTACLE: i32 = ExplOccCell::Obstacle.into_i32();

        /// constructs a new navigation grid using the given aabb
        #[new]
        fn __new__(
            min: &Bound<'_, PyAny>,
            max: &Bound<'_, PyAny>,
            unit_size: f32,
        ) -> PyResult<Self> {
            let min = py_parse_fvec3(min)?;
            let max = py_parse_fvec3(max)?;

            if !min
                .coords
                .iter()
                .chain(max.coords.iter())
                .all(|v| v.is_finite())
            {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "min and max must be finite.",
                ));
            }

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

        /// the size of a unit cell
        pub fn unit_size(&self) -> f32 {
            self.inner.unit_size()
        }

        /// the world dimensions of the grid
        pub fn dimensions(&self) -> Vec<f32> {
            py_emit_fvec3(self.inner.dimensions())
        }

        /// the number of cells in each dimension
        pub fn cell_dimensions(&self) -> Vec<isize> {
            py_emit_ivec3(self.inner.cell_dimensions())
        }

        /// the world aabb
        pub fn aabb(&self) -> (Vec<f32>, Vec<f32>) {
            py_emit_faabb(self.inner.aabb())
        }

        /// whether a world-coordinate point is in bounds
        pub fn is_in_bounds(&self, point: &Bound<'_, PyAny>) -> PyResult<bool> {
            let p = py_parse_fvec3(point)?;
            Ok(self.inner.is_in_bounds(p))
        }

        /// gets the state of the cell at the given position
        /// check against
        ///  - NavGrid.SAFETY_OCCUPIED
        ///  - NavGrid.UNEXPLORED
        ///  - NavGrid.OBSTACLE
        ///  - NavGrid.EMPTY
        pub fn cell_at(&self, position: &Bound<'_, PyAny>) -> PyResult<i32> {
            let p = py_parse_fvec3(position)?;
            let cell = self.inner.cell_at(p);

            Ok(cell.into())
        }

        /// gets the empty position nearest to the given position
        pub fn nearest_safe_position(
            &self,
            position: &Bound<'_, PyAny>,
        ) -> PyResult<Option<Vec<f32>>> {
            let pos = py_parse_fvec3(position)?;

            Ok(self.inner.nearest_safe_position(pos).map(py_emit_fvec3))
        }

        /// handles a simulated lidar scan were
        /// start - the sensor position
        /// endpoints - the detected lidar endpoint RELATIVE to the start position
        /// range - how far an endpoint can travel before being ignored
        /// safety_margin - how far to expand a detected obstacle for pathfinding purposes
        ///
        /// returns a list containing the positions of every newly-detected obstacle
        pub fn handle_scan(
            &mut self,
            start: &Bound<'_, PyAny>,
            endpoints: &Bound<'_, PyAny>,
            range: f32,
            safety_margin: f32,
        ) -> PyResult<Vec<Vec<f32>>> {
            let start = py_parse_fvec3(start)?;

            if !range.is_finite() || range < 0.0 {
                return Err(PyValueError::new_err(
                    "range must be finite and non-negative",
                ));
            }
            if !safety_margin.is_finite() || safety_margin < 0.0 {
                return Err(PyValueError::new_err(
                    "safety_margin must be finite and non-negative",
                ));
            }

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

        /// pathfind between two points using the given search size
        ///
        /// makes a best effort to get to the end point, returning the found path that
        /// ends nearest to the specified end point
        ///
        /// returns a tuple: (path, success)
        ///  - path is the best-effort path to the endpoint
        ///     - if the start or endpoints are invalid, path will be empty
        ///  - success is whether or not the endpoint was successfully found
        pub fn pathfind(
            &self,
            start: &Bound<'_, PyAny>,
            end: &Bound<'_, PyAny>,
            search_size: usize,
        ) -> PyResult<(Vec<Vec<f32>>, bool)> {
            let start = py_parse_fvec3(start)?;
            let end = py_parse_fvec3(end)?;

            let (path, success) = self.inner.pathfind(start, end, search_size);

            Ok((
                path.into_iter().map(|p| py_emit_fvec3(p)).collect(),
                success,
            ))
        }

        /// gets the y coordinate of the top of the lowest detected obstacle in this column
        /// returns NaN if no floor exists
        pub fn find_floor(&self, x: f32, z: f32) -> f32 {
            if let Some(vector) = self.inner.find_floor(x, z) {
                vector.y
            } else {
                f32::NAN
            }
        }

        /// attempts to export the obstacles detected in the navigation grid to
        /// a .obj file
        pub fn export_obstacle_obj(&self, path: &str) -> PyResult<()> {
            self.inner.export_obstacle_obj(PathBuf::from_str(path)?)?;
            Ok(())
        }

        /// attempts to export the occupied cells in the navigation grid to
        /// a .obj file
        pub fn export_occupied_obj(&self, path: &str) -> PyResult<()> {
            self.inner.export_occupied_obj(PathBuf::from_str(path)?)?;
            Ok(())
        }

        /// attempts to export the explored, unoccupied cells in the navigation grid to
        /// a .obj file
        pub fn export_explored_obj(&self, path: &str) -> PyResult<()> {
            self.inner.export_explored_obj(PathBuf::from_str(path)?)?;
            Ok(())
        }
    }

    #[pyclass(name = "PathPlanningProblem")]
    pub struct PyPathPlanningProblem {
        inner: PathPlanningProblem,
    }

    impl PyPathPlanningProblem {
        fn py_emit_soln(&self, soln: Vec<usize>) -> Vec<Vec<isize>> {
            self.inner
                .scores
                .map_indices(soln.into_iter())
                .map(py_emit_ivec2)
                .collect()
        }
    }

    #[pymethods]
    impl PyPathPlanningProblem {
        /// constructs a new path planning problem
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
            let scores_grid =
                ScoreGrid::new(scores).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

            let inner = PathPlanningProblem {
                start_position,
                scores: scores_grid,
                node_cost,
                distance_multiplier,
                max_cost,
            };

            inner
                .validate()
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

            Ok(Self { inner })
        }

        /// solves the path planning problem, returning the positions of cells
        /// optimization time is provided in milliseconds
        pub fn solve(&self, optimization_time: f64) -> Vec<Vec<isize>> {
            self.py_emit_soln(
                self.inner
                    .optimized_greedy_score_distance_ratio_soln(Duration::from_secs_f64(
                        optimization_time * 1000.0,
                    ))
                    .path,
            )
        }

        /// solves the path planning problem with a genetic algorithm of the given
        /// generation count
        pub fn genetic_solve(
            &self,
            generation_count: usize,
            population_size: usize,
        ) -> Vec<Vec<isize>> {
            self.py_emit_soln(self.inner.genetic_soln(generation_count, population_size))
        }
    }
}
