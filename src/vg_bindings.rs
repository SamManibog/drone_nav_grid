use pyo3::{prelude::*, types::PyString};

use crate::{
    math::{Aabb, FVec3, IVec3},
    nav_grid::OccupiedExploredNavGrid,
};

#[pyclass(name = "NavGrid")]
pub struct PyNavGrid {
    inner: OccupiedExploredNavGrid,
}

fn py_parse_fvec3(py_any: &Bound<'_, PyAny>) -> PyResult<FVec3> {
    Ok(FVec3::new(
        py_any.get_item(0)?.extract::<f32>()?,
        py_any.get_item(1)?.extract::<f32>()?,
        py_any.get_item(2)?.extract::<f32>()?,
    ))
}

fn py_emit_fvec3(v: FVec3) -> Vec<f32> {
    vec![v.x, v.y, v.z]
}

fn py_emit_ivec3(v: IVec3) -> Vec<isize> {
    vec![v.x, v.y, v.z]
}

fn py_emit_faabb(aabb: Aabb<FVec3>) -> (Vec<f32>, Vec<f32>) {
    (py_emit_fvec3(aabb.min), py_emit_fvec3(aabb.max))
}

#[pymethods]
impl PyNavGrid {
    #[classattr]
    const OCCUPIED: i32 = 1;

    #[classattr]
    const EMPTY_EXPLORED: i32 = 0;

    #[classattr]
    const EMPTY_UNEXPLORED: i32 = -1;

    #[new]
    #[pyo3(signature = (min, max, unit_size))]
    fn __new__(min: &Bound<'_, PyAny>, max: &Bound<'_, PyAny>, unit_size: f32) -> PyResult<Self> {
        let min = py_parse_fvec3(min)?;
        let max = py_parse_fvec3(max)?;

        Ok(Self {
            inner: OccupiedExploredNavGrid::new(Aabb::new(min, max), unit_size),
        })
    }

    #[pyo3(signature = ())]
    pub fn unit_size(&self) -> f32 {
        self.inner.unit_size()
    }

    #[pyo3(signature = ())]
    pub fn dimensions(&self) -> Vec<f32> {
        py_emit_fvec3(self.inner.dimensions())
    }

    pub fn cell_dimensions(&self) -> Vec<isize> {
        py_emit_ivec3(self.inner.cell_dimensions())
    }

    pub fn aabb(&self) -> (Vec<f32>, Vec<f32>) {
        py_emit_faabb(self.inner.aabb())
    }

    #[pyo3(signature = (point))]
    pub fn is_in_bounds(&self, point: &Bound<'_, PyAny>) -> PyResult<bool> {
        let p = py_parse_fvec3(point)?;
        Ok(self.inner.is_in_bounds(p))
    }

    #[pyo3(signature = (position))]
    pub fn cell_at(&self, position: &Bound<'_, PyAny>) -> PyResult<i32> {
        let p = py_parse_fvec3(position)?;
        let cell = self.inner.cell_at(p);

        Ok(cell.into())
    }

    #[pyo3(signature = (position))]
    pub fn nearest_safe_position(&self, position: &Bound<'_, PyAny>) -> PyResult<Option<Vec<f32>>> {
        let pos = py_parse_fvec3(position)?;

        Ok(self.inner.nearest_safe_position(pos).map(py_emit_fvec3))
    }

    #[pyo3(signature = (start, endpoints, range, safety_margin))]
    pub fn handle_scan(
        &mut self,
        start: &Bound<'_, PyAny>,
        endpoints: &Bound<'_, PyAny>,
        range: f32,
        safety_margin: f32,
    ) -> PyResult<()> {
        let start = py_parse_fvec3(start)?;
        let mut ep_error = Ok(());

        let mut index = 0;
        let ep_len = endpoints.len()?;
        let ep_iter = std::iter::from_fn(|| {
            if index >= ep_len {
                return None;
            }

            match py_parse_fvec3(&endpoints.get_item(index).ok()?) {
                Ok(ep) => {
                    index += 1;
                    Some(ep)
                }
                Err(e) => {
                    ep_error = Err(e);
                    None
                }
            }
        });

        self.inner.handle_scan(start, ep_iter, range, safety_margin);
        ep_error
    }

    #[pyo3(signature = (start, end, search_size))]
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

    #[pyo3(signature = (x, z))]
    pub fn find_floor(&self, x: f32, z: f32) -> Option<Vec<f32>> {
        self.inner.find_floor(x, z).map(py_emit_fvec3)
    }

    #[pyo3(signature = (path))]
    pub fn export_occupied_obj(&self, path: &Bound<'_, PyString>) -> PyResult<()> {
        self.inner.export_occupied_obj(path.to_str()?)?;
        Ok(())
    }

    #[pyo3(signature = (path))]
    pub fn export_explored_obj(&self, path: &Bound<'_, PyString>) -> PyResult<()> {
        self.inner.export_explored_obj(path.to_str()?)?;
        Ok(())
    }
}

#[pymodule]
fn drone_nav_system(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNavGrid>()?;
    Ok(())
}
