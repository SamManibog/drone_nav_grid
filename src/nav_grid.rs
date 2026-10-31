use std::{collections::HashSet, io, path::Path};

use nalgebra::SimdPartialOrd;

use crate::{
    math::{Aabb, FVec3, IDistanceHeuristic, IDistanceSquared, IVec3, LineSegment, floor_fvec3},
    voxel_grid::{
        cell_identifiers::{CellID, CellPosition},
        chunks::ChunkSize64,
        voxel_grid::VoxelGrid,
    },
};

/// basic grid cells tracking the following qualities
///  - explored vs unexplored
///  - empty vs occupied
/// note that if a cell is occupied, it is necessarily explored
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ExplOccCell {
    #[default]
    EmptyUnexplored,
    EmptyExplored,
    Occupied,
}

impl ExplOccCell {
    pub fn is_explored(&self) -> bool {
        *self != Self::EmptyUnexplored
    }

    pub fn is_occupied(&self) -> bool {
        *self == Self::Occupied
    }
}

type NavCellID = CellID<ChunkSize64>;

/// a navigation grid of ExplOccCells
/// designed for set-path navigation strategy where
///  - navigation points are predetermined
///  - drone attempts to path to an "acceptably close" point with a limitted search size
///    - if pathing fails, another point is chosen
///    - if pathing succeeds, that point is removed from the set
///  - if pathing fails for all points in set, the region is considered fully explored
pub struct OccupiedExploredNavGrid {
    /// world position of the min corner of cell (0, 0, 0)
    offset: FVec3,

    /// world-space edge length of a (cubic) cell
    unit_size: f32,

    /// world-space dimensions of the entire voxel grid (cell_dimensions * unit_size)
    dimensions: FVec3,

    /// voxels in the nav grid
    voxels: VoxelGrid<ExplOccCell, ChunkSize64, ()>,
}

impl OccupiedExploredNavGrid {
    /// creates a new instance covering at least `aabb`, made of cubic cells of edge length `unit_size`.
    /// The grid is rounded up to a whole number of cells, so `self.aabb()` may be slightly larger
    /// than the requested aabb (it always starts at `aabb.min`).
    ///
    /// panics if `unit_size` is not finite and positive, or if the aabb is empty / not finite
    pub fn new(aabb: Aabb<FVec3>, unit_size: f32) -> Self {
        assert!(
            unit_size.is_finite() && unit_size > 0.0,
            "unit_size must be finite and positive"
        );

        let size = aabb.max - aabb.min;
        assert!(
            size.x.is_finite()
                && size.y.is_finite()
                && size.z.is_finite()
                && size.x > 0.0
                && size.y > 0.0
                && size.z > 0.0,
            "aabb must be finite and non-empty"
        );

        let cells_along = |len: f32| ((len / unit_size).ceil() as isize).max(1);
        let cell_dimensions = IVec3::new(
            cells_along(size.x),
            cells_along(size.y),
            cells_along(size.z),
        );

        Self {
            offset: aabb.min,
            unit_size,
            dimensions: FVec3::new(
                cell_dimensions.x as f32 * unit_size,
                cell_dimensions.y as f32 * unit_size,
                cell_dimensions.z as f32 * unit_size,
            ),
            voxels: VoxelGrid::from_cell_dimensions(cell_dimensions),
        }
    }

    /// translate world coordinates to voxel coordinates
    /// (a voxel coordinate of (i, j, k) is the min corner of cell (i, j, k))
    fn static_transform_point(offset: FVec3, unit_size: f32, point: FVec3) -> FVec3 {
        FVec3::from((point - offset) / unit_size)
    }

    /// voxel coordinates to world coordinates
    fn static_inv_transform_point(offset: FVec3, unit_size: f32, point: FVec3) -> FVec3 {
        offset + point.coords * unit_size
    }

    /// gets the cell position based on world coordinates, clamped into the grid
    fn static_cell_position_from_world_point(
        offset: FVec3,
        unit_size: f32,
        cell_dimensions: IVec3,
        point: FVec3,
    ) -> CellPosition {
        let grid_position = Self::static_transform_point(offset, unit_size, point);
        CellPosition(floor_fvec3(&grid_position).simd_clamp(
            IVec3::new(0, 0, 0),
            cell_dimensions - IVec3::new(1, 1, 1).coords,
        ))
    }

    /// gets the world coordinates of the center of a cell
    fn static_world_point_from_cell_center(
        offset: FVec3,
        unit_size: f32,
        cell_position: CellPosition,
    ) -> FVec3 {
        let grid_pos: FVec3 = cell_position.0.cast::<f32>() + FVec3::new(0.5, 0.5, 0.5).coords;
        Self::static_inv_transform_point(offset, unit_size, grid_pos)
    }

    /// gets the cell position based on world coordinates
    fn cell_position_from_world_point(&self, point: FVec3) -> CellPosition {
        Self::static_cell_position_from_world_point(
            self.offset,
            self.unit_size,
            self.cell_dimensions(),
            point,
        )
    }

    /// gets the world coordinates of the center of a cell
    fn world_point_from_cell_center(&self, cell_position: CellPosition) -> FVec3 {
        Self::static_world_point_from_cell_center(self.offset, self.unit_size, cell_position)
    }

    /// gets the size of a single grid unit
    pub fn unit_size(&self) -> f32 {
        self.unit_size
    }

    /// translate world coordinates to voxel coordinates
    fn transform_point(&self, point: FVec3) -> FVec3 {
        Self::static_transform_point(self.offset, self.unit_size, point)
    }

    /// voxel coordinates to world coordinates
    #[allow(dead_code)]
    fn inv_transform_point(&self, point: FVec3) -> FVec3 {
        Self::static_inv_transform_point(self.offset, self.unit_size, point)
    }

    /// gets the dimensions of this grid
    pub fn dimensions(&self) -> FVec3 {
        self.dimensions
    }

    /// gets the dimensions of this grid in cells
    pub fn cell_dimensions(&self) -> IVec3 {
        self.voxels.cell_dimensions()
    }

    /// gets the aabb of this grid
    pub fn aabb(&self) -> Aabb<FVec3> {
        Aabb {
            min: self.offset,
            max: self.offset + self.dimensions.coords,
        }
    }

    /// checks if a world position is in bounds
    pub fn is_in_bounds(&self, point: FVec3) -> bool {
        let aabb = self.aabb();

        point.x >= aabb.min.x
            && point.y >= aabb.min.y
            && point.z >= aabb.min.z
            && point.x < aabb.max.x
            && point.y < aabb.max.y
            && point.z < aabb.max.z
    }

    /// gets the state of the cell containing a world position
    /// (positions outside the grid are clamped to the nearest cell)
    pub fn cell_at(&self, point: FVec3) -> ExplOccCell {
        self.voxels
            .get_cell_id(self.cell_position_from_world_point(point))
            .and_then(|id| self.voxels.get_cell(id))
            .unwrap_or_default()
    }

    /// locate center point of the nearest explored empty cell
    pub fn nearest_safe_cell(&self, position: FVec3) -> Option<FVec3> {
        let cell_position = self.cell_position_from_world_point(position);

        let bfs_iter = self
            .voxels
            .breadth_first_iter(self.voxels.get_cell_id(cell_position)?);

        for id in bfs_iter {
            // unwrap is safe because bfs iter can only yield valid cells
            let state = self.voxels.get_cell(id).unwrap();

            // if cell is found, emit it
            if state.is_explored() && !state.is_occupied() {
                let cell_pos = self.voxels.get_cell_position(id).unwrap();
                return Some(self.world_point_from_cell_center(cell_pos));
            }
        }

        None
    }

    /// finds the nearest (manhattan, by flood fill) cell that is not occupied,
    /// which is `cell` itself when it is free
    fn nearest_unoccupied_cell(&self, cell: NavCellID) -> Option<NavCellID> {
        self.voxels
            .breadth_first_iter(cell)
            .find(|&id| self.voxels.get_cell(id).is_some_and(|c| !c.is_occupied()))
    }

    /// handle lidar scan
    ///
    ///  - start - world position of the scanner
    ///  - endpoints - world positions of the points hit (or max-range points) of each beam
    ///  - range - maximum sensor range, in world units. Beams ending further away are truncated
    ///    and do NOT produce obstacles
    ///  - safety_margin - world-space radius around each in-range hit that is marked occupied
    ///
    /// Cells along every beam become explored-empty (unless already occupied). Afterwards every
    /// cell whose center lies within `safety_margin` of an in-range hit cell's center is occupied.
    pub fn handle_scan(
        &mut self,
        start: FVec3,
        endpoints: impl Iterator<Item = FVec3>,
        range: f32,
        safety_margin: f32,
    ) {
        // map start, range, safety_margin to voxel coords (1 unit == 1 cell)
        let start_v = self.transform_point(start);
        let range_v = range / self.unit_size;
        let margin_v = (safety_margin / self.unit_size).max(0.0);
        let cell_dims = self.cell_dimensions();

        // set of cells containing in-range hits
        let mut hit_cells: HashSet<CellPosition> = HashSet::new();

        for endpoint in endpoints {
            let end_v = self.transform_point(endpoint);
            let delta = end_v - start_v;
            let len = delta.norm();
            if !len.is_finite() {
                continue;
            }

            // clamp the beam to the sensor range
            let (beam_end, in_range) = if len > range_v {
                (start_v + delta * (range_v / len), false)
            } else {
                (end_v, true)
            };

            // every cell on the path (in-range or not) is explored-empty unless already occupied
            self.voxels.segment_cell_initializing_for_each_mut(
                LineSegment {
                    start: start_v,
                    end: beam_end,
                },
                |_, cell, _| {
                    if *cell != ExplOccCell::Occupied {
                        *cell = ExplOccCell::EmptyExplored;
                    }
                },
            );

            // an in-range endpoint inside the grid is an obstacle hit
            if in_range {
                let hit = floor_fvec3(&end_v);
                if hit.x >= 0
                    && hit.y >= 0
                    && hit.z >= 0
                    && hit.x < cell_dims.x
                    && hit.y < cell_dims.y
                    && hit.z < cell_dims.z
                {
                    hit_cells.insert(CellPosition(hit));
                }
            }
        }

        // stamp a safety sphere (in cell units) around every hit cell.
        // compares squared distances between cell centers; both are integer cell positions
        let radius = margin_v.ceil() as isize;
        let margin_sq = margin_v * margin_v;

        for hit in hit_cells {
            let region = Aabb::new(
                IVec3::new(hit.x - radius, hit.y - radius, hit.z - radius),
                IVec3::new(hit.x + radius + 1, hit.y + radius + 1, hit.z + radius + 1),
            );

            self.voxels
                .aabb_cell_initializing_for_each_mut(region, |pos, cell, _| {
                    if IDistanceSquared::apply(*pos, *hit) as f32 <= margin_sq {
                        *cell = ExplOccCell::Occupied;
                    }
                });
        }
    }

    ///

    /// generate a a path using funnelled astar algorithm to get as near as possible to the given point
    /// attempts to get as near as possible to the end point within the given search size
    /// if no path exists, returns an empty vector
    /// if end is out of bounds, returns an empty vector
    /// if start is out of bounds, the nearest in-bounds position is chosen instead as a starting point
    /// if start is inside an occupied cell (e.g. within a safety margin), the nearest unoccupied
    /// cell is used as the starting point so the agent can path out of the margin
    /// if the end cell itself is occupied, no path is returned
    pub fn pathfind(&self, start: FVec3, end: FVec3, search_size: usize) -> Vec<FVec3> {
        if !self.is_in_bounds(end) {
            return Vec::with_capacity(0);
        }

        let Some(start_cell) = self
            .voxels
            .get_cell_id(self.cell_position_from_world_point(start))
            .and_then(|id| self.nearest_unoccupied_cell(id))
        else {
            return Vec::with_capacity(0);
        };
        let Some(end_cell) = self
            .voxels
            .get_cell_id(self.cell_position_from_world_point(end))
        else {
            return Vec::with_capacity(0);
        };

        self.voxels
            .funnelled_astar::<IDistanceSquared>(start_cell, end_cell, search_size, |_, c| {
                ExplOccCell::is_occupied(&c)
            })
            .into_iter()
            .map(|p| self.world_point_from_cell_center(self.voxels.get_cell_position(p).unwrap()))
            .collect()
    }

    /// writes a Wavefront .obj mesh of every occupied cell
    pub fn export_occupied_obj(&self, path: impl AsRef<Path>) -> io::Result<()> {
        self.voxels
            .export_obj(|_, cstate, _| cstate.is_occupied(), path)
    }

    /// writes a Wavefront .obj mesh of every explored, unoccupied cell
    pub fn export_explored_obj(&self, path: impl AsRef<Path>) -> io::Result<()> {
        self.voxels
            .export_obj(|_, cstate, _| cstate == ExplOccCell::EmptyExplored, path)
    }
}
