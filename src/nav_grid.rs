use crate::{
    math::FVec3,
    voxel_grid::{chunks::ChunkSize64, voxel_grid::VoxelGrid},
};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum GridCell {
    #[default]
    EmptyUnexplored,
    EmptyExplored,
    Occupied,
}

pub struct NavGrid {
    /// offset of cell (0, 0, 0)
    offset: FVec3,

    /// dimensions of entire voxel grid
    dimensions: FVec3,

    /// voxels in the nav grid
    voxels: VoxelGrid<GridCell, ChunkSize64, ()>,
}

impl NavGrid {
    /// translate world coordinates to voxel coordinates
    fn transform_point(&self, point: FVec3) -> FVec3 {
        todo!()
    }

    /// voxel coordinates to world coordinates
    fn inv_transform_point(&self, point: FVec3) -> FVec3 {
        todo!()
    }

    /// handle lidar scan
    pub fn handle_scan(&mut self, start: FVec3, endpoints: impl Iterator<Item = FVec3>, length: f32) {
        todo!()
    }

    /// generate a a path using funnelled astar algorithm to get as near as possible to the given point
    /// attempts to get as near as possible to the end point within the given search size
    /// if no path exists, returns an empty vector
    pub fn pathfind(&self, start: FVec3, end: FVec3, search_size: usize) -> Vec<FVec3> {
        todo!()
    }
}
