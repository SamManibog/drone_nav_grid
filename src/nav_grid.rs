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
    offset: FVec3,
    dimensions: FVec3,
    voxels: VoxelGrid<GridCell, ChunkSize64, ()>,
}
