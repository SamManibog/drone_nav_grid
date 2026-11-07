use std::cmp::Ordering;

use bit_set::BitSet;
use bit_vec::BitVec;

use crate::math::IVec2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Corner {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

impl Corner {
    /// gets the opposite corner
    pub fn opposite(self) -> Self {
        match self {
            Corner::TopLeft => Corner::BottomRight,
            Corner::TopRight => Corner::BottomLeft,
            Corner::BottomLeft => Corner::TopRight,
            Corner::BottomRight => Corner::TopLeft,
        }
    }

    /// gets the pair of nearest corners between the two cells
    pub fn corner_pair(cell1: IVec2, cell2: IVec2) -> (Corner, Corner) {
        match (cell1.x.cmp(&cell2.x), cell1.y.cmp(&cell2.y)) {
            (Ordering::Equal, Ordering::Equal)      => (Corner::TopLeft, Corner::TopLeft),

            (Ordering::Equal, Ordering::Less)       => (Corner::TopLeft, Corner::BottomLeft),
            (Ordering::Equal, Ordering::Greater)    => (Corner::BottomLeft, Corner::TopLeft),

            (Ordering::Less, Ordering::Equal)       => (Corner::TopRight, Corner::BottomRight),
            (Ordering::Greater, Ordering::Equal)    => (Corner::BottomRight, Corner::TopRight),

            (Ordering::Less, Ordering::Less)        => todo!(),
            (Ordering::Less, Ordering::Greater)     => todo!(),
            (Ordering::Greater, Ordering::Less)     => todo!(),
            (Ordering::Greater, Ordering::Greater)  => todo!(),
        }
    }

    fn from_is_top_is_left(is_top: bool, is_left: bool) -> Self {
        match (is_top, is_left) {
            (true, true)    => Corner::TopLeft,
            (true, false)   => Corner::TopRight,
            (false, true)   => Corner::BottomLeft,
            (false, false)  => Corner::BottomRight,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum WaypointMoveCmd {
    /// repeat command until cant anymore
    Inf(IVec2),

    /// repeat command once
    One(IVec2),
}

// movement pattern for top left to bottom right corner
// assumes positive y is down, positive x is right
const MOVEMENT_PATTERN: [WaypointMoveCmd; 6] = [
    WaypointMoveCmd::Inf(IVec2::new(0, 1)),
    WaypointMoveCmd::One(IVec2::new(1, 0)),
    WaypointMoveCmd::Inf(IVec2::new(0, -1)),
    WaypointMoveCmd::Inf(IVec2::new(1, 0)),
    WaypointMoveCmd::One(IVec2::new(0, 1)),
    WaypointMoveCmd::Inf(IVec2::new(-1, 0)),
];

#[derive(Debug, Clone)]
pub struct ScoreGrid {
    dimensions: IVec2,
    scores: Vec<f32>,
}

fn grid_index(pos: IVec2, x_dimension: isize) -> usize {
    (pos.x + x_dimension * pos.y) as usize
}

fn grid_pos(index: usize, x_dimension: isize) -> IVec2 {
    IVec2::new(
        (index as isize).rem_euclid(x_dimension),
        index as isize / x_dimension,
    )
}

impl ScoreGrid {
    pub fn new(score_matrix: Vec<Vec<f32>>) -> Result<Self, String> {
        if score_matrix.is_empty() {
            return Err("Score matrix cannot be empty".to_string());
        }
        let width = score_matrix[0].len();
        if width == 0 {
            return Err("Score matrix rows cannot be empty".to_string());
        }
        let height = score_matrix.len();

        let mut scores = Vec::with_capacity(width * height);
        for (y, row) in score_matrix.into_iter().enumerate() {
            if row.len() != width {
                return Err(format!(
                    "Row {} has length {}, expected {}",
                    y,
                    row.len(),
                    width
                ));
            }
            for val in row {
                if !val.is_finite() {
                    return Err("Score values must be finite".to_string());
                }
                scores.push(val);
            }
        }

        Ok(Self {
            dimensions: IVec2::new(width as isize, height as isize),
            scores,
        })
    }

    pub fn map_indices(&self, indices: impl Iterator<Item = usize>) -> impl Iterator<Item = IVec2> {
        indices.map(|i| self.node_pos(i))
    }

    pub fn node_index(&self, pos: IVec2) -> usize {
        grid_index(pos, self.dimensions.x)
    }

    pub fn contains_pos(&self, pos: IVec2) -> bool {
        pos.x >= 0 && pos.x < self.dimensions.x && pos.y >= 0 && pos.y < self.dimensions.y
    }

    pub fn node_pos(&self, index: usize) -> IVec2 {
        grid_pos(index, self.dimensions.x)
    }

    pub fn get_score(&self, pos: IVec2) -> f32 {
        self.scores[self.node_index(pos)]
    }

    fn get_score_by_index(&self, index: usize) -> f32 {
        self.scores[index]
    }

    pub fn node_count(&self) -> usize {
        (self.dimensions.x * self.dimensions.y) as usize
    }

    pub fn dimensions(&self) -> IVec2 {
        self.dimensions
    }
}

/// generates a snaking path through a grid with the given dimensions
pub fn snaking_path(dimensions: IVec2) -> Vec<usize> {
    let mut path = Vec::new();
    for i in 0..dimensions.x {
        if i % 2 == 0 {
            for j in 0..dimensions.y {
                path.push(grid_index(IVec2::new(i, j), dimensions.x))
            }
        } else {
            for j in (0..dimensions.y).into_iter().rev() {
                path.push(grid_index(IVec2::new(i, j), dimensions.x))
            }
        }
    }
    path
}

#[derive(Debug, Clone)]
pub struct PathPlanningProblem {
    /// the starting node (need not be explored as first move)
    pub start_position: IVec2,

    /// the score of each node (want paths that maximize score)
    pub scores: ScoreGrid,

    /// the cost of visiting a node
    pub node_cost: f32,

    /// distance_multiplier * distance between nodes is the cost of traversing an edge betweeen nodes
    pub distance_multiplier: f32,

    /// the maximum cost that can be accumulated before the path ends
    pub max_cost: f32,
}

impl PathPlanningProblem {
    /// ensures the problem definition is valid
    pub fn validate(&self) -> Result<(), String> {
        if self.max_cost < 0.0 {
            return Err("Max cost must be non-negative".to_string());
        }

        if self.node_cost < 0.0 || !self.node_cost.is_finite() {
            return Err("Node cost must be non-negative and finite".to_string());
        }

        if self.distance_multiplier < 0.0 || !self.distance_multiplier.is_finite() {
            return Err("Distance multiplier must be non-negative and finite".to_string());
        }

        if !self.scores.contains_pos(self.start_position) {
            return Err("Start position is outside grid dimensions".to_string());
        }

        Ok(())
    }

    /// Greedy algorithm that plans an exploration path between grid cells
    ///
    /// the next chosen cell is the one that maximizes (score/distance)
    ///
    /// O(n^2)
    pub fn greedy_prob_distance_ratio_soln(&self) -> Vec<usize> {
        let mut path = Vec::new();

        let mut remaining_cost = self.max_cost;
        let mut current_pos: IVec2 = self.start_position;

        let mut unexplored: BitSet<u64> =
            BitSet::from_bit_vec_general(BitVec::from_elem_general(self.scores.node_count(), true));

        while remaining_cost > 0.0 {
            let mut max_pos = current_pos;
            let mut max_sd_ratio = f32::NEG_INFINITY;
            let mut max_edge_cost = 0.0;

            'unex_loop: for i in unexplored.iter() {
                let i_pos = self.scores.node_pos(i);
                let rel_pos = current_pos - i_pos;
                let distance = f32::sqrt((rel_pos.x * rel_pos.x + rel_pos.y * rel_pos.y) as f32);
                let edge_cost = distance * self.distance_multiplier;
                if edge_cost >= remaining_cost {
                    continue 'unex_loop;
                }

                let score = self.scores.get_score_by_index(i);
                let sd_ratio = if distance == 0.0 {
                    score * 1000.0
                } else {
                    score / distance
                };

                if sd_ratio > max_sd_ratio {
                    max_sd_ratio = sd_ratio;
                    max_pos = i_pos;
                    max_edge_cost = edge_cost;
                }
            }

            // remaining nodes exhausted or unable to navigate to next node in time
            if !max_sd_ratio.is_finite() {
                return path;
            }

            // remaining nodes don't improve score
            if max_sd_ratio <= 0.0 {
                return path;
            }

            let max_idx = self.scores.node_index(max_pos);
            current_pos = max_pos;
            unexplored.remove(max_idx);
            path.push(max_idx);

            remaining_cost -= max_edge_cost + self.node_cost;
        }

        path
    }

    /// plans an efficient exploration path between grid cells
    ///
    /// algorithm overview:
    ///     - initial solutions to speed up convergence time:
    ///         - greedy algorithm that selects the highest probability unvisited node first
    ///         - greedy algorithm that selects the nearest unvisited node first
    ///         - greedy algorithm that selects unvisited node with greatest (probability/distance to node)
    ///         - greedy algorithm that spirals out from starting position
    ///     - end condition/output: a list of nodes representing visited
    ///     - the fitness function: sum of visited node scores
    ///     - mutations:
    ///         - swap order of visitation of random chunks
    ///         - change chunk to nearby, unvisited chunk
    pub fn genetic_soln(&self) -> Vec<usize> {
        const PASS_SIZE: usize = 8;

        // trivial case
        if self.max_cost <= 0.0 {
            return Vec::new();
        }

        todo!("complete genetic pathfinding in sprint 3")
    }
}
