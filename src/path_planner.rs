use std::{
    cmp::Ordering,
    collections::BinaryHeap,
    ops::Range,
    time::{Duration, Instant},
};

use crate::{
    heap_entries::{MaxHeapEntry, MinHeapEntry},
    math::IVec2,
};
use bit_set::BitSet;
use bit_vec::BitVec;
use priority_queue::PriorityQueue;
use rand::{Rng, RngExt, seq::SliceRandom};

#[derive(Debug, Clone)]
pub struct ScoreGrid {
    dimensions: IVec2,
    scores: Vec<f32>,
    total_score: f32,
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

fn grid_dist(pos1: IVec2, pos2: IVec2) -> f32 {
    let rel_pos = pos1 - pos2.coords;
    f32::sqrt((rel_pos.x * rel_pos.x + rel_pos.y * rel_pos.y) as f32)
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

        let mut total_score = 0.0;

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
                total_score += val;
                scores.push(val);
            }
        }

        Ok(Self {
            dimensions: IVec2::new(width as isize, height as isize),
            scores,
            total_score,
        })
    }

    pub fn total_score(&self) -> f32 {
        self.total_score
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

    /// returns the score at the given position
    ///
    /// # Panics
    /// panics with a descriptive message if `pos` is outside the grid
    /// (a negative coordinate would otherwise wrap into a huge index)
    pub fn get_score(&self, pos: IVec2) -> f32 {
        assert!(self.contains_pos(pos), "position is outside the score grid");
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
    /// the number of genetic iterations to run before recalculating scores and costs
    const UNDRIFT_GENETIC_ITERATIONS: usize = 10;

    /// the smallest population the genetic algorithm will run with. guarantees the
    /// elite/performance/diversity bookkeeping always has individuals to work with
    const MIN_POPULATION_SIZE: usize = 16;

    /// the ratio of completely preserved individuals to total population size per generation
    /// elite individuals are not mutated, but may be selected for crossover
    const ELITE_FRACTION: f32 = 0.025;

    /// the ratio of top-performing individuals selected to pass on to the next generation
    const PERFORMANCE_FRACTION: f32 = 0.4 - Self::ELITE_FRACTION;

    /// the ratio of non-top-performing individuals selected randomly to pass on to the next generation
    /// used to diversify the gene pool
    const DIVERSITY_FRACTION: f32 = 0.1;

    /// the probability any given gene is mutated
    const MUTATION_PROBABILITY: f32 = 0.01;

    /// the probability of a crossover
    const CROSSOVER_PROBABILITY: f32 = 0.75;

    /// the maximum number of Dinkelbach iterations used to find the max-efficiency substring
    /// (converges in a handful of iterations in practice)
    const MAX_EFFICIENCY_ITERATIONS: usize = 16;

    const PASS_FRACTION: f32 =
        Self::ELITE_FRACTION + Self::PERFORMANCE_FRACTION + Self::DIVERSITY_FRACTION;

    /// evaluated at module level (see the `const _` items below the impl block);
    /// associated consts are otherwise never evaluated unless referenced
    pub const PASS_SIZE_ASSERTION: () = assert!(Self::PASS_FRACTION < 1.0_f32);
    pub const ELITE_FRACTION_ASSERTION: () =
        assert!(Self::ELITE_FRACTION <= Self::PERFORMANCE_FRACTION);

    const NUDGE_MUTATION_DELTAS: &[IVec2] = &[
        IVec2::new(2, -1),
        IVec2::new(-2, 1),
        IVec2::new(-1, 1),
        IVec2::new(0, 2),
        IVec2::new(-1, -1),
        IVec2::new(2, 0),
        IVec2::new(1, 1),
        IVec2::new(2, -2),
        IVec2::new(2, 2),
        IVec2::new(-1, -2),
        IVec2::new(-1, 0),
        IVec2::new(0, -2),
        IVec2::new(-2, 2),
        IVec2::new(0, -1),
        IVec2::new(1, -1),
        IVec2::new(-2, -1),
        IVec2::new(-2, 0),
        IVec2::new(1, 2),
        IVec2::new(2, 1),
        IVec2::new(1, 0),
        IVec2::new(-2, -2),
        IVec2::new(1, -2),
        IVec2::new(0, 1),
        IVec2::new(-1, 2),
    ];

    /// ensures the problem definition is valid
    pub fn validate(&self) -> Result<(), String> {
        // `!(x >= 0.0)` also rejects NaN
        if !(self.max_cost >= 0.0) || !self.max_cost.is_finite() {
            return Err("Max cost must be non-negative and finite".to_string());
        }

        if !(self.node_cost >= 0.0) || !self.node_cost.is_finite() {
            return Err("Node cost must be non-negative and finite".to_string());
        }

        if !(self.distance_multiplier >= 0.0) || !self.distance_multiplier.is_finite() {
            return Err("Distance multiplier must be non-negative and finite".to_string());
        }

        if !self.scores.contains_pos(self.start_position) {
            return Err("Start position is outside grid dimensions".to_string());
        }

        Ok(())
    }

    /// the cost saved by removing node `i` from the path described by `adj`, including the node
    /// cost. handles head (no prev), tail (no next), and a lone node (neither).
    ///
    /// s(i) = score of node i
    /// c(i, j) = cost of edge between i and j (distance scaled by the distance multiplier)
    /// r = root/start position
    /// C = node cost
    ///
    /// let path p = x, ..., y, ..., z
    ///   head:  c(r, x) + c(x, x + 1) - c(r, x + 1) + C
    ///   inner: c(y - 1, y) + c(y, y + 1) - c(y - 1, y + 1) + C
    ///   tail:  c(z - 1, z) + C
    fn removal_savings(&self, adj: &LinkedPath, i: usize) -> f32 {
        let (prev_opt, next_opt) = adj.neighbours(i);
        let pos = self.scores.node_pos(i);
        let prev_pos = if let Some(prev) = prev_opt {
            self.scores.node_pos(prev)
        } else {
            self.start_position
        };

        let dist = if let Some(next) = next_opt {
            let next_pos = self.scores.node_pos(next);
            grid_dist(prev_pos, pos) + grid_dist(pos, next_pos) - grid_dist(prev_pos, next_pos)
        } else {
            grid_dist(prev_pos, pos)
        };

        dist * self.distance_multiplier + self.node_cost
    }

    /// smart deduplication step for a single duplicated node: keeps only the occurrence whose
    /// removal would save the least edge cost, and adjusts score and cost accordingly
    ///
    /// the incremental cost update treats each removed occurrence independently, which is
    /// inexact when removed occurrences are adjacent; `smart_dedup` recomputes score and cost
    /// afterwards to correct for that
    ///
    /// O(n)
    fn smart_dedup_node_helper(&self, soln: &mut SolutionInfo, node_index: usize) {
        // Record path indices of duplicates
        let mut path_indices = Vec::new();
        for (i, node) in soln.path.iter().copied().enumerate() {
            if node == node_index {
                path_indices.push(i);
            }
        }

        // edge cost (in cost units, i.e. scaled by the distance multiplier) saved by removing
        // the occurrence at path index `idx`
        let comparative_path_idx_savings = |idx: usize| -> f32 {
            let pos = self.scores.node_pos(soln.path[idx]);

            let prev_pos = if idx == 0 {
                self.start_position
            } else {
                self.scores.node_pos(soln.path[idx - 1])
            };

            let dist = if idx == soln.path.len() - 1 {
                grid_dist(prev_pos, pos)
            } else {
                let next_pos = self.scores.node_pos(soln.path[idx + 1]);
                grid_dist(prev_pos, pos) + grid_dist(pos, next_pos) - grid_dist(prev_pos, next_pos)
            };

            dist * self.distance_multiplier
        };

        // Find the occurrence with the least savings
        let mut min_savings = comparative_path_idx_savings(path_indices[0]);
        let mut min_path_index = path_indices[0];

        let mut savings_cache = Vec::with_capacity(path_indices.len());
        savings_cache.push(min_savings);

        for path_index in path_indices.iter().copied().skip(1) {
            let savings = comparative_path_idx_savings(path_index);
            savings_cache.push(savings);

            if savings < min_savings {
                min_savings = savings;
                min_path_index = path_index;
            }
        }

        // Find which position in `path_indices` corresponds to our `min_path_index`
        let min_i = path_indices
            .iter()
            .position(|&idx| idx == min_path_index)
            .unwrap();

        // Adjust for removed node cost and score
        let node_score = self.scores.get_score_by_index(node_index);
        let removed_count = (path_indices.len() - 1) as f32;
        soln.score -= removed_count * node_score;
        soln.cost -= removed_count * self.node_cost;

        // Remove duplicates, adjust for edge cost/savings using occurrence index `i`
        let mut i = 0;
        soln.path.retain(|n_index| {
            if *n_index == node_index {
                let keep = i == min_i;
                if !keep {
                    soln.cost -= savings_cache[i];
                }
                i += 1;
                keep
            } else {
                true
            }
        });
    }

    /// smart deduplication step for a solution with duplicate nodes
    ///
    /// score and cost are recomputed from scratch afterwards, since the per-node incremental
    /// updates are inexact when removed duplicates are adjacent to each other
    ///
    /// O(n^2) in the worst case
    fn smart_dedup(&self, soln: &mut SolutionInfo) {
        // construct list of duplicate nodes
        let mut dupes = Vec::new();
        {
            let mut found = BitSet::<u64>::with_capacity_general(self.scores.node_count());
            let mut pushed = BitSet::<u64>::with_capacity_general(self.scores.node_count());
            for index in soln.path.iter().copied() {
                if found.contains(index) {
                    if !pushed.contains(index) {
                        dupes.push(index);
                        pushed.insert(index);
                    }
                } else {
                    found.insert(index);
                }
            }
        };

        if dupes.is_empty() {
            return;
        }

        // smart dedup on specific node
        for dupe_node in dupes {
            self.smart_dedup_node_helper(soln, dupe_node);
        }

        self.update_score_and_cost(soln);
    }

    /// shrinks a solution until it is valid, prioritizing the removal of inefficient edges
    ///
    /// the solution's path must not contain duplicate nodes (see `smart_dedup`) and its
    /// score and cost must be correct
    ///
    /// O(nlogn)
    fn greedy_efficiency_path_shrink(&self, soln: &mut SolutionInfo) {
        let mut path = LinkedPath::from_slice(&soln.path, self.scores.node_count());

        // efficiency = score created per unit of cost that removal would save.
        // The queue is a max-heap, so negate: the least efficient node pops first.
        // Removals that save nothing are treated as infinitely efficient (never
        // chosen early).
        let priority = |adj: &LinkedPath, i: usize| -> FPriority {
            let saved = self.removal_savings(adj, i);
            let efficiency = if saved <= 0.0 {
                f32::INFINITY
            } else {
                self.scores.get_score_by_index(i) / saved
            };
            FPriority(-efficiency)
        };

        // push nodes to priority queue (least efficient nodes are removed first)
        let mut queue = PriorityQueue::with_capacity(path.len());
        for node in path.iter() {
            queue.push(node, priority(&path, node));
        }

        // shrink until solution has valid cost or all nodes are removed
        while soln.cost > self.max_cost {
            let Some((node_index, _)) = queue.pop() else {
                break;
            };

            let (prev_opt, next_opt) = path.neighbours(node_index);

            // compute before unlinking
            soln.cost -= self.removal_savings(&path, node_index);
            soln.score -= self.scores.get_score_by_index(node_index);

            path.remove(node_index);

            // only the two neighbours' savings change
            if let Some(prev) = prev_opt {
                queue.change_priority(&prev, priority(&path, prev));
            }
            if let Some(next) = next_opt {
                queue.change_priority(&next, priority(&path, next));
            }
        }

        soln.path = path.to_vec();
    }

    fn random_spanning_path<T: Rng>(&self, rng: &mut T) -> SolutionInfo {
        let node_count = self.scores.node_count();
        if node_count == 0 {
            return SolutionInfo {
                path: Vec::new(),
                score: 0.0,
                cost: 0.0,
            };
        }

        let mut unselected = Vec::from_iter(0..node_count);
        unselected.shuffle(rng);

        let mut path = Vec::with_capacity(self.scores.node_count());

        let mut total_cost = self.node_cost * node_count as f32;
        let mut total_score = 0.0;

        let mut prev_position = self.start_position;

        while let Some(node_index) = unselected.pop() {
            let node_position = self.scores.node_pos(node_index);
            total_cost += grid_dist(prev_position, node_position) * self.distance_multiplier;
            total_score += self.scores.get_score_by_index(node_index);

            path.push(node_index);
            prev_position = node_position;
        }

        SolutionInfo {
            path,
            score: total_score,
            cost: total_cost,
        }
    }

    /// Greedy algorithm that plans an exploration path between grid cells
    ///
    /// designed to provide seeds for genetic solutions
    ///
    /// first generates a random, likely cost-exceeding path between nodes
    /// then, removes the least efficient entries until cost is no longer exceeded
    ///
    /// O(nlogn)
    fn greedy_random_subtractive_soln<T: Rng>(&self, rng: &mut T) -> SolutionInfo {
        let mut soln = self.random_spanning_path(rng);
        self.greedy_efficiency_path_shrink(&mut soln);
        soln
    }

    /// Greedy algorithm that plans an exploration path between grid cells
    ///
    /// the next chosen cell is the with the greatest score that doesnt exceed cost
    ///
    /// O(nlogn)
    pub fn greedy_score_soln(&self) -> Vec<usize> {
        let mut path = Vec::new();

        let mut heap = BinaryHeap::from_iter(
            (0..self.scores.node_count())
                .map(|i| MaxHeapEntry(self.scores.get_score_by_index(i), i)),
        );

        let mut remaining_cost = self.max_cost;

        let mut current_pos = self.start_position;

        // check the remaining cost *before* popping so no entry is discarded needlessly
        while remaining_cost > 0.0 {
            let Some(MaxHeapEntry(_, index)) = heap.pop() else {
                break;
            };

            let i_pos = self.scores.node_pos(index);
            let distance = grid_dist(current_pos, i_pos);
            let edge_cost = distance * self.distance_multiplier;
            if edge_cost + self.node_cost >= remaining_cost {
                continue;
            }

            path.push(index);

            current_pos = i_pos;
            remaining_cost -= edge_cost + self.node_cost;
        }

        path
    }

    /// modifies the given path either by shrinking to stay within max cost, or by expanding to
    /// use up remaining cost and increase score
    ///
    /// the path must not contain duplicate nodes, and its score and cost must be correct
    ///
    /// current method for shrinking is to iteratively remove the nodes that create the least
    /// score per unit of cost saved (see `greedy_efficiency_path_shrink`)
    ///
    /// current method for expansion is running greedy distance/prob algorithm
    fn greedy_path_adjust(&self, entry: &mut SolutionInfo) {
        if entry.cost > self.max_cost {
            self.greedy_efficiency_path_shrink(entry);
        }

        // only need to attempt expansion if a new node could feasibly be introduced.
        // distinct grid nodes are always at least distance 1 apart, so the cheapest possible
        // new node costs at least `node_cost + distance_multiplier`
        if entry.cost + self.node_cost + self.distance_multiplier < self.max_cost {
            self.greedy_score_distance_ratio_path_expand(entry)
        }
    }

    /// shrinks the given path to a valid cost by removing lowest-scoring nodes first,
    /// updating the entry's path, score and cost
    ///
    /// the path must not contain duplicate nodes, and its score and cost must be correct
    ///
    /// (currently unused; `greedy_efficiency_path_shrink` is preferred)
    #[allow(dead_code)]
    fn greedy_score_path_shrink(&self, entry: &mut SolutionInfo) {
        let mut path = LinkedPath::from_slice(&entry.path, self.scores.node_count());

        // construct a min heap of score
        let mut heap = BinaryHeap::from_iter(
            entry
                .path
                .iter()
                .map(|&i| MinHeapEntry(self.scores.get_score_by_index(i), i)),
        );

        // pop from score while invalid
        while entry.cost > self.max_cost {
            let Some(MinHeapEntry(_, node_index)) = heap.pop() else {
                break;
            };

            // compute before unlinking
            entry.cost -= self.removal_savings(&path, node_index);
            entry.score -= self.scores.get_score_by_index(node_index);

            path.remove(node_index);
        }

        entry.path = path.to_vec();
    }

    /// expands the given path based on score/distance greedy algorithm
    ///
    /// new nodes are appended to the end of the path, starting from the path's last node
    /// (or the start position if the path is empty)
    ///
    /// assumes the passed score and cost are correct and the path has no duplicate nodes
    ///
    /// updates the entry's path, score and cost
    fn greedy_score_distance_ratio_path_expand(&self, entry: &mut SolutionInfo) {
        let mut remaining_cost = self.max_cost - entry.cost;
        let mut current_pos: IVec2 = entry
            .path
            .last()
            .map_or_else(|| self.start_position, |x| self.scores.node_pos(*x));

        // construct bitset of unexplored nodes (all nodes not already on the path)
        let mut unexplored: BitSet<u64> =
            BitSet::from_bit_vec_general(BitVec::from_elem_general(self.scores.node_count(), true));
        for i in entry.path.iter() {
            unexplored.remove(*i);
        }

        while remaining_cost > 0.0 {
            if self.greedy_score_distance_ratio_soln_helper(
                &mut current_pos,
                &mut unexplored,
                &mut remaining_cost,
                &mut entry.score,
                &mut entry.path,
            ) {
                break;
            }
        }

        entry.cost = self.max_cost - remaining_cost;
    }

    /// recalculates the generation entry's score and cost
    fn update_score_and_cost(&self, entry: &mut SolutionInfo) {
        let (score, cost) = self.path_score_and_cost(&entry.path);
        entry.score = score;
        entry.cost = cost;
    }

    /// returns the path's score and cost in that order (score, cost)
    pub fn path_score_and_cost(&self, path: &[usize]) -> (f32, f32) {
        let mut score = 0.0;

        let node_cost = self.node_cost * path.len() as f32;
        let mut edge_cost = 0.0;

        let mut prev_pos = self.start_position;

        for curr in path.iter() {
            score += self.scores.get_score_by_index(*curr);

            let curr_pos = self.scores.node_pos(*curr);
            let dist = grid_dist(curr_pos, prev_pos);

            edge_cost += dist * self.distance_multiplier;

            prev_pos = curr_pos;
        }

        (score, node_cost + edge_cost)
    }

    /// Greedy algorithm that plans an exploration path between grid cells
    ///
    /// the next chosen cell is the one that maximizes (score + 1)^2 / distance
    ///
    /// returns (path, score)
    ///
    /// O(n^2)
    pub fn greedy_score_distance_ratio_soln(&self) -> SolutionInfo {
        let mut path = Vec::new();

        let mut remaining_cost = self.max_cost;
        let mut current_pos: IVec2 = self.start_position;
        let mut total_score: f32 = 0.0;

        let mut unexplored: BitSet<u64> =
            BitSet::from_bit_vec_general(BitVec::from_elem_general(self.scores.node_count(), true));

        while remaining_cost > 0.0 {
            if self.greedy_score_distance_ratio_soln_helper(
                &mut current_pos,
                &mut unexplored,
                &mut remaining_cost,
                &mut total_score,
                &mut path,
            ) {
                break;
            }
        }

        SolutionInfo {
            path,
            score: total_score,
            cost: self.max_cost - remaining_cost,
        }
    }

    /// runs an iteration of greedy_score_distance_ratio_soln,
    /// returns true if the current path should be returned
    fn greedy_score_distance_ratio_soln_helper(
        &self,
        current_pos: &mut IVec2,
        unexplored: &mut BitSet<u64>,
        remaining_cost: &mut f32,
        total_score: &mut f32,
        path: &mut Vec<usize>,
    ) -> bool {
        let mut max_pos = *current_pos;
        let mut max_sd_ratio = f32::NEG_INFINITY;
        let mut max_edge_cost = 0.0;
        let mut max_score = 0.0;

        'unex_loop: for i in unexplored.iter() {
            let i_pos = self.scores.node_pos(i);
            let distance = grid_dist(*current_pos, i_pos);
            let edge_cost = distance * self.distance_multiplier;
            if edge_cost + self.node_cost >= *remaining_cost {
                continue 'unex_loop;
            }

            let score = self.scores.get_score_by_index(i);
            let sd_ratio = if distance == 0.0 {
                score * 1000.0
            } else {
                let score_adjusted = score + 1.0;
                score_adjusted * score_adjusted / distance
            };

            if sd_ratio > max_sd_ratio {
                max_sd_ratio = sd_ratio;
                max_pos = i_pos;
                max_edge_cost = edge_cost;
                max_score = score;
            }
        }

        // remaining nodes exhausted or unable to navigate to next node in time
        if !max_sd_ratio.is_finite() {
            return true;
        }

        // remaining nodes don't improve score
        if max_sd_ratio <= 0.0 {
            return true;
        }

        let max_idx = self.scores.node_index(max_pos);
        *current_pos = max_pos;
        unexplored.remove(max_idx);
        path.push(max_idx);
        *total_score += max_score;

        *remaining_cost -= max_edge_cost + self.node_cost;

        false
    }

    /// plans an efficient exploration path between grid cells
    ///
    /// algorithm overview:
    ///     - initial population: greedy random subtractive solutions (a random spanning path
    ///       shrunk by repeatedly removing the least efficient node until it fits the cost)
    ///     - end condition/output: a list of nodes representing visited nodes, after `passes`
    ///       generations
    ///     - the fitness function: score^2 / cost for positive-scoring individuals (rewards both
    ///       score and efficiency); non-positive scores rank below all positive ones
    ///     - crossover: heuristic crossover on disjoint pairs, inserting each parent's most
    ///       efficient substring into the other parent, followed by deduplication and greedy
    ///       shrinking/expansion so offspring are valid
    ///     - mutation: nudge random nodes to a nearby, unvisited grid cell; individuals pushed
    ///       over max cost by a mutation are greedily shrunk back to validity
    ///     - selection each generation:
    ///         - a few elites are preserved untouched
    ///         - top-performing individuals plus a random sample of the rest survive
    ///         - the population is refilled to its original size by breeding survivors
    ///     - output: the highest-scoring valid individual, after recomputing every individual's
    ///       score and cost to remove accumulated floating point drift
    pub fn genetic_soln(&self, generations: usize, population_size: usize) -> Vec<usize> {
        // trivial cases
        if self.max_cost <= 0.0 || self.scores.node_count() == 0 {
            return Vec::new();
        }

        let mut rng = rand::rng();

        let population_size = population_size.max(Self::MIN_POPULATION_SIZE);
        let elite_count = f32::ceil(population_size as f32 * Self::ELITE_FRACTION) as usize;
        let performance_count =
            f32::ceil(population_size as f32 * Self::PERFORMANCE_FRACTION) as usize;
        let diversity_count = f32::ceil(population_size as f32 * Self::DIVERSITY_FRACTION) as usize;

        // always leave at least one individual that does not pass, so the diversity sampling
        // below always has a non-empty range to draw from
        let pass_count = (performance_count + diversity_count).min(population_size - 1);
        let performance_count = performance_count.min(pass_count);
        let elite_count = elite_count.min(performance_count);

        // higher is better. non-positive scores are ranked by score (below every positive
        // individual) so that squaring cannot make negative scores look good
        let fitness = |soln: &SolutionInfo| {
            let v = if soln.score <= 0.0 {
                soln.score
            } else if soln.cost <= 0.0 {
                f32::INFINITY
            } else {
                soln.score * soln.score / soln.cost
            };
            if v.is_nan() { f32::NEG_INFINITY } else { v }
        };

        // get initial population
        let mut population = Vec::with_capacity(population_size);
        for _ in 0..population_size {
            population.push(self.greedy_random_subtractive_soln(&mut rng));
        }

        for generation_num in 0..generations {
            if generation_num % Self::UNDRIFT_GENETIC_ITERATIONS == 0 {
                for soln in population.iter_mut() {
                    self.update_score_and_cost(soln);
                }
            }

            // cull population
            population.sort_unstable_by(|l, r| fitness(r).total_cmp(&fitness(l)));

            // copy elites to vector
            let elites: Vec<SolutionInfo> = population[0..elite_count].to_vec();

            // pick some random solutions for more genetic diversity
            for swap_index in performance_count..pass_count {
                let swap_with = rng.random_range(swap_index..population.len());
                population.swap(swap_index, swap_with);
            }

            // drop those that didnt pass
            population.truncate(pass_count);

            // crossover among the survivors
            self.genetic_crossover(&mut rng, &mut population);

            // refill the population back to its original size by breeding random survivors
            let target_size = population_size - elites.len();
            let parent_count = population.len();
            while population.len() < target_size {
                let p1 = rng.random_range(0..parent_count);
                let mut p2 = rng.random_range(0..parent_count);
                if parent_count > 1 && p2 == p1 {
                    p2 = (p1 + 1) % parent_count;
                }

                let [off1, off2] =
                    self.efficiency_heuristic_crossover(&mut rng, &population[p1], &population[p2]);
                population.push(off1);
                if population.len() < target_size {
                    population.push(off2);
                }
            }

            // mutate, then repair any individuals that mutation pushed over max cost
            for soln in population.iter_mut() {
                self.nudge_mutation(&mut rng, soln);
                if soln.cost > self.max_cost {
                    self.greedy_efficiency_path_shrink(soln);
                }
            }

            // re-introduce elites (never mutated)
            population.extend(elites);
        }

        // remove accumulated drift and make sure every individual is valid before choosing
        for soln in population.iter_mut() {
            self.update_score_and_cost(soln);
            while soln.cost > self.max_cost {
                self.greedy_efficiency_path_shrink(soln);
                self.update_score_and_cost(soln);
            }
        }

        // return the valid population member with greatest score
        population
            .into_iter()
            .max_by(|a, b| a.score.total_cmp(&b.score))
            .map(|soln| soln.path)
            .unwrap_or_default()
    }

    /// performs genetic crossover on an entire generation
    ///
    /// the population is shuffled and paired into disjoint pairs, each of which crosses over
    /// with probability CROSSOVER_PROBABILITY (an odd individual out is left untouched)
    fn genetic_crossover<T: Rng>(&self, rng: &mut T, population: &mut Vec<SolutionInfo>) {
        population.shuffle(rng);
        for p1_idx in (0..population.len().saturating_sub(1)).step_by(2) {
            let p2_idx = p1_idx + 1;

            if rng.random_range(0.0..1.0_f32) >= Self::CROSSOVER_PROBABILITY {
                continue;
            }

            let [off1, off2] =
                self.efficiency_heuristic_crossover(rng, &population[p1_idx], &population[p2_idx]);

            population[p1_idx] = off1;
            population[p2_idx] = off2;
        }
    }

    /// detects the max-efficiency substring of nodes in a path
    /// returning the end-exclusive range
    ///
    /// efficiency of a substring is its total score divided by its total cost, where the cost
    /// is the node cost of each node plus the scaled edge cost between consecutive nodes *inside*
    /// the substring (the edge entering the substring is excluded, as it disappears once the
    /// substring is extracted and inserted elsewhere)
    ///
    /// maximizing a ratio is not directly solvable with Kadane's algorithm, so this uses
    /// Dinkelbach's method: given a candidate efficiency `lambda`, Kadane's algorithm finds the
    /// substring maximizing (score - lambda * cost); if that is positive, the substring beats
    /// `lambda`, so it becomes the new candidate. this converges to the exact optimum in a few
    /// iterations
    ///
    /// returns an empty range for an empty path
    ///
    /// O(n) per iteration
    ///
    /// used for heuristic crossover
    fn max_efficiency_substring(&self, path: &[usize]) -> Range<usize> {
        let n = path.len();
        if n == 0 {
            return 0..0;
        }

        // keeps the ratio finite if every cost is zero
        const MIN_COST: f32 = 1e-12;

        let scores: Vec<f32> = path
            .iter()
            .map(|&node| self.scores.get_score_by_index(node))
            .collect();

        // cost of extending a substring so that it includes node k (k >= 1)
        let mut extend_costs = vec![self.node_cost; n];
        for k in 1..n {
            let dist = grid_dist(
                self.scores.node_pos(path[k - 1]),
                self.scores.node_pos(path[k]),
            );
            extend_costs[k] = dist * self.distance_multiplier + self.node_cost;
        }

        let ratio = |range: &Range<usize>| -> f32 {
            let score: f32 = scores[range.clone()].iter().sum();
            let cost: f32 =
                self.node_cost + extend_costs[range.start + 1..range.end].iter().sum::<f32>();
            score / cost.max(MIN_COST)
        };

        // best substring maximizing sum(score - lambda * cost), and that sum
        let kadane = |lambda: f32| -> (Range<usize>, f32) {
            let mut best = 0..1;
            let mut best_value = f32::NEG_INFINITY;

            let mut cur_value = 0.0;
            let mut cur_start = 0;

            for k in 0..n {
                let start_value = scores[k] - lambda * self.node_cost;
                let extend_value = cur_value + scores[k] - lambda * extend_costs[k];

                if k == 0 || start_value >= extend_value {
                    cur_value = start_value;
                    cur_start = k;
                } else {
                    cur_value = extend_value;
                }

                if cur_value > best_value {
                    best_value = cur_value;
                    best = cur_start..k + 1;
                }
            }

            (best, best_value)
        };

        let mut best = 0..1;
        let mut lambda = ratio(&best);

        for _ in 0..Self::MAX_EFFICIENCY_ITERATIONS {
            let (candidate, value) = kadane(lambda);
            if value <= 1e-6 {
                break;
            }

            let candidate_ratio = ratio(&candidate);
            if candidate_ratio <= lambda {
                break;
            }

            best = candidate;
            lambda = candidate_ratio;
        }

        best
    }

    /// performs a heuristic double crossover using the max_efficiency_substring algorithm
    ///
    /// the first child inserts parent1's best genes on parent2's genome
    /// and the second child inserts parent2's best genes onto parent1's genome
    fn efficiency_heuristic_crossover<T: Rng>(
        &self,
        rng: &mut T,
        parent1: &SolutionInfo,
        parent2: &SolutionInfo,
    ) -> [SolutionInfo; 2] {
        let p1_best = self.max_efficiency_substring(&parent1.path);
        let p2_best = self.max_efficiency_substring(&parent2.path);

        [
            self.heuristic_crossover_insert_helper(
                rng,
                parent1,
                p1_best.clone(),
                parent2,
                p2_best.clone(),
            ),
            self.heuristic_crossover_insert_helper(rng, parent2, p2_best, parent1, p1_best),
        ]
    }

    /// performs a heuristic crossover using the max_efficiency_substring algorithm
    ///
    /// the child is a copy of dst_parent with src_parent's best genes (`src_range`) inserted at a
    /// random position that does not split dst_parent's own best genes (`dst_safe_range`),
    /// followed by deduplication and greedy adjustment so the child is valid
    fn heuristic_crossover_insert_helper<T: Rng>(
        &self,
        rng: &mut T,
        src_parent: &SolutionInfo,
        src_range: Range<usize>,
        dst_parent: &SolutionInfo,
        dst_safe_range: Range<usize>,
    ) -> SolutionInfo {
        // choose an insertion index in 0..=len that does not fall strictly inside the safe
        // range. valid indices are 0..=safe.start and safe.end..=len, which are
        // (safe.start + 1) + (len - safe.end + 1) choices in total
        let len = dst_parent.path.len();
        let before_count = dst_safe_range.start + 1;
        let after_count = len - dst_safe_range.end + 1;

        let choice = rng.random_range(0..before_count + after_count);
        let index = if choice < before_count {
            choice
        } else {
            dst_safe_range.end + (choice - before_count)
        };

        // insert src genes into that position
        let path: Vec<usize> = dst_parent.path[0..index]
            .iter()
            .copied()
            .chain(src_parent.path[src_range].iter().copied())
            .chain(dst_parent.path[index..len].iter().copied())
            .collect();

        let mut offspring = SolutionInfo {
            path,
            score: f32::NEG_INFINITY,
            cost: f32::INFINITY,
        };

        // ensure offspring is valid
        self.update_score_and_cost(&mut offspring);
        self.smart_dedup(&mut offspring);
        self.greedy_path_adjust(&mut offspring);

        offspring
    }

    /// replaces the node at `index` of the path with `new_node`, updating the solution's
    /// score and cost incrementally
    fn nudge_node(&self, soln: &mut SolutionInfo, index: usize, new_node: usize) {
        let old_node = soln.path[index];
        let old_pos = self.scores.node_pos(old_node);

        let new_pos = self.scores.node_pos(new_node);

        let prev_pos = if index == 0 {
            self.start_position
        } else {
            self.scores.node_pos(soln.path[index - 1])
        };

        // modify cost
        let distance_delta = if index + 1 == soln.path.len() {
            // last node: there is no outgoing edge
            grid_dist(prev_pos, new_pos) - grid_dist(prev_pos, old_pos)
        } else {
            let next_pos = self.scores.node_pos(soln.path[index + 1]);

            let init_dist = grid_dist(prev_pos, old_pos) + grid_dist(old_pos, next_pos);
            let final_dist = grid_dist(prev_pos, new_pos) + grid_dist(new_pos, next_pos);

            final_dist - init_dist
        };
        soln.cost += distance_delta * self.distance_multiplier;

        // modify score
        soln.score +=
            self.scores.get_score_by_index(new_node) - self.scores.get_score_by_index(old_node);

        // apply the change
        soln.path[index] = new_node;
    }

    /// mutates a path by moving a few random nodes to a nearby, unvisited grid cell
    ///
    /// mutations may cause the individual to exceed max cost; callers are responsible for
    /// repairing such individuals (see `genetic_soln`)
    fn nudge_mutation<T: Rng>(&self, rng: &mut T, entry: &mut SolutionInfo) {
        // record which grid nodes are already on the path
        let mut used = BitSet::<u64>::with_capacity_general(self.scores.node_count());
        for i in &entry.path {
            used.insert(*i);
        }

        // note that entry length will not change by nudging entries
        'gene_loop: for path_index in 0..entry.path.len() {
            if rng.random_range(0.0_f32..1.0_f32) >= Self::MUTATION_PROBABILITY {
                continue 'gene_loop;
            }

            let old_node = entry.path[path_index];
            let old_position = self.scores.node_pos(old_node);

            // attempt each possible nudge once, starting at a random one, until one is valid
            let start_delta = rng.random_range(0..Self::NUDGE_MUTATION_DELTAS.len());
            for offset in 0..Self::NUDGE_MUTATION_DELTAS.len() {
                let delta_index = (start_delta + offset) % Self::NUDGE_MUTATION_DELTAS.len();
                let new_position = old_position + Self::NUDGE_MUTATION_DELTAS[delta_index].coords;

                if !self.scores.contains_pos(new_position) {
                    continue;
                }

                let new_node = self.scores.node_index(new_position);
                if used.contains(new_node) {
                    continue;
                }

                self.nudge_node(entry, path_index, new_node);

                used.remove(old_node);
                used.insert(new_node);

                continue 'gene_loop;
            }
        }
    }

    /// 2-opt optimized greedy score-distance ratio solution
    pub fn optimized_greedy_score_distance_ratio_soln(
        &self,
        two_opt_compute_time: Duration,
    ) -> SolutionInfo {
        let mut soln = self.greedy_score_distance_ratio_soln();
        self.two_opt_soln(&mut soln, two_opt_compute_time);
        soln
    }

    /// greedy solution with a multi-pass strategy
    ///
    /// the general strategy is as follows
    ///  - collect nodes into high-scoring clusters (see `detect_score_clusters`)
    ///    - determine the top scoring node not yet in a cluster
    ///    - if adjacent (8-connected) to a cluster, add it to that cluster (joining all the
    ///      clusters it is adjacent to into one), otherwise it starts a cluster of its own
    ///    - repeat until max(1, node_count * TOP_NODE_FRACTION) nodes have been added to a
    ///      cluster, or no remaining node has a positive score
    ///  - determine path between clusters
    ///    - for each cluster, determine
    ///      - total score
    ///      - excess = average score of its nodes above the baseline (the average score of all
    ///        nodes on the grid), floored at zero
    ///      - expected cost: the node cost is exact, and the edge cost is lower bounded by
    ///        distance_multiplier * (cluster node count - 1), as the nodes are adjacent
    ///        (the edge entering the cluster is accounted for when choosing the cluster)
    ///    - greedily select the next cluster that maximizes
    ///      (total score + 1) * (excess + 1)^2 / travel_cost
    ///      - travel_cost = distance(nearest node in current, nearest node in next) *
    ///        distance_multiplier (for the first cluster, the distance is from the start
    ///        position). the cost of the cluster itself is not weighed when ranking clusters
    ///      - selection stops once the estimated cost (which does include each selected
    ///        cluster's own lower-bound cost) exceeds max_cost, as the path built from the
    ///        selected clusters could not fit any more of them
    ///  - build a minimal path between the key nodes of the clusters based on the following key nodes
    ///      - the "entry" node of each cluster
    ///      - the "exit" node of each cluster
    ///      - the "center" node of each cluster; that is, the node in the cluster closest to the cluster's
    ///        score-weighted center of mass
    ///    the minimal path visits each cluster's entry, then center, then exit. if it exceeds
    ///    max_cost, nodes are removed from the end of the path until it fits
    ///  - run greedy insertion path expand on that minimal path
    ///
    /// cluster selection is O(T^2) in the number of clustered nodes T in the worst case,
    /// and gap filling is O(n * path_len) per inserted node
    pub fn greedy_cluster_inserting_solution(&self) -> SolutionInfo {
        /// the ratio of nodes (to total node count) that are added to clusters
        const TOP_NODE_FRACTION: f32 = 0.01;
        /// keeps the selection ratio finite when the travel cost is zero
        const MIN_TRAVEL_COST: f32 = 1e-6;

        let empty = || SolutionInfo {
            path: Vec::new(),
            score: 0.0,
            cost: 0.0,
        };

        let node_count = self.scores.node_count();
        if node_count == 0 || self.max_cost <= 0.0 {
            return empty();
        }

        let clusters = self.detect_score_clusters(TOP_NODE_FRACTION);
        if clusters.is_empty() {
            return empty();
        }
        let k = clusters.len();

        // the average score of a node on the grid
        let baseline = ((0..node_count)
            .map(|i| self.scores.get_score_by_index(i) as f64)
            .sum::<f64>()
            / node_count as f64) as f32;

        // total score, excess score and expected (lower bound) cost of each cluster
        let mut cluster_totals = Vec::with_capacity(k);
        let mut cluster_excesses = Vec::with_capacity(k);
        let mut cluster_costs = Vec::with_capacity(k);
        for members in &clusters {
            let count = members.len() as f32;
            let total: f32 = members
                .iter()
                .map(|&n| self.scores.get_score_by_index(n))
                .sum();

            cluster_totals.push(total);
            cluster_excesses.push((total / count - baseline).max(0.0));
            cluster_costs.push(self.node_cost * count + self.distance_multiplier * (count - 1.0));
        }

        // greedily choose the order of the clusters
        // for the i-th chosen cluster: `entries[i]` is the node it is entered at, and `exits[i]`
        // is the node it is left from (None for the last chosen cluster)
        let mut visited = vec![false; k];
        let mut order: Vec<usize> = Vec::new();
        let mut entries: Vec<usize> = Vec::new();
        let mut exits: Vec<Option<usize>> = Vec::new();
        let mut estimated_cost = 0.0;

        while order.len() < k {
            // (ratio, cluster, exit node of current, entry node of cluster, step cost)
            let mut best: Option<(f32, usize, usize, usize, f32)> = None;

            for c in 0..k {
                if visited[c] {
                    continue;
                }

                let (dist, exit, entry) = match order.last() {
                    Some(&current) => self.closest_node_pair(&clusters[current], &clusters[c]),
                    None => {
                        let (dist, entry) =
                            self.closest_node_to_position(self.start_position, &clusters[c]);
                        (dist, usize::MAX, entry)
                    }
                };

                // clusters are ranked by the cost of travelling to them only. the cost of the
                // cluster itself is still counted in the estimate used to stop selecting
                let travel_cost = dist * self.distance_multiplier;
                let step_cost = cluster_costs[c] + travel_cost;
                let excess = cluster_excesses[c] + 1.0;
                let ratio =
                    (cluster_totals[c] + 1.0) * excess * excess / travel_cost.max(MIN_TRAVEL_COST);

                if best.map_or(true, |b| ratio > b.0) {
                    best = Some((ratio, c, exit, entry, step_cost));
                }
            }

            // there is always an unvisited cluster inside this loop
            let (_, c, exit, entry, step_cost) = best.unwrap();

            if let Some(last_exit) = exits.last_mut() {
                *last_exit = Some(exit);
            }
            visited[c] = true;
            order.push(c);
            entries.push(entry);
            exits.push(None);

            estimated_cost += step_cost;
            if estimated_cost > self.max_cost {
                break;
            }
        }

        // build the minimal path through the key nodes of each cluster: entry, center, exit
        let mut path: Vec<usize> = Vec::new();
        for (i, &c) in order.iter().enumerate() {
            let members = &clusters[c];
            let entry = entries[i];

            let exit = exits[i].map(|exit_candidate| {
                if exit_candidate != entry || members.len() == 1 {
                    exit_candidate
                } else {
                    // entering and leaving at the same node would leave the cluster's other
                    // nodes unrepresented, so leave from the other member closest to the next
                    // cluster's entry
                    let next_entry_pos = self.scores.node_pos(entries[i + 1]);
                    members
                        .iter()
                        .copied()
                        .filter(|&m| m != entry)
                        .min_by(|&a, &b| {
                            grid_dist(self.scores.node_pos(a), next_entry_pos)
                                .total_cmp(&grid_dist(self.scores.node_pos(b), next_entry_pos))
                        })
                        .unwrap()
                }
            });
            let center = self.cluster_center_node(members);

            // key nodes are only added once, even where they coincide
            path.push(entry);
            if center != entry && Some(center) != exit {
                path.push(center);
            }
            if let Some(exit) = exit {
                if exit != entry {
                    path.push(exit);
                }
            }
        }

        let mut soln = SolutionInfo {
            path,
            score: 0.0,
            cost: 0.0,
        };
        self.update_score_and_cost(&mut soln);

        // the minimal path must itself fit in max cost: remove nodes from the end if not
        if soln.cost > self.max_cost {
            let mut cost = 0.0;
            let mut prev_pos = self.start_position;
            let mut keep = 0;

            for &node in &soln.path {
                let pos = self.scores.node_pos(node);
                let next_cost =
                    cost + self.node_cost + grid_dist(prev_pos, pos) * self.distance_multiplier;
                if next_cost > self.max_cost {
                    break;
                }
                cost = next_cost;
                prev_pos = pos;
                keep += 1;
            }

            soln.path.truncate(keep);
            self.update_score_and_cost(&mut soln);
        }

        // use up the remaining cost on the most efficient insertions
        self.greedy_insertion_path_expand(&mut soln);
        self.update_score_and_cost(&mut soln);

        soln
    }

    /// the node of a non-empty cluster closest to the cluster's score-weighted center of mass
    ///
    /// nodes with non-positive scores carry no weight. if no node has a positive score, all
    /// nodes are weighted equally
    fn cluster_center_node(&self, members: &[usize]) -> usize {
        let weight = |node: usize| self.scores.get_score_by_index(node).max(0.0) as f64;

        let use_scores = members.iter().any(|&n| weight(n) > 0.0);
        let weight_of = |node: usize| if use_scores { weight(node) } else { 1.0 };

        let mut total_weight = 0.0;
        let mut center_x = 0.0;
        let mut center_y = 0.0;
        for &n in members {
            let pos = self.scores.node_pos(n);
            let w = weight_of(n);
            total_weight += w;
            center_x += w * pos.x as f64;
            center_y += w * pos.y as f64;
        }
        center_x /= total_weight;
        center_y /= total_weight;

        let dist_sq = |node: usize| {
            let pos = self.scores.node_pos(node);
            let dx = pos.x as f64 - center_x;
            let dy = pos.y as f64 - center_y;
            dx * dx + dy * dy
        };

        members
            .iter()
            .copied()
            .min_by(|&a, &b| dist_sq(a).total_cmp(&dist_sq(b)))
            .unwrap()
    }

    /// greedy solution with a multi-pass strategy
    ///
    /// the general strategy is as follows
    ///  - collect nodes into high-scoring clusters (see `detect_score_clusters`)
    ///    - determine the top scoring node not yet in a cluster
    ///    - if adjacent (8-connected) to a cluster, add it to that cluster (joining all the
    ///      clusters it is adjacent to into one), otherwise it starts a cluster of its own
    ///    - repeat until max(1, node_count * TOP_NODE_FRACTION) nodes have been added to a
    ///      cluster, or no remaining node has a positive score
    ///  - determine path between clusters
    ///    - for each cluster, determine
    ///      - total score
    ///      - excess = average score of its nodes above the baseline (the average score of all
    ///        nodes on the grid), floored at zero
    ///      - expected cost: the node cost is exact, and the edge cost is lower bounded by
    ///        distance_multiplier * (cluster node count - 1), as the nodes are adjacent
    ///        (the edge entering the cluster is accounted for when choosing the cluster)
    ///    - greedily select the next cluster that maximizes
    ///      (total score + 1) * (excess + 1)^2 / travel_cost
    ///      - travel_cost = distance(nearest node in current, nearest node in next) *
    ///        distance_multiplier (for the first cluster, the distance is from the start
    ///        position). the cost of the cluster itself is not weighed when ranking clusters
    ///      - selection stops once the estimated cost (which does include each selected
    ///        cluster's own lower-bound cost) exceeds the truncation limit, as anything
    ///        selected afterwards would be truncated away anyway
    ///  - for each cluster, determine the path through the nodes in the cluster (entry and
    ///    exit nodes are determined by the incoming/outgoing cluster links)
    ///    - join nodes by nearest neighbour
    ///    - run two-opt on that path (with the entry and, if there is one, the exit node fixed)
    ///  - if generated path exceeds max_cost * TRUNCATION_COST_FACTOR (this leaves some room
    ///    for the more "efficient" optimizations below)
    ///    - remove nodes from end of the path until generated path.cost <= that limit
    ///  - (after previous if statement) if path cost exceeds max_cost,
    ///    - run greedy_efficiency_path_shrink
    ///  - else
    ///    - the generated path can be extended, so greedily insert the node and edge to insert
    ///      on that maximize `node_score^2 / additional_cost`
    ///
    /// cluster selection is O(T^2) in the number of clustered nodes T in the worst case,
    /// and gap filling is O(n * path_len) per inserted node
    pub fn greedy_cluster_truncating_soln(&self) -> SolutionInfo {
        /// the ratio of nodes (to total node count) that are added to clusters
        const TOP_NODE_FRACTION: f32 = 0.01;
        /// the generated path is cut down to this multiple of max cost before the efficiency
        /// based shrinking, so that shrinking has extra nodes to choose between
        const TRUNCATION_COST_FACTOR: f32 = 1.2;
        /// keeps the selection ratio finite for free clusters
        const MIN_STEP_COST: f32 = 1e-6;

        let empty = || SolutionInfo {
            path: Vec::new(),
            score: 0.0,
            cost: 0.0,
        };

        if self.scores.node_count() == 0 || self.max_cost <= 0.0 {
            return empty();
        }

        let clusters = self.detect_score_clusters(TOP_NODE_FRACTION);
        if clusters.is_empty() {
            return empty();
        }
        let k = clusters.len();

        // the average score of a node on the grid. visiting an ordinary node earns about this
        // much, so only score above it is what makes a cluster worth travelling to
        let baseline = ((0..self.scores.node_count())
            .map(|i| self.scores.get_score_by_index(i) as f64)
            .sum::<f64>()
            / self.scores.node_count() as f64) as f32;

        // value and expected (lower bound) cost of each cluster
        // value = average score above the baseline per node, scaled by size^SIZE_WEIGHT
        let (cluster_values, cluster_costs, cluster_totals) = {
            let mut cluster_values = Vec::with_capacity(clusters.len());
            let mut cluster_costs = Vec::with_capacity(clusters.len());
            let mut cluster_totals = Vec::with_capacity(clusters.len());

            for members in &clusters {
                let count = members.len() as f32;
                let total_cost: f32 = members
                    .iter()
                    .map(|&n| self.scores.get_score_by_index(n))
                    .sum();
                let total_excess = total_cost - (members.len() as f32 * baseline);

                cluster_values.push((total_excess / count).max(0.0));

                cluster_costs.push(self.node_cost * members.len() as f32
                    + self.distance_multiplier * (members.len() - 1) as f32);

                cluster_totals.push(total_cost);
            }

            (cluster_values, cluster_costs, cluster_totals)
        };

        // greedily choose the order of the clusters
        // for the i-th chosen cluster: `entries[i]` is the node it is entered at, and `exits[i]`
        // is the node it is left from (None for the last chosen cluster)
        let truncation_limit = self.max_cost * TRUNCATION_COST_FACTOR;
        let mut visited = vec![false; k];
        let mut order: Vec<usize> = Vec::new();
        let mut entries: Vec<usize> = Vec::new();
        let mut exits: Vec<Option<usize>> = Vec::new();
        let mut estimated_cost = 0.0;

        while order.len() < k {
            // (ratio, cluster, exit node of current, entry node of cluster, step cost)
            let mut best: Option<(f32, usize, usize, usize, f32)> = None;

            for c in 0..k {
                if visited[c] {
                    continue;
                }

                let (dist, exit, entry) = match order.last() {
                    Some(&current) => self.closest_node_pair(&clusters[current], &clusters[c]),
                    None => {
                        let (dist, entry) =
                            self.closest_node_to_position(self.start_position, &clusters[c]);
                        (dist, usize::MAX, entry)
                    }
                };

                // clusters are ranked by the cost of travelling to them only. the cost of the
                // cluster itself is still counted in the estimate used to stop selecting
                let travel_cost = dist * self.distance_multiplier;
                let step_cost = cluster_costs[c] + travel_cost;
                let value = cluster_values[c] + 1.0;
                let total_mult = cluster_totals[c] + 1.0;
                let ratio = total_mult * value * value / travel_cost.max(MIN_STEP_COST);

                if best.map_or(true, |b| ratio > b.0) {
                    best = Some((ratio, c, exit, entry, step_cost));
                }
            }

            // there is always an unvisited cluster inside this loop
            let (_, c, exit, entry, step_cost) = best.unwrap();

            if let Some(last_exit) = exits.last_mut() {
                *last_exit = Some(exit);
            }
            visited[c] = true;
            order.push(c);
            entries.push(entry);
            exits.push(None);

            estimated_cost += step_cost;
            if estimated_cost > truncation_limit {
                break;
            }
        }

        // order the nodes inside each cluster, entering and leaving at the cluster links
        let mut path = Vec::new();
        for (i, &c) in order.iter().enumerate() {
            let members = &clusters[c];
            let entry = entries[i];

            let exit = exits[i].map(|exit_candidate| {
                if exit_candidate != entry || members.len() == 1 {
                    exit_candidate
                } else {
                    // entering and leaving at the same node would skip the cluster's interior,
                    // so leave from the other member closest to the next cluster's entry
                    let next_entry_pos = self.scores.node_pos(entries[i + 1]);
                    members
                        .iter()
                        .copied()
                        .filter(|&m| m != entry)
                        .min_by(|&a, &b| {
                            grid_dist(self.scores.node_pos(a), next_entry_pos)
                                .total_cmp(&grid_dist(self.scores.node_pos(b), next_entry_pos))
                        })
                        .unwrap()
                }
            });

            path.extend(self.cluster_node_order(members, entry, exit));
        }

        let mut soln = SolutionInfo {
            path,
            score: 0.0,
            cost: 0.0,
        };
        self.update_score_and_cost(&mut soln);

        // truncate from the end of the path until the cost is within the truncation limit
        if soln.cost > truncation_limit {
            let mut cost = 0.0;
            let mut prev_pos = self.start_position;
            let mut keep = 0;

            for &node in &soln.path {
                let pos = self.scores.node_pos(node);
                let next_cost =
                    cost + self.node_cost + grid_dist(prev_pos, pos) * self.distance_multiplier;
                if next_cost > truncation_limit {
                    break;
                }
                cost = next_cost;
                prev_pos = pos;
                keep += 1;
            }

            soln.path.truncate(keep);
            self.update_score_and_cost(&mut soln);
        }

        if soln.cost > self.max_cost {
            // shrink until valid (recomputing to remove drift)
            while soln.cost > self.max_cost && !soln.path.is_empty() {
                self.greedy_efficiency_path_shrink(&mut soln);
                self.update_score_and_cost(&mut soln);
            }
        } else {
            self.greedy_insertion_path_expand(&mut soln);
            self.update_score_and_cost(&mut soln);
        }

        soln
    }

    /// detects clusters of high-scoring nodes, returning the nodes of each cluster
    ///
    ///  - nodes are considered in order of descending score. the top scoring node not yet in a
    ///    cluster joins the cluster of any adjacent (8-connected) node, and all the clusters it
    ///    is adjacent to are joined into one. if there is no adjacent cluster it starts a
    ///    cluster of its own
    ///  - this repeats until max(1, node_count * node_fraction) nodes have been added to
    ///    clusters, or no remaining node has a positive score
    ///
    /// the clusters are therefore the 8-connected components of the top `node_fraction` of
    /// nodes. the returned clusters are disjoint, non-empty, and each is 8-connected
    ///
    /// O(n log n)
    fn detect_score_clusters(&self, node_fraction: f32) -> Vec<Vec<usize>> {
        const UNASSIGNED: usize = usize::MAX;

        let node_count = self.scores.node_count();
        if node_count == 0 {
            return Vec::new();
        }

        let max_added = ((node_count as f32 * node_fraction) as usize).max(1);

        // nodes from highest to lowest score
        let mut ranked = Vec::from_iter(0..node_count);
        ranked.sort_unstable_by(|&a, &b| {
            self.scores
                .get_score_by_index(b)
                .total_cmp(&self.scores.get_score_by_index(a))
        });

        // add the top nodes, joining each to the clusters of the nodes adjacent to it
        // note that disjoint set expects to be for the entire grid
        let mut sets = DisjointNodeSet::new(node_count);
        let mut selected = vec![false; node_count];
        let mut top_nodes = Vec::with_capacity(max_added);

        for node in ranked {
            if top_nodes.len() >= max_added {
                break;
            }

            // later nodes score no higher, so nothing further can qualify
            if self.scores.get_score_by_index(node) <= 0.0 {
                break;
            }

            for adjacent in self.adjacent_nodes(node) {
                if selected[adjacent] {
                    sets.union(node, adjacent);
                }
            }

            selected[node] = true;
            top_nodes.push(node);
        }

        // gather the members of each cluster
        let mut root_to_cluster = vec![UNASSIGNED; node_count];
        let mut clusters: Vec<Vec<usize>> = Vec::new();
        for node in top_nodes {
            let root = sets.find(node);
            if root_to_cluster[root] == UNASSIGNED {
                root_to_cluster[root] = clusters.len();
                clusters.push(Vec::new());
            }
            clusters[root_to_cluster[root]].push(node);
        }

        clusters
    }

    /// the indices of the (up to 8) grid nodes adjacent to the given node, diagonals included
    fn adjacent_nodes(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        let pos = self.scores.node_pos(node);

        (-1..=1isize)
            .flat_map(move |dy| (-1..=1isize).map(move |dx| (dx, dy)))
            .filter(|&(dx, dy)| dx != 0 || dy != 0)
            .map(move |(dx, dy)| IVec2::new(pos.x + dx, pos.y + dy))
            .filter(move |&p| self.scores.contains_pos(p))
            .map(move |p| self.scores.node_index(p))
    }

    /// the closest pair of nodes between two non-empty node sets
    ///
    /// returns (distance, node in `from`, node in `to`)
    fn closest_node_pair(&self, from: &[usize], to: &[usize]) -> (f32, usize, usize) {
        let mut best = (f32::INFINITY, from[0], to[0]);

        for &a in from {
            let a_pos = self.scores.node_pos(a);
            for &b in to {
                let dist = grid_dist(a_pos, self.scores.node_pos(b));
                if dist < best.0 {
                    best = (dist, a, b);
                }
            }
        }

        best
    }

    /// the node of a non-empty node set closest to a position
    ///
    /// returns (distance, node)
    fn closest_node_to_position(&self, pos: IVec2, nodes: &[usize]) -> (f32, usize) {
        let mut best = (f32::INFINITY, nodes[0]);

        for &n in nodes {
            let dist = grid_dist(pos, self.scores.node_pos(n));
            if dist < best.0 {
                best = (dist, n);
            }
        }

        best
    }

    /// greedy solution with a multi-pass strategy
    ///
    /// the general strategy is as follows
    ///  - collect nodes into high-scoring clusters
    ///    - determine the average score and standard deviation across all nodes
    ///    - select top nodes (those with score >= average + TOP_STD_DEVS * standard deviation,
    ///      and a positive score)
    ///    - join all adjacent (8-connected) nodes in the top group into clusters
    ///  - determine the shortest path between clusters
    ///    - the distance between two clusters is the distance between their closest pair of
    ///      nodes, and that pair is where the path leaves one cluster and enters the other
    ///    - create a tour over the cluster graph (nearest neighbour), run two-opt on that tour
    ///    - convert the tour into a non-cycle path: begin at the cluster containing the node
    ///      nearest to `start_position`, and break the longer of that cluster's two tour edges
    ///  - for each cluster, determine the path through the nodes in the cluster (entry and
    ///    exit nodes are determined by the incoming/outgoing cluster links)
    ///    - join nodes by nearest neighbour
    ///    - run two-opt on that path (with the entry and, if there is one, the exit node fixed)
    ///  - fill gaps
    ///    - if the generated path exceeds cost, run `greedy_efficiency_path_shrink`
    ///    - if the generated path can be extended, greedily insert the node and edge to insert
    ///      on that maximize `node_score^2 / additional_cost`
    ///
    /// the cluster distance matrix is O(k^2) in memory for k clusters, and computing it is
    /// O(t^2) in the number of top nodes t. gap filling is O(n * path_len) per inserted node
    pub fn greedy_clustering_soln(&self) -> SolutionInfo {
        /// how many standard deviations above the average score a node must be to count as a
        /// top node (higher values give fewer, tighter clusters)
        const TOP_STD_DEVS: f32 = 1.0;
        /// minimum improvement to accept a tour two-opt move
        const EPSILON: f32 = 1e-4;
        /// cap on two-opt passes over the cluster tour
        const MAX_TOUR_PASSES: usize = 50;

        let empty = || SolutionInfo {
            path: Vec::new(),
            score: 0.0,
            cost: 0.0,
        };

        let node_count = self.scores.node_count();
        if node_count == 0 || self.max_cost <= 0.0 {
            return empty();
        }

        // average score and (population) standard deviation across all nodes
        // (accumulated in f64 to limit rounding error on large grids)
        let mean = (0..node_count)
            .map(|i| self.scores.get_score_by_index(i) as f64)
            .sum::<f64>()
            / node_count as f64;
        let variance = (0..node_count)
            .map(|i| {
                let d = self.scores.get_score_by_index(i) as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / node_count as f64;
        let threshold = (mean + TOP_STD_DEVS as f64 * variance.sqrt()) as f32;

        // select top nodes, ignoring nodes that cannot add score
        // (on a perfectly uniform grid the deviation is zero, so every positive node qualifies)
        let top_nodes: Vec<usize> = (0..node_count)
            .filter(|&i| {
                let score = self.scores.get_score_by_index(i);
                score > 0.0 && score >= threshold
            })
            .collect();
        if top_nodes.is_empty() {
            return empty();
        }

        // make clusters (sets) of adjacent nodes
        // note that disjoint set expects to be for the entire grid
        let mut in_top = vec![false; node_count];
        for &n in &top_nodes {
            in_top[n] = true;
        }

        let mut sets = DisjointNodeSet::new(node_count);
        for &n in &top_nodes {
            let pos = self.scores.node_pos(n);
            for dy in -1..=1isize {
                for dx in -1..=1isize {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let neighbour = IVec2::new(pos.x + dx, pos.y + dy);
                    if !self.scores.contains_pos(neighbour) {
                        continue;
                    }
                    let neighbour_index = self.scores.node_index(neighbour);
                    if in_top[neighbour_index] {
                        sets.union(n, neighbour_index);
                    }
                }
            }
        }

        // gather the members of each cluster
        let mut root_to_cluster = vec![usize::MAX; node_count];
        let mut cluster_of = vec![0usize; node_count];
        let mut clusters: Vec<Vec<usize>> = Vec::new();
        for &n in &top_nodes {
            let root = sets.find(n);
            if root_to_cluster[root] == usize::MAX {
                root_to_cluster[root] = clusters.len();
                clusters.push(Vec::new());
            }
            let c = root_to_cluster[root];
            clusters[c].push(n);
            cluster_of[n] = c;
        }
        let k = clusters.len();

        // min distance between each cluster pair, and the node pair that achieves it
        // `links[a * k + b]` is (node in a, node in b)
        let mut dist = vec![f32::INFINITY; k * k];
        let mut links = vec![(0usize, 0usize); k * k];
        for c in 0..k {
            dist[c * k + c] = 0.0;
        }
        for (ai, &a) in top_nodes.iter().enumerate() {
            let ca = cluster_of[a];
            let pa = self.scores.node_pos(a);
            for &b in &top_nodes[ai + 1..] {
                let cb = cluster_of[b];
                if ca == cb {
                    continue;
                }
                let d = grid_dist(pa, self.scores.node_pos(b));
                if d < dist[ca * k + cb] {
                    dist[ca * k + cb] = d;
                    dist[cb * k + ca] = d;
                    links[ca * k + cb] = (a, b);
                    links[cb * k + ca] = (b, a);
                }
            }
        }

        // the cluster containing the top node nearest to the start position begins the path
        let start_node = top_nodes
            .iter()
            .copied()
            .min_by(|&a, &b| {
                grid_dist(self.start_position, self.scores.node_pos(a))
                    .total_cmp(&grid_dist(self.start_position, self.scores.node_pos(b)))
            })
            .unwrap();
        let start_cluster = cluster_of[start_node];

        // nearest neighbour tour over the clusters
        let mut tour = Vec::with_capacity(k);
        let mut visited = vec![false; k];
        let mut cur = start_cluster;
        visited[cur] = true;
        tour.push(cur);
        while tour.len() < k {
            let next = (0..k)
                .filter(|&c| !visited[c])
                .min_by(|&a, &b| dist[cur * k + a].total_cmp(&dist[cur * k + b]))
                .unwrap();
            visited[next] = true;
            tour.push(next);
            cur = next;
        }

        // two-opt on the (cyclic) tour
        if k >= 4 {
            for _ in 0..MAX_TOUR_PASSES {
                let mut improved = false;
                for i in 0..k - 1 {
                    for j in i + 2..k {
                        // edges (i, i+1) and (j, j+1) are adjacent when they share the wrap node
                        if i == 0 && j == k - 1 {
                            continue;
                        }
                        let (a, b) = (tour[i], tour[i + 1]);
                        let (c, d) = (tour[j], tour[(j + 1) % k]);
                        let delta =
                            dist[a * k + c] + dist[b * k + d] - dist[a * k + b] - dist[c * k + d];
                        if delta < -EPSILON {
                            tour[i + 1..=j].reverse();
                            improved = true;
                        }
                    }
                }
                if !improved {
                    break;
                }
            }
        }

        // turn the cycle into a path beginning at the start cluster by breaking the longer of
        // the start cluster's two edges
        let start_index = tour.iter().position(|&c| c == start_cluster).unwrap();
        tour.rotate_left(start_index);
        if k >= 3 && dist[start_cluster * k + tour[1]] > dist[start_cluster * k + tour[k - 1]] {
            tour[1..].reverse();
        }

        // order the nodes inside each cluster, entering and leaving at the cluster links
        let mut path = Vec::with_capacity(top_nodes.len());
        for (p, &c) in tour.iter().enumerate() {
            let members = &clusters[c];

            let entry = if p == 0 {
                members
                    .iter()
                    .copied()
                    .min_by(|&a, &b| {
                        grid_dist(self.start_position, self.scores.node_pos(a))
                            .total_cmp(&grid_dist(self.start_position, self.scores.node_pos(b)))
                    })
                    .unwrap()
            } else {
                links[tour[p - 1] * k + c].1
            };

            let exit = if p + 1 < k {
                let (exit_candidate, next_entry) = links[c * k + tour[p + 1]];
                if exit_candidate != entry || members.len() == 1 {
                    Some(exit_candidate)
                } else {
                    // entering and leaving at the same node would skip the cluster's interior,
                    // so leave from the other member closest to the next cluster's entry
                    let next_entry_pos = self.scores.node_pos(next_entry);
                    members
                        .iter()
                        .copied()
                        .filter(|&m| m != entry)
                        .min_by(|&a, &b| {
                            grid_dist(self.scores.node_pos(a), next_entry_pos)
                                .total_cmp(&grid_dist(self.scores.node_pos(b), next_entry_pos))
                        })
                }
            } else {
                None
            };

            path.extend(self.cluster_node_order(members, entry, exit));
        }

        let mut soln = SolutionInfo {
            path,
            score: 0.0,
            cost: 0.0,
        };

        // fill gaps: shrink until valid (recomputing to remove drift), then expand
        loop {
            self.update_score_and_cost(&mut soln);
            if soln.cost <= self.max_cost || soln.path.is_empty() {
                break;
            }
            self.greedy_efficiency_path_shrink(&mut soln);
        }
        self.greedy_insertion_path_expand(&mut soln);
        self.update_score_and_cost(&mut soln);

        soln
    }

    /// orders the nodes of a single cluster into a path that begins at `entry` and, if `exit`
    /// is given, ends at `exit` (nearest neighbour followed by two-opt)
    ///
    /// `entry` and `exit` must be members of the cluster. every member appears exactly once
    fn cluster_node_order(
        &self,
        members: &[usize],
        entry: usize,
        exit: Option<usize>,
    ) -> Vec<usize> {
        let mut remaining: Vec<usize> = members
            .iter()
            .copied()
            .filter(|&m| m != entry && Some(m) != exit)
            .collect();

        let mut order = Vec::with_capacity(members.len());
        order.push(entry);
        let mut current_pos = self.scores.node_pos(entry);

        while !remaining.is_empty() {
            let (idx, _) = remaining
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    grid_dist(current_pos, self.scores.node_pos(*a.1))
                        .total_cmp(&grid_dist(current_pos, self.scores.node_pos(*b.1)))
                })
                .unwrap();
            let node = remaining.swap_remove(idx);
            current_pos = self.scores.node_pos(node);
            order.push(node);
        }

        let fixed_end = match exit {
            Some(e) if e != entry => {
                order.push(e);
                true
            }
            _ => false,
        };

        self.two_opt_open_path(&mut order, fixed_end);
        order
    }

    /// two-opt on an open path of nodes. the first node is always fixed, and so is the last
    /// when `fixed_end` is set
    fn two_opt_open_path(&self, path: &mut [usize], fixed_end: bool) {
        /// minimum improvement to accept a move (prevents cycling on float noise)
        const EPSILON: f32 = 1e-4;
        /// cap on passes over the path
        const MAX_PASSES: usize = 250;

        let m = path.len();
        if m < 3 {
            return;
        }

        // last index that may be moved
        let max_j = if fixed_end { m - 2 } else { m - 1 };
        let pos = |node: usize| self.scores.node_pos(node);

        for _ in 0..MAX_PASSES {
            let mut improved = false;

            for i in 1..max_j {
                for j in i + 1..=max_j {
                    let a = pos(path[i - 1]);
                    let b = pos(path[i]);
                    let c = pos(path[j]);

                    let mut delta = grid_dist(a, c) - grid_dist(a, b);
                    if j + 1 < m {
                        let d = pos(path[j + 1]);
                        delta += grid_dist(b, d) - grid_dist(c, d);
                    }

                    if delta < -EPSILON {
                        path[i..=j].reverse();
                        improved = true;
                    }
                }
            }

            if !improved {
                break;
            }
        }
    }

    /// expands the given path by repeatedly inserting the unvisited node (at the position along
    /// the path, including the start and end) that maximizes `node_score^2 / additional_cost`,
    /// as long as the path stays within max cost
    ///
    /// assumes the passed score and cost are correct and the path has no duplicate nodes
    ///
    /// updates the entry's path, score and cost
    fn greedy_insertion_path_expand(&self, soln: &mut SolutionInfo) {
        /// keeps the ratio finite for free insertions
        const MIN_ADDITIONAL_COST: f32 = 1e-6;

        let node_count = self.scores.node_count();
        let mut on_path = BitSet::<u64>::with_capacity_general(node_count);
        for &n in &soln.path {
            on_path.insert(n);
        }

        loop {
            let len = soln.path.len();
            // (ratio, node, insertion index, additional cost)
            let mut best: Option<(f32, usize, usize, f32)> = None;

            for node in 0..node_count {
                if on_path.contains(node) {
                    continue;
                }
                let score = self.scores.get_score_by_index(node);
                if score <= 0.0 {
                    continue;
                }
                let node_pos = self.scores.node_pos(node);

                for slot in 0..=len {
                    let prev_pos = if slot == 0 {
                        self.start_position
                    } else {
                        self.scores.node_pos(soln.path[slot - 1])
                    };

                    let edge = if slot < len {
                        let next_pos = self.scores.node_pos(soln.path[slot]);
                        grid_dist(prev_pos, node_pos) + grid_dist(node_pos, next_pos)
                            - grid_dist(prev_pos, next_pos)
                    } else {
                        grid_dist(prev_pos, node_pos)
                    };

                    let additional = edge * self.distance_multiplier + self.node_cost;
                    if soln.cost + additional > self.max_cost {
                        continue;
                    }

                    let ratio = score * score / additional.max(MIN_ADDITIONAL_COST);
                    if best.map_or(true, |b| ratio > b.0) {
                        best = Some((ratio, node, slot, additional));
                    }
                }
            }

            let Some((_, node, slot, additional)) = best else {
                break;
            };

            soln.path.insert(slot, node);
            on_path.insert(node);
            soln.cost += additional;
            soln.score += self.scores.get_score_by_index(node);
        }
    }

    /// applies 2-opt optimization to an existing solution
    /// performs up to the specified amount of passes as a proportion of path width
    fn two_opt_soln(&self, soln: &mut SolutionInfo, compute_time: Duration) {
        /// how often (in candidate moves) to check for exceeding compute time
        const TIME_CHECK_INTERVAL: u32 = 512;
        /// minimum improvement to accept a move (prevents cycling on float noise)
        const EPSILON: f32 = 1e-4;

        let n = soln.path.len();
        if n < 2 {
            return;
        }

        let end_time = Instant::now() + compute_time;
        let mut checks = 0u32;

        // position of tour index t, where t = 0 is the start and t >= 1 is path[t - 1]
        let pos = |path: &[usize], t: usize| -> IVec2 {
            if t == 0 {
                self.start_position
            } else {
                self.scores.node_pos(path[t - 1])
            }
        };

        let mut improved = true;
        while improved {
            improved = false;

            for i in 0..n {
                for j in i + 1..=n {
                    checks = checks.wrapping_add(1);
                    if checks % TIME_CHECK_INTERVAL == 0 && Instant::now() >= end_time {
                        return;
                    }

                    let a = pos(&soln.path, i);
                    let b = pos(&soln.path, i + 1);
                    let c = pos(&soln.path, j);

                    let mut delta = grid_dist(a, c) - grid_dist(a, b);
                    if j < n {
                        let d = pos(&soln.path, j + 1);
                        delta += grid_dist(b, d) - grid_dist(c, d);
                    }

                    if delta < -EPSILON {
                        soln.path[i..j].reverse();
                        soln.cost += delta * self.distance_multiplier;
                        improved = true;
                    }
                }
            }
        }
    }
}

// force evaluation of the compile-time sanity checks on the genetic algorithm's constants
const _: () = PathPlanningProblem::PASS_SIZE_ASSERTION;
const _: () = PathPlanningProblem::ELITE_FRACTION_ASSERTION;

/// a path stored as a doubly linked list inside a flat array indexed by grid node id,
/// allowing O(1) removal of a node
#[derive(Debug, Clone)]
struct LinkedPath {
    head: usize,
    tail: usize,
    /// (prev, next) links for every grid node id; (NONE, NONE) for nodes not on the path
    adj_map: Vec<(usize, usize)>,
    len: usize,
}

impl LinkedPath {
    const NONE: usize = usize::MAX;

    pub fn neighbours(&self, node_index: usize) -> (Option<usize>, Option<usize>) {
        let tuple = self.adj_map[node_index];
        (
            (tuple.0 != Self::NONE).then_some(tuple.0),
            (tuple.1 != Self::NONE).then_some(tuple.1),
        )
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    /// the last node of the path, if any
    #[allow(dead_code)]
    pub fn tail(&self) -> Option<usize> {
        (self.tail != Self::NONE).then_some(self.tail)
    }

    pub fn iter(&self) -> impl Iterator<Item = usize> {
        let mut node = self.head;

        std::iter::from_fn(move || {
            if node != Self::NONE {
                let stored = node;
                node = self.adj_map[node].1;
                Some(stored)
            } else {
                None
            }
        })
    }

    /// removes the given node from the path
    ///
    /// the node must currently be on the path
    pub fn remove(&mut self, node_index: usize) {
        let neighbors = self.neighbours(node_index);
        self.adj_map[node_index] = (Self::NONE, Self::NONE);

        match neighbors {
            (Some(prev), Some(next)) => {
                self.adj_map[prev].1 = next;
                self.adj_map[next].0 = prev;
            }
            (Some(prev), None) => {
                self.adj_map[prev].1 = Self::NONE;
                self.tail = prev;
            }
            (None, Some(next)) => {
                self.adj_map[next].0 = Self::NONE;
                self.head = next;
            }
            (None, None) => {
                self.head = Self::NONE;
                self.tail = Self::NONE;
            }
        }

        self.len -= 1;
    }

    pub fn to_vec(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.len);

        for node in self.iter() {
            out.push(node)
        }

        out
    }

    /// builds a linked path from a slice of grid node ids
    ///
    /// `node_count` is the number of nodes in the grid, since links are indexed by grid node id
    /// (not by position in the slice). the slice must not contain duplicate nodes
    pub fn from_slice(slice: &[usize], node_count: usize) -> Self {
        let mut adj_map = vec![(Self::NONE, Self::NONE); node_count];

        if slice.is_empty() {
            return Self {
                head: Self::NONE,
                tail: Self::NONE,
                adj_map,
                len: 0,
            };
        }

        let head = slice[0];
        let mut prev = head;

        for &node in slice.iter().skip(1) {
            adj_map[prev].1 = node;
            adj_map[node].0 = prev;

            prev = node;
        }

        Self {
            head,
            tail: prev,
            adj_map,
            len: slice.len(),
        }
    }
}

/// a totally ordered f32 wrapper (via `f32::total_cmp`), so NaN can never cause a panic
#[derive(Debug, Clone, Copy)]
struct FPriority(pub f32);

impl PartialEq for FPriority {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for FPriority {}
impl PartialOrd for FPriority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for FPriority {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

#[derive(Debug, Clone)]
pub struct SolutionInfo {
    pub path: Vec<usize>,
    pub score: f32,
    pub cost: f32,
}

/// disjoint set datastructure for nodes
#[derive(Debug, Clone)]
struct DisjointNodeSet {
    map: Vec<usize>,
    ranks: Vec<usize>,
}

impl DisjointNodeSet {
    /// nodes are initialized with themselves as parents and ranks as 0
    pub fn new(size: usize) -> Self {
        Self {
            map: (0..size).collect(),
            ranks: vec![0; size],
        }
    }

    /// gets the representative node in the set of the given node (with path compression)
    pub fn find(&mut self, node_index: usize) -> usize {
        let mut root = node_index;
        while root != self.map[root] {
            root = self.map[root];
        }

        // Path compression: point nodes along the path directly to the root
        let mut curr = node_index;
        while curr != root {
            let next = self.map[curr];
            self.map[curr] = root;
            curr = next;
        }

        root
    }

    /// unions the sets containing the two nodes (using union by rank)
    pub fn union(&mut self, node1: usize, node2: usize) -> bool {
        let root1 = self.find(node1);
        let root2 = self.find(node2);

        if root1 == root2 {
            return false; // Already in the same set
        }

        // Attach the smaller rank tree under the root of the higher rank tree
        if self.ranks[root1] < self.ranks[root2] {
            self.map[root1] = root2;
        } else if self.ranks[root1] > self.ranks[root2] {
            self.map[root2] = root1;
        } else {
            self.map[root2] = root1;
            self.ranks[root1] += 1;
        }

        true
    }
}
