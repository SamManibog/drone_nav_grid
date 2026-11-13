use std::{cmp::Ordering, collections::BinaryHeap, hint::cold_path};

use crate::{
    heap_entries::{Comparator, MappedMaxHeapEntry, MaxHeapEntry, MinHeapEntry},
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
    /// the initial population size as a proportion of maximum possible gene count
    const POPULATION_INIT_MULTIPIER: f32 = 8.0;

    /// the proportion of the initial population that is a mutation of the existing greedy solutions
    const SEEDED_POPULATION_RATIO: f32 = Self::ELITE_FRACTION * 0.25;

    /// the ratio of completely preserved individuals to total population size per generation
    /// elite individuals are not mutated, but may be selected for crossover
    const ELITE_FRACTION: f32 = 0.025;

    /// the ratio of top-performing individuals selected to pass on to the next generation
    const PERFORMANCE_FRACTION: f32 = 0.4 - Self::ELITE_FRACTION;

    /// the ratio of non-top-performing individuals selected randomly to pass on to the next generation
    /// used to diversify the gene pool
    const DIVERSITY_FRACTION: f32 = 0.1;

    /// the ratio of population size to average gene count
    const POPULATION_GENE_RATIO: f32 = 10.0;

    /// the probability any given gene is mutated
    const MUTATION_PROBABILITY: f32 = 0.01;

    const PASS_FRACTION: f32 =
        Self::ELITE_FRACTION + Self::PERFORMANCE_FRACTION + Self::DIVERSITY_FRACTION;

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
        if self.max_cost < 0.0 && !self.max_cost.is_nan() {
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
    /// first generates a random, likely cost-exceeding path between nodes
    /// then, removes the least efficient entries until cost is no longer exceeded
    ///
    /// O(nlogn)
    fn greedy_random_subtractive_soln<T: Rng>(&self, rng: &mut T) -> SolutionInfo {
        const NONE: usize = usize::MAX;

        let node_count = self.scores.node_count();
        if node_count == 0 {
            return SolutionInfo {
                path: Vec::new(),
                score: 0.0,
                cost: 0.0,
            };
        }

        // first, add entire path to grid randomly
        let mut unselected = Vec::from_iter(0..node_count);
        unselected.shuffle(rng);

        // build initial path as a doubly linked list: (prev, next) per node
        let mut node_adjacency_map = vec![(NONE, NONE); node_count];

        // the first item in the path
        let mut head_index = unselected.pop().unwrap();

        // the last item in the path (only needed while building)
        let mut tail_index = head_index;

        let mut total_cost = self.node_cost * node_count as f32;
        let mut total_score = self.scores.get_score_by_index(head_index);

        {
            let mut prev_position = self.scores.node_pos(head_index);
            total_cost += grid_dist(self.start_position, prev_position) * self.distance_multiplier;

            while let Some(node_index) = unselected.pop() {
                let node_position = self.scores.node_pos(node_index);
                total_cost += grid_dist(prev_position, node_position) * self.distance_multiplier;
                total_score += self.scores.get_score_by_index(node_index);

                node_adjacency_map[tail_index].1 = node_index;
                node_adjacency_map[node_index].0 = tail_index;

                prev_position = node_position;
                tail_index = node_index;
            }
        }

        // s(i) = score of node i
        // c(i, j) = cost of edge between i and j
        // r = root/start position
        // C = node cost
        //
        // Cost saved by removing node i given the current links. Handles head
        // (prev == NONE), tail (next == NONE), and a lone node (both NONE).
        //
        // let path p = x, ..., y, ..., z
        //   head:  c(r, x) + c(x, x + 1) - c(r, x + 1) + C
        //   inner: c(y - 1, y) + c(y, y + 1) - c(y - 1, y + 1) + C
        //   tail:  c(z - 1, z) + C
        //
        // (distances are scaled by the distance multiplier)
        let savings = |adj: &[(usize, usize)], i: usize| -> f32 {
            let (prev, next) = adj[i];
            let pos = self.scores.node_pos(i);
            let prev_pos = if prev == NONE {
                self.start_position
            } else {
                self.scores.node_pos(prev)
            };

            let dist = if next == NONE {
                grid_dist(prev_pos, pos)
            } else {
                let next_pos = self.scores.node_pos(next);
                grid_dist(prev_pos, pos) + grid_dist(pos, next_pos) - grid_dist(prev_pos, next_pos)
            };

            dist * self.distance_multiplier + self.node_cost
        };

        // efficiency = score created per unit of cost that removal would save.
        // The queue is a max-heap, so negate: the least efficient node pops first.
        // Removals that save nothing are treated as infinitely efficient (never
        // chosen early).
        let priority = |adj: &[(usize, usize)], i: usize| -> FPriority {
            let saved = savings(adj, i);
            let efficiency = if saved <= 0.0 {
                f32::INFINITY
            } else {
                self.scores.get_score_by_index(i) / saved
            };
            FPriority(-efficiency)
        };

        // push nodes to priority queue (least efficient nodes are removed first)
        let mut queue = PriorityQueue::with_capacity(node_count);
        for i in 0..node_count {
            queue.push(i, priority(&node_adjacency_map, i));
        }

        // shrink until solution has valid cost or all nodes are removed
        let mut path_length = node_count;
        while total_cost > self.max_cost {
            let Some((node_index, _)) = queue.pop() else {
                break;
            };
            let (prev_node, next_node) = node_adjacency_map[node_index];

            // compute before unlinking
            total_cost -= savings(&node_adjacency_map, node_index);
            total_score -= self.scores.get_score_by_index(node_index);
            path_length -= 1;

            // unlink
            if prev_node != NONE {
                node_adjacency_map[prev_node].1 = next_node;
            } else {
                head_index = next_node;
            }
            if next_node != NONE {
                node_adjacency_map[next_node].0 = prev_node;
            }

            // only the two neighbours' savings change
            if prev_node != NONE {
                queue.change_priority(&prev_node, priority(&node_adjacency_map, prev_node));
            }
            if next_node != NONE {
                queue.change_priority(&next_node, priority(&node_adjacency_map, next_node));
            }
        }

        // everything was removed; return exact zeros rather than float drift
        if head_index == NONE {
            return SolutionInfo {
                path: Vec::new(),
                score: 0.0,
                cost: 0.0,
            };
        }

        // perform path reconstruction
        let mut path = Vec::with_capacity(path_length);
        {
            let mut node_index = head_index;
            while node_index != NONE {
                path.push(node_index);
                node_index = node_adjacency_map[node_index].1;
            }
        }

        SolutionInfo {
            path,
            cost: total_cost,
            score: total_score,
        }
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
                .into_iter()
                .map(|i| MaxHeapEntry(self.scores.get_score_by_index(i), i)),
        );

        let mut remaining_cost = self.max_cost;

        let mut current_pos = self.start_position;

        while let Some(MaxHeapEntry(_, index)) = heap.pop()
            && remaining_cost > 0.0
        {
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

    /// modifies the given path either by expanding to maximize score, or shrinking to stay within
    /// max score
    ///
    /// DOES NOT PERFORM DUPLICATE DETECTION/REMOVAL
    ///
    /// current method for shrinking is to iteratively select and remove the lowest-scoring included
    /// nodes
    ///
    /// current method for expansion is running greedy distance/prob algorithm
    pub fn greedy_score_dist_ratio_path_adjust(&self, entry: &mut SolutionInfo) {
        if entry.cost > self.max_cost {
            Self::greedy_score_path_shrink(&self, entry);
        }

        // only need to recompute if a new node could feasibly be introduced
        if entry.cost + self.node_cost + self.distance_multiplier < self.max_cost {
            Self::greedy_score_distance_ratio_path_expand(&self, entry)
        }
    }

    /// shrinks the given path to a valid cost by removing lowest-scoring nodes first
    ///
    /// returns the new score
    fn greedy_score_path_shrink(&self, entry: &mut SolutionInfo) {
        // construct a min heap of score
        let mut heap = BinaryHeap::from_iter(
            entry
                .path
                .iter()
                .map(|i| MinHeapEntry(self.scores.get_score_by_index(*i), *i)),
        );

        // pop from score while invalid
        while let Some(node) = heap.pop()
            && entry.cost > self.max_cost
        {
            let i = node.1;

            let node_pos = self.scores.node_pos(i);

            // adjust path cost
            if i == 0 && i == entry.path.len() {
                // resulting path is empty. no cost, no score
                entry.path.pop();
                entry.cost = 0.0;
                entry.score = 0.0;
                return;
            } else if i == entry.path.len() - 1 {
                entry.cost -= grid_dist(self.scores.node_pos(i - 1), node_pos)
            } else {
                let prev_pos = if i == 0 {
                    self.start_position
                } else {
                    self.scores.node_pos(i - 1)
                };

                let next_pos = self.scores.node_pos(i + 1);

                let original_dist = grid_dist(prev_pos, node_pos) + grid_dist(node_pos, next_pos);
                let new_dist = grid_dist(prev_pos, next_pos);

                entry.cost -= (new_dist - original_dist) * self.distance_multiplier;
            };

            // adjust score
            entry.score -= self.scores.get_score_by_index(i);

            // remove from path
            entry.path.remove(i);
        }
    }

    /// expands the given path based on score/distance greedy algorithm
    /// assumes the path is non-empty
    ///
    /// assumes the passed total score and cost are correct
    ///
    /// returns the new score
    fn greedy_score_distance_ratio_path_expand(&self, entry: &mut SolutionInfo) {
        let mut remaining_cost = self.max_cost - entry.cost;
        let mut current_pos: IVec2 = entry
            .path
            .first()
            .map_or_else(|| self.start_position, |x| self.scores.node_pos(*x));

        // construct bitset
        let mut unexplored: BitSet<u64> =
            BitSet::from_bit_vec_general(BitVec::from_elem_general(self.scores.node_count(), true));
        for i in entry.path.iter() {
            unexplored.insert(*i);
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

        return (score, node_cost + edge_cost);
    }

    /// Greedy algorithm that plans an exploration path between grid cells
    ///
    /// the next chosen cell is the one that maximizes (score/distance)
    ///
    /// returns (path, score)
    ///
    /// O(n^2)
    pub fn greedy_score_distance_ratio_soln(&self) -> (Vec<usize>, f32) {
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
                return (path, total_score);
            }
        }

        (path, total_score)
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
                score / distance
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
    ///     - initial solutions to speed up convergence time:
    ///         - greedy algorithm that selects the highest probability unvisited node first
    ///         - greedy algorithm that selects unvisited node with greatest (probability/distance to node)
    ///     - end condition/output: a list of nodes representing visited
    ///     - the fitness function: sum of visited node scores
    ///     - crossover: ox1 crossover followed by greedy shrinking and expansion
    ///     - mutations:
    ///         - swap order of visitation of random chunks
    ///         - change chunk to nearby, unvisited chunk
    ///     - selection after crossover and mutation:
    ///         - certain number of top-scoring w or w/o crossover and/or mutation
    ///         - certain amount of top scoring with crossover and mutation
    pub fn genetic_soln(&self, passes: usize) -> Vec<usize> {
        // trivial case
        if self.max_cost <= 0.0 {
            return Vec::new();
        }

        // get initial solutions

        todo!("complete genetic pathfinding in sprint 3");
    }

    /// performs a selection step on the population
    /// returns the vector of cloned elites
    fn genetic_selection<T: Rng>(
        &self,
        rng: &mut T,
        population: &mut Population,
    ) -> Vec<SolutionInfo> {
        let total_size = population.len();
        let elite_count: usize = f32::ceil(total_size as f32 * Self::ELITE_FRACTION) as usize;
        let performance_count: usize =
            f32::ceil(total_size as f32 * Self::PERFORMANCE_FRACTION) as usize;
        let diverse_count: usize = f32::ceil(total_size as f32 * Self::DIVERSITY_FRACTION) as usize;

        let top_scorers = population.get_top_performers(elite_count);

        population.cull(rng, performance_count, diverse_count);

        top_scorers
    }

    /// performs raw crossover
    /// crossover may create an offspring that exceeds max cost
    fn ox1_crossover(&self, parent1: &SolutionInfo, parent2: &SolutionInfo) -> SolutionInfo {
        todo!()
    }

    fn nudge_entry(&self, entry: &mut SolutionInfo, index: usize, new_node: usize) {
        // determine cost of connected edges
        // determine cost of modified edge
        // compute cost delta connected edge -> modified edge
        // subtract cost delta from entry's total cost
        todo!()
    }

    /// mutates a path by moving a few random nodes by a random amount
    /// mutations may cause the individual to exceed max cost
    fn nudge_mutation<T: Rng>(&self, rng: &mut T, entry: &mut SolutionInfo) {
        // add contained nodes to range
        let mut used = BitSet::<u64>::with_capacity_general(entry.path.len());
        for i in &entry.path {
            used.insert(*i);
        }

        // note that entry length will not change by nudging entries
        'gene_loop: for node_index in 0..entry.path.len() {
            let random_var: f32 = rng.random_range(0.0_f32..=1.0_f32);
            if random_var > Self::MUTATION_PROBABILITY {
                continue 'gene_loop;
            }

            let node_position = self.scores.node_pos(node_index);

            let mut nudge_index = rng.random_range(0..Self::NUDGE_MUTATION_DELTAS.len());
            let stop_index = nudge_index;

            // attempt to nudge for all possible nudges
            'nudge_attempt_loop: {
                let new_position = node_position + Self::NUDGE_MUTATION_DELTAS[nudge_index].coords;
                if self.scores.contains_pos(node_position) {
                    let new_node_index = self.scores.node_index(new_position);
                    if !used.contains(new_node_index) {
                        self.nudge_entry(entry, node_index, new_node_index);

                        used.remove(node_index);
                        used.insert(new_node_index);

                        break 'nudge_attempt_loop;
                    }
                }

                nudge_index += 1;
                nudge_index %= Self::NUDGE_MUTATION_DELTAS.len();

                if nudge_index == stop_index {
                    break 'nudge_attempt_loop;
                }
            }
        }
    }
}

#[derive(Debug)]
struct Population {
    /// individuals, sorted by highest to lowest scoring
    individuals: Vec<SolutionInfo>,
}

impl Population {
    pub fn new() -> Self {
        Self {
            individuals: Default::default(),
        }
    }

    /// gets the size of the populaion
    pub const fn len(&self) -> usize {
        self.individuals.len()
    }

    /// inserts an individual into the population
    pub fn insert(&mut self, individual: SolutionInfo) {
        let bs_result = self
            .individuals
            .binary_search_by(|a| a.score.partial_cmp(&individual.score).unwrap());
        let index = match bs_result {
            Ok(i) => i,
            Err(i) => i,
        };
        self.individuals.insert(index, individual);
    }

    /// clones the top n performers into an array
    pub fn get_top_performers(&self, n: usize) -> Vec<SolutionInfo> {
        self.individuals[0..n.min(self.individuals.len())].to_vec()
    }

    /// culls the population keeping a specific amount of top-performers and some random
    /// non-top-performers
    ///
    /// assumes that tops_to_keep + randos_to_keep <= self.len()
    pub fn cull<T: Rng>(&mut self, rng: &mut T, tops_to_keep: usize, randos_to_keep: usize) {
        let stop_index = tops_to_keep + randos_to_keep;

        // perform swaps into part of array to keep
        for swap_index in tops_to_keep..stop_index {
            let swap_with = rng.random_range(tops_to_keep..self.individuals.len());
            self.individuals.swap(swap_index, swap_with);
        }

        // sort changed values
        (&mut self.individuals[tops_to_keep..stop_index])
            .sort_unstable_by(|l, r| r.score.partial_cmp(&l.score).unwrap());

        // drain unused values
        self.individuals.drain(stop_index..self.individuals.len());
    }

    /// gets a mutable iterator over the population in order from most to least score
    pub fn individuals_mut(&mut self) -> impl Iterator<Item = &mut SolutionInfo> {
        self.individuals.iter_mut()
    }

    /// gets an iterator over the population in order from most to least score
    pub fn individuals(&mut self) -> impl Iterator<Item = &SolutionInfo> {
        self.individuals.iter()
    }
}

#[derive(Debug, Clone, Copy)]
struct FPriority(pub f32);

impl PartialEq for FPriority {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
    fn ne(&self, other: &Self) -> bool {
        self.0 != other.0
    }
}
impl Eq for FPriority {}
impl PartialOrd for FPriority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}
impl Ord for FPriority {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap()
    }
}

#[derive(Debug, Clone)]
struct SolutionInfo {
    path: Vec<usize>,
    score: f32,
    cost: f32,
}

struct EntryScoreCmp;
impl Comparator for EntryScoreCmp {
    type Value = SolutionInfo;

    fn cmp(v1: &Self::Value, v2: &Self::Value) -> std::cmp::Ordering {
        v1.score.partial_cmp(&v2.score).unwrap()
    }
}
