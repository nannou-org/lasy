//! A small library dedicated to LASER path optimisation.
//!
//! The goal for the library is to provide a suite of useful tools for optimising linear vector
//! images for galvanometer-based LASER projection. To allow *lasy* LASERs to get more done.
//!
//! This crate implements the full suite of optimisations covered in *Accurate and Efficient
//! Drawing Method for Laser Projection* by Purkhet Abderyim et al. These include Graph
//! optimisation, draw order optimisation, blanking delays and sharp angle delays. See [the
//! paper](https://art-science.org/journal/v7n4/v7n4pp155/artsci-v7n4pp155.pdf) for more details.
//!
//! ## Optimising a Vector Image
//!
//! Optimising a single frame of `input_points` is achieved by the following steps:
//!
//! 1. Convert the list of points to a list of segments, tracking whether they are blank or lit.
//! 2. Convert the segments to a graph where the nodes are points. Only a single node for each
//!    unique position is stored.
//! 3. Convert the point graph to a euler graph, determining the minimum number of blanks necessary
//!    to do so.
//! 4. Find the optimal euler circuit within the euler graph. This represents the new path that the
//!    LASER should travel.
//! 5. Interpolate the euler circuit. This step will attempt to distribute the "target count"
//!    number of points across the euler circuit while accounting for blanking and sharp angle
//!    delays in accordance with the provided configuration.
//!
//! In code, these steps look like so:
//!
//! ```rust,ignore
//! let segs = points_to_segments(&input_points);
//! let pg = segments_to_point_graph(&input_points, segs);
//! let eg = point_graph_to_euler_graph(&pg);
//! let ec = euler_graph_to_euler_circuit(&input_points, &eg);
//! let output_points = interpolate_euler_circuit(&input_points, &ec, &eg, target_count, config);
//! ```
//!
//! ## Optimising Animation
//!
//! It is worth keeping in mind that, if you are working with an animated stream of frames, some
//! consideration must be taken in how to move from the end of one frame to the beginning of the
//! next.
//!
//! For the most part, this simply requires blanking from the last point of the previous frame to
//! the first point of the current one (or in other words, the first point of the euler circuit for
//! the current frame).
//!
//! The `blank_segment_points` function can be useful for generating these points, as this same
//! function is used for generating blank segment points during interpolation.
//!
//! ## Point Types
//!
//! In order to have your points be compatible with the functions provided by this crate, the
//! `Position`, `IsBlank`, `Weight`, `Blanked` and `Lerp` traits should be implemented as
//! necessary. Please see their respective documentation for more information.
//!
//! Note that the optimisation steps above are intended to work with two different point types: an
//! *input* point type describing the user's original frame and an *output* (or *raw*) point type
//! representing the fully interpolated, optimised path. The main difference between these two
//! types is that the *output* point type does not to store its *weight*, as point weighting is
//! applied during the interpolation process. See the `Weight` docs for more details.
//!
//! ## Graph Storage
//!
//! The graphs used within the optimisation steps above will never store points directly, and
//! instead will store indices into the original slice of `input_points`. This allows for a smaller
//! memory footprint and to avoid complicated generics propagating up through the graph types.

mod lerp;

pub use crate::lerp::Lerp;

use petgraph::visit::EdgeRef;
use petgraph::Undirected;
pub use petgraph::{Direction, Incoming, Outgoing};
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

// ----------------------------------------------------------------------------

/// Point types that may describe position.
///
/// *Input* point types must be able to produce their normalised position along both axes (-1.0 to
/// 1.0) in order to allow for the optimisation steps to reason about things like distance and
/// angular variance.
pub trait Position {
    /// The position of the point, where `[-1.0, -1.0]` is the bottom left and `[1.0, 1.0]` is the
    /// top right.
    ///
    /// *Note: If an axis is inverted you should be fine, but I haven't tested this.*
    fn position(&self) -> [f32; 2];
}

/// Point types that can describe whether or not they are blank (black).
///
/// Input point types must implement this in order to allow for euler graph construction and for
/// the interpolation process to account for blanking delay.
pub trait IsBlank {
    /// Whether or not the point is blank.
    fn is_blank(&self) -> bool;
}

/// Point types that have a weight associated with them.
///
/// This must be implemented for *input* point types. This weight represents the number of time
/// units that should be spent drawing this point in order to accentuate it within the path.
///
/// Note that the *output* point type does *not* require storing this weight, and as such you can
/// request a different output point type (e.g. that does not store the `weight`) from the
/// `interpolate_euler_circuit` function.
pub trait Weight {
    /// The minimum number of extra times this point should be drawn.
    ///
    /// `0` is the default used for drawing sequences of smooth line segments.
    ///
    /// Values greater than `0` are useful for accenting individual points.
    fn weight(&self) -> u32;
}

/// Point types that can produce a blanked copy of themselves.
pub trait Blanked {
    /// Produce a point of the same type and position but that is blank.
    fn blanked(&self) -> Self;
}

// ----------------------------------------------------------------------------

/// An index into a slice of points.
pub type PointIndex = u32;

/// Represents a line segment over which the laser scanner will travel.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Segment {
    pub start: PointIndex,
    pub end: PointIndex,
    pub kind: SegmentKind,
}

/// describes whether a line segment between two points is blank or not.
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub enum SegmentKind {
    Blank,
    Lit,
}

/// A type used to represent graph describing the points in a frame and how they are joined.
///
/// Only lit edges are represented in this representation.
pub type PointGraph = petgraph::Graph<PointIndex, (), Undirected, u32>;

/// A type used to represent a graph of points that contains at least one euler circuit.
pub type EulerGraph = petgraph::Graph<PointIndex, SegmentKind, Undirected, u32>;

/// A type used to represent a eulerian circuit through an eulerian graph.
///
/// The `EdgeIndex` referes to an index into the `EulerGraph`. The `Direction` describes whether
/// traversal is performed from `source` to `target` (`Direction::Outgoing`) or `target` to
/// `source` (`Direction::Incoming`).
pub type EulerCircuit = Vec<(EdgeIndex, Direction)>;

type EdgeIndex = petgraph::graph::EdgeIndex<u32>;
type NodeIndex = petgraph::graph::NodeIndex<u32>;
type Edge<E> = petgraph::graph::Edge<E, u32>;
type EgEdge = Edge<SegmentKind>;

/// An iterator yielding all lit line segments.
#[derive(Clone)]
pub struct Segments<I>
where
    I: Iterator,
{
    points: I,
    last: Option<I::Item>,
}

/// Configuration options for eulerian circuit interpolation.
#[repr(C)]
#[derive(Clone, Debug, PartialEq)]
pub struct InterpolationConfig {
    /// The minimum distance the interpolator can travel along an edge before a new point is
    /// required.
    ///
    /// By default, this value is `InterpolationConfig::DEFAULT_DISTANCE_PER_POINT`.
    pub distance_per_point: f32,
    /// The number of points to insert at the end of a blank to account for light modulator delay.
    ///
    /// By default, this value is `InterpolationConfig::DEFAULT_BLANK_DELAY_POINTS`.
    pub blank_delay_points: u32,
    /// The amount of delay to add based on the angle of the corner in radians.
    ///
    /// By default, this value is `InterpolationConfig::DEFAULT_RADIANS_PER_POINT`.
    pub radians_per_point: f32,
}

/// For the blank ab: `[a, a.blanked(), b.blanked(), (0..delay).map(|_| b.blanked())]`.
pub const BLANK_MIN_POINTS: u32 = 3;

// ----------------------------------------------------------------------------

impl InterpolationConfig {
    /// The default distance the interpolator can travel before a new point is required.
    pub const DEFAULT_DISTANCE_PER_POINT: f32 = 0.1;
    /// The default number of points inserted for the end of each blank segment.
    pub const DEFAULT_BLANK_DELAY_POINTS: u32 = 10;
    /// The default radians per point of delay to reduce corner inertia.
    pub const DEFAULT_RADIANS_PER_POINT: f32 = 0.6;
}

// ----------------------------------------------------------------------------

impl<'a, T> Position for &'a T
where
    T: Position,
{
    fn position(&self) -> [f32; 2] {
        (**self).position()
    }
}

impl<'a, T> IsBlank for &'a T
where
    T: IsBlank,
{
    fn is_blank(&self) -> bool {
        (**self).is_blank()
    }
}

impl<'a, T> Weight for &'a T
where
    T: Weight,
{
    fn weight(&self) -> u32 {
        (**self).weight()
    }
}

impl Position for [f32; 2] {
    fn position(&self) -> [f32; 2] {
        self.clone()
    }
}

impl Default for InterpolationConfig {
    fn default() -> Self {
        Self {
            distance_per_point: Self::DEFAULT_DISTANCE_PER_POINT,
            blank_delay_points: Self::DEFAULT_BLANK_DELAY_POINTS,
            radians_per_point: Self::DEFAULT_RADIANS_PER_POINT,
        }
    }
}

impl<I, P> Iterator for Segments<I>
where
    I: Iterator<Item = (PointIndex, P)>,
    P: Clone + IsBlank + Position,
{
    type Item = Segment;
    fn next(&mut self) -> Option<Self::Item> {
        while let Some((end_ix, end)) = self.points.next() {
            let (start_ix, start) = match self.last.replace((end_ix, end.clone())) {
                None => continue,
                Some(last) => last,
            };

            // Skip duplicates.
            let kind = if start.position() == end.position() {
                if !start.is_blank() && !end.is_blank() {
                    SegmentKind::Lit
                } else {
                    continue;
                }
            } else if start.is_blank() && end.is_blank() {
                SegmentKind::Blank
            } else {
                SegmentKind::Lit
            };

            let start = start_ix;
            let end = end_ix;
            return Some(Segment { start, end, kind });
        }
        None
    }
}

// ----------------------------------------------------------------------------

/// Create an iterator yielding segments from an iterator yielding points.
pub fn points_to_segments<I>(points: I) -> impl Iterator<Item = Segment>
where
    I: IntoIterator,
    I::Item: Clone + IsBlank + Position,
{
    let points = points
        .into_iter()
        .enumerate()
        .map(|(i, p)| (i as PointIndex, p));
    let last = None;
    Segments { points, last }
}

/// Convert the given laser frame vector segments to a graph of points.
///
/// The point type `P` must be able to be hashed for node deduplication.
pub fn segments_to_point_graph<P, I>(points: &[P], segments: I) -> PointGraph
where
    P: Hash + Weight,
    I: IntoIterator<Item = Segment>,
{
    struct Node {
        ix: NodeIndex,
        weight: u32,
    }

    fn point_hash<P: Hash>(p: &P) -> u64 {
        let mut hasher = DefaultHasher::new();
        p.hash(&mut hasher);
        hasher.finish()
    }

    let mut g = PointGraph::default();
    let mut pt_to_id = HashMap::new();

    // Build the graph.
    let mut segments = segments.into_iter().peekable();
    let mut prev_seg_kind = None;
    while let Some(seg) = segments.next() {
        let prev_kind = prev_seg_kind;
        prev_seg_kind = Some(seg.kind);

        if let SegmentKind::Blank = seg.kind {
            continue;
        }
        let pa = &points[seg.start as usize];
        let pb = &points[seg.end as usize];
        let ha = point_hash(&pa);
        let hb = point_hash(&pb);
        let na = {
            let n = pt_to_id.entry(ha).or_insert_with(|| {
                let ix = g.add_node(seg.start);
                let weight = pa.weight();
                Node { ix, weight }
            });
            n.weight = std::cmp::max(n.weight, pa.weight());
            n.ix
        };
        let nb = {
            let n = pt_to_id.entry(hb).or_insert_with(|| {
                let ix = g.add_node(seg.end);
                let weight = pb.weight();
                Node { ix, weight }
            });
            n.weight = std::cmp::max(n.weight, pb.weight());
            n.ix
        };

        // If the edge is a loop, only keep it if it describes a lone point.
        if na == nb {
            let prev_edge_lit = match prev_kind {
                Some(SegmentKind::Lit) => true,
                _ => false,
            };
            let mut next_edge_lit = || match segments.peek() {
                Some(s) if s.kind == SegmentKind::Lit => true,
                _ => false,
            };
            if prev_edge_lit || next_edge_lit() {
                continue;
            }
        }

        if g.find_edge(na, nb).is_none() {
            g.add_edge(na, nb, ());
        }
    }

    g
}

/// Convert a point graph to a euler graph.
///
/// This determines the minimum number of blank segments necessary to create a euler circuit from
/// the given point graph. A euler circuit is useful as it represents a graph that can be drawn
/// unicursally (one continuous path that covers all nodes while only traversing each edge once).
pub fn point_graph_to_euler_graph(pg: &PointGraph) -> EulerGraph {
    // Find the connected components.
    let ccs = petgraph::algo::kosaraju_scc(pg);

    // Whether or not the edge is a loop (starts and ends at the same node).
    //
    // These should be ignored in euler graph construction.
    fn edge_is_loop<E>(e: &E) -> bool
    where
        E: EdgeRef,
        E::NodeId: PartialEq,
    {
        e.source() == e.target()
    }

    // Whether or not the given node has an even degree.
    //
    // A node has an even degree if it has an even number of non-loop edges. A loop edge is an edge
    // that starts and ends at the same node.
    fn node_has_even_degree(pg: &PointGraph, n: NodeIndex) -> bool {
        let non_loop_edges = pg.edges(n).filter(|e| !edge_is_loop(e)).count();
        non_loop_edges % 2 == 0
    }

    // The indices of the connected components whose nodes all have an even degree.
    let euler_components: HashSet<_> = ccs
        .iter()
        .enumerate()
        .filter(|(_, cc)| cc.iter().all(|&n| node_has_even_degree(pg, n)))
        .map(|(i, _)| i)
        .collect();

    // Represents the nodes to be connected for a single component.
    struct ToConnect {
        // Connection to the previous component.
        prev: NodeIndex,
        // Consecutive connections within the component.
        inner: Vec<NodeIndex>,
        // Connection to the next component.
        next: NodeIndex,
    }

    // Collect the free nodes from each connected component that are to be connected by blanks.
    let mut to_connect = vec![];
    for (i, cc) in ccs.iter().enumerate() {
        if euler_components.contains(&i) {
            // Take the first point.
            let n = cc[0];
            to_connect.push(ToConnect {
                prev: n,
                inner: vec![],
                next: n,
            });
        } else {
            let v: Vec<_> = cc
                .iter()
                .filter(|&&n| !node_has_even_degree(pg, n))
                .collect();

            // If there's a single point, connect to itself.
            if v.len() == 1 {
                let p = *v[0];
                let prev = p;
                let inner = vec![];
                let next = p;
                to_connect.push(ToConnect { prev, inner, next });
                continue;

            // Otherwise convert to a euler component.
            } else {
                assert_eq!(
                    v.len() % 2,
                    0,
                    "expected even number of odd-degree nodes for non-Euler component, found {}",
                    v.len(),
                );
                let prev = *v[0];
                let inner = v[1..v.len() - 1].iter().map(|&&n| n).collect();
                let next = *v[v.len() - 1];
                to_connect.push(ToConnect { prev, inner, next });
            }
        }
    }

    // Convert the `to_connect` Vec containing the nodes to be connected for each connected
    // component to a `Vec` containing the pairs of nodes which will be directly connected.
    let mut pairs = vec![];
    let mut iter = to_connect.iter().enumerate().peekable();
    while let Some((i, this)) = iter.next() {
        for ch in this.inner.chunks(2) {
            pairs.push((ch[0], ch[1]));
        }
        match iter.peek() {
            Some((_, next)) => pairs.push((this.next, next.prev)),
            None if i > 0 => pairs.push((this.next, to_connect[0].prev)),
            None => match euler_components.contains(&0) {
                // If there is only one component and it is euler, we are done.
                true => (),
                // If there is only one non-euler, connect it to itself.
                false => pairs.push((this.next, this.prev)),
            },
        }
    }

    // Turn the graph into a euler graph by adding the blanks.
    let mut eg = pg.map(|_n_ix, n| n.clone(), |_e_ix, _| SegmentKind::Lit);
    for (na, nb) in pairs {
        eg.add_edge(na, nb, SegmentKind::Blank);
    }

    eg
}

/// Given some Euler Circuit edge `e` and its direction `d`, return the index of the node that
/// represents the *start* of the edge.
pub fn ec_edge_start(eg: &EulerGraph, e: EdgeIndex, d: Direction) -> NodeIndex {
    fn edge_start(edge: &EgEdge, d: Direction) -> NodeIndex {
        match d {
            Outgoing => edge.source(),
            Incoming => edge.target(),
        }
    }
    edge_start(&eg.raw_edges()[e.index()], d)
}

/// Given some Euler Circuit edge `e` and its direction `d`, return the index of the node that
/// represents the *end* of the edge.
pub fn ec_edge_end(eg: &EulerGraph, e: EdgeIndex, d: Direction) -> NodeIndex {
    fn edge_end(edge: &EgEdge, d: Direction) -> NodeIndex {
        match d {
            Outgoing => edge.target(),
            Incoming => edge.source(),
        }
    }
    edge_end(&eg.raw_edges()[e.index()], d)
}

/// Given a Euler Graph describing the vector image to be drawn, return the optimal Euler Circuit
/// describing the path over which the laser should travel.
///
/// This is Hierholzer's Algorithm with the amendment that during traversal of each vertex the edge
/// with the closest angle to a straight line is always chosen.
pub fn euler_graph_to_euler_circuit<P>(points: &[P], eg: &EulerGraph) -> EulerCircuit
where
    P: Position,
{
    // If there is one or less nodes, there's no place for edges.
    if eg.node_count() == 0 || eg.node_count() == 1 {
        return vec![];
    }

    // Begin the traversals to build the circuit, starting at `v0`.
    let start_n = eg
        .node_indices()
        .next()
        .expect("expected at least two nodes, found none");
    let mut visited: HashSet<EdgeIndex> = HashSet::new();
    let mut visit_order: Vec<(EdgeIndex, Direction)> = vec![];
    loop {
        // Find a node in the visit order with untraversed edges, or pick one to begin if we're
        // just starting. We will do a traversal from this node. Keep track of where in the
        // existing `visit_order` we should merge this new traversal. If there are no nodes with
        // untraversed edges, we are done.
        let (merge_ix, n) = match visit_order.is_empty() {
            true => (0, start_n),
            false => {
                match visit_order
                    .iter()
                    .map(|&(e, dir)| ec_edge_start(eg, e, dir))
                    .enumerate()
                    .find(|&(_i, n)| eg.edges(n).any(|e| !visited.contains(&e.id())))
                {
                    Some(n) => n,
                    None => break,
                }
            }
        };

        let traversal = traverse_unvisited(points, n, eg, &mut visited);
        let new_visit_order = visit_order
            .iter()
            .take(merge_ix)
            .cloned()
            .chain(traversal)
            .chain(visit_order.iter().skip(merge_ix).cloned())
            .collect();
        visit_order = new_visit_order;
    }

    visit_order
}

// A traversal through unvisited edges of the graph starting from `n`.
//
// Traversal ends when `n` is reached again.
//
// The returned `Vec` contains the index of each edge traversed.
fn traverse_unvisited<P>(
    points: &[P],
    start: NodeIndex,
    eg: &EulerGraph,
    visited: &mut HashSet<EdgeIndex>,
) -> Vec<(EdgeIndex, Direction)>
where
    P: Position,
{
    let mut n = start;
    let mut traversal: Vec<(EdgeIndex, Direction)> = vec![];
    loop {
        // Find the straightest edge that hasn't yet been traversed.
        let e_ref = {
            let mut untraversed_edges = eg
                // Specifies that `n` should be the `source` for each edge reference yielded.
                .edges_directed(n, Outgoing)
                .filter(|e_ref| !visited.contains(&e_ref.id()));

            let init_e_ref = untraversed_edges
                .next()
                .expect("expected a strongly connected euler graph");

            match traversal
                .last()
                .map(|&(e, dir)| ec_edge_start(eg, e, dir))
                .map(|n| points[eg[n] as usize].position())
            {
                // If this is the first edge in the traversal, use the first ref.
                None => init_e_ref,

                // Retrieve the three positions used to determine the angle.
                Some(prev_source_p) => {
                    let source_p = points[eg[init_e_ref.source()] as usize].position();
                    let target_p = points[eg[init_e_ref.target()] as usize].position();
                    let init_dist = straight_angle_variance(prev_source_p, source_p, target_p);
                    let init = (init_e_ref, init_dist);
                    let (e_ref, _) = untraversed_edges.fold(init, |best, e_ref| {
                        let (_, best_dist) = best;
                        let target_p = points[eg[e_ref.target()] as usize].position();
                        let dist = straight_angle_variance(prev_source_p, source_p, target_p);
                        if dist < best_dist {
                            (e_ref, dist)
                        } else {
                            best
                        }
                    });
                    e_ref
                }
            }
        };

        // Add the edge into our visitation record.
        let e = e_ref.id();
        let dir = if n == eg.raw_edges()[e.index()].source() {
            Direction::Outgoing
        } else {
            Direction::Incoming
        };
        n = e_ref.target();
        visited.insert(e);
        traversal.push((e, dir));

        // If this edge brings us back to the start, we have finished this traversal.
        if e_ref.target() == start {
            break;
        }
    }

    traversal
}

// Given an angle described by points a -> b -> c, return the variance from a straight angle in
// radians.
fn straight_angle_variance([ax, ay]: [f32; 2], [bx, by]: [f32; 2], [cx, cy]: [f32; 2]) -> f32 {
    let [ux, uy] = [bx - ax, by - ay];
    let [vx, vy] = [cx - bx, cy - by];
    let ur = uy.atan2(ux);
    let vr = vy.atan2(vx);
    let diff_rad = vr - ur;

    // Convert the radians to the angular distance.
    fn angular_dist(rad: f32) -> f32 {
        let rad = rad.abs();
        if rad > std::f32::consts::PI {
            -rad + std::f32::consts::PI * 2.0
        } else {
            rad
        }
    }

    angular_dist(diff_rad)
}

fn distance_squared(a: [f32; 2], b: [f32; 2]) -> f32 {
    let [ax, ay] = a;
    let [bx, by] = b;
    let [abx, aby] = [bx - ax, by - ay];
    abx * abx + aby * aby
}

/// The number of points used per blank segment given the `blank_delay_points` from a config.
pub fn blank_segment_point_count(a_weight: u32, blank_delay_points: u32) -> u32 {
    a_weight + BLANK_MIN_POINTS + blank_delay_points
}

/// Returns the points used to blank between two given lit points *a* and *b*.
///
/// The point type `A` is expected to know its weight, while the point type `B` does not need to.
pub fn blank_segment_points<A, B>(a: A, br: B, blank_delay_points: u32) -> impl Iterator<Item = B>
where
    A: Into<B> + Weight,
    B: Blanked + Clone,
{
    let a_weight = a.weight();
    let ar: B = a.into();
    let ar_blanked = ar.blanked();
    let br_blanked = br.blanked();
    Some(ar.clone())
        .into_iter()
        .chain((0..a_weight).map(move |_| ar.clone()))
        .chain(Some(ar_blanked))
        .chain(Some(br_blanked.clone()))
        .chain((0..blank_delay_points).map(move |_| br_blanked.clone()))
}

/// The number of points added at a lit corner given its angle and angular delay rate.
pub fn corner_point_count(rad: f32, corner_delay_radians_per_point: f32) -> u32 {
    (rad / corner_delay_radians_per_point) as _
}

/// The minimum points for traversing a lit segment (not including end corner delays).
pub fn distance_min_point_count(dist: f32, min_distance_per_point: f32) -> u32 {
    // There must be at least one point at the beginning of the line.
    const MIN_COUNT: u32 = 1;
    MIN_COUNT + (dist * min_distance_per_point) as u32
}

/// The minimum number of points used for a lit segment of the given distance and end angle.
///
/// `a_weight` refers to the weight of the point at the beginning of the segment.
pub fn lit_segment_min_point_count(
    distance: f32,
    end_corner_radians: f32,
    distance_per_point: f32,
    radians_per_point: f32,
    a_weight: u32,
) -> u32 {
    a_weight
        + corner_point_count(end_corner_radians, radians_per_point)
        + distance_min_point_count(distance, distance_per_point)
}

/// Returns the points that make up a lit segment between *a* and *b* including delay for the end
/// corner.
///
/// `excess_points` are distributed across the distance point count. This is used to allow the
/// interpolation process to evenly distribute left-over points across a frame.
///
/// Point type `A` is expected to know its weight, while the point type `B` does not need to.
///
/// Point type `B` should support linear interpolation of all of its attributes.
pub fn lit_segment_points<A, B>(
    a: A,
    br: B,
    corner_point_count: u32,
    distance_min_point_count: u32,
    excess_points: u32,
) -> impl Iterator<Item = B>
where
    A: Into<B> + Weight,
    B: Clone + Lerp<Scalar = f32>,
{
    let dist_point_count = distance_min_point_count + excess_points;
    let a_weight = a.weight();
    let ar: B = a.into();
    let ar2 = ar.clone();
    let br2 = br.clone();
    let weight_points = (0..a_weight).map(move |_| ar.clone());
    let dist_points = (0..dist_point_count).map(move |i| {
        let lerp_amt = i as f32 / dist_point_count as f32;
        ar2.clone().lerp(&br2.clone(), lerp_amt)
    });
    let corner_points = (0..corner_point_count).map(move |_| br.clone());
    weight_points.chain(dist_points).chain(corner_points)
}

/// Interpolate the given `EulerCircuit` with the given configuration in order to produce a path
/// ready to be submitted to the DAC.
///
/// The interpolation process will attempt to generate `target_points` number of points along the
/// circuit, but may generate *more* points in the user's `InterpolationConfig` indicates that more
/// are required for interpolating the specified circuit.
///
/// Performs the following steps:
///
/// 1. Determine the minimum number of required points:
///     - 1 for each edge plus the 1 for the end.
///     - The number of points required for each edge.
///         - For lit edges:
///             - The distance of each edge accounting for minimum points per distance.
///             - The angular distance to the following lit edge (none if blank).
///         - For blank edges:
///             - The specified blank delay.
/// 2. If the total is greater than `target_points`, we're done. If not, goto 3.
/// 3. Determine a weight per lit edge based on the distance of each edge.
/// 4. Distribute the remaining points between each lit edge distance based on their weights.
///
/// **Panic!**s if the given graph is not actually a `EulerCircuit`.
pub fn interpolate_euler_circuit<P, R>(
    points: &[P],
    ec: &EulerCircuit,
    eg: &EulerGraph,
    target_points: u32,
    conf: &InterpolationConfig,
) -> Vec<R>
where
    P: Clone + Into<R> + Position + Weight,
    R: Blanked + Clone + Lerp<Scalar = f32>,
{
    // Capture a profile of each edge to assist with interpolation.
    #[derive(Debug)]
    struct EdgeProfile {
        a_weight: u32,
        kind: EdgeProfileKind,
    }

    #[derive(Debug)]
    enum EdgeProfileKind {
        Blank,
        Lit { distance: f32, end_corner: f32 },
    }

    impl EdgeProfile {
        // Create an `EdgeProfile` for the edge at the given index.
        fn from_index<P>(points: &[P], ix: usize, ec: &EulerCircuit, eg: &EulerGraph) -> Self
        where
            P: Position + Weight,
        {
            let (ab, ab_dir) = ec[ix];
            let ab_kind = eg[ab];
            let a_ix = ec_edge_start(eg, ab, ab_dir);
            let a = &points[eg[a_ix] as usize];
            let a_weight = a.weight();
            let kind = match ab_kind {
                SegmentKind::Blank => EdgeProfileKind::Blank,
                SegmentKind::Lit => {
                    let a_pos = a.position();
                    let b_ix = ec_edge_end(eg, ab, ab_dir);
                    let b = &points[eg[b_ix] as usize];
                    let b_pos = b.position();
                    let distance = distance_squared(a_pos, b_pos).sqrt();
                    let next_ix = (ix + 1) % ec.len();
                    let (bc, bc_dir) = ec[next_ix];
                    let c_ix = ec_edge_end(eg, bc, bc_dir);
                    let c = &points[eg[c_ix] as usize];
                    let c_pos = c.position();
                    let end_corner = straight_angle_variance(a_pos, b_pos, c_pos);
                    EdgeProfileKind::Lit {
                        distance,
                        end_corner,
                    }
                }
            };
            EdgeProfile { a_weight, kind }
        }

        fn is_lit(&self) -> bool {
            match self.kind {
                EdgeProfileKind::Lit { .. } => true,
                EdgeProfileKind::Blank => false,
            }
        }

        // The lit distance covered by this edge.
        fn lit_distance(&self) -> f32 {
            match self.kind {
                EdgeProfileKind::Lit { distance, .. } => distance,
                _ => 0.0,
            }
        }

        // The minimum number of points required to draw the edge.
        fn min_points(&self, conf: &InterpolationConfig) -> u32 {
            match self.kind {
                EdgeProfileKind::Blank => {
                    blank_segment_point_count(self.a_weight, conf.blank_delay_points)
                }
                EdgeProfileKind::Lit {
                    distance,
                    end_corner,
                } => lit_segment_min_point_count(
                    distance,
                    end_corner,
                    conf.distance_per_point,
                    conf.radians_per_point,
                    self.a_weight,
                ),
            }
        }

        // The points for this edge.
        fn points<P, R>(
            &self,
            points: &[P],
            e: EdgeIndex,
            e_dir: Direction,
            eg: &EulerGraph,
            conf: &InterpolationConfig,
            excess_points: u32,
        ) -> Vec<R>
        where
            P: Clone + Into<R> + Position + Weight,
            R: Blanked + Clone + Lerp<Scalar = f32>,
        {
            let a_ix = ec_edge_start(eg, e, e_dir);
            let b_ix = ec_edge_end(eg, e, e_dir);
            let a = points[eg[a_ix] as usize].clone();
            let b = points[eg[b_ix] as usize].clone();
            let br: R = b.into();
            match self.kind {
                EdgeProfileKind::Blank => {
                    blank_segment_points(a, br, conf.blank_delay_points).collect()
                }
                EdgeProfileKind::Lit {
                    end_corner,
                    distance,
                } => {
                    let dist_point_count =
                        distance_min_point_count(distance, conf.distance_per_point);
                    let corner_point_count = corner_point_count(end_corner, conf.radians_per_point);
                    lit_segment_points(a, br, corner_point_count, dist_point_count, excess_points)
                        .collect()
                }
            }
        }
    }

    // If the circuit is empty, so is our path.
    if ec.is_empty() || target_points == 0 {
        return vec![];
    }

    // Create a profile of each edge containing useful information for interpolation.
    let edge_profiles = (0..ec.len())
        .map(|ix| EdgeProfile::from_index(points, ix, ec, eg))
        .collect::<Vec<_>>();

    // TODO: If the circuit doesn't contain any lit edges, what should we do?
    if !edge_profiles.iter().any(|ep| ep.is_lit()) {
        return vec![];
    }
    // The minimum number of points required to display the image.
    let min_points = edge_profiles
        .iter()
        .map(|ep| ep.min_points(conf))
        .fold(0, |acc, n| acc + n);

    // The target number of points not counting the last to be added at the end.
    let target_points_minus_last = target_points - 1;

    // The excess points distributed across all edges.
    let edge_excess_point_counts = if min_points < target_points_minus_last {
        // A multiplier for determining excess points. This should be distributed across distance.
        let excess_points = target_points_minus_last - min_points;
        // The lit distance covered by each edge.
        let edge_lit_dists = edge_profiles
            .iter()
            .map(|ep| (ep.is_lit(), ep.lit_distance()))
            .collect::<Vec<_>>();
        // The total lit distance covered by the traversal.
        let total_lit_dist = edge_lit_dists.iter().fold(0.0, |acc, &(_, d)| acc + d);
        // Determine the weights for each edge based on distance.
        let edge_weights: Vec<(bool, f32)> = match total_lit_dist <= std::f32::EPSILON {
            true => {
                // If there was no total distance, distribute evenly.
                let n_lit_edges = edge_lit_dists.iter().filter(|&&(b, _)| b).count();
                edge_lit_dists
                    .iter()
                    .map(|&(is_lit, _)| (is_lit, 1.0 / n_lit_edges as f32))
                    .collect()
            }
            false => {
                // Otherwise weight by distance.
                edge_lit_dists
                    .iter()
                    .map(|&(is_lit, dist)| (is_lit, dist / total_lit_dist))
                    .collect()
            }
        };

        // Multiply the weight by the excess points. Track fractional error and distribute.
        let mut v = Vec::with_capacity(ec.len());
        let mut err = 0.0;
        let mut count = 0;
        for (is_lit, w) in edge_weights {
            if !is_lit {
                v.push(0);
                continue;
            }
            let nf = w * excess_points as f32 + err;
            err = nf.fract();
            let n = nf as u32;
            count += n;
            v.push(n);
        }

        // Check for rounding error.
        if count == (excess_points - 1) {
            // Find first lit edge index.
            let (i, _) = edge_profiles
                .iter()
                .enumerate()
                .find(|&(_, ep)| ep.is_lit())
                .expect("expected at least one lit edge");
            v[i] += 1;
            count += 1;
        }

        // Sanity check that rounding errors have been handled.
        debug_assert_eq!(count, excess_points);

        v
    } else {
        vec![0; ec.len()]
    };

    // Collect all points.
    let total_points = std::cmp::max(min_points, target_points);
    let mut new_points = Vec::with_capacity(total_points as usize);
    for elem in ec.iter().zip(&edge_profiles).zip(&edge_excess_point_counts) {
        let ((&(e_ix, e_dir), ep), &excess) = elem;
        new_points.extend(ep.points(points, e_ix, e_dir, eg, conf, excess));
    }

    // Push the last point.
    let last_point = {
        let &(e, dir) = ec.last().unwrap();
        let end = ec_edge_end(eg, e, dir);
        &points[eg[end] as usize]
    };
    new_points.push(last_point.clone().into());

    // Sanity check that we generated at least `target_points`.
    debug_assert!(new_points.len() >= target_points as usize);

    new_points
}

// ----------------------------------------------------------------------------

#[cfg(test)]
mod test {
    use super::{
        euler_graph_to_euler_circuit, point_graph_to_euler_graph, points_to_segments,
        segments_to_point_graph,
    };
    use super::{EulerGraph, Outgoing, PointGraph, SegmentKind};
    use crate::{Blanked, IsBlank, Position, Weight};
    use std::collections::HashSet;
    use std::hash::{Hash, Hasher};

    #[derive(Copy, Clone, Debug, PartialEq)]
    struct Point {
        position: [f32; 2],
        rgb: [f32; 3],
        weight: u32,
    }

    #[derive(Eq, Hash, PartialEq)]
    struct HashPoint {
        pos: [i32; 2],
        rgb: [u32; 3],
    }

    #[derive(Copy, Clone, Debug, PartialEq)]
    struct RawPoint {
        position: [f32; 2],
        rgb: [f32; 3],
    }

    impl From<Point> for HashPoint {
        fn from(p: Point) -> Self {
            let [px, py] = p.position;
            let [pr, pg, pb] = p.rgb;
            let x = (px * std::i16::MAX as f32) as i32;
            let y = (py * std::i16::MAX as f32) as i32;
            let r = (pr * std::u16::MAX as f32) as u32;
            let g = (pg * std::u16::MAX as f32) as u32;
            let b = (pb * std::u16::MAX as f32) as u32;
            let pos = [x, y];
            let rgb = [r, g, b];
            HashPoint { pos, rgb }
        }
    }

    impl Blanked for RawPoint {
        fn blanked(&self) -> Self {
            RawPoint {
                position: self.position,
                rgb: [0.0; 3],
            }
        }
    }

    impl Hash for Point {
        fn hash<H: Hasher>(&self, hasher: &mut H) {
            HashPoint::from(self.clone()).hash(hasher)
        }
    }

    impl IsBlank for Point {
        fn is_blank(&self) -> bool {
            self.rgb == [0.0; 3]
        }
    }

    impl Position for Point {
        fn position(&self) -> [f32; 2] {
            self.position
        }
    }

    impl Position for RawPoint {
        fn position(&self) -> [f32; 2] {
            self.position
        }
    }

    impl Weight for Point {
        fn weight(&self) -> u32 {
            self.weight
        }
    }

    fn graph_eq<N, E, Ty, Ix>(
        a: &petgraph::Graph<N, E, Ty, Ix>,
        b: &petgraph::Graph<N, E, Ty, Ix>,
    ) -> bool
    where
        N: PartialEq,
        E: PartialEq,
        Ty: petgraph::EdgeType,
        Ix: petgraph::graph::IndexType + PartialEq,
    {
        let a_ns = a.raw_nodes().iter().map(|n| &n.weight);
        let b_ns = b.raw_nodes().iter().map(|n| &n.weight);
        let a_es = a
            .raw_edges()
            .iter()
            .map(|e| (e.source(), e.target(), &e.weight));
        let b_es = b
            .raw_edges()
            .iter()
            .map(|e| (e.source(), e.target(), &e.weight));
        a_ns.eq(b_ns) && a_es.eq(b_es)
    }

    fn is_euler_graph<N, E, Ty, Ix>(g: &petgraph::Graph<N, E, Ty, Ix>) -> bool
    where
        Ty: petgraph::EdgeType,
        Ix: petgraph::graph::IndexType,
    {
        let even_degree = g.node_indices().all(|n| g.edges(n).count() % 2 == 0);
        let strongly_connected = petgraph::algo::kosaraju_scc(g).len() == 1;
        even_degree && strongly_connected
    }

    fn white_pt(position: [f32; 2]) -> Point {
        Point {
            position,
            rgb: [1.0; 3],
            weight: 0,
        }
    }

    fn blank_pt(position: [f32; 2]) -> Point {
        Point {
            position,
            rgb: [0.0; 3],
            weight: 0,
        }
    }

    fn square_pts() -> [Point; 5] {
        let a = white_pt([-1.0, -1.0]);
        let b = white_pt([-1.0, 1.0]);
        let c = white_pt([1.0, 1.0]);
        let d = white_pt([1.0, -1.0]);
        [a, b, c, d, a]
    }

    fn two_vertical_lines_pts() -> [Point; 8] {
        let a = [-1.0, -1.0];
        let b = [-1.0, 1.0];
        let c = [1.0, -1.0];
        let d = [1.0, 1.0];
        [
            white_pt(a),
            white_pt(b),
            blank_pt(b),
            blank_pt(c),
            white_pt(c),
            white_pt(d),
            blank_pt(d),
            blank_pt(a),
        ]
    }

    #[test]
    fn test_points_to_point_graph_no_blanks() {
        let pts = square_pts();
        let segs = points_to_segments(pts.iter().cloned());
        let pg = segments_to_point_graph(&pts, segs);

        let mut expected = PointGraph::default();
        let na = expected.add_node(0);
        let nb = expected.add_node(1);
        let nc = expected.add_node(2);
        let nd = expected.add_node(3);
        expected.add_edge(na, nb, ());
        expected.add_edge(nb, nc, ());
        expected.add_edge(nc, nd, ());
        expected.add_edge(nd, na, ());

        assert!(graph_eq(&pg, &expected));
    }

    #[test]
    fn test_points_to_point_graph_with_blanks() {
        let pts = two_vertical_lines_pts();
        let segs = points_to_segments(pts.iter().cloned());
        let pg = segments_to_point_graph(&pts, segs);

        let mut expected = PointGraph::default();
        let na = expected.add_node(0);
        let nb = expected.add_node(1);
        let nc = expected.add_node(4);
        let nd = expected.add_node(5);
        expected.add_edge(na, nb, ());
        expected.add_edge(nc, nd, ());

        assert!(graph_eq(&pg, &expected));
    }

    #[test]
    fn test_point_graph_to_euler_graph_no_blanks() {
        let pts = square_pts();
        let segs = points_to_segments(pts.iter().cloned());
        let pg = segments_to_point_graph(&pts, segs);
        let eg = point_graph_to_euler_graph(&pg);

        let mut expected = EulerGraph::default();
        let na = expected.add_node(0);
        let nb = expected.add_node(1);
        let nc = expected.add_node(2);
        let nd = expected.add_node(3);
        expected.add_edge(na, nb, SegmentKind::Lit);
        expected.add_edge(nb, nc, SegmentKind::Lit);
        expected.add_edge(nc, nd, SegmentKind::Lit);
        expected.add_edge(nd, na, SegmentKind::Lit);

        assert!(graph_eq(&eg, &expected));
    }

    #[test]
    fn test_point_graph_to_euler_graph_with_blanks() {
        let pts = two_vertical_lines_pts();
        let segs = points_to_segments(pts.iter().cloned());
        let pg = segments_to_point_graph(&pts, segs);
        let eg = point_graph_to_euler_graph(&pg);

        assert!(is_euler_graph(&eg));

        let pg_ns: Vec<_> = pg.raw_nodes().iter().map(|n| n.weight).collect();
        let eg_ns: Vec<_> = eg.raw_nodes().iter().map(|n| n.weight).collect();
        assert_eq!(pg_ns, eg_ns);

        assert_eq!(
            eg.raw_edges()
                .iter()
                .filter(|e| e.weight == SegmentKind::Blank)
                .count(),
            2
        );
        assert_eq!(
            eg.raw_edges()
                .iter()
                .filter(|e| e.weight == SegmentKind::Lit)
                .count(),
            2
        );
    }

    #[test]
    fn test_euler_graph_to_euler_circuit_no_blanks() {
        let pts = square_pts();
        let segs = points_to_segments(pts.iter().cloned());
        let pg = segments_to_point_graph(&pts, segs);
        let eg = point_graph_to_euler_graph(&pg);
        let ec = euler_graph_to_euler_circuit(&pts, &eg);

        let mut ns = eg.node_indices();
        let na = ns.next().unwrap();
        let nb = ns.next().unwrap();
        let nc = ns.next().unwrap();
        let nd = ns.next().unwrap();

        let expected = vec![
            (eg.find_edge(na, nb).unwrap(), Outgoing),
            (eg.find_edge(nb, nc).unwrap(), Outgoing),
            (eg.find_edge(nc, nd).unwrap(), Outgoing),
            (eg.find_edge(nd, na).unwrap(), Outgoing),
        ];

        assert_eq!(ec, expected);
    }

    #[test]
    fn test_euler_graph_to_euler_circuit_with_blanks() {
        let pts = two_vertical_lines_pts();
        let segs = points_to_segments(pts.iter().cloned());
        let pg = segments_to_point_graph(&pts, segs);
        let eg = point_graph_to_euler_graph(&pg);
        let ec = euler_graph_to_euler_circuit(&pts, &eg);

        assert_eq!(ec.len(), eg.edge_count());

        let mut visited = HashSet::new();
        let mut walk = ec
            .iter()
            .cycle()
            .map(|&(e, _)| (e, &eg.raw_edges()[e.index()]));
        while visited.len() < 4 {
            let (e_id, _) = walk.next().unwrap();
            assert!(visited.insert(e_id));
        }
    }

    #[test]
    fn test_euler_circuit_duplicate_points() {
        let pts = [white_pt([0., 0.]), white_pt([0., 1.]), white_pt([0., 1.])];
        let segs = points_to_segments(pts.iter().cloned());
        let pg = segments_to_point_graph(&pts, segs);
        let eg = point_graph_to_euler_graph(&dbg!(pg));
        let _ = euler_graph_to_euler_circuit(&pts, &dbg!(eg));
    }

    #[test]
    fn test_single_point() {
        let pts = [white_pt([0., 0.]), white_pt([0., 0.])];
        let segs = points_to_segments(pts.iter().cloned());
        let pg = segments_to_point_graph(&pts, segs);
        let eg = point_graph_to_euler_graph(&dbg!(pg));
        let ec = euler_graph_to_euler_circuit(&pts, &dbg!(eg));
        assert_eq!(ec.len(), 0);
    }
}
