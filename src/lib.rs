#![allow(unsafe_op_in_unsafe_fn)]

use meval::{Context, ContextProvider, Expr};
use numpy::{
    Element, IxDyn, PyArrayDyn, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyDict};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use std::cell::RefCell;
use std::str::FromStr;
use thiserror::Error;

const TIME_EPSILON: f64 = 1e-12;

fn flatten_pyarray2<T: Copy + Element>(
    array: PyReadonlyArray2<T>,
    name: &str,
) -> Result<(usize, usize, Vec<T>), SimError> {
    let shape = array.shape();
    if shape.len() != 2 {
        return Err(SimError::Shape(format!("{name} must be 2-dimensional")));
    }
    let data = array
        .as_slice()
        .map_err(|_| SimError::Shape(format!("{name} must be contiguous")))?;
    Ok((shape[0], shape[1], data.to_vec()))
}

fn read_array1<T: Copy + Element>(
    array: PyReadonlyArray1<T>,
    expected: usize,
    name: &str,
) -> Result<Vec<T>, SimError> {
    let slice = array
        .as_slice()
        .map_err(|_| SimError::Shape(format!("{name} array must be contiguous")))?;
    if slice.len() != expected {
        return Err(SimError::Shape(format!(
            "{name} length {} does not match reaction count {}",
            slice.len(),
            expected
        )));
    }
    Ok(slice.to_vec())
}

fn extract_param_rows(
    params: Option<PyReadonlyArray2<f64>>,
    n_reactions: usize,
) -> Result<Option<Vec<Vec<f64>>>, SimError> {
    params
        .map(|arr| {
            let (rows, width, flat) = flatten_pyarray2(arr, "reaction parameter array")?;
            if rows != n_reactions {
                return Err(SimError::Shape(format!(
                    "reaction parameter rows {} do not match reaction count {}",
                    rows, n_reactions
                )));
            }
            let mut out = Vec::with_capacity(rows);
            if width == 0 {
                out.resize_with(rows, Vec::new);
            } else {
                for chunk in flat.chunks(width) {
                    out.push(chunk.to_vec());
                }
            }
            Ok(out)
        })
        .transpose()
}

fn extract_expression_infos(
    expressions: Option<Bound<'_, PyAny>>,
    n_reactions: usize,
    n_species: usize,
) -> Result<Option<Vec<Option<ExpressionInfo>>>, SimError> {
    if let Some(obj) = expressions {
        if obj.is_none() {
            return Ok(None);
        }
        let entries: Vec<Option<String>> = obj.extract().map_err(|_| {
            SimError::InvalidArgument(
                "reaction_expressions must be a sequence of optional strings".into(),
            )
        })?;
        if entries.len() != n_reactions {
            return Err(SimError::Shape(format!(
                "reaction_expressions length {} does not match reaction count {}",
                entries.len(),
                n_reactions
            )));
        }
        let mut compiled = Vec::with_capacity(n_reactions);
        for (idx, maybe_expr) in entries.into_iter().enumerate() {
            if let Some(expr_str) = maybe_expr {
                let expr = Expr::from_str(&expr_str).map_err(|err| {
                    SimError::InvalidArgument(format!(
                        "reaction {} expression parse error: {}",
                        idx, err
                    ))
                })?;
                let species_refs = collect_species_refs(&expr_str, idx, n_species)?;
                compiled.push(Some(ExpressionInfo { expr, species_refs }));
            } else {
                compiled.push(None);
            }
        }
        Ok(Some(compiled))
    } else {
        Ok(None)
    }
}

fn collect_species_refs(
    expr_str: &str,
    reaction_idx: usize,
    n_species: usize,
) -> Result<Vec<usize>, SimError> {
    let mut refs = Vec::new();
    let bytes = expr_str.as_bytes();
    let mut idx = 0;
    while idx < bytes.len() {
        let ch = bytes[idx];
        if ch == b's' || ch == b'S' {
            let mut end = idx + 1;
            while end < bytes.len() && bytes[end].is_ascii_digit() {
                end += 1;
            }
            if end > idx + 1 {
                let digits = &expr_str[idx + 1..end];
                let species_idx = digits.parse::<usize>().map_err(|_| {
                    SimError::InvalidArgument(format!(
                        "reaction {} expression contains invalid species index '{}'",
                        reaction_idx, digits
                    ))
                })?;
                if species_idx >= n_species {
                    return Err(SimError::InvalidArgument(format!(
                        "reaction {} expression species index {} exceeds number of species {}",
                        reaction_idx, species_idx, n_species
                    )));
                }
                if !refs.contains(&species_idx) {
                    refs.push(species_idx);
                }
                idx = end;
                continue;
            }
        }
        idx += 1;
    }
    Ok(refs)
}

fn parse_species_variable(name: &str) -> Option<usize> {
    let digits = name.strip_prefix('s').or_else(|| name.strip_prefix('S'))?;
    if digits.is_empty() {
        return None;
    }
    digits.parse::<usize>().ok()
}

fn parse_interventions(
    interventions: Option<Bound<'_, PyAny>>,
    n_species: usize,
    n_reactions: usize,
) -> Result<InterventionPlan, SimError> {
    if let Some(obj) = interventions {
        if obj.is_none() {
            return Ok(InterventionPlan::default());
        }
        let py = obj.py();
        let entries: Vec<Py<PyAny>> = obj.extract().map_err(|_| {
            SimError::InvalidArgument(
                "interventions must be a sequence of mappings with time/action fields".into(),
            )
        })?;
        let mut events = Vec::with_capacity(entries.len());
        let mut last_time = -f64::INFINITY;
        for raw in entries {
            let bound = raw.bind(py);
            let dict = bound.cast::<PyDict>().map_err(|_| {
                SimError::InvalidArgument(
                    "each intervention entry must be a mapping with named fields".into(),
                )
            })?;
            let time_obj = dict
                .get_item("time")
                .map_err(|_| SimError::InvalidArgument("intervention entry missing 'time'".into()))?
                .ok_or_else(|| {
                    SimError::InvalidArgument("intervention entry missing 'time'".into())
                })?;
            let time: f64 = time_obj.extract().map_err(|_| {
                SimError::InvalidArgument("intervention time must be a float".into())
            })?;
            if time.is_nan() || time < 0.0 {
                return Err(SimError::InvalidArgument(
                    "intervention times must be non-negative numbers".into(),
                ));
            }
            if time + TIME_EPSILON < last_time {
                return Err(SimError::InvalidArgument(
                    "intervention times must be sorted".into(),
                ));
            }
            last_time = time;
            let mut actions = Vec::new();
            if let Some(list_obj) = dict.get_item("species_delta").map_err(|_| {
                SimError::InvalidArgument(
                    "failed to read species_delta from intervention entry".into(),
                )
            })? {
                if !list_obj.is_none() {
                    let list: Vec<(usize, i32)> = list_obj.extract().map_err(|_| {
                        SimError::InvalidArgument(
                            "species_delta must be a sequence of (species, delta) pairs".into(),
                        )
                    })?;
                    for (species, delta) in list {
                        if species >= n_species {
                            return Err(SimError::InvalidArgument(format!(
                                "species_delta refers to invalid species index {}",
                                species
                            )));
                        }
                        if delta != 0 {
                            actions.push(InterventionAction::SpeciesDelta { species, delta });
                        }
                    }
                }
            }
            if let Some(list_obj) = dict.get_item("species_set").map_err(|_| {
                SimError::InvalidArgument(
                    "failed to read species_set from intervention entry".into(),
                )
            })? {
                if !list_obj.is_none() {
                    let list: Vec<(usize, i32)> = list_obj.extract().map_err(|_| {
                        SimError::InvalidArgument(
                            "species_set must be a sequence of (species, value) pairs".into(),
                        )
                    })?;
                    for (species, value) in list {
                        if species >= n_species {
                            return Err(SimError::InvalidArgument(format!(
                                "species_set refers to invalid species index {}",
                                species
                            )));
                        }
                        actions.push(InterventionAction::SpeciesSet { species, value });
                    }
                }
            }
            if let Some(list_obj) = dict.get_item("rate_set").map_err(|_| {
                SimError::InvalidArgument("failed to read rate_set from intervention entry".into())
            })? {
                if !list_obj.is_none() {
                    let list: Vec<(usize, f64)> = list_obj.extract().map_err(|_| {
                        SimError::InvalidArgument(
                            "rate_set must be a sequence of (reaction, value) pairs".into(),
                        )
                    })?;
                    for (reaction, value) in list {
                        if reaction >= n_reactions {
                            return Err(SimError::InvalidArgument(format!(
                                "rate_set refers to invalid reaction index {}",
                                reaction
                            )));
                        }
                        actions.push(InterventionAction::RateSet { reaction, value });
                    }
                }
            }
            if actions.is_empty() {
                return Err(SimError::InvalidArgument(
                    "each intervention must specify at least one action".into(),
                ));
            }
            events.push(InterventionEvent { time, actions });
        }
        return Ok(InterventionPlan { events });
    }
    Ok(InterventionPlan::default())
}

#[derive(Debug, Error)]
enum SimError {
    #[error("shape mismatch: {0}")]
    Shape(String),
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("thread pool error: {0}")]
    ThreadPool(String),
}

impl From<SimError> for PyErr {
    fn from(err: SimError) -> Self {
        PyValueError::new_err(err.to_string())
    }
}

#[derive(Clone, Debug)]
struct Reactant {
    species: usize,
    count: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReactionTypeCode {
    MassAction = 0,
    Hill = 1,
    MichaelisMenten = 2,
    Expression = 3,
}

impl TryFrom<i32> for ReactionTypeCode {
    type Error = SimError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::MassAction),
            1 => Ok(Self::Hill),
            2 => Ok(Self::MichaelisMenten),
            3 => Ok(Self::Expression),
            other => Err(SimError::InvalidArgument(format!(
                "unsupported reaction type code {} (expected 0=mass-action, 1=Hill, 2=Michaelis-Menten, or 3=expression)",
                other
            ))),
        }
    }
}

#[derive(Clone, Debug)]
enum ReactionKind {
    MassAction,
    Hill {
        activator: usize,
        hill_n: f64,
        #[allow(dead_code)] // Kept for debugging, k_half_pow_n is used for performance
        k_half: f64,
        k_half_pow_n: f64, // Cached k_half^hill_n
    },
    MichaelisMenten {
        substrate: usize,
        k_m: f64,
    },
    Expression {
        expr: Expr,
        species_refs: Vec<usize>,
    },
}

#[derive(Clone, Debug)]
struct Reaction {
    rate_constant: f64,
    reactants: Vec<Reactant>,
    kind: ReactionKind,
}

#[derive(Clone, Debug)]
struct ExpressionInfo {
    expr: Expr,
    species_refs: Vec<usize>,
}

#[derive(Clone, Debug)]
enum InterventionAction {
    SpeciesDelta { species: usize, delta: i32 },
    SpeciesSet { species: usize, value: i32 },
    RateSet { reaction: usize, value: f64 },
}

#[derive(Clone, Debug)]
struct InterventionEvent {
    time: f64,
    actions: Vec<InterventionAction>,
}

#[derive(Clone, Debug, Default)]
struct InterventionPlan {
    events: Vec<InterventionEvent>,
}

struct SpeciesContext<'a> {
    state: &'a [i32],
}

impl<'a> ContextProvider for SpeciesContext<'a> {
    fn get_var(&self, name: &str) -> Option<f64> {
        parse_species_variable(name).map(|idx| self.state[idx].max(0) as f64)
    }
}

impl Reaction {
    #[inline]
    fn propensity(&self, rate_constant: f64, state: &[i32]) -> f64 {
        match self.kind {
            ReactionKind::MassAction => {
                let mut propensity = rate_constant;
                for reactant in &self.reactants {
                    let available = state[reactant.species];
                    if available < reactant.count {
                        return 0.0;
                    }
                    propensity *= falling_factorial(available, reactant.count);
                }
                propensity
            }
            ReactionKind::Hill {
                activator,
                hill_n,
                k_half_pow_n,
                ..
            } => {
                let concentration = state[activator].max(0) as f64;
                // concentration is guaranteed >= 0.0 after max(0)
                let power = concentration.powf(hill_n);
                let denom = k_half_pow_n + power;
                if denom == 0.0 {
                    0.0
                } else {
                    rate_constant * power / denom
                }
            }
            ReactionKind::MichaelisMenten { substrate, k_m } => {
                let substrate_count = state[substrate].max(0) as f64;
                let denom = k_m + substrate_count;
                if denom == 0.0 {
                    0.0
                } else {
                    rate_constant * substrate_count / denom
                }
            }
            ReactionKind::Expression { ref expr, .. } => {
                let ctx = (SpeciesContext { state }, Context::new());
                expr.eval_with_context(ctx).unwrap_or(0.0)
            }
        }
    }
}

impl ReactionKind {
    fn from_code(
        code: ReactionTypeCode,
        reaction_idx: usize,
        n_species: usize,
        params: Option<&[f64]>,
        expression: Option<&ExpressionInfo>,
    ) -> Result<Self, SimError> {
        match code {
            ReactionTypeCode::MassAction => Ok(Self::MassAction),
            ReactionTypeCode::Hill => {
                let params = params.ok_or_else(|| {
                    SimError::InvalidArgument(format!(
                        "reaction {} requires Hill parameters",
                        reaction_idx
                    ))
                })?;
                if params.len() < 3 {
                    return Err(SimError::InvalidArgument(format!(
                        "reaction {} Hill parameters require [activator, hill_n, K]",
                        reaction_idx
                    )));
                }
                let activator = params[0] as isize;
                if activator < 0 {
                    return Err(SimError::InvalidArgument(format!(
                        "reaction {} activator index must be non-negative",
                        reaction_idx
                    )));
                }
                let hill_n = params[1];
                let k_half = params[2];
                if hill_n <= 0.0 || k_half <= 0.0 {
                    return Err(SimError::InvalidArgument(format!(
                        "reaction {} Hill parameters must be positive",
                        reaction_idx
                    )));
                }
                let activator = activator as usize;
                if activator >= n_species {
                    return Err(SimError::InvalidArgument(format!(
                        "reaction {} activator index {} exceeds number of species {}",
                        reaction_idx, activator, n_species
                    )));
                }
                Ok(Self::Hill {
                    activator,
                    hill_n,
                    k_half,
                    k_half_pow_n: k_half.powf(hill_n),
                })
            }
            ReactionTypeCode::MichaelisMenten => {
                let params = params.ok_or_else(|| {
                    SimError::InvalidArgument(format!(
                        "reaction {} requires Michaelis-Menten parameters",
                        reaction_idx
                    ))
                })?;
                if params.len() < 2 {
                    return Err(SimError::InvalidArgument(format!(
                        "reaction {} Michaelis-Menten parameters require [substrate_index, k_m]",
                        reaction_idx
                    )));
                }
                let substrate = params[0] as isize;
                if substrate < 0 {
                    return Err(SimError::InvalidArgument(format!(
                        "reaction {} substrate index must be non-negative",
                        reaction_idx
                    )));
                }
                let k_m = params[1];
                if k_m <= 0.0 {
                    return Err(SimError::InvalidArgument(format!(
                        "reaction {} Michaelis-Menten k_m must be positive",
                        reaction_idx
                    )));
                }
                let substrate = substrate as usize;
                if substrate >= n_species {
                    return Err(SimError::InvalidArgument(format!(
                        "reaction {} substrate index {} exceeds number of species {}",
                        reaction_idx, substrate, n_species
                    )));
                }
                Ok(Self::MichaelisMenten { substrate, k_m })
            }
            ReactionTypeCode::Expression => {
                let info = expression.ok_or_else(|| {
                    SimError::InvalidArgument(format!(
                        "reaction {} requires an expression string",
                        reaction_idx
                    ))
                })?;
                Ok(Self::Expression {
                    expr: info.expr.clone(),
                    species_refs: info.species_refs.clone(),
                })
            }
        }
    }
}

#[derive(Clone, Debug)]
struct SpeciesDelta {
    species: usize,
    delta: i32,
}

struct Model {
    n_species: usize,
    n_reactions: usize,
    reaction_deltas: Vec<Vec<SpeciesDelta>>,
    reactions: Vec<Reaction>,
    dependencies: Vec<Vec<usize>>,
}

impl Model {
    fn from_inputs(
        stoich: PyReadonlyArray2<i32>,
        rate_constants: PyReadonlyArray1<f64>,
        reaction_type_codes: PyReadonlyArray1<i32>,
        reaction_type_params: Option<PyReadonlyArray2<f64>>,
        reaction_expressions: Option<Bound<'_, PyAny>>,
    ) -> Result<Self, SimError> {
        let (n_reactions, n_species, stoich_vec) =
            flatten_pyarray2(stoich, "stoichiometry matrix")?;
        if n_reactions == 0 || n_species == 0 {
            return Err(SimError::InvalidArgument(
                "stoichiometry must contain at least one reaction and one species".into(),
            ));
        }

        let rate_constants_vec = read_array1(rate_constants, n_reactions, "rate constant")?;
        let reaction_type_vec = read_array1(reaction_type_codes, n_reactions, "reaction type")?;

        let reaction_param_rows = extract_param_rows(reaction_type_params, n_reactions)?;
        let expression_infos =
            extract_expression_infos(reaction_expressions, n_reactions, n_species)?;

        let mut reactions = Vec::with_capacity(n_reactions);
        for (idx, row) in stoich_vec.chunks_exact(n_species).enumerate() {
            let reactants: Vec<_> = row
                .iter()
                .enumerate()
                .filter_map(|(species, &delta)| {
                    (delta < 0).then_some(Reactant {
                        species,
                        count: -delta,
                    })
                })
                .collect();
            let params = reaction_param_rows
                .as_ref()
                .and_then(|rows| rows.get(idx))
                .map(Vec::as_slice);
            let expression_info = expression_infos
                .as_ref()
                .and_then(|list| list.get(idx))
                .and_then(Option::as_ref);
            let reaction_type = ReactionTypeCode::try_from(reaction_type_vec[idx])?;
            if reaction_type != ReactionTypeCode::Expression && expression_info.is_some() {
                return Err(SimError::InvalidArgument(format!(
                    "reaction {} provided an expression but is not marked as ReactionType.EXPRESSION",
                    idx
                )));
            }
            let kind =
                ReactionKind::from_code(reaction_type, idx, n_species, params, expression_info)?;
            reactions.push(Reaction {
                rate_constant: rate_constants_vec[idx],
                reactants,
                kind,
            });
        }

        let reaction_deltas = build_reaction_deltas(n_species, &stoich_vec);
        let dependencies = build_dependency_graph(n_species, &reaction_deltas, &reactions);

        Ok(Self {
            n_species,
            n_reactions,
            reaction_deltas,
            reactions,
            dependencies,
        })
    }
}

fn build_dependency_graph(
    n_species: usize,
    reaction_deltas: &[Vec<SpeciesDelta>],
    reactions: &[Reaction],
) -> Vec<Vec<usize>> {
    let mut species_dependents: Vec<Vec<usize>> = vec![Vec::new(); n_species];
    for (idx, reaction) in reactions.iter().enumerate() {
        for reactant in &reaction.reactants {
            species_dependents[reactant.species].push(idx);
        }
        if let ReactionKind::Hill { activator, .. } = reaction.kind {
            species_dependents[activator].push(idx);
        }
        if let ReactionKind::MichaelisMenten { substrate, .. } = reaction.kind {
            species_dependents[substrate].push(idx);
        }
        if let ReactionKind::Expression {
            ref species_refs, ..
        } = reaction.kind
        {
            for &species in species_refs {
                species_dependents[species].push(idx);
            }
        }
    }

    let mut dependencies = vec![Vec::new(); reactions.len()];
    let mut visit_markers = vec![0usize; reactions.len()];
    let mut stamp = 1usize;
    for (r, deps) in dependencies.iter_mut().enumerate() {
        if stamp == usize::MAX {
            visit_markers.fill(0);
            stamp = 1;
        }
        let mark = stamp;
        stamp += 1;

        deps.clear();
        visit_markers[r] = mark;
        deps.push(r);
        for delta in &reaction_deltas[r] {
            for &dep in &species_dependents[delta.species] {
                if visit_markers[dep] != mark {
                    visit_markers[dep] = mark;
                    deps.push(dep);
                }
            }
        }
    }
    dependencies
}

fn build_reaction_deltas(n_species: usize, stoich: &[i32]) -> Vec<Vec<SpeciesDelta>> {
    stoich
        .chunks_exact(n_species)
        .map(|row| {
            row.iter()
                .enumerate()
                .filter_map(|(species, &delta)| {
                    (delta != 0).then_some(SpeciesDelta { species, delta })
                })
                .collect()
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputMode {
    Timeseries,
    FinalOnly,
}

struct SimulationOptions {
    t_end: f64,
    t_points: Vec<f64>,
    mode: OutputMode,
    interventions: InterventionPlan,
}

struct SimulationOutput {
    data: Vec<i32>,
    n_traj: usize,
    n_species: usize,
    n_times: usize,
    mode: OutputMode,
}

#[derive(Clone, Debug)]
struct PropensityTree {
    len: usize,
    leaf_count: usize,
    data: Vec<f64>,
}

impl PropensityTree {
    fn new(len: usize) -> Self {
        let base = len.max(1);
        let leaf_count = base.next_power_of_two();
        Self {
            len,
            leaf_count,
            data: vec![0.0; leaf_count * 2],
        }
    }

    fn rebuild(&mut self, values: &[f64]) {
        debug_assert_eq!(values.len(), self.len);
        self.data.fill(0.0);
        for (idx, &value) in values.iter().enumerate() {
            self.data[self.leaf_count + idx] = value;
        }
        for idx in (1..self.leaf_count).rev() {
            self.data[idx] = self.data[idx << 1] + self.data[idx << 1 | 1];
        }
    }

    fn total(&self) -> f64 {
        self.data[1]
    }

    fn update(&mut self, idx: usize, value: f64) {
        let mut pos = self.leaf_count + idx;
        self.data[pos] = value;
        while pos > 1 {
            pos >>= 1;
            self.data[pos] = self.data[pos << 1] + self.data[pos << 1 | 1];
        }
    }

    fn select(&self, mut target: f64) -> usize {
        debug_assert!(self.len > 0);
        debug_assert!(target >= 0.0);
        let mut node = 1usize;
        while node < self.leaf_count {
            let left = self.data[node << 1];
            if left > 0.0 && target <= left {
                node <<= 1;
            } else {
                target -= left;
                node = (node << 1) | 1;
            }
        }
        let idx = node - self.leaf_count;
        if idx >= self.len { self.len - 1 } else { idx }
    }
}

impl Default for PropensityTree {
    fn default() -> Self {
        Self::new(1)
    }
}

impl SimulationOutput {
    fn into_py(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dims = match self.mode {
            OutputMode::Timeseries => vec![self.n_traj, self.n_times, self.n_species],
            OutputMode::FinalOnly => vec![self.n_traj, self.n_species],
        };
        let array = unsafe { PyArrayDyn::<i32>::new(py, IxDyn(&dims), false) };
        unsafe {
            array
                .as_slice_mut()
                .map_err(|_| PyValueError::new_err("failed to export data"))?
                .copy_from_slice(&self.data);
        }
        Ok(array.into_any().unbind())
    }
}

#[cfg_attr(not(test), allow(dead_code))]
struct TrajectoryRecord {
    states: Vec<i32>,
}

struct StateRecorder<'a> {
    buffer: &'a mut [i32],
    write_idx: usize,
    n_species: usize,
}

impl<'a> StateRecorder<'a> {
    fn new(buffer: &'a mut [i32], n_species: usize) -> Self {
        Self {
            buffer,
            write_idx: 0,
            n_species,
        }
    }

    fn record(&mut self, state: &[i32]) {
        let end = self.write_idx + self.n_species;
        debug_assert!(end <= self.buffer.len());
        self.buffer[self.write_idx..end].copy_from_slice(state);
        self.write_idx = end;
    }

    fn len(&self) -> usize {
        self.write_idx
    }
}

#[derive(Default)]
struct TrajectoryScratch {
    state: Vec<i32>,
    rate_constants: Vec<f64>,
    propensities: Vec<f64>,
    prop_tree: PropensityTree,
}

impl TrajectoryScratch {
    fn ensure(&mut self, n_species: usize, n_reactions: usize) {
        if self.state.len() != n_species {
            self.state.resize(n_species, 0);
        }
        if self.rate_constants.len() != n_reactions {
            self.rate_constants.resize(n_reactions, 0.0);
        }
        if self.propensities.len() != n_reactions {
            self.propensities.resize(n_reactions, 0.0);
        }
        if self.prop_tree.len != n_reactions {
            self.prop_tree = PropensityTree::new(n_reactions);
        }
    }
}

thread_local! {
    static TRAJECTORY_SCRATCH: RefCell<TrajectoryScratch> = RefCell::new(TrajectoryScratch::default());
}

fn recompute_propensities(
    reactions: &[Reaction],
    rate_constants: &[f64],
    state: &[i32],
    propensities: &mut [f64],
) -> f64 {
    let mut total = 0.0;
    for (idx, reaction) in reactions.iter().enumerate() {
        let value = reaction.propensity(rate_constants[idx], state);
        total += value;
        propensities[idx] = value;
    }
    total
}

fn apply_interventions(
    plan: &InterventionPlan,
    next_idx: &mut usize,
    current_time: f64,
    state: &mut [i32],
    rate_constants: &mut [f64],
    reactions: &[Reaction],
    propensities: &mut [f64],
    total_propensity: &mut f64,
    tree: &mut PropensityTree,
) {
    if plan.events.is_empty() {
        return;
    }
    let mut applied = false;
    while let Some(event) = plan.events.get(*next_idx) {
        if current_time + TIME_EPSILON < event.time {
            break;
        }
        for action in &event.actions {
            match *action {
                InterventionAction::SpeciesDelta { species, delta } => {
                    state[species] += delta;
                }
                InterventionAction::SpeciesSet { species, value } => {
                    state[species] = value;
                }
                InterventionAction::RateSet { reaction, value } => {
                    rate_constants[reaction] = value;
                }
            }
        }
        *next_idx += 1;
        applied = true;
    }
    if applied {
        recompute_propensities(reactions, rate_constants, state, propensities);
        tree.rebuild(propensities);
        *total_propensity = tree.total();
    }
}

fn run_ensemble(
    model: &Model,
    initial_state: &[i32],
    per_traj_initial_states: Option<&[i32]>,
    per_traj_initial_times: Option<&[f64]>,
    options: &SimulationOptions,
    n_trajectories: usize,
    n_threads: Option<usize>,
    seed: Option<u64>,
) -> Result<SimulationOutput, SimError> {
    if initial_state.len() != model.n_species {
        return Err(SimError::Shape(format!(
            "initial state length {} does not match number of species {}",
            initial_state.len(),
            model.n_species
        )));
    }
    if let Some(states) = per_traj_initial_states {
        let expected = n_trajectories.checked_mul(model.n_species).ok_or_else(|| {
            SimError::Shape("initial_states size exceeds allowable limits".into())
        })?;
        if states.len() != expected {
            return Err(SimError::Shape(format!(
                "initial_states length {} does not match {} trajectories × {} species (expected {})",
                states.len(),
                n_trajectories,
                model.n_species,
                expected
            )));
        }
    }
    if let Some(times) = per_traj_initial_times {
        if times.len() != n_trajectories {
            return Err(SimError::Shape(format!(
                "initial_times length {} does not match number of trajectories {}",
                times.len(),
                n_trajectories
            )));
        }
    }
    if n_trajectories == 0 {
        return Err(SimError::InvalidArgument(
            "number of trajectories must be greater than zero".into(),
        ));
    }
    if options.t_end <= 0.0 {
        return Err(SimError::InvalidArgument("t_end must be positive".into()));
    }

    let n_times = match options.mode {
        OutputMode::Timeseries => options.t_points.len(),
        OutputMode::FinalOnly => 1,
    };
    let stride = n_times * model.n_species;
    let mut data = vec![0i32; n_trajectories * stride];
    let species_len = model.n_species;
    let per_traj_states = per_traj_initial_states;
    let per_traj_times = per_traj_initial_times;

    let mut simulate = || -> Result<(), SimError> {
        data.par_chunks_mut(stride)
            .enumerate()
            .try_for_each(|(traj_idx, chunk)| {
                let state_slice = if let Some(data) = per_traj_states {
                    let start = traj_idx * species_len;
                    &data[start..start + species_len]
                } else {
                    initial_state
                };
                let start_time = per_traj_times
                    .map(|times| times[traj_idx].max(0.0))
                    .unwrap_or(0.0);
                let mut rng = ChaCha8Rng::seed_from_u64(derive_seed(seed, traj_idx as u64));
                simulate_single_into(model, state_slice, start_time, options, &mut rng, chunk)
            })
    };

    match n_threads {
        Some(n) => ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .map_err(|e| SimError::ThreadPool(e.to_string()))?
            .install(|| simulate())?,
        None => simulate()?,
    };

    Ok(SimulationOutput {
        data,
        n_traj: n_trajectories,
        n_species: model.n_species,
        n_times,
        mode: options.mode,
    })
}

#[cfg_attr(not(test), allow(dead_code))]
fn simulate_single(
    model: &Model,
    initial_state: &[i32],
    start_time: f64,
    options: &SimulationOptions,
    rng: &mut ChaCha8Rng,
) -> Result<TrajectoryRecord, SimError> {
    let record_len = match options.mode {
        OutputMode::Timeseries => options.t_points.len(),
        OutputMode::FinalOnly => 1,
    } * model.n_species;
    let mut recorded_states = vec![0i32; record_len];
    simulate_single_into(
        model,
        initial_state,
        start_time,
        options,
        rng,
        &mut recorded_states,
    )?;
    Ok(TrajectoryRecord {
        states: recorded_states,
    })
}

fn simulate_single_into(
    model: &Model,
    initial_state: &[i32],
    start_time: f64,
    options: &SimulationOptions,
    rng: &mut ChaCha8Rng,
    output: &mut [i32],
) -> Result<(), SimError> {
    TRAJECTORY_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        scratch.ensure(model.n_species, model.n_reactions);
        let TrajectoryScratch {
            state,
            rate_constants,
            propensities,
            prop_tree,
        } = &mut *scratch;
        state[..model.n_species].copy_from_slice(initial_state);
        for (dst, reaction) in rate_constants.iter_mut().zip(model.reactions.iter()) {
            *dst = reaction.rate_constant;
        }
        simulate_with_scratch(
            model,
            start_time,
            options,
            rng,
            output,
            state,
            rate_constants,
            propensities,
            prop_tree,
        )
    })
}

fn simulate_with_scratch(
    model: &Model,
    start_time: f64,
    options: &SimulationOptions,
    rng: &mut ChaCha8Rng,
    output: &mut [i32],
    state: &mut [i32],
    rate_constants: &mut [f64],
    propensities: &mut [f64],
    prop_tree: &mut PropensityTree,
) -> Result<(), SimError> {
    recompute_propensities(&model.reactions, rate_constants, state, propensities);
    prop_tree.rebuild(propensities);
    let mut total_propensity = prop_tree.total();

    let mut current_time = start_time.max(0.0);
    if current_time > options.t_end {
        current_time = options.t_end;
    }
    let mut next_intervention_idx = 0usize;
    while let Some(event) = options.interventions.events.get(next_intervention_idx) {
        if event.time + TIME_EPSILON < current_time {
            next_intervention_idx += 1;
        } else {
            break;
        }
    }
    apply_interventions(
        &options.interventions,
        &mut next_intervention_idx,
        current_time,
        state,
        rate_constants,
        &model.reactions,
        propensities,
        &mut total_propensity,
        prop_tree,
    );
    let mut recorder = StateRecorder::new(output, model.n_species);
    let mut next_time_idx = 0usize;
    if options.mode == OutputMode::Timeseries {
        while let Some(&tp) = options.t_points.get(next_time_idx) {
            if tp + TIME_EPSILON < current_time {
                next_time_idx += 1;
            } else {
                break;
            }
        }
    }
    record_due(
        options.mode,
        &options.t_points,
        &mut next_time_idx,
        current_time,
        state,
        &mut recorder,
    );

    while current_time < options.t_end - TIME_EPSILON {
        let next_intervention_time = options
            .interventions
            .events
            .get(next_intervention_idx)
            .map(|event| event.time)
            .unwrap_or(f64::INFINITY);

        let next_boundary = next_intervention_time.min(options.t_end);
        let time_to_boundary = if next_boundary > current_time {
            next_boundary - current_time
        } else {
            0.0
        };

        let mut tau = f64::INFINITY;
        if total_propensity > 0.0 {
            let u1: f64 = rng.r#gen();
            tau = -u1.ln() / total_propensity;
        }

        if tau + TIME_EPSILON >= time_to_boundary {
            // Boundary reached (intervention or t_end)
            current_time = next_boundary;
            apply_interventions(
                &options.interventions,
                &mut next_intervention_idx,
                current_time,
                state,
                rate_constants,
                &model.reactions,
                propensities,
                &mut total_propensity,
                prop_tree,
            );
        } else {
            // Reaction occurs
            current_time += tau;
            let u2: f64 = rng.r#gen();
            let chosen = prop_tree.select(u2 * total_propensity);

            for delta in &model.reaction_deltas[chosen] {
                state[delta.species] += delta.delta;
            }

            for &dep in &model.dependencies[chosen] {
                let new_value = model.reactions[dep].propensity(rate_constants[dep], state);
                propensities[dep] = new_value;
                prop_tree.update(dep, new_value);
            }
            total_propensity = prop_tree.total();
        }

        record_due(
            options.mode,
            &options.t_points,
            &mut next_time_idx,
            current_time,
            state,
            &mut recorder,
        );
    }

    finalize_recording(options.mode, options.t_points.len(), &mut recorder, state);

    Ok(())
}

#[inline]
fn record_due(
    mode: OutputMode,
    t_points: &[f64],
    next_idx: &mut usize,
    current_time: f64,
    state: &[i32],
    recorder: &mut StateRecorder<'_>,
) {
    if mode != OutputMode::Timeseries {
        return;
    }
    while let Some(&tp) = t_points.get(*next_idx) {
        if current_time + TIME_EPSILON >= tp {
            recorder.record(state);
            *next_idx += 1;
        } else {
            break;
        }
    }
}

fn finalize_recording(
    mode: OutputMode,
    t_points_len: usize,
    recorder: &mut StateRecorder<'_>,
    state: &[i32],
) {
    match mode {
        OutputMode::Timeseries => {
            let target_len = t_points_len * state.len();
            while recorder.len() < target_len {
                recorder.record(state);
            }
        }
        OutputMode::FinalOnly => {
            if recorder.len() < state.len() {
                recorder.record(state);
            }
        }
    }
}

#[inline]
fn falling_factorial(value: i32, count: i32) -> f64 {
    match count {
        0 => 1.0,
        1 => value as f64,
        2 if value >= 2 => (value * (value - 1)) as f64,
        3 if value >= 3 => (value * (value - 1) * (value - 2)) as f64,
        _ if value < count => 0.0,
        _ => {
            let mut acc = 1.0;
            for i in 0..count {
                acc *= (value - i) as f64;
            }
            acc
        }
    }
}

fn derive_seed(seed: Option<u64>, trajectory: u64) -> u64 {
    const GOLDEN_GAMMA: u64 = 0x9E3779B97F4A7C15;
    let base = seed.unwrap_or(0xDEADBEEFCAFEBABE);
    let mut z = base ^ (trajectory.wrapping_mul(GOLDEN_GAMMA));
    // SplitMix64
    z = z.wrapping_add(GOLDEN_GAMMA);
    let mut result = z;
    result = (result ^ (result >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    result = (result ^ (result >> 27)).wrapping_mul(0x94D049BB133111EB);
    result ^ (result >> 31)
}

fn parse_t_points(array: Option<PyReadonlyArray1<f64>>) -> Result<Vec<f64>, SimError> {
    if let Some(arr) = array {
        let slice = arr
            .as_slice()
            .map_err(|_| SimError::Shape("t_points array must be contiguous".into()))?;
        if slice.windows(2).any(|w| w[0] > w[1] + TIME_EPSILON) {
            return Err(SimError::InvalidArgument(
                "t_points must be sorted in ascending order".into(),
            ));
        }
        Ok(slice.to_vec())
    } else {
        Ok(Vec::new())
    }
}

fn parse_mode(mode: Option<&str>, has_t_points: bool) -> Result<OutputMode, SimError> {
    match mode {
        Some(m) if m.eq_ignore_ascii_case("final") => Ok(OutputMode::FinalOnly),
        Some(m) if m.eq_ignore_ascii_case("timeseries") => {
            if !has_t_points {
                Err(SimError::InvalidArgument(
                    "timeseries mode requires t_points".into(),
                ))
            } else {
                Ok(OutputMode::Timeseries)
            }
        }
        Some(other) => Err(SimError::InvalidArgument(format!(
            "unrecognized mode '{}'",
            other
        ))),
        None => {
            if has_t_points {
                Ok(OutputMode::Timeseries)
            } else {
                Ok(OutputMode::FinalOnly)
            }
        }
    }
}

#[pyfunction(signature = (
    stoich,
    initial_state,
    rate_constants,
    reaction_type_codes,
    t_end,
    n_trajectories,
    initial_states=None,
    initial_times=None,
    t_points=None,
    reaction_type_params=None,
    reaction_expressions=None,
    interventions=None,
    n_threads=None,
    seed=None,
    mode=None
))]
pub fn simulate_ensemble(
    py: Python<'_>,
    stoich: PyReadonlyArray2<i32>,
    initial_state: PyReadonlyArray1<i32>,
    rate_constants: PyReadonlyArray1<f64>,
    reaction_type_codes: PyReadonlyArray1<i32>,
    t_end: f64,
    n_trajectories: usize,
    initial_states: Option<PyReadonlyArray2<i32>>,
    initial_times: Option<PyReadonlyArray1<f64>>,
    t_points: Option<PyReadonlyArray1<f64>>,
    reaction_type_params: Option<PyReadonlyArray2<f64>>,
    reaction_expressions: Option<Py<PyAny>>,
    interventions: Option<Py<PyAny>>,
    n_threads: Option<usize>,
    seed: Option<u64>,
    mode: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let t_points_vec = parse_t_points(t_points)?;
    let has_t_points = !t_points_vec.is_empty();
    let output_mode = parse_mode(mode, has_t_points)?;
    if output_mode == OutputMode::Timeseries && !has_t_points {
        return Err(PyValueError::new_err(
            "timeseries mode requires at least one t_point",
        ));
    }
    if output_mode == OutputMode::Timeseries
        && t_points_vec.last().copied().unwrap_or(0.0) > t_end + TIME_EPSILON
    {
        return Err(PyValueError::new_err("t_points cannot exceed t_end"));
    }
    let initial_state_vec = initial_state
        .as_slice()
        .map_err(|_| PyValueError::new_err("initial_state must be contiguous"))?
        .to_vec();
    let per_traj_initial_states_info = initial_states
        .map(|arr| flatten_pyarray2(arr, "initial_states"))
        .transpose()
        .map_err(PyErr::from)?;
    let per_traj_initial_times_vec = if let Some(arr) = initial_times {
        let slice = arr.as_slice().map_err(|_| {
            PyValueError::new_err("initial_times array must be contiguous f64 vector")
        })?;
        Some(slice.to_vec())
    } else {
        None
    };
    let expr_bound = reaction_expressions.as_ref().map(|obj| obj.bind(py));
    let interventions_bound = interventions.as_ref().map(|obj| obj.bind(py));
    let model = Model::from_inputs(
        stoich,
        rate_constants,
        reaction_type_codes,
        reaction_type_params,
        expr_bound.cloned(),
    )?;
    let plan = parse_interventions(
        interventions_bound.cloned(),
        model.n_species,
        model.n_reactions,
    )?;
    let mut per_traj_initial_states_vec = None;
    let mut per_traj_initial_times = None;
    if let Some((rows, cols, data)) = per_traj_initial_states_info {
        if rows != n_trajectories {
            return Err(PyValueError::new_err(format!(
                "initial_states rows {} do not match number of trajectories {}",
                rows, n_trajectories
            )));
        }
        if cols != model.n_species {
            return Err(PyValueError::new_err(format!(
                "initial_states columns {} do not match number of species {}",
                cols, model.n_species
            )));
        }
        per_traj_initial_states_vec = Some(data);
    }
    if let Some(times) = per_traj_initial_times_vec {
        if times.len() != n_trajectories {
            return Err(PyValueError::new_err(format!(
                "initial_times length {} does not match number of trajectories {}",
                times.len(),
                n_trajectories
            )));
        }
        if times
            .iter()
            .any(|&t| t.is_nan() || t < -TIME_EPSILON || t > t_end + TIME_EPSILON)
        {
            return Err(PyValueError::new_err(
                "initial_times entries must be finite and between 0 and t_end",
            ));
        }
        per_traj_initial_times = Some(times);
    }
    let options = SimulationOptions {
        t_end,
        t_points: t_points_vec,
        mode: output_mode,
        interventions: plan,
    };
    let sim_result = py.detach(move || {
        run_ensemble(
            &model,
            &initial_state_vec,
            per_traj_initial_states_vec.as_deref(),
            per_traj_initial_times.as_deref(),
            &options,
            n_trajectories,
            n_threads,
            seed,
        )
    })?;
    sim_result.into_py(py)
}

#[pymodule]
fn reactors(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(simulate_ensemble, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests;
