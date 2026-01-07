use super::*;
use rand::SeedableRng;
use std::str::FromStr;

fn birth_death_model() -> Model {
    let n_species = 1;
    let stoich = vec![1, -1];
    let reaction_deltas = build_reaction_deltas(n_species, &stoich);
    let reactants_birth = Vec::new();
    let reactants_death = vec![Reactant {
        species: 0,
        count: 1,
    }];
    let reactions = vec![
        Reaction {
            rate_constant: 5.0,
            reactants: reactants_birth,
            kind: ReactionKind::MassAction,
        },
        Reaction {
            rate_constant: 1.0,
            reactants: reactants_death,
            kind: ReactionKind::MassAction,
        },
    ];
    let dependencies = build_dependency_graph(n_species, &reaction_deltas, &reactions);
    Model {
        n_species,
        n_reactions: 2,
        reaction_deltas,
        reactions,
        dependencies,
    }
}

fn death_only_model() -> Model {
    let n_species = 1;
    let stoich = vec![-1];
    let reaction_deltas = build_reaction_deltas(n_species, &stoich);
    let reactions = vec![Reaction {
        rate_constant: 1.0,
        reactants: vec![Reactant {
            species: 0,
            count: 1,
        }],
        kind: ReactionKind::MassAction,
    }];
    let dependencies = build_dependency_graph(n_species, &reaction_deltas, &reactions);
    Model {
        n_species,
        n_reactions: 1,
        reaction_deltas,
        reactions,
        dependencies,
    }
}

fn inert_model() -> Model {
    let n_species = 1;
    let stoich = vec![0];
    let reaction_deltas = build_reaction_deltas(n_species, &stoich);
    let reactions = vec![Reaction {
        rate_constant: 0.0,
        reactants: Vec::new(),
        kind: ReactionKind::MassAction,
    }];
    let dependencies = build_dependency_graph(n_species, &reaction_deltas, &reactions);
    Model {
        n_species,
        n_reactions: 1,
        reaction_deltas,
        reactions,
        dependencies,
    }
}

fn boolean_follow_model() -> Model {
    let n_species = 2;
    let stoich = vec![1, 0, -1, 0, 0, 1, 0, -1];
    let reaction_deltas = build_reaction_deltas(n_species, &stoich);
    let mut reactions = Vec::new();
    let expr_specs = [
        ("50.0 * (1.0 - s0)", vec![0]),
        ("0.0", Vec::new()),
        ("50.0 * (1.0 - s1) * s0", vec![0, 1]),
        ("50.0 * s1 * (1.0 - s0)", vec![0, 1]),
    ];
    for (expr_str, species_refs) in expr_specs {
        reactions.push(Reaction {
            rate_constant: 0.0,
            reactants: Vec::new(),
            kind: ReactionKind::Expression {
                expr: Expr::from_str(expr_str).unwrap(),
                species_refs,
            },
        });
    }
    let dependencies = build_dependency_graph(n_species, &reaction_deltas, &reactions);
    Model {
        n_species,
        n_reactions: reactions.len(),
        reaction_deltas,
        reactions,
        dependencies,
    }
}

#[test]
fn falling_factorial_basics() {
    assert_eq!(falling_factorial(5, 0), 1.0);
    assert_eq!(falling_factorial(5, 2), 20.0);
    assert_eq!(falling_factorial(3, 4), 0.0);
}

#[test]
fn derive_seed_is_deterministic() {
    let s1 = derive_seed(Some(42), 5);
    let s2 = derive_seed(Some(42), 5);
    assert_eq!(s1, s2);
    assert_ne!(derive_seed(Some(42), 5), derive_seed(Some(42), 6));
}

#[test]
fn single_trajectory_is_reproducible() {
    let model = birth_death_model();
    let options = SimulationOptions {
        t_end: 1.0,
        t_points: vec![1.0],
        mode: OutputMode::Timeseries,
        interventions: InterventionPlan::default(),
    };
    let initial = [0i32];
    let result = run_ensemble(
        &model,
        &initial,
        None,
        None,
        &options,
        1,
        Some(1),
        Some(123),
    )
    .unwrap();
    let result2 = run_ensemble(
        &model,
        &initial,
        None,
        None,
        &options,
        1,
        Some(1),
        Some(123),
    )
    .unwrap();
    assert_eq!(result.data, result2.data);
}

#[test]
fn timeseries_record_counts_match() {
    let model = birth_death_model();
    let options = SimulationOptions {
        t_end: 2.0,
        t_points: vec![0.5, 1.0],
        mode: OutputMode::Timeseries,
        interventions: InterventionPlan::default(),
    };
    let initial = [0i32];
    let result =
        run_ensemble(&model, &initial, None, None, &options, 3, Some(2), Some(77)).unwrap();
    assert_eq!(result.n_times, 2);
    assert_eq!(result.data.len(), 3 * 2 * model.n_species);
}

#[test]
fn final_mode_returns_single_time() {
    let model = birth_death_model();
    let options = SimulationOptions {
        t_end: 2.0,
        t_points: Vec::new(),
        mode: OutputMode::FinalOnly,
        interventions: InterventionPlan::default(),
    };
    let initial = [0i32];
    let result = run_ensemble(&model, &initial, None, None, &options, 4, None, Some(5)).unwrap();
    assert_eq!(result.n_times, 1);
    assert_eq!(result.data.len(), 4 * model.n_species);
}

#[test]
fn michaelis_menten_propensity_behaves() {
    let reaction = Reaction {
        rate_constant: 8.0,
        reactants: vec![Reactant {
            species: 0,
            count: 1,
        }],
        kind: ReactionKind::MichaelisMenten {
            substrate: 0,
            k_m: 4.0,
        },
    };
    let state = [6];
    let propensity = reaction.propensity(reaction.rate_constant, &state);
    assert!((propensity - (8.0 * 6.0 / (4.0 + 6.0))).abs() < 1e-12);
}

#[test]
fn hill_kinetics_propensity_behaves() {
    // rate = V_max * [A]^n / (K^n + [A]^n)
    // V_max = 10.0, [A] = 4, n = 2, K = 3
    // rate = 10 * 16 / (9 + 16) = 160 / 25 = 6.4
    let reaction = Reaction {
        rate_constant: 10.0,
        reactants: Vec::new(),
        kind: ReactionKind::Hill {
            activator: 0,
            hill_n: 2.0,
            k_half: 3.0,
            k_half_pow_n: 9.0, // 3^2
        },
    };
    let state = [4];
    let propensity = reaction.propensity(reaction.rate_constant, &state);
    assert!((propensity - 6.4).abs() < 1e-12);
}

#[test]
fn reaction_type_code_conversion_is_strict() {
    assert!(ReactionTypeCode::try_from(0).is_ok());
    assert!(ReactionTypeCode::try_from(1).is_ok());
    assert!(ReactionTypeCode::try_from(2).is_ok());
    assert!(ReactionTypeCode::try_from(3).is_ok());
    assert!(ReactionTypeCode::try_from(4).is_err());
}

#[test]
fn expression_propensity_evaluates() {
    let expr = Expr::from_str("2.0 * s0 + s1").unwrap();
    let reaction = Reaction {
        rate_constant: 1.0,
        reactants: Vec::new(),
        kind: ReactionKind::Expression {
            expr,
            species_refs: vec![0, 1],
        },
    };
    let state = [3, 5];
    let propensity = reaction.propensity(reaction.rate_constant, &state);
    assert!((propensity - 11.0).abs() < 1e-12);
}

#[test]
fn collect_species_refs_deduplicates_and_is_case_insensitive() {
    let refs = collect_species_refs("2*s0 + 3*S0 + s2", 0, 3).unwrap();
    assert_eq!(refs, vec![0, 2]);
}

#[test]
fn collect_species_refs_rejects_out_of_range_indices() {
    let err = collect_species_refs("s5 + 1", 1, 2).unwrap_err();
    assert!(matches!(err, SimError::InvalidArgument(msg) if msg.contains("exceeds")));
}

#[test]
fn parse_mode_defaults_and_validates_inputs() {
    assert_eq!(
        parse_mode(Some("final"), false).unwrap(),
        OutputMode::FinalOnly
    );
    assert_eq!(parse_mode(None, true).unwrap(), OutputMode::Timeseries);
    assert_eq!(parse_mode(None, false).unwrap(), OutputMode::FinalOnly);
    assert!(matches!(
        parse_mode(Some("timeseries"), false),
        Err(SimError::InvalidArgument(_))
    ));
    assert!(matches!(
        parse_mode(Some("bogus"), true),
        Err(SimError::InvalidArgument(_))
    ));
}

#[test]
fn propensity_tree_selects_expected_indices() {
    let props = vec![1.0, 3.0, 6.0];
    let mut tree = PropensityTree::new(props.len());
    tree.rebuild(&props);
    let total = tree.total();
    assert_eq!(tree.select(0.0), 0);
    assert_eq!(tree.select(0.1 * total), 0);
    assert_eq!(tree.select(0.2 * total), 1);
    assert_eq!(tree.select(0.6 * total), 2);
    assert_eq!(tree.select(0.95 * total), 2);
}

#[test]
fn propensity_tree_handles_zero_entries() {
    let props = vec![0.0, 2.0, 0.0, 5.0];
    let mut tree = PropensityTree::new(props.len());
    tree.rebuild(&props);
    let total = tree.total();
    assert_eq!(tree.select(0.01 * total), 1);
    assert_eq!(tree.select(0.4 * total), 3);
    assert_eq!(tree.select(0.9 * total), 3);
}

#[test]
fn propensity_tree_updates_after_modifications() {
    let props = vec![2.0, 3.0];
    let mut tree = PropensityTree::new(props.len());
    tree.rebuild(&props);
    assert_eq!(tree.total(), 5.0);
    tree.update(1, 1.0);
    assert!((tree.total() - 3.0).abs() < 1e-12);
    assert_eq!(tree.select(0.5), 0);
    assert_eq!(tree.select(2.1), 1);
}

#[test]
fn boolean_network_converges_to_known_state() {
    let model = boolean_follow_model();
    let options = SimulationOptions {
        t_end: 2.0,
        t_points: Vec::new(),
        mode: OutputMode::FinalOnly,
        interventions: InterventionPlan::default(),
    };
    let initial = [0i32, 0];
    let result = run_ensemble(
        &model,
        &initial,
        None,
        None,
        &options,
        8,
        Some(2),
        Some(1234),
    )
    .unwrap();
    assert_eq!(result.n_times, 1);
    for chunk in result.data.chunks(model.n_species) {
        assert_eq!(chunk, &[1, 1]);
    }
}

#[test]
fn run_ensemble_validates_initial_state_and_counts() {
    let model = birth_death_model();
    let options = SimulationOptions {
        t_end: 1.0,
        t_points: vec![0.5],
        mode: OutputMode::Timeseries,
        interventions: InterventionPlan::default(),
    };
    let bad_initial = [0i32, 0];
    let err = match run_ensemble(&model, &bad_initial, None, None, &options, 1, None, None) {
        Ok(_) => panic!("run_ensemble unexpectedly succeeded with mismatched initial state"),
        Err(err) => err,
    };
    assert!(matches!(err, SimError::Shape(msg) if msg.contains("initial state length")));

    let zero_traj_err = match run_ensemble(&model, &[0], None, None, &options, 0, None, None) {
        Ok(_) => panic!("run_ensemble unexpectedly succeeded with zero trajectories"),
        Err(err) => err,
    };
    assert!(matches!(
        zero_traj_err,
        SimError::InvalidArgument(msg) if msg.contains("number of trajectories")
    ));

    let bad_options = SimulationOptions {
        t_end: 0.0,
        t_points: vec![],
        mode: OutputMode::FinalOnly,
        interventions: InterventionPlan::default(),
    };
    let non_positive_t_err =
        match run_ensemble(&model, &[0], None, None, &bad_options, 1, None, None) {
            Ok(_) => panic!("run_ensemble unexpectedly succeeded with non-positive t_end"),
            Err(err) => err,
        };
    assert!(matches!(
        non_positive_t_err,
        SimError::InvalidArgument(msg) if msg.contains("t_end")
    ));
}

#[test]
fn simulate_single_records_even_when_propensities_are_zero() {
    let model = death_only_model();
    let options = SimulationOptions {
        t_end: 1.0,
        t_points: vec![0.0, 0.5, 1.0],
        mode: OutputMode::Timeseries,
        interventions: InterventionPlan::default(),
    };
    let mut rng = ChaCha8Rng::seed_from_u64(123);
    let trajectory = simulate_single(&model, &[0], 0.0, &options, &mut rng).unwrap();
    assert_eq!(trajectory.states, vec![0, 0, 0]);
}

#[test]
fn interventions_can_modify_species_without_reactions() {
    let model = inert_model();
    let plan = InterventionPlan {
        events: vec![InterventionEvent {
            time: 0.5,
            actions: vec![InterventionAction::SpeciesSet {
                species: 0,
                value: 7,
            }],
        }],
    };
    let options = SimulationOptions {
        t_end: 1.0,
        t_points: Vec::new(),
        mode: OutputMode::FinalOnly,
        interventions: plan,
    };
    let result = run_ensemble(&model, &[0], None, None, &options, 1, None, Some(9)).unwrap();
    assert_eq!(result.data, vec![7]);
}

#[test]
fn rate_set_intervention_updates_propensities() {
    let model = inert_model();
    let mut rate_constants = vec![0.0];
    let mut propensities = vec![0.0];
    let mut state = vec![0];
    let mut total = 0.0;
    let mut tree = PropensityTree::new(model.n_reactions);
    tree.rebuild(&propensities);
    let plan = InterventionPlan {
        events: vec![InterventionEvent {
            time: 0.0,
            actions: vec![InterventionAction::RateSet {
                reaction: 0,
                value: 3.5,
            }],
        }],
    };
    let mut idx = 0usize;
    apply_interventions(
        &plan,
        &mut idx,
        0.0,
        &mut state,
        &mut rate_constants,
        &model.reactions,
        &mut propensities,
        &mut total,
        &mut tree,
    );
    assert!((rate_constants[0] - 3.5).abs() < 1e-12);
    assert!((propensities[0] - 3.5).abs() < 1e-12);
    assert!((total - 3.5).abs() < 1e-12);
}

#[test]
fn per_trajectory_initial_states_are_honored() {
    let model = inert_model();
    let options = SimulationOptions {
        t_end: 1.0,
        t_points: Vec::new(),
        mode: OutputMode::FinalOnly,
        interventions: InterventionPlan::default(),
    };
    let shared_initial = [0i32];
    let per_traj = vec![3, 7];
    let result = run_ensemble(
        &model,
        &shared_initial,
        Some(&per_traj),
        None,
        &options,
        2,
        None,
        Some(11),
    )
    .unwrap();
    assert_eq!(result.data, per_traj);
}

#[test]
fn per_trajectory_initial_states_validate_lengths() {
    let model = inert_model();
    let options = SimulationOptions {
        t_end: 1.0,
        t_points: Vec::new(),
        mode: OutputMode::FinalOnly,
        interventions: InterventionPlan::default(),
    };
    let shared_initial = [0i32];
    let per_traj = vec![1, 2]; // length 2, but we'll request 3 trajectories
    let err = match run_ensemble(
        &model,
        &shared_initial,
        Some(&per_traj),
        None,
        &options,
        3,
        None,
        None,
    ) {
        Ok(_) => panic!("expected shape validation to fail"),
        Err(err) => err,
    };
    assert!(matches!(err, SimError::Shape(msg) if msg.contains("initial_states length")));
}

#[test]
fn per_trajectory_initial_times_skip_past_events() {
    let model = inert_model();
    let options = SimulationOptions {
        t_end: 10.0,
        t_points: Vec::new(),
        mode: OutputMode::FinalOnly,
        interventions: InterventionPlan {
            events: vec![
                InterventionEvent {
                    time: 1.0,
                    actions: vec![InterventionAction::SpeciesSet {
                        species: 0,
                        value: 50,
                    }],
                },
                InterventionEvent {
                    time: 6.0,
                    actions: vec![InterventionAction::SpeciesDelta {
                        species: 0,
                        delta: 1,
                    }],
                },
            ],
        },
    };
    let shared_initial = [0i32];
    let per_traj = vec![0i32];
    let start_times = vec![5.0];
    let result = run_ensemble(
        &model,
        &shared_initial,
        Some(&per_traj),
        Some(&start_times),
        &options,
        1,
        None,
        Some(7),
    )
    .unwrap();
    assert_eq!(result.data, vec![1]);
}

#[test]
fn per_trajectory_initial_times_validate_length() {
    let model = inert_model();
    let options = SimulationOptions {
        t_end: 5.0,
        t_points: Vec::new(),
        mode: OutputMode::FinalOnly,
        interventions: InterventionPlan::default(),
    };
    let shared_initial = [0i32];
    let per_traj_states = vec![0, 0];
    let times = vec![1.0]; // too short for 2 trajectories
    let err = match run_ensemble(
        &model,
        &shared_initial,
        Some(&per_traj_states),
        Some(&times),
        &options,
        2,
        None,
        None,
    ) {
        Ok(_) => panic!("expected initial_times validation to fail"),
        Err(err) => err,
    };
    assert!(matches!(err, SimError::Shape(msg) if msg.contains("initial_times length")));
}
