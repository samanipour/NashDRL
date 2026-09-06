# Configuration

## Training controls

| Key | Meaning |
|---|---|
| `training.episodes` | number of learning episodes |
| `training.gamma` | TD discount factor |
| `training.actor_lr` | Actor learning rate |
| `training.critic_lr` | Critic learning rate |
| `training.exploration_sigma` | initial Gaussian action noise |
| `training.exploration_sigma_final` | final Gaussian action noise |
| `training.exploration_decay_episodes` | linear exploration-decay horizon |
| `training.reward_scale` | constant applied to rewards only for learning; reporting stays in raw units |
| `training.max_grad_norm` | gradient clipping threshold |
| `training.replay_enabled` | enable transition replay |
| `training.replay_capacity` | replay buffer capacity |
| `training.replay_batch_size` | transitions sampled per optimizer update |
| `training.replay_warmup` | transitions required before learning starts |
| `training.updates_per_step` | optimizer batches learned after each environment step |
| `training.target_update_interval` | Main Critic → Target Critic hard-copy interval |
| `training.visualization` | use SUMO-GUI while training |

## Mathematical environment

| Key | Meaning |
|---|---|
| `environment.energy_rate_kwh_per_km` | energy consumption rate |
| `environment.charging_overhead` | charging-price overhead `O` |
| `environment.charging_fixed_cost` | fixed unit charging cost `C` |
| `environment.charging_floor_price` | minimum charging unit price |
| `environment.congestion_alpha` | congestion parameter `alpha` |
| `environment.congestion_beta` | congestion parameter `beta` |
| `environment.use_model_travel_time` | use Equation (5) for learning reward; when false use SUMO telemetry |

## Reward

| Key | Meaning |
|---|---|
| `reward.travel_time_weight` | `w_T` |
| `reward.charging_cost_weight` | `w_C` |
| `reward.budget_penalty` | hard-budget penalty `P` |

## Mock-data feasibility

| Key | Meaning |
|---|---|
| `mock_data.seed` | reproducible generator seed |
| `mock_data.num_vehicles` | number of vehicles |
| `mock_data.min_trips_per_vehicle` | minimum trips |
| `mock_data.max_trips_per_vehicle` | maximum trips |
| `mock_data.graph_nodes` | graph node count |
| `mock_data.graph_extra_edges` | additional directed edges |
| `mock_data.road_min_length_km` | minimum edge length |
| `mock_data.road_max_length_km` | maximum edge length |
| `mock_data.budget_min` | lower configured budget bound |
| `mock_data.budget_max` | upper configured budget bound |
| `mock_data.budget_feasibility_multiplier` | normal-budget center relative to the shortest-path feasibility cost |
| `mock_data.budget_resample_limit` | maximum lower-tail rejection samples |

The feasible-budget generation does not change the reward equation; it changes only how synthetic datasets are constructed so that at least one known reference routing assignment satisfies the hard budget constraint.
