# Simulation Design

## Scope

This document defines the non-learning execution layer. It provides a deterministic demonstration path before Actor/Critic training is introduced.

## Execution sequence

1. Load YAML configuration.
2. Seed all pseudo-random generators.
3. Generate mock JSON dataset when `mode=mock`, or load an existing dataset when `mode=real`.
4. Construct `ProblemDefinition`.
5. Build a SUMO network and route file.
6. Start `sumo` or `sumo-gui` through TraCI.
7. Advance the simulation one configured step at a time.
8. Collect vehicle/edge/system telemetry.
9. Write the raw trace.
10. Optionally build the analytical report.
11. Optionally replay the trajectory using salabim.

## SUMO boundary

`SumoScenarioBuilder` is responsible only for translating Nash-DRL domain data to SUMO artifacts. `run_sumo` is responsible for TraCI execution and telemetry. The rest of the Python project does not need to know about TraCI calls.

## Route semantics in this stage

Because DRL is intentionally excluded, the demo uses a deterministic shortest-path policy based on edge length. Every vehicle's complete trip set is converted into one concatenated SUMO route. This is a baseline execution mechanism, not the final Nash-DRL action policy.

## Visualization

`visualization=true` selects `sumo-gui`, giving a live microscopic traffic visualization. After the run, salabim can replay the recorded vehicle trajectory as a lightweight Python animation. The salabim replay is deliberately decoupled from the simulator state so it cannot change the simulation results.

## Reproducibility

The random seed is present in the YAML configuration and saved into generated dataset metadata and run metadata. The same mock configuration and seed generate the same graph, vehicles, and trips.

## Turnaround handling

Synthetic multi-trip scenarios can produce a valid trip sequence in which the destination of one trip is followed by a next trip whose destination requires the vehicle to reverse direction immediately. In SUMO this is a turnaround movement, and the network must contain a corresponding turnaround connection. The demo therefore defaults `simulation.sumo.allow_turnarounds` to `true` and passes `--no-turnarounds false --no-turnarounds.geometry false` to `netconvert`. This avoids the route failure caused by globally disabling turnaround connections. SUMO documents `--no-turnarounds` as disabling turnaround construction and notes that its default is `false`; geometry-like turnaround construction is controlled separately. citeturn465325search0turn465325search1

The scenario builder also validates every generated route before writing the SUMO route file. It checks that every adjacent pair of edge IDs is topologically connected (`previous.destination == current.source`) so disconnected routes fail before TraCI starts.
