# mc_ai_bt

AI-BT mission manager, planner validator and execution shell for the `fan_bt`
line.

Local lightweight build:

```bash
colcon build --base-paths mc_one mc_multimodal mc_resource_authority mc_world_state mc_ai_bt mc_voice_pipeline_legacy --packages-select mc_one mc_multimodal mc_resource_authority mc_world_state mc_ai_bt mc_voice_pipeline_legacy --symlink-install --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
```

For local Docker images without pushing first, run
`mc_one_codey/desktop/build_fan_bt_core_local.sh` from this workspace.

Run the core AI-BT services from the workspace root:

```bash
source install/setup.bash
ros2 launch mc_ai_bt fan_bt_core.launch.py
```

With mission journaling enabled:

```bash
ros2 launch mc_ai_bt fan_bt_core.launch.py \
  mission_journal_path:=/tmp/mc_ai_bt_missions.jsonl
```

Run a local ROS smoke without Isaac by starting fake navigation/animator action
servers and fake object/person perception:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_LOG_DIR=/tmp/ros2_launch_logs
ros2 launch mc_ai_bt fan_bt_local_fake.launch.py \
  mission_journal_path:=/tmp/mc_ai_bt_fake_missions.jsonl
```

In another terminal:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 service call /mc_ai_bt/submit_task mc_one/srv/SubmitTaskIntent \
  "{source: manual, operator_id: user, intent_text: 'go to kitchen then find the cup', parent_mission_id: '', priority: 10, allow_queue: true, context_json: '{}'}"

ros2 service call /mc_ai_bt/submit_task mc_one/srv/SubmitTaskIntent \
  "{source: manual, operator_id: user, intent_text: 'point at the cup with left hand', parent_mission_id: '', priority: 10, allow_queue: true, context_json: '{}'}"

ros2 service call /mc_ai_bt/list_missions mc_one/srv/ListMissions \
  "{mission_id: '', include_terminal: true}"
```

The fake launch is only a wiring smoke. It does not replace Isaac; it proves the
AI-BT service/action/world-state/resource path can run locally before the
simulation stack is introduced.

To automate the same fake-launch smoke from one terminal:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_LOG_DIR=/tmp/ros2_launch_logs
ros2 run mc_ai_bt ai_bt_local_fake_ros_smoke
```

The runner starts `fan_bt_local_fake.launch.py`, submits
`go to kitchen then point at the cup`, waits for the mission to succeed,
checks read-only world queries, verifies active resource leases drain to zero,
prints a JSON summary, and stops the launch. If the stack is already running,
use `--no-launch`.

Useful read-only inspection services:

```bash
ros2 run mc_ai_bt ai_bt_doctor

# When nav/animator or the fake skill launch should also be up:
ros2 run mc_ai_bt ai_bt_doctor --require-skill-actions

ros2 service call /mc_ai_bt/list_missions mc_one/srv/ListMissions \
  "{mission_id: '', include_terminal: true}"

ros2 service call /mc_resource_authority/list mc_one/srv/ListResourceLeases \
  "{resources: [], include_inactive: false}"
```

If the process cannot write to `~/.ros/log`, set a writable log directory:

```bash
export ROS_LOG_DIR=/tmp/ros2_launch_logs
```

Run the logic-only smoke without ROS/DDS:

```bash
python3 -m mc_ai_bt.local_smoke go to kitchen --print-plan
```

VisualCheck smoke can be run by passing a compact world snapshot:

```bash
python3 -m mc_ai_bt.local_smoke find the cup \
  --world-json '{"facts":{"objects":{"object:cup":{"value":{"object_name":"cup","score":0.9}}}}}'
```

After building the overlay, the same smoke is available as:

```bash
source install/setup.bash
ros2 run mc_ai_bt ai_bt_local_smoke go to kitchen --print-plan
```

On the `fan_bt` branch, the legacy voice pipeline keeps ordinary chat on the
old LLM route but sends clear robot tasks, task-control utterances, and
world-state questions to AI-BT by default. To temporarily bypass this bridge:

```bash
ros2 param set /mc_voice_pipeline_legacy/pipeline ai_bt_route_final_utterances false
```

Legacy inline physical skills consult the global resource authority by default
on `fan_bt`. To temporarily bypass that gate:

```bash
ros2 param set /mc_voice_pipeline_legacy/pipeline resource_authority_enforce_inline_skills false
```

When AI-BT routing is enabled, short task-control utterances are routed to the
mission-control services instead of being submitted as new missions:

- cancel current mission: `stop`, `cancel`, `halt`, `abort`, `停下来`, `取消任务`
- pause current mission: `pause`, `hold on`, `wait`, `暂停一下`, `先等一下`
- resume current mission: `resume`, `continue`, `keep going`, `继续任务`, `接着做`

If AI-BT explicitly rejects a submitted task or task-control request, the
legacy voice node speaks a short failure acknowledgement. If the AI-BT service
is not ready at all, the utterance still falls back to the legacy dialogue path.

Current bootstrap planner routes these common intents into validated BT:

- `go to kitchen` -> `say` then await `/mc_navigation/go_to_place`
- `come to me` -> `say` then await `/mc_navigation/come_to_me`
- `move forward`, `move left 90` -> `say` then await `/mc_navigation/simple_move`
- `wave hello` or `play animation wave` -> await `/mc_animator/play`
- `look left`, `look at front_up` -> await `/mc_animator/play` with `look_at`
- `point at the cup`, `point at the cup with left hand` -> await
  `/mc_animator/play` with `right_point` / `left_point`
- `find the cup`, `look for someone`, `check for the cup` -> `say` then `VisualCheck`
- First Chinese aliases for the same bootstrapped skills:
  `去厨房`, `来我这里`, `左转90度`, `挥手`, `看左边`, `指一下杯子`, `找杯子`
- simple compounds with `then` / `and then`, for example
  `go to kitchen then wave` or `go to kitchen then find the cup`, become a
  single sequential BT with the final step as the mission `GoalSpec`.

The real LLM planner will produce the same `BT + GoalSpec` shape. The bootstrap
planner is only a deterministic local route for early simulation tests.

Successful skills best-effort write compact facts back to `mc_world_state`
through `/mc_world_state/update_facts`:

- `go_to_place` -> `navigation/current_place`
- `come_to_me` -> `robot/near_interaction_owner`
- `simple_move` -> `robot/last_relative_motion`
- `play_animation` -> `robot/last_animation`
- `look_at` -> `robot/last_animation`
- `point_at` -> `robot/last_animation`
- `say` -> `robot/last_utterance`

The writer is intentionally non-fatal. If `mc_world_state` is not running, a
skill that already succeeded still returns success; GoalCheck can still use the
action result facts for the current mission.

AI-BT skill adapters acquire global resource leases before using embodied
lanes. Base actions lease `base`, animator actions lease `body`, and `say`
leases `voice, face` while submitting the utterance to the legacy TTS path.
Leases opened by a mission carry that mission's
`mission_id / plan_version / execution_id` through acquire, renew, and release,
so resource-authority events can be traced back to the exact active runner.
Long-running action leases are renewed in the background before their TTL
expires, then released when the action returns or is canceled.
The current `say` lease covers submission, not full audio playback completion;
that can be tightened later once the TTS path exposes a completion event.

Mission lifecycle events also best-effort project compact task facts:

- `tasks/active_mission`
- `tasks/mission_index`

Visual `goal_spec` entries that match the first supported object/person
visibility patterns are projected as structured `object_visible` or
`person_visible` goals while keeping the original query for debugging. This lets
`mc_world_state` compare incoming perception facts against the active goal and
emit replan-worthy `GOAL_EVIDENCE_CHANGED` events.

Optional mission journaling can be enabled with `mission_journal_path`. When
set, every `TaskEvent` appends a JSONL record containing the full mission
snapshot. On node restart, latest journal entries are loaded; non-terminal
missions are restored as `BLOCKED` with `recovered_after_restart` so they do
not silently disappear or automatically replay robot actions. Recovered
missions are published as startup `TaskStatus` snapshots three times and task
projection facts are refreshed, without appending new journal events.

During BT execution, `BtExecutor` reports progress at leaf nodes
(`Action`, `Wait`, `Condition`, `VisualCheck`, `GoalCheck`, `NoAction`). `AiBtNode` stores
that label in `TaskStatus.active_node` and publishes updated `progress` without
creating extra lifecycle `TaskEvent` records. This makes multi-step local and
simulation runs inspectable while keeping the event journal compact.

Read-only world queries are available through:

```bash
ros2 service call /mc_ai_bt/query_world mc_one/srv/QueryWorld \
  "{source: manual, query_text: 'where are you?', scopes: [], max_age_sec: 5.0, context_json: '{}'}"
```

The first deterministic query engine answers current-place, current-task,
simple object-visibility, visible-people, and safety-stop questions from
`mc_world_state`. Unknown evidence is returned as `CERTAINTY_UNKNOWN`, not made up.

`mc_ai_bt` subscribes to `/mc_world_state/events`. Every world event first goes
through `TriggerManager`, which returns one of the frozen decisions:
`NO_ACTION`, `LOCAL_HANDLED`, `REPLAN`, `PLAN_NEW_MISSION`, or `BLOCKED`.
Regular `FACT_UPDATED` events are ignored for mission control; safety-stop
event types block the active mission; replan-worthy event types cancel the
current runner, reserve a new `plan_version`, clear the old execution id, and
run the planning pipeline again. Idle social events such as
`PERSON_APPROACHED` may create a low-priority world-event mission only after
TriggerManager chooses `PLAN_NEW_MISSION`. Late planner/action results whose
identity no longer matches are ignored. For `GOAL_EVIDENCE_CHANGED`, a payload
with `"replan_recommended": false` is handled locally without replanning. If
that payload carries `mission_id` and `plan_version`, they must match the
current active mission or the replan trigger is ignored as stale.

When the legacy voice node has `ai_bt_route_final_utterances=true`, its
conservative world-query classifier calls this service for questions such as
`where are you?`, `what are you doing?`, `do you see the cup?`, `你在哪？`,
`有人吗？`, and `你看到杯子了吗？`. General chat/questions stay on the legacy
LLM path.

Implemented BT runtime nodes:

- `Sequence`
- `Fallback`
- `Action`
- `Wait`
- `Retry`
- `Parallel`
- `Timeout`
- `Condition`
- `VisualCheck`
- `GoalCheck`
- `NoAction`

`Condition`, `VisualCheck`, and `GoalCheck` use the shared `GoalChecker` and
preserve `TRUE/FALSE/UNKNOWN`: `UNKNOWN` blocks the mission instead of being
treated as ordinary failure. `VisualCheck` first uses structured world-state
evidence for recognized object/person predicates, then falls back to the
configured `mc_multimodal` visual-check action for concise true/false visual
queries.

Structured `GoalCheck` also verifies `person_at_place` and `object_at_place`
through the same PlaceRegion evaluator used by `mc_world_state`, plus explicit
`person_following` facts from execution or world state. Missing evidence stays
`UNKNOWN`.

Planner context is built by `ContextBuilder` before planning. It includes the
incoming mission, caller context, a bounded world snapshot, current mission
summaries, the skill catalog and planner constraints. `LlmJsonPlanner` defines
the injectable JSON-planner interface for the later production LLM route, while
the node still defaults to `BootstrapPlanner`.

After JSON schema validation, `PolicyGuard` enforces robot execution policy:
bounded BT nodes, bounded multi-step embodied actions, capped physical/base/
body/VisualCheck counts, bounded wait/retry budget, conservative `simple_move`
ranges, bounded `look_at` directions/holds, bounded `point_at` target forms,
short `say` text, static rejection of overlapping skill resources in `Parallel`
branches, and alignment between structured goals and the
physical action that is supposed to satisfy them. Current defaults allow up to
32 BT nodes, 12 actions, 4 physical actions, 3 base actions, 3 body actions,
and 4 VisualCheck nodes per plan.

Planner backend parameters:

- `planner_backend`: `bootstrap` by default; `langchain_openai` is optional.
- `planner_model_name`: required only for `langchain_openai`.
- `planner_temperature`: default `0.0`.
- `planner_timeout`: default `15.0` seconds.
- `mission_journal_path`: empty by default; set to a writable JSONL path to
  enable restart audit/recovery snapshots.

If an optional planner backend cannot be built, the node logs a warning and
falls back to `BootstrapPlanner` so the local/simulation stack can still boot.
The `mc_ai_bt` Docker image installs `langchain-openai` for this backend; a
non-Docker local run needs the same Python dependency available in the active
environment.
