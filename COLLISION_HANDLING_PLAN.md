# Collision Handling Plan for Heuristic Nearest Fire Script

## Problem Analysis

### Current System State
- **Agents cannot overlap**: In `wildfire_environment/core/agent.py:150`, `can_overlap()` returns `False` for agents
- **Sequential movement**: In `wildfire_environment/envs/wildfire.py:1152`, agents move in random order
- **Collision detection**: Movement is only allowed if `next_cell is None or next_cell.can_overlap()`
- **Supply source is shared**: All agents have the same `home_pos` (급수원): `(11, 11)` by default

### Current Heuristic Behavior (Line 119-123)
When an agent is in RETURNING state:
```python
if hasattr(agent, 'state') and agent.state == AgentState.RETURNING:
    if hasattr(agent, 'home_pos') and agent.home_pos is not None:
        return _move_towards(agent_pos, agent.home_pos)
```

This directly moves toward home using `_move_towards()` which returns a movement action in one of 8 directions.

---

## Errors That Can Arise When Multiple Agents Go to Supply Source Simultaneously

### 1. **Collision at Supply Source Entrance**
**Scenario**: Multiple agents are in RETURNING state and trying to move toward the same supply source `(11, 11)`.

**What happens in environment.py's step() function**:
- Lines 1152-1153: Agents move in random order
- Lines 1177-1180: Each agent checks `if next_cell is None or next_cell.can_overlap()`
- Since agents `can_overlap() = False`, the movement is blocked if there's another agent at the destination

**Error manifestation**:
```
Agent A tries to move from (12, 11) → (11, 11)
Agent B tries to move from (11, 12) → (11, 11)
Only the first agent (in random order) succeeds
The second agent stays in RETURNING state and cannot progress
```

### 2. **Deadlock at Supply Source Bottleneck**
When multiple agents converge from different directions (e.g., 4 agents from north, south, east, west), they can get stuck around the supply source because:
- Each agent path is blocked by another agent one step away
- No agent can complete the final move into the supply source
- Agents remain in RETURNING state indefinitely (or until water runs out → forced recharge)

### 3. **Inefficient Queueing / Agent Starvation**
**Scenario**: Agent A moves into supply source first, but Agent B is blocked and cannot advance
- Agent A transitions to RECHARGING (stays in place for `recharge_time` steps)
- During recharge, Agent A occupies the supply source position
- Other agents waiting in RETURNING state cannot enter
- The heuristic has no "queue management" - agents don't know to wait in a specific order

**Problems**:
- Water running out prematurely (Agent B forced to return while still RETURNING)
- Inefficient recharge scheduling (all agents eventually need recharge at the same time)
- Agents may get stuck in RETURNING state if water depletes before reaching home

### 4. **Race Condition in Simultaneous Movement**
When agents move in random order within the same step:
```
Step t:
  - Order: [2, 0, 1, 3, 4, 5]
  - Agent 2 moves toward (11,11), succeeds
  - Agent 0 tries to move to same location, fails (blocked by Agent 2)
  - Agent 1 tries to move, succeeds
  - ... unpredictable queueing behavior
```

The heuristic doesn't account for this - it always tries to move directly toward home without any fallback behavior.

### 5. **Supply Source Saturation**
If all 6 agents are in RETURNING state simultaneously:
- Only one agent can occupy `(11, 11)` at a time
- 5 agents are stuck in RETURNING state around the supply source
- They cannot move further without water/time to complete fire suppression first
- This creates a "traffic jam" at the supply source

---

## Issues in Current Heuristic Implementation

### Issue 1: No Collision Awareness
The `select_heuristic_action()` function only looks at:
- Current agent position
- Nearest fire position (or home position if RETURNING)
- It does NOT check if the next position is occupied by another agent

### Issue 2: No Fallback Behavior
If the direct path to supply source is blocked:
```python
return _move_towards(agent_pos, agent.home_pos)
```
Always returns the "optimal" direction toward home. If that cell is occupied, the movement fails in `env.step()`, but the heuristic doesn't know about this failure.

### Issue 3: No Queue Management
There's no mechanism to:
- Detect when supply source is "busy"
- Have agents wait at a designated location
- Prioritize which agent gets to recharge first

### Issue 4: Water Depletion During Return
If agents spend too many steps stuck in RETURNING state:
- `agent.water_remaining` decreases from movement/suppression
- Water might hit 0 before agent reaches home
- Agent is forced back to RETURNING state again (but already is)
- Wasted recharge cycles

---

## Solution Strategy

### Phase 1: Improve Collision Awareness
**Goal**: Make heuristic aware of other agents' positions and avoid direct collisions

**Approach**:
1. In `select_heuristic_action()`, check adjacent cells for other agents
2. When moving to supply source, identify alternative adjacent positions
3. If direct path is blocked, move to an adjacent position near supply source to form a "queue"

**Key changes**:
```python
def select_heuristic_action(env, agent_id, obs, obs_dict, env_config, partial_obs):
    # ... existing code ...

    if agent.state == AgentState.RETURNING:
        target_pos = agent.home_pos
        next_move = _move_towards(agent_pos, target_pos)
        next_pos = _get_next_position(agent_pos, next_move)

        # CHECK: Is next position occupied by another agent?
        next_cell = env.grid.get(*next_pos)
        if next_cell and hasattr(next_cell, 'index'):  # It's an agent
            # Try alternative move (one step away from direct path)
            alt_move = _find_alternative_move(agent_pos, target_pos, env)
            return alt_move if alt_move else 0  # STILL if no alternative

        return next_move
```

### Phase 2: Supply Source Queue Management
**Goal**: Organize agents into orderly recharge cycles

**Approach**:
1. Define "queue positions" around supply source (e.g., at distance 2-3)
2. Agents in RETURNING state move to nearest queue position
3. From queue position, wait for supply source to free up, then move in
4. Implement simple priority (e.g., "furthest along gets priority")

**Implementation options**:
- Option A (Simple): Just avoid blocking, let queue form naturally
- Option B (Moderate): Define explicit queue positions and move to them
- Option C (Complex): Add communication between agents (not heuristic anymore)

### Phase 3: Robustness Improvements
**Goal**: Handle edge cases and failures gracefully

**Approaches**:
1. **Timeout handling**: If agent is RETURNING for too long, force a move in any valid direction
2. **Water management**: Check if water will run out during return, transition earlier
3. **Deadlock detection**: If multiple agents are stuck, allow temporary overlap or rotate movements

---

## Recommended Implementation (Phase 1 + Phase 2 Simplified)

### Step 1: Add Helper Function to Check for Agent Collisions
```python
def _is_position_occupied_by_agent(env, pos):
    """Check if a position is occupied by another agent"""
    cell = env.grid.get(*pos)
    return cell is not None and hasattr(cell, 'index')  # Agent check
```

### Step 2: Add Function to Get Next Position from Action
```python
def _get_next_position(from_pos, action):
    """Calculate next position given current position and action"""
    x, y = from_pos
    # Map action to direction vector and return new position
    direction_vectors = {
        0: (0, 0),    # STILL
        1: (0, -1),   # NORTH
        2: (1, -1),   # NORTH_EAST
        3: (1, 0),    # EAST
        # ... etc
    }
    dx, dy = direction_vectors[action]
    return (x + dx, y + dy)
```

### Step 3: Add Fallback Movement Logic
```python
def _find_safe_direction(env, from_pos, to_pos):
    """
    Try to move toward to_pos, but avoid other agents.
    Returns safe action or 0 (STILL) if blocked.
    """
    primary_action = _move_towards(from_pos, to_pos)
    next_pos = _get_next_position(from_pos, primary_action)

    if not _is_position_occupied_by_agent(env, next_pos):
        return primary_action

    # Try adjacent moves (orthogonal directions first)
    for alternative_action in [1, 3, 5, 7, 2, 4, 6, 8]:  # Cardinal + diagonal
        alt_pos = _get_next_position(from_pos, alternative_action)
        if not _is_position_occupied_by_agent(env, alt_pos):
            # Check if this alternative gets us closer to goal
            dist_primary = abs(to_pos[0] - from_pos[0]) + abs(to_pos[1] - from_pos[1])
            dist_alt = abs(to_pos[0] - alt_pos[0]) + abs(to_pos[1] - alt_pos[1])
            if dist_alt <= dist_primary:  # Same or closer
                return alternative_action

    return 0  # STILL - all adjacent positions blocked
```

### Step 4: Update RETURNING State Logic
```python
# In select_heuristic_action():
if hasattr(agent, 'state') and agent.state == AgentState.RETURNING:
    if hasattr(agent, 'home_pos') and agent.home_pos is not None:
        return _find_safe_direction(env, agent_pos, agent.home_pos)  # NEW!
    else:
        return 0
```

### Step 5: Add Supply Source Queue Detection
```python
def _is_near_supply_source(env, agent_pos, home_pos, queue_distance=2):
    """Check if agent is within queue distance of supply source"""
    dist = abs(agent_pos[0] - home_pos[0]) + abs(agent_pos[1] - home_pos[1])
    return dist <= queue_distance

def _count_agents_near_supply_source(env, home_pos, queue_distance=2):
    """Count how many agents are near supply source"""
    count = 0
    for agent in env.agents:
        if _is_near_supply_source(env, tuple(agent.pos), home_pos, queue_distance):
            count += 1
    return count
```

### Step 6: Add Wait Logic When Supply Source is Busy
```python
# In select_heuristic_action() RETURNING logic:
if agent is within supply source queue:
    if supply_source is occupied:
        return 0  # STILL - wait for supply source to free up
    else:
        return _find_safe_direction(...)  # Try to move into supply source
```

---

## Testing Strategy

### Test Case 1: Two Agents Approaching Same Supply Source
```
Setup: 2 agents, one starting at (10, 11), one at (11, 10)
Both in RETURNING state heading to (11, 11)
Expected: Both reach supply source without deadlock, queue naturally
```

### Test Case 2: Multiple Agents Simultaneous Return
```
Setup: 6 agents, all run fire suppression for 100 steps, then all transition to RETURNING
Expected: All agents eventually reach supply source and recharge without blocking
```

### Test Case 3: Supply Source Saturation
```
Setup: All 6 agents return to supply source at same time
Expected: Agents queue around supply source, one recharges at a time, no permanent deadlock
```

### Test Case 4: Fallback Movement Chains
```
Setup: Multiple agents with blocked direct paths
Expected: Agents find alternative routes that make progress toward home
```

---

## Expected Improvements

| Aspect | Before | After |
|--------|--------|-------|
| Collision at supply source | Frequent blocking | Rare, handled with queue |
| Agent stuck in RETURNING | Possible indefinitely | Resolved within queue_distance steps |
| Water depletion during return | Wastes recharge cycles | Better planned with safe paths |
| Heuristic adaptability | No awareness of others | Reactive to other agents' positions |
| Code complexity | Simple but flaky | Moderate but robust |

---

## Files to Modify

1. **`train_marllib_self/heuristic_nearest_fire.py`**
   - Add helper functions: `_is_position_occupied_by_agent()`
   - Add helper functions: `_get_next_position()`
   - Add helper functions: `_find_safe_direction()`
   - Add helper functions: `_is_near_supply_source()` (optional)
   - Add helper functions: `_count_agents_near_supply_source()` (optional)
   - Update `select_heuristic_action()` to use new functions
   - Update RETURNING state handling to call `_find_safe_direction()`

2. **No changes needed to**:
   - `wildfire_environment/envs/wildfire.py` (collision system works as-is)
   - `wildfire_environment/core/agent.py` (can_overlap() intentionally False)
   - `train_marllib_self/environment.py` (config unchanged)

---

## Implementation Order

1. Add `_get_next_position()` (basic direction calculation)
2. Add `_is_position_occupied_by_agent()` (collision checking)
3. Add `_find_safe_direction()` (fallback movement logic)
4. Update `select_heuristic_action()` to use new functions
5. Test with simple 2-agent scenario
6. Test with full 6-agent scenario
7. Optionally add queue management functions (Phase 2)

---

## Potential Limitations

1. **No guaranteed collision-free path**: If all adjacent cells are blocked, agent must STILL
2. **No planning ahead**: Heuristic doesn't predict future collisions, only reacts
3. **Supply source queue still natural**: With Phase 1, we get a queue, but not an "optimal" one
4. **Deadlock possible but rare**: With queue_distance, theoretical deadlock becomes very unlikely

---

## Success Metrics

- ✓ No permanent agent blockage at supply source
- ✓ All agents complete return to supply source successfully
- ✓ Water depletion during return is minimal
- ✓ Agents transition through RETURNING → RECHARGING → ACTIVE smoothly
- ✓ Heuristic runs without exceptions or warnings
- ✓ GIF visualization shows orderly agent movement, not stuck agents

