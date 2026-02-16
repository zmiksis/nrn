# Investigation: SPINE vs NEURON Voltage Divergence

## Problem Statement

Running the same ion channel model (HH + slow K + low-threshold Ca) in SPINE and
NEURON produces matching behavior for most of the simulation, but diverges near the
end: NEURON's voltage levels out around -50 mV, while SPINE's voltage keeps
decreasing.

---

## Root Cause: Implicit vs Explicit Treatment of Ionic Currents

**This is the primary and most consequential difference between the two simulators.**

### NEURON: Linearized Implicit Ionic Currents

NEURON assembles the cable equation as a linear system where ionic currents are
**linearized about the current voltage** and the voltage-proportional part is moved
to the left-hand side (matrix diagonal):

```
(Cm/dt + sum(g_i)) * V_new = Cm/dt * V_old + sum(g_i * E_i) + I_ext
```

This is implemented across several files:

1. **`src/nrnoc/treeset.cpp:175-188`** — The mathematical formulation:
   ```
   cm/dt*(Vnew - Vold) + (di/dvm*(Vnew - Vold) + i(Vold)) = ...
   ```
   Rearranging: `(cm/dt + di/dvm) * Vnew = cm/dt*Vold + di/dvm*Vold - i(Vold) + ...`

2. **`src/nrnoc/treeset.cpp:520-525`** — The Jacobian (di/dv = conductance) is
   computed by each mechanism's `jacob()` function and **added to the matrix
   diagonal** `vec_d[i]`:
   ```cpp
   jacob(sorted_token, _nt, tml->ml, tml->index);  // adds g to diagonal
   ```

3. **`src/nrnoc/passive0.cpp:39-46`** — Concrete example showing exactly how this
   works for a passive channel:
   ```cpp
   // current function: subtracts i = g*(V-E) from RHS
   static void pas_cur(...) {
       NODERHS(vnode[i]) -= g * (V - E);   // RHS -= g*V - g*E
   }
   // jacobian function: adds g to matrix diagonal
   static void pas_jacob(...) {
       vec_d[ni[i]] += g;                   // LHS diagonal += g
   }
   ```

4. **`src/nrnoc/capac.cpp:39-47`** — Capacitance `Cm/dt` is added to diagonal:
   ```cpp
   vec_d[ni[i]] += cfac * cm;  // where cfac = 0.001 * cj, cj = 1/dt
   ```

The resulting matrix equation for a single compartment:

```
(Cm/dt + g_Na + g_K + g_leak + g_slowK + g_CaT) * V_new
    = Cm/dt * V_old + g_Na*E_Na + g_K*E_K + g_leak*E_leak + ... + I_ext
```

Solving for V_new:

```
V_new = (Cm/dt * V_old + sum(g_i * E_i) + I_ext) / (Cm/dt + sum(g_i))
```

**This is a weighted average of V_old and the reversal potentials.** V_new is
mathematically bounded between the most negative E_i and the most positive E_i.
The voltage **cannot run away** — it is always pulled back toward the equilibrium
by every conductance in the denominator.

### SPINE: Fully Explicit Ionic Currents

SPINE uses SBDF (Semi-implicit Backward Differentiation Formulas) where **only the
cable diffusion operator is implicit**. All ionic currents are evaluated at the old
voltage and placed entirely on the right-hand side:

```
(I - gamma*dt*D) * V^{n+1} = SBDF_rhs(V_history, f_history)
```

Where:
- `D` = cable diffusion operator `(1/Rm*Cm) * nabla^2 V` (implicit, on LHS)
- `f^n = -sum(g_i/Cm * (V^n - E_i))` = total ionic current at V^n (explicit, on RHS)

From `src/spine/cable_eq/current_model.py`:
```python
def current_profile(self, comm, comm_iter):
    V = 1e-3 * self.neuron.sol.V    # Current voltage V^n
    for channel in self.channels:
        I_total -= channel.compute_current(V, C, self.Cm)  # All at V^n
    self.profile = I_total + self.Ext
```

**There is no conductance (`di/dV`) term on the LHS.** The entire current
`g*(V_old - E)` is a source term on the right side.

### Why This Matters

Consider what happens when voltage is near an equilibrium and a perturbation pushes
it slightly lower:

**NEURON (implicit):**
- The conductances `g_i` appear in the denominator: `V_new = (...) / (Cm/dt + sum(g_i))`
- Larger conductances (e.g., the 400 pS low-threshold Ca channel) create **stronger
  damping**, pulling V_new toward a weighted average of reversal potentials
- The equilibrium is **inherently stable** — the implicit treatment cannot overshoot

**SPINE (explicit):**
- The current `g_CaT * (V_old - 120mV)` is a large negative number (inward current)
  evaluated at V_old
- This large inward current pushes V_new lower
- At the next step, V is even more negative → even larger driving force → even larger
  inward current → V pushed lower still
- The cable diffusion on the LHS only helps if there are spatial gradients; for a
  uniform voltage (or small geometry like the 0.25 μm beam), it provides no restoring
  force
- Result: **slow drift away from equilibrium**

This is not classical numerical instability (oscillation/divergence) — it's a subtle
bias where the explicit treatment of currents allows the solution to creep past
equilibria that the implicit method would naturally find and hold.

---

## Contributing Factor: TABLE Statement Boundary Clamping

### NEURON

The HH model in NEURON uses TABLE lookups (`src/nrnoc/hh.mod:97`):
```nmodl
TABLE minf, mtau, hinf, htau, ninf, ntau DEPEND celsius FROM -100 TO 100 WITH 200
```

When voltage goes below -100 mV (or above +100 mV), rate functions are **clamped at
the boundary values**. This effectively:
- Prevents gating kinetics from producing extreme values at very negative voltages
- Limits how much the steady-state and time-constant functions can change
- Acts as an implicit "rate cap" at extreme voltages

### SPINE

SPINE computes rate functions analytically at every voltage with no domain clamping.
At very negative voltages (< -100 mV), the exponentials in the HH rate equations can
produce values that the TABLE mechanism in NEURON would have clamped.

This difference would compound with the explicit current treatment: SPINE allows
gating variables to reach more extreme values, which produce larger currents, which
push voltage further, in a feedback loop that NEURON's TABLE clamping interrupts.

---

## Contributing Factor: Forward Euler Gating Variables

The test script uses `update_method = 'forward_euler'`. In SPINE:

```python
def update_forward_euler(self, kinetics_fn, V, dt):
    state = self.current
    x_inf, tau = kinetics_fn(state, V)
    dstate = (x_inf - state) / tau
    self.current = state + dstate * dt
```

This is a simple forward Euler step. When `dt/tau > 2`, forward Euler **overshoots**
the steady-state and can oscillate. For the low-threshold calcium channel's `u` gate,
`tau_u` can become very small at certain voltages, making `dt/tau` large.

**In NEURON**, even though gating variables are also updated with their respective
methods (cnexp is typical for HH), the **voltage** is always solved implicitly. This
means even if gating variables overshoot slightly, the implicit voltage solve
compensates by damping the resulting current. In SPINE, both the gating overshoot
AND the explicit current evaluation compound.

Note: with `update_method = 'exact'` in SPINE, the gating variable update uses the
exponential integrator (`x_inf + (x - x_inf) * exp(-dt/tau)`), which cannot
overshoot. This would reduce but not eliminate the divergence, since the fundamental
issue (explicit ionic currents) remains.

---

## Contributing Factor: Model Parameters

The specific parameter choices make this model particularly sensitive:

| Parameter | Value | Effect |
|-----------|-------|--------|
| `V_rest` | 0.0 mV | Unusual; far from typical -65 mV |
| `leak.g` | 1.0e-16 S/μm² | Essentially zero leak conductance |
| `low_threshold_Ca.g_max` | 400 pS | Very large calcium conductance |
| `low_threshold_Ca.E_rev` | +120 mV | Large driving force at negative V |
| `dt` | 10 μs | May be insufficient for explicit methods |

The near-zero leak conductance is critical: in NEURON, even a tiny leak conductance
on the LHS diagonal helps stabilize the implicit solve. With `g_leak ≈ 0`, the only
stabilizing conductances are the voltage-gated channels themselves, making the system
more sensitive to the implicit-vs-explicit distinction.

---

## Summary of All Differences

| Aspect | NEURON | SPINE | Impact |
|--------|--------|-------|--------|
| **Ionic current treatment** | Implicit (linearized, g on LHS diagonal) | Explicit (full current on RHS) | **PRIMARY CAUSE** — NEURON's implicit solve bounds voltage between reversal potentials; SPINE's explicit treatment allows drift |
| **TABLE rate clamping** | Rates clamped at ±100 mV boundaries | No domain clamping | Prevents extreme gating values in NEURON |
| **Voltage solver** | Hines tridiagonal (implicit for all terms) | SBDF (implicit diffusion only) | NEURON implicitly handles all local currents; SPINE only handles spatial coupling implicitly |
| **di/dV linearization** | Yes — `jacob()` adds conductance to matrix diagonal | None | The mechanism by which NEURON achieves implicit current treatment |
| **Gating-voltage coupling** | Gating at V_old, but V_new solved implicitly accounting for conductances | Gating at V_old, V_new solved with explicit currents only | Both use operator splitting, but NEURON's implicit V solve compensates |
| **Forward Euler gating** | Not typically used (cnexp is default for HH) | Used as specified by user | Can overshoot in SPINE; NEURON's implicit V solve would damp the effect |

---

## Recommendations

1. **Add implicit linearization to SPINE's voltage solve.** The most impactful change
   would be to move the conductance term to the LHS:
   - Currently: `(I - γ*dt*D) * V_new = RHS + dt*f(V_old)`
   - Proposed: `(I - γ*dt*D + dt*G/Cm) * V_new = RHS + dt*(sum(g_i*E_i)/Cm + I_ext/Cm)`
   - Where `G = diag(sum(g_i))` is the total conductance matrix from all channels
   - This requires channels to report their conductance `g_i` separately from the full
     current, so the `g*V` part can go to the LHS and the `g*E` part to the RHS

2. **Use the `exact` (exponential) method for gating variables** instead of
   forward Euler, to eliminate gating overshoot as a compounding factor.

3. **Consider implementing TABLE-like rate clamping** in SPINE for voltages outside
   a physiological range (e.g., ±100 mV from rest).

4. **Reduce dt** when using forward Euler to ensure `dt/tau < 1` for all gating
   variables at all voltages encountered during the simulation.
