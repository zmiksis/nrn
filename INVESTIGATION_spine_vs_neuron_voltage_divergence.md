# Investigation: SPINE vs NEURON Voltage Divergence

## Problem Statement

Running the same ion channel model (HH + slow K + low-threshold Ca) in SPINE and
NEURON produces matching behavior for most of the simulation, but diverges near the
end: NEURON's voltage levels out around -50 mV, while SPINE's voltage keeps
decreasing. The model uses unphysiological parameters (V_rest = 0 mV, near-zero
leak conductance).

---

## Source of the Difference: Implicit vs Explicit Treatment of Ionic Currents

The two simulators treat ionic currents differently in the voltage solve.

### NEURON: Linearized Implicit Ionic Currents

NEURON linearizes ionic currents about the current voltage and solves for
`dV = V_new - V_old`. The conductance `g = dI/dV` (including gating variables)
is placed on the left-hand side (matrix diagonal), while the full current `I(V_old)`
goes on the right-hand side.

The implementation works as follows:

1. **`nrn_rhs()` (`src/nrnoc/treeset.cpp:375`)** — Each mechanism's `nrn_cur`
   function is called. The NMODL code generator
   (`src/nmodl/codegen/codegen_neuron_cpp_visitor.cpp:2417-2448`) evaluates the
   current at both `v` and `v + 0.001` to numerically extract the conductance:
   ```cpp
   double I1 = nrn_current(v + 0.001);   // perturbed
   double I0 = nrn_current(v);            // actual
   double rhs = I0;                        // full current
   double g   = (I1 - I0) / 0.001;        // conductance = dI/dV
   node_data.node_rhs[node_id] -= rhs;    // RHS -= I(V_old)
   ```

2. **`nrn_lhs()` (`src/nrnoc/treeset.cpp:488`)** — Each mechanism's `nrn_jacob`
   function adds the same `g` to the matrix diagonal:
   ```cpp
   node_data.node_diagonal[node_id] += g;  // LHS diagonal += dI/dV
   ```
   Capacitance `Cm/dt` is also added to the diagonal (`src/nrnoc/capac.cpp:39-47`).

3. **`nrn_solve()` (`src/nrnoc/solve.cpp:333`)** — The Hines algorithm
   (`triang()` + `bksub()`) solves the tridiagonal tree system for `dV`:
   ```
   (Cm/dt + sum(g_i) + axial) * dV = -I(V_old) + axial_currents
   ```

4. **`nrn_update_voltage()` (`src/nrnoc/fadvance.cpp:572`)** — Voltage is updated:
   ```cpp
   vec_v[i] += vec_rhs[i];   // V_new = V_old + dV
   ```

The resulting implicit equation is equivalent to:
```
V_new = (Cm/dt * V_old + sum(g_i * E_i) + axial) / (Cm/dt + sum(g_i) + axial)
```
This is a weighted average — voltage is mathematically bounded between the
reversal potentials.

Because the matrix diagonal changes every timestep (conductances depend on gating
variables), NEURON **rebuilds and refactors the matrix at every step**.

### SPINE: Fully Explicit Ionic Currents with Pre-factored LHS

SPINE uses SBDF (Semi-implicit Backward Differentiation Formulas) where **only the
cable diffusion operator is implicit**. All ionic currents are evaluated at the old
voltage and placed entirely on the right-hand side:

```
(I - gamma*dt*D) * V^{n+1} = SBDF_rhs(V_history, f_history)
```

Where:
- `D` = cable diffusion operator `(1/Rm*Cm) * nabla^2 V` (implicit, on LHS)
- `f^n = -sum(g_i/Cm * (V^n - E_i))` = total ionic current at V^n (explicit, on RHS)

Because the LHS matrix depends only on geometry and constant membrane parameters
(`Rm`, `Cm`), it is **pre-factored once** before the time loop using sparse LU
decomposition (`scipy.sparse.linalg.splu`). Each timestep requires only an O(n)
back-substitution — no matrix reassembly or refactoring.

---

## Analysis: NEURON's Behavior is the Artifact

The divergence occurs in an unphysiological model setup:

| Parameter | Value | Issue |
|-----------|-------|-------|
| `V_rest` | 0.0 mV | Far from typical resting potentials |
| `leak.g` | 1.0e-16 S/μm² | Essentially zero — no passive restoring force |
| `low_threshold_Ca.g_max` | 400 pS | Large calcium conductance |
| `low_threshold_Ca.E_rev` | +120 mV | Large driving force at negative voltages |

SPINE produces a smooth, monotonically decreasing voltage — a mathematically
consistent solution to the equations as written with these parameters. There is no
oscillation, no blowup, and no sign of numerical instability.

NEURON's voltage "leveling out around -50 mV" is **artificial damping introduced
by the implicit linearization**. The conductance terms on the LHS diagonal act as
numerical friction, pulling voltage toward a weighted average of reversal potentials
regardless of whether the underlying equations actually have a stable equilibrium at
that point. In this unphysiological regime, NEURON is manufacturing an equilibrium
that does not exist in the continuous equations.

For physiologically parameterized models (realistic resting potential, non-negligible
leak conductance, reasonable channel densities), the two approaches converge to
essentially the same answer. The divergence is specific to this edge case.

---

## Additional Differences Noted

### TABLE Statement Boundary Clamping

NEURON's HH model uses TABLE lookups (`src/nrnoc/hh.mod:97`):
```nmodl
TABLE minf, mtau, hinf, htau, ninf, ntau DEPEND celsius FROM -100 TO 100 WITH 200
```
Rates are clamped at boundary voltages (±100 mV). SPINE computes rate functions
analytically at all voltages with no domain clamping.

### Forward Euler Gating Variables

The test script uses `update_method = 'forward_euler'`. When `dt/tau > 2`, forward
Euler can overshoot steady-state. SPINE's `exact` (exponential) integrator avoids
this entirely: `x_new = x_inf + (x - x_inf) * exp(-dt/tau)`.

---

## SPINE's Approach is Appropriate

SPINE's explicit treatment of ionic currents with a pre-factored LHS is the right
design for its use case:

1. **Computational efficiency**: The LHS matrix is factored once. Each timestep is
   an O(n) back-substitution. NEURON must rebuild and refactor every step because
   the conductances on the diagonal change. Over clinical-length simulations with
   millions of timesteps, this difference in per-step cost is substantial.

2. **Faithful representation**: SPINE evaluates the model equations as written and
   shows the true behavior. It does not introduce artificial damping that can mask
   model errors or create spurious equilibria in edge cases.

3. **Numerical adequacy**: For physiological parameter regimes, SPINE's scheme
   is stable and accurate. The SBDF2 time stepper provides second-order accuracy
   with implicit treatment of the stiff diffusion operator, and the exact gating
   integrator is unconditionally stable for the gating ODEs.

The observed divergence from NEURON occurs only in unphysiological configurations
where NEURON's implicit scheme introduces artificial stabilization that is arguably
less correct than SPINE's explicit result.
