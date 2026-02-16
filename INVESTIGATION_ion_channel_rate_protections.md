# Investigation: Ion Channel Rate Protections in NEURON

## Overview

This document catalogs every mechanism in the NEURON codebase that caps, clamps,
constrains, or otherwise protects ion channel rate computations, state variables,
and related numerical quantities. The protections are organized from most to least
directly relevant to ion channel behavior.

---

## 1. Singularity Avoidance in Rate Functions

### 1a. `vtrap()` — Voltage Trap for HH-style Rate Equations

**File:** `src/nrnoc/hh.mod:121-127`
**Also:** `src/coreneuron/mechanism/mech/modfile/hh.mod:117-121`

```nmodl
FUNCTION vtrap(x,y) {  :Traps for 0 in denominator of rate eqns.
        if (fabs(x/y) < 1e-6) {
                vtrap = y*(1 - x/y/2)
        }else{
                vtrap = x/(exp(x/y) - 1)
        }
}
```

**What it does:** The classic HH alpha rate functions contain the form `x/(exp(x/y) - 1)`,
which has a removable singularity at `x = 0` (0/0). At that point, `exp(x/y) - 1 ≈ x/y`,
so the true limit is `y`. `vtrap` switches to a first-order Taylor expansion
`y*(1 - x/(2y))` when `|x/y| < 1e-6`, avoiding the division by zero.

**Where used in `hh.mod`:**
- Line 102: `alpha = .1 * vtrap(-(v+40), 10)` — sodium m-gate activation
- Line 114: `alpha = .01 * vtrap(-(v+55), 10)` — potassium n-gate activation

**Impact:** Without this, simulations would produce NaN/Inf at specific voltages
(v = -40 mV for Na activation, v = -55 mV for K activation). This is a
**model-level** convention — every .mod file must implement its own singularity
protection. NEURON does NOT insert this automatically.

### 1b. `efun()` — Safe Exponential for GHK Equation

**File:** `src/nrnoc/eion.cpp:341-347`

```cpp
static double efun(double x) {
    if (fabs(x) < 1e-4) {
        return 1. - x / 2.;
    } else {
        return x / (exp(x) - 1);
    }
}
```

**What it does:** Identical mathematical form to `vtrap` but used internally in
`nrn_ghk()` (Goldman-Hodgkin-Katz current equation, line 349). Computes
`z*v/ktf / (exp(z*v/ktf) - 1)` safely when voltage is near zero.

**Where used:** `nrn_ghk()` at `src/nrnoc/eion.cpp:349-355`, called whenever a
model uses the GHK equation for ion permeation currents.

---

## 2. The `cnexp` Integration Method — Implicit Rate Bounding

**File:** `src/nmodl/parser/diffeq_context.cpp:119-136`

```cpp
std::string DiffEqContext::get_cnexp_solution() const {
    auto a = cvode_deriv();    // derivative of RHS w.r.t. state (the "rate")
    auto b = cvode_eqnrhs();   // steady-state value
    std::string result;
    if (a == "0.0") {
        // ... forward Euler fallback ...
        result = state + " = " + state + "-dt*(" + b + ")";
    } else {
        result = state + " = " + state + "+(1.0-exp(dt*(" + a + ")))*(" + b + "-" + state + ")";
    }
    return result;
}
```

**What it does:** For a linear ODE `dy/dt = a*(y - b)` (where `a < 0` for stable
channels), `cnexp` computes the exact analytical solution:

```
y(t+dt) = y(t) + (1 - exp(a*dt)) * (b - y(t))
```

**This is the most important "rate capping" in NEURON.** Here's why:

- When `a < 0` (always true for stable gating variables), `exp(a*dt)` is between 0 and 1
- Therefore `(1 - exp(a*dt))` is between 0 and 1
- The state moves toward `b` (the steady-state) by a **fraction** of the distance
- Even with enormous rates (`|a| >> 1/dt`), the state **cannot overshoot** `b`
- As `|a| -> infinity`, `exp(a*dt) -> 0`, so the state simply jumps to `b`

**Contrast with forward Euler:** `y(t+dt) = y(t) + dt * a * (y(t) - b)` would
overshoot and oscillate when `|a*dt| > 2`, and diverge when `|a*dt| > 2`.

This is why the HH model (line 65 of `hh.mod`) uses `SOLVE states METHOD cnexp`
and writes derivatives in the `(inf - state) / tau` form (lines 84-86):

```nmodl
DERIVATIVE states {
    rates(v)
    m' = (minf - m) / mtau
    h' = (hinf - h) / htau
    n' = (ninf - n) / ntau
}
```

The NMODL compiler recognizes this linear form, extracts `a = -1/tau` and `b = inf`,
and generates the exponential update. **The rate is effectively capped by the
mathematical structure of the solution, not by an explicit clamp.**

---

## 3. The `TABLE` Statement — Discretized Rate Lookup

**File:** `src/nrnoc/hh.mod:97`

```nmodl
TABLE minf, mtau, hinf, htau, ninf, ntau DEPEND celsius FROM -100 TO 100 WITH 200
```

**What it does:** Pre-computes `minf`, `mtau`, `hinf`, `htau`, `ninf`, `ntau` at
201 evenly spaced voltage points from -100 to +100 mV. During simulation, values
are looked up via linear interpolation rather than recomputing exponentials.

**Rate-capping implications:**
1. **Domain restriction:** Voltages outside [-100, +100] mV use the boundary values
   (the table is clamped at its edges). This effectively caps rates at whatever they
   are at +/- 100 mV.
2. **Smoothing:** Linear interpolation between table entries eliminates any
   sub-millivolt spikes in rate functions.
3. **Performance:** Avoids recomputing `exp()` calls at every timestep.

**Implementation:** When TABLE is present, the NMODL compiler generates a
`_check_table_thread` function (registered via `_nrn_thread_table_reg` at
`src/nrnoc/init.cpp:1251-1252`) that rebuilds the table when dependencies (like
`celsius`) change. The `usetable` flag (default: on) can be toggled to disable
table lookup.

---

## 4. The `derivimplicit` / Newton Solver — Implicit Stability

**File:** `src/nmodl/solver/newton/newton.hpp:79-117`

```cpp
static constexpr int MAX_ITER = 50;
static constexpr double EPS = 1e-13;

template <int N, typename FUNC>
int newton_solver(Eigen::Matrix<double, N, 1>& X, FUNC functor,
                  double eps = EPS, int max_iter = MAX_ITER) {
    // ... Newton iteration: X_{n+1} = X_n - J^{-1} * F(X_n) ...
}
```

**What it does:** For `SOLVE states METHOD derivimplicit` (used with nonlinear or
stiff kinetic schemes), NEURON uses backward Euler with a Newton solver. The
implicit formulation solves:

```
(y(t+dt) - y(t)) / dt = f(y(t+dt))
```

**Rate-capping implications:**
- Backward Euler is **unconditionally stable** — it cannot diverge regardless of
  how large rates are, unlike forward Euler
- The Newton solver converges (or fails after 50 iterations) rather than allowing
  unbounded state excursions
- If the Jacobian is singular (Crout decomposition at `src/nmodl/solver/crout/crout.hpp`
  with pivot threshold `1e-20`), the solver returns -1 rather than producing garbage
- Template specializations for N<=4 use explicit matrix inverse (lines 125-150)

**No explicit clamping:** The Newton solver does NOT clamp states to [0,1]. States
can go negative or exceed 1 if the model equations allow it.

---

## 5. Ion Concentration Limits

**File:** `src/nrnoc/eion.cpp:182-183`

```cpp
hoc_symbol_limits(hoc_lookup(buf[2].c_str()), 1e-12, 1e9);  // internal concentration
hoc_symbol_limits(hoc_lookup(buf[3].c_str()), 1e-12, 1e9);  // external concentration
```

**What it does:** Every ion species (na, k, ca, etc.) gets its internal and external
concentrations registered with bounds `[1e-12, 1e9]` mM.

**How limits are enforced:** `hoc_symbol_limits` stores the bounds on the Symbol
(`src/oc/code2.cpp:102-108`). The actual clamping function is `check_domain_limits`
(`src/oc/code2.cpp:116-125`):

```cpp
double check_domain_limits(float* limits, double val) {
    if (limits) {
        if (val < limits[0]) return (double) limits[0];
        else if (val > limits[1]) return (double) limits[1];
    }
    return val;
}
```

**Important caveat:** This function is primarily called from the GUI panel editor
(`src/ivoc/xmenu.cpp:1927-1928`) when users interactively change parameters. It is
**NOT called during the integration loop**. During simulation, concentrations
can transiently exceed these limits. The limits are advisory metadata, not runtime
clamps.

### Nernst Equation Safeguards

**File:** `src/nrnoc/eion.cpp:279-291`

```cpp
double nrn_nernst(double ci, double co, double z) {
    if (z == 0) return 0.;
    if (ci <= 0.) return 1e6;
    else if (co <= 0.) return -1e6;
    else return ktf / z * log(co / ci);
}
```

**What it does:** Returns extreme voltages (+/- 1 MV) if concentrations are zero
or negative, instead of producing `log(0)` = -Inf or `log(negative)` = NaN. This
is a **hard runtime protection** that fires during integration.

---

## 6. Global Parameter Limits

**File:** `src/nrnoc/init.cpp:120-128`

```cpp
static HocParmLimits _hoc_parm_limits[] = {
    {"Ra",          {1e-6, 1e9}},
    {"L",           {1e-4, 1e20}},
    {"diam",        {1e-9, 1e9}},
    {"cm",          {0.,   1e9}},
    {"rallbranch",  {1.,   1e9}},
    {"nseg",        {1.,   1e9}},
    {"celsius",     {-273., 1e6}},
    {"dt",          {1e-9, 1e15}},
    {nullptr,       {0., 0.}}
};
```

**What it does:** Registers bounds for core morphological and simulation parameters.
Same enforcement mechanism as ion concentrations (advisory/GUI only, not runtime).

### PARAMETER Bounds in .mod Files

**File:** `src/nrnoc/hh.mod:38-41`

```nmodl
PARAMETER {
    gnabar = .12 (S/cm2) <0,1e9>
    gkbar = .036 (S/cm2) <0,1e9>
    gl = .0003 (S/cm2)   <0,1e9>
    el = -54.3 (mV)
}
```

The `<low,high>` syntax in PARAMETER declarations is compiled by the NMODL
translator (`src/nocmodl/nocpout.cpp:673-689`) into a `_hoc_parm_limits` array,
which is registered via `hoc_register_limits` (`src/nrnoc/init.cpp:1092-1107`).
These are **advisory bounds** — enforced by GUI editors but not during integration.

---

## 7. PROTECT Statement (Thread Safety, NOT Rate Limiting)

**File:** `src/nmodl/language/nmodl.yaml:1403`

```nmodl
PROTECT g_cnt = g_cnt + 1
```

**What it does:** Generates an atomic/mutex-protected assignment for thread safety
when writing to GLOBAL variables from multiple threads. This has **nothing to do
with rate limiting** — it prevents data races, not numerical overflow.

---

## 8. Sparse Matrix Solver Numerical Safeguards

**File:** `src/sparse13/spfactor.cpp`

The sparse matrix factorization used in implicit solvers includes:
- Partial pivoting to avoid near-zero diagonal elements
- Markowitz product minimization for fill-in reduction
- Overflow prevention during LU decomposition

These prevent the **linear algebra** from failing but do not directly cap rates
or states.

---

## Summary Table

| Mechanism | Location | Type | Runtime? | Scope |
|-----------|----------|------|----------|-------|
| `vtrap()` / singularity avoidance | `hh.mod:121` | Prevents NaN at specific voltages | Yes | Per-model (user-written) |
| `efun()` for GHK | `eion.cpp:341` | Prevents NaN near V=0 | Yes | All GHK computations |
| `cnexp` exponential integrator | `diffeq_context.cpp:133` | Prevents overshoot via analytical solution | Yes | All `METHOD cnexp` models |
| `TABLE` statement | `hh.mod:97` | Clamps rates at table boundaries | Yes | Per-model (optional) |
| `derivimplicit` Newton solver | `newton.hpp:80` | Unconditionally stable implicit method | Yes | All `METHOD derivimplicit` models |
| Nernst safeguards | `eion.cpp:279` | Returns +/-1e6 mV for zero concentration | Yes | All reversal potential calculations |
| Ion concentration limits | `eion.cpp:182` | Bounds to [1e-12, 1e9] mM | GUI only | All ions |
| Parameter `<lo,hi>` bounds | `nocpout.cpp:674` | Advisory bounds from .mod PARAMETER | GUI only | Per-model |
| `PROTECT` | `nmodl.yaml:1403` | Thread-safe atomic writes | Yes | Thread safety only |

## Key Takeaway

**There is no single global "rate cap" in NEURON.** The apparent rate limiting you
observe is most likely the `cnexp` method (mechanism #2 above), which analytically
solves the gating ODE so that states approach their steady-state values
exponentially without overshooting, regardless of how large the rate constants
become. This, combined with the `TABLE` statement's domain clamping and the
`vtrap` singularity avoidance, creates the behavior of rates appearing to be
"capped" even at extreme voltages.
