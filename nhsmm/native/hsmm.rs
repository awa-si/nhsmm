fn add_log(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let hi = a.max(b);
    hi + (-(a - b).abs()).exp().ln_1p()
}
fn sum_log(xs: &[f64]) -> f64 {
    let mut v = f64::NEG_INFINITY;
    for &x in xs {
        v = add_log(v, x);
    }
    v
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsmm_forward_backward(
    t: usize,
    k: usize,
    d: usize,
    initial_ptr: *const f64,
    emission_ptr: *const f64,
    transition_ptr: *const f64,
    end_ptr: *const f64,
    continuation_ptr: *const f64,
    likelihood_ptr: *mut f64,
    gi_ptr: *mut f64,
    ge_ptr: *mut f64,
    ga_ptr: *mut f64,
    gh_ptr: *mut f64,
    gc_ptr: *mut f64,
) -> i32 {
    let Some(m) = k.checked_mul(d) else { return 1 };
    let Some(tm) = t.checked_mul(m) else { return 1 };
    let Some(tk) = t.checked_mul(k) else { return 1 };
    let Some(tmk) = tm.checked_mul(k) else {
        return 1;
    };
    if t == 0
        || k == 0
        || d == 0
        || [
            initial_ptr,
            emission_ptr,
            transition_ptr,
            end_ptr,
            continuation_ptr,
        ]
        .iter()
        .any(|p| p.is_null())
        || [likelihood_ptr, gi_ptr, ge_ptr, ga_ptr, gh_ptr, gc_ptr]
            .iter()
            .any(|p| p.is_null())
    {
        return 1;
    }
    // SAFETY: the caller supplies contiguous Float64 buffers of the declared sizes.
    let pi = unsafe { std::slice::from_raw_parts(initial_ptr, k) };
    let em = unsafe { std::slice::from_raw_parts(emission_ptr, tk) };
    let a = unsafe { std::slice::from_raw_parts(transition_ptr, tmk) };
    let h = unsafe { std::slice::from_raw_parts(end_ptr, tm) };
    let c = unsafe { std::slice::from_raw_parts(continuation_ptr, tm) };
    let gi = unsafe { std::slice::from_raw_parts_mut(gi_ptr, k) };
    let ge = unsafe { std::slice::from_raw_parts_mut(ge_ptr, tk) };
    let ga = unsafe { std::slice::from_raw_parts_mut(ga_ptr, tmk) };
    let gh = unsafe { std::slice::from_raw_parts_mut(gh_ptr, tm) };
    let gc = unsafe { std::slice::from_raw_parts_mut(gc_ptr, tm) };
    gi.fill(0.0);
    ge.fill(0.0);
    ga.fill(0.0);
    gh.fill(0.0);
    gc.fill(0.0);
    let mut alpha = vec![f64::NEG_INFINITY; tm];
    let mut beta = vec![f64::NEG_INFINITY; tm];
    let mut scale = vec![0.0; t];
    for i in 0..k {
        alpha[i * d] = pi[i] + em[i];
    }
    scale[0] = sum_log(&alpha[..m]);
    if !scale[0].is_finite() {
        return 2;
    }
    for v in &mut alpha[..m] {
        *v -= scale[0];
    }
    for row in 1..t {
        let (before, after) = alpha.split_at_mut(row * m);
        let prev = &before[(row - 1) * m..row * m];
        let cur = &mut after[..m];
        for i in 0..k {
            for age in 0..d {
                let src = i * d + age;
                let mass = prev[src];
                let edge = (row - 1) * m + src;
                if !mass.is_finite() {
                    continue;
                }
                for j in 0..k {
                    cur[j * d] = add_log(cur[j * d], mass + h[edge] + a[edge * k + j]);
                }
                let next_age = (age + 1).min(d - 1);
                cur[i * d + next_age] = add_log(cur[i * d + next_age], mass + c[edge]);
            }
        }
        for i in 0..k {
            for age in 0..d {
                cur[i * d + age] += em[row * k + i];
            }
        }
        scale[row] = sum_log(cur);
        if !scale[row].is_finite() {
            return 2;
        }
        for v in cur {
            *v -= scale[row];
        }
    }
    let ll: f64 = scale.iter().sum();
    if !ll.is_finite() {
        return 2;
    }
    unsafe {
        *likelihood_ptr = ll;
    }
    for v in &mut beta[(t - 1) * m..] {
        *v = 0.0;
    }
    for row in (0..t - 1).rev() {
        let (before, after) = beta.split_at_mut((row + 1) * m);
        let cur = &mut before[row * m..(row + 1) * m];
        let next = &after[..m];
        for i in 0..k {
            for age in 0..d {
                let src = i * d + age;
                let edge = row * m + src;
                let mut mass = f64::NEG_INFINITY;
                for j in 0..k {
                    mass = add_log(
                        mass,
                        h[edge] + a[edge * k + j] + em[(row + 1) * k + j] + next[j * d],
                    );
                }
                mass = add_log(
                    mass,
                    c[edge] + em[(row + 1) * k + i] + next[i * d + (age + 1).min(d - 1)],
                );
                cur[src] = mass - scale[row + 1];
            }
        }
    }
    for row in 0..t {
        for i in 0..k {
            for age in 0..d {
                let src = row * m + i * d + age;
                let posterior = (alpha[src] + beta[src]).exp();
                ge[row * k + i] += posterior;
                if row == 0 && age == 0 {
                    gi[i] = posterior;
                }
                if row == t - 1 {
                    continue;
                }
                let edge = src;
                for j in 0..k {
                    let x = (alpha[src]
                        + h[edge]
                        + a[edge * k + j]
                        + em[(row + 1) * k + j]
                        + beta[(row + 1) * m + j * d]
                        - scale[row + 1])
                        .exp();
                    ga[edge * k + j] = x;
                    gh[edge] += x;
                }
                let x = (alpha[src]
                    + c[edge]
                    + em[(row + 1) * k + i]
                    + beta[(row + 1) * m + i * d + (age + 1).min(d - 1)]
                    - scale[row + 1])
                    .exp();
                gc[edge] = x;
            }
        }
    }
    0
}
