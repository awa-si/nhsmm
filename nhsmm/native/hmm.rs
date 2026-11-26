// Research-only static finite-HMM CPU kernel. No production API or autograd contract.
#[inline]
fn logsumexp(values: &[f64]) -> f64 {
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !maximum.is_finite() {
        return maximum;
    }
    maximum + values.iter().map(|x| (x - maximum).exp()).sum::<f64>().ln()
}

/// Inputs: log initial [K], log transition [K,K], emission scores [T,K].
/// Outputs: likelihood, d/d log initial [K], d/d log transition [K,K],
/// d/d emission [T,K]. Only finite, positive-support CPU Float64 is probed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hmm_forward_backward(
    t: usize,
    k: usize,
    initial_ptr: *const f64,
    transition_ptr: *const f64,
    emission_ptr: *const f64,
    likelihood_ptr: *mut f64,
    initial_grad_ptr: *mut f64,
    transition_grad_ptr: *mut f64,
    emission_grad_ptr: *mut f64,
) -> i32 {
    let Some(tk) = t.checked_mul(k) else { return 1 };
    let Some(kk) = k.checked_mul(k) else { return 1 };
    if t == 0
        || k == 0
        || initial_ptr.is_null()
        || transition_ptr.is_null()
        || emission_ptr.is_null()
        || likelihood_ptr.is_null()
        || initial_grad_ptr.is_null()
        || transition_grad_ptr.is_null()
        || emission_grad_ptr.is_null()
    {
        return 1;
    }
    // SAFETY: the research caller owns contiguous buffers of the documented sizes.
    let pi = unsafe { std::slice::from_raw_parts(initial_ptr, k) };
    let a = unsafe { std::slice::from_raw_parts(transition_ptr, kk) };
    let e = unsafe { std::slice::from_raw_parts(emission_ptr, tk) };
    let gi = unsafe { std::slice::from_raw_parts_mut(initial_grad_ptr, k) };
    let ga = unsafe { std::slice::from_raw_parts_mut(transition_grad_ptr, kk) };
    let ge = unsafe { std::slice::from_raw_parts_mut(emission_grad_ptr, tk) };

    let mut fwd = vec![f64::NEG_INFINITY; tk];
    let mut bwd = vec![0.0; tk];
    let mut candidates = vec![0.0; k];
    for j in 0..k {
        fwd[j] = pi[j] + e[j];
    }
    for row in 1..t {
        for j in 0..k {
            for i in 0..k {
                candidates[i] = fwd[(row - 1) * k + i] + a[i * k + j];
            }
            fwd[row * k + j] = logsumexp(&candidates) + e[row * k + j];
        }
    }
    let ll = logsumexp(&fwd[(t - 1) * k..t * k]);
    if !ll.is_finite() {
        return 2;
    }
    unsafe { *likelihood_ptr = ll };
    for row in (0..t - 1).rev() {
        for i in 0..k {
            for j in 0..k {
                candidates[j] = a[i * k + j] + e[(row + 1) * k + j] + bwd[(row + 1) * k + j];
            }
            bwd[row * k + i] = logsumexp(&candidates);
        }
    }
    ga.fill(0.0);
    for row in 0..t {
        for i in 0..k {
            ge[row * k + i] = (fwd[row * k + i] + bwd[row * k + i] - ll).exp();
        }
    }
    gi.copy_from_slice(&ge[..k]);
    for row in 0..t - 1 {
        for i in 0..k {
            for j in 0..k {
                ga[i * k + j] += (fwd[row * k + i]
                    + a[i * k + j]
                    + e[(row + 1) * k + j]
                    + bwd[(row + 1) * k + j]
                    - ll)
                    .exp();
            }
        }
    }
    0
}
