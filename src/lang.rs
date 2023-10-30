use crate::slot::Slot::{Branch, FoldHood, Nbr, Rep};
use crate::vm::round_vm::RoundVM;

pub mod builtins;
pub mod execution;
pub mod macros;

/// Observes the value of an expression across neighbors, producing a “field of fields”.
///
/// # Arguments
///
/// * `vm` the current VM
/// * `expr` the expression to evaluate
///
/// # Generic Parameters
///
/// * `A` The type of value returned by the expression.
/// * `F` - The type of the closure, which must be a closure that takes a `RoundVM` as argument and returns a tuple `(RoundVM, A)`.
///
/// # Returns
///
/// the value of the expression
pub fn nbr<A: Copy + 'static, F>(mut vm: RoundVM, expr: F) -> (RoundVM, A)
where
    F: Fn(RoundVM) -> (RoundVM, A),
{
    vm.nest_in(Nbr(vm.index().clone()));
    let (mut vm_, val) = match vm.neighbor() {
        Some(nbr) if nbr.clone() != vm.self_id() => match vm.neighbor_val::<A>() {
            Some(val) => (vm.clone(), val.clone()),
            _ => expr(vm.clone()),
        },
        _ => expr(vm),
    };
    let res = vm_.nest_write(vm_.unless_folding_on_others(), val);
    vm_.nest_out(true);
    (vm_, res)
}

/// Iteratively updates the value of the input expression at each device using the last computed value.
///
/// # Arguments
///
/// * `vm` the current VM
/// * `init` the initial value
/// * `fun` the function to apply to the value
///
/// # Generic Parameters
///
/// * `A` The type of value returned by the expression.
/// * `F` - The type of the closure, which must be a closure that takes no arguments and returns a value of type `A`.
/// * `G` - The type of the closure, which must be a closure that takes a tuple `(RoundVM, A)` and returns a tuple `(RoundVM, A)`.
///
/// # Returns
///
/// the updated value
pub fn rep<A: Copy + 'static, F, G>(mut vm: RoundVM, init: F, fun: G) -> (RoundVM, A)
where
    F: Fn(RoundVM) -> (RoundVM, A),
    G: Fn(RoundVM, A) -> (RoundVM, A),
{
    vm.nest_in(Rep(vm.index().clone()));
    let (mut vm_, val) = locally(vm, |vm1| {
        if vm1.previous_round_val::<A>().is_some() {
            let prev = vm1.previous_round_val::<A>().unwrap().clone();
            fun(vm1, prev)
        } else {
            let init_args = init(vm1);
            fun(init_args.0, init_args.1)
        }
    });
    let res = vm_.nest_write(vm_.unless_folding_on_others(), val);
    vm_.nest_out(true);
    (vm_, res)
}

/// Aggregates the results of the neighbor computation.
///
/// # Arguments
///
/// * `vm` the current VM
/// * `init` the initial value
/// * `aggr` the function to apply to the value
/// * `expr` the expression to evaluate
///
/// # Generic Parameters
///
/// * `A` The type of value returned by the expression.
/// * `F` - The type of inti, which must be a closure that takes no arguments and returns a value of type `A`.
/// * `G` - The type of aggr, which must be a closure that takes a tuple `(A, A)` and returns a value of type `A`.
/// * `H` - The type of expr, which must be a closure that takes a `RoundVM` as argument and returns a tuple `(RoundVM, A)`.
///
/// # Returns
///
/// the aggregated value
pub fn foldhood<A: Copy + 'static, F, G, H>(
    mut vm: RoundVM,
    init: F,
    aggr: G,
    expr: H,
) -> (RoundVM, A)
where
    F: Fn(RoundVM) -> (RoundVM, A),
    G: Fn(A, A) -> A,
    H: Fn(RoundVM) -> (RoundVM, A) + Copy,
{
    vm.nest_in(FoldHood(vm.index().clone()));
    let nbrs = vm.aligned_neighbours::<A>().clone();
    let (vm_, local_init) = locally(vm, |vm_| init(vm_));
    let temp_vec: Vec<A> = Vec::new();
    let (vm__, nbrs_vec) = nbrs_computation(vm_, expr, temp_vec, nbrs, local_init);
    let (mut vm___, res) = isolate(vm__, |vm_| {
        let val = nbrs_vec.iter().fold(local_init, |x, y| aggr(x, y.clone()));
        (vm_, val)
    });
    let res_ = vm___.nest_write(true, res);
    vm___.nest_out(true);
    (vm___, res_)
}

/// A utility function used by the `foldhood` function.
fn nbrs_computation<A: Copy + 'static>(
    vm: RoundVM,
    expr: impl Fn(RoundVM) -> (RoundVM, A),
    mut tmp: Vec<A>,
    mut ids: Vec<i32>,
    init: A,
) -> (RoundVM, Vec<A>) {
    if ids.len() == 0 {
        return (vm, tmp);
    } else {
        let current_id = ids.pop();
        let (vm_, res, expr_) = folded_eval(vm, expr, current_id);
        tmp.push(res.unwrap_or(init).clone());
        nbrs_computation(vm_, expr_, tmp, ids, init)
    }
}

/// Partitions the domain into two subspaces that do not interact with each other.
///
/// # Arguments
///
/// * `vm` the current VM
/// * `cond` the condition to evaluate
/// * `thn` the expression to evaluate if the condition is true
/// * `els` the expression to evaluate if the condition is false
///
/// # Generic Parameters
///
/// * `A` The type of value returned by the expression.
/// * `B` - The type of cond, which must be a closure that takes no arguments and returns a value of type `bool`.
/// * `F` - The type of thn and els, which must be a closure that takes a `RoundVM` as argument and returns a tuple `(RoundVM, A)`.
///
/// # Returns
///
/// the value of the expression
pub fn branch<A: Copy + 'static, B, TH, EL>(
    mut vm: RoundVM,
    cond: B,
    thn: TH,
    els: EL,
) -> (RoundVM, A)
where
    B: Fn() -> bool,
    TH: Fn(RoundVM) -> (RoundVM, A),
    EL: Fn(RoundVM) -> (RoundVM, A),
{
    vm.nest_in(Branch(vm.index().clone()));
    let (vm, tag) = locally(vm, |_vm1| (_vm1, cond()));
    let (mut vm_, val): (RoundVM, A) = match vm.neighbor() {
        Some(nbr) if nbr.clone() != vm.self_id() => {
            let val_clone = vm.neighbor_val::<A>().unwrap().clone();
            (vm, val_clone)
        }
        _ => {
            if tag {
                locally(vm, |vm1| thn(vm1))
            } else {
                locally(vm, |vm1| els(vm1))
            }
        }
    };
    let res = vm_.nest_write(vm_.unless_folding_on_others(), val);
    vm_.nest_out(tag);
    (vm_, res)
}

/// Returns the id of the current device.
///
/// # Arguments
///
/// * `vm` the current VM
///
/// # Returns
///
/// the id of the current device
pub fn mid(vm: RoundVM) -> (RoundVM, i32) {
    let mid = vm.self_id().clone();
    (vm, mid)
}

/// Evaluates the given expression locally and return the result.
///
/// # Arguments
///
/// * `expr` The expression to evaluate.
///
/// # Generic Parameters
///
/// * `A` - The type of value returned by the expression.
/// * `F` - The type of the closure, which must be a mutable closure that takes no arguments and returns a value of type `A`.
///
/// # Returns
///
/// The result of the closure `expr`.
fn locally<A: Copy + 'static>(
    mut vm: RoundVM,
    expr: impl Fn(RoundVM) -> (RoundVM, A),
) -> (RoundVM, A) {
    let current_neighbour = vm.neighbor().map(|id| id.clone());
    vm.status = vm.status.fold_out();
    let (mut vm_, result) = expr(vm);
    vm_.status = vm_.status.fold_into(current_neighbour);
    (vm_, result)
}

fn isolate<A: Copy + 'static, F>(mut vm: RoundVM, expr: F) -> (RoundVM, A)
where
    F: Fn(RoundVM) -> (RoundVM, A),
{
    vm.status = vm.status.fold_out();
    let (mut vm_, result) = expr(vm);
    vm_.status = vm_.status.fold_into(None);
    (vm_, result)
}

/// Perform a folded evaluation of the given expression in the given neighbor and return the result.
/// Used by the `foldhood` function.
/// Same behavior as the folded_eval function in src/core/lang/round_vm.rs.
///
/// # Arguments
///
/// * `vm` - The current VM.
/// * `expr` - The expression to evaluate, which should return a value of type `A`.
/// * `id` - The id of the neighbor. It is of type `i32`.
///
/// # Generic Parameters
///
/// * `A` - The type of value returned by the expression.
/// * `F` - The type of the expr, which must be a closure that takes o `RoundVM` as an argument and returns a tuple of type `(RoundVM, A)`.
///
/// # Returns
///
/// A tuple of type `(RoundVM, Option<A>, F)`.
fn folded_eval<A: Copy + 'static, F>(
    mut vm: RoundVM,
    expr: F,
    id: Option<i32>,
) -> (RoundVM, Option<A>, F)
where
    F: Fn(RoundVM) -> (RoundVM, A),
{
    vm.status = vm.status.push();
    vm.status = vm.status.fold_into(id);
    let (mut vm_, res) = expr(vm);
    vm_.status = vm_.status.pop();
    (vm_, Some(res), expr)
}
