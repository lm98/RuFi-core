use crate::slot::Slot::{Branch, FoldHood, Nbr, Rep};
use crate::vm::round_vm::RoundVM;
use std::str::FromStr;

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
pub fn nbr<A: Clone + 'static + FromStr, F>(mut vm: RoundVM, expr: F) -> (RoundVM, A)
where
    F: Fn(RoundVM) -> (RoundVM, A),
{
    vm.nest_in(Nbr(vm.index().clone()));
    let (mut vm_, val) = match vm.neighbor() {
        Some(nbr) if nbr != vm.self_id() => match vm.neighbor_val::<A>() {
            Ok(val) => (vm.clone(), val.clone()),
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
pub fn rep<A: Clone + 'static + FromStr, F, G>(mut vm: RoundVM, init: F, fun: G) -> (RoundVM, A)
where
    F: Fn(RoundVM) -> (RoundVM, A),
    G: Fn(RoundVM, A) -> (RoundVM, A),
{
    vm.nest_in(Rep(vm.index().clone()));
    let (mut vm_, val) = vm.locally(|vm1| {
        if vm1.previous_round_val::<A>().is_ok() {
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
pub fn foldhood<A: Clone + 'static + FromStr, F, G, H>(
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
    let (vm_, local_init) = vm.locally(|vm_| init(vm_));
    let temp_vec: Vec<A> = Vec::new();
    let (mut vm__, nbrs_vec) = nbrs_computation(vm_, expr, temp_vec, nbrs, local_init.clone());
    let (mut vm___, res) = vm__.isolate(|vm_| {
        let val = nbrs_vec
            .iter()
            .fold(local_init.clone(), |x, y| aggr(x, y.clone()));
        (vm_, val)
    });
    let res_ = vm___.nest_write(true, res);
    vm___.nest_out(true);
    (vm___, res_)
}

/// A utility function used by the `foldhood` function.
fn nbrs_computation<A: Clone + 'static, F>(
    mut vm: RoundVM,
    expr: F,
    mut tmp: Vec<A>,
    mut ids: Vec<i32>,
    init: A,
) -> (RoundVM, Vec<A>)
where
    F: Fn(RoundVM) -> (RoundVM, A) + Copy,
{
    if ids.len() == 0 {
        return (vm, tmp);
    } else {
        let current_id = ids.pop();
        let (vm_, res) = vm.folded_eval(expr, current_id.unwrap());
        tmp.push(res.unwrap_or(init.clone()).clone());
        nbrs_computation(vm_, expr, tmp, ids, init)
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
pub fn branch<A: Clone + 'static + FromStr, B, TH, EL>(
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
    let (mut vm, tag) = vm.locally(|_vm1| (_vm1, cond()));
    let (mut vm_, val): (RoundVM, A) = match vm.neighbor() {
        Some(nbr) if nbr != vm.self_id() => {
            let val_clone = vm.neighbor_val::<A>().unwrap().clone();
            (vm, val_clone)
        }
        _ => {
            if tag {
                //locally(vm, thn);
                vm.locally(thn)
            } else {
                //locally(vm, els)
                vm.locally(els)
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