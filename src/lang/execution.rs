use crate::vm::round_vm::RoundVM;

pub fn round<A: Copy + 'static>(
    vm: RoundVM,
    program: impl Fn(RoundVM) -> (RoundVM, A),
) -> (RoundVM, A) {
    let (mut vm_, res) = program(vm);
    vm_.register_root(res);
    let res = vm_.export_data().root::<A>().clone();
    (vm_, res)
}
