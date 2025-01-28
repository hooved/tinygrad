# TODO: merge with examples/tinychat/compile.py

import os, re
from extra.export_model import compile_net, jit_model, export_model
from examples.llama3 import build_transformer
from tinygrad import Device, Variable, Tensor, dtypes
from tinygrad.helpers import fetch, Context
from typing import List, Tuple, Any, NamedTuple
from tinygrad.nn.state import get_state_dict
from tinygrad.ops import Ops
from collections import OrderedDict
from examples.tinychat.compile import q6k_to_f32

# from nn.state.ggml_data_to_tensor: TODO: move to namespace of nn.state so we can import?
def q_to_uint8(t: Tensor, b: int) -> Tensor:
  # TODO: rewrite with arange?
  shift_tensor, bitmask = Tensor.stack(*[ Tensor(2**(i*b), device=t.device, dtype=t.dtype) for i in range(8//b) ]), 0xff >> (8 - b)
  return t.unsqueeze(-1).expand((*t.shape,8//b)).idiv(shift_tensor).bitwise_and(bitmask).transpose(-1, -2).flatten(-2)

# TODO: refactor to merge with q6k_to_f32
def q6k_to_f16(x: Tensor) -> Tensor:
  blocks = x.reshape((-1, 210))
  xl, xh = q_to_uint8(blocks[:,:128].reshape((-1, 2, 64)), 4), q_to_uint8(blocks[:,128:192].reshape((-1, 2, 32)), 2).lshift(4)
  scales = blocks[:,192:208].bitcast(dtypes.int8).unsqueeze(-1).expand((-1, 16, 16)).reshape((-1, 256))
  #d = blocks[:,-2:].bitcast(dtypes.float16).cast(dtypes.float32).expand((-1, 256))
  # TODO: parameterize function for returning fp16 or fp32, or cast fp32 back to fp16 downstream
  d = blocks[:,-2:].bitcast(dtypes.float16).expand((-1, 256))
  return (d * (xl.bitwise_or(xh).bitcast(dtypes.int8) - 32).flatten(-2) * scales).flatten()

def q6k_to_int8(x: Tensor, shape: Tuple) -> Tuple:
  #x = q6k_to_f16(x).reshape(shape)
  x = q6k_to_f32(x).reshape(shape)
  scale = x.abs().max(axis=1) / 127.0
  int8_weight = (x.T/scale).T.cast(dtype=dtypes.int8)
  return scale, int8_weight

if __name__=="__main__":
  Device.DEFAULT = "CLANG"
  model_path = fetch("https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf", "Llama-3.2-1B-Instruct-Q6_K.gguf", subdir="llama3-1b-instruct")
  model_size="1B"
  Tensor.no_grad = True
  f32_fn = os.path.join(os.path.dirname(__file__), "llama3_1B_f32.safetensors")
  max_context=1024
  model = build_transformer(model_path, model_size=model_size, quantize="int8", max_context=max_context)

  TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P = 0.95, 0, 0.0, 0.0, 0.0

  tok = 128000
  model_input = [Tensor([[tok]]), 0, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P]
  out = model.forward(*model_input)

  class Step(NamedTuple):
    name: str = ""
    input: List[Tensor] = []
    forward: Any = None
    model: Any = {}

  start_pos = Variable("start_pos", 0, max_context).bind(0)
  sub_steps = [
    Step(name = "transformer", input = [Tensor([[tok]]), start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P], forward = model.forward, model=model),
    #Step(name = "q6k_to_f16", input = [Tensor.randn(430_080, dtype=dtypes.uint8)], forward = q6k_to_f16),
    Step(name = "q6k_to_f32", input = [Tensor.randn(430_080, dtype=dtypes.uint8)], forward = q6k_to_f32),
  ]
  quant_weight_shapes = ((2048, 2048), (2048, 512), (2048, 8192), (8192, 2048)) # these are shapes in gguf file, need to reverse
  sub_steps += [Step(name = f"q6k_to_int8_{s1}_{s0}", input = [Tensor.randn((s1 * s0 * 210) // 256, dtype=dtypes.uint8), (s1, s0)], forward = q6k_to_int8) for s0, s1 in quant_weight_shapes]

  # TODO: refactor to move some corrected CLANG rendering to export_model.py
  def compile_step(step: Step):
    run, special_names = jit_model(step, *step.input)
    functions, statements, bufs, bufs_to_save = compile_net(run, special_names)
    state = get_state_dict(step.model)
    weightbuf_to_name = {id(x.lazydata.base.realized): name for name, x in state.items()}
    # this omits saving the random seeds, which therefore will be set in client by default to 0,0 (2x uint32)
    bufs_to_save = {k:v for k,v in bufs.items() if v[2] in weightbuf_to_name}
    cprog = ["#include <tgmath.h>", "#include <stddef.h>"]

    # declare buffers that we'll load weights into from javascript
    # TODO: import the same type names used in each function declaration. Below mapping is not comprehensive, and may go out of date
    dtype_map = {dtypes.int: "int", dtypes.float: "float", dtypes.uchar: "unsigned char", dtypes.char: "signed char", dtypes.half: "__fp16", dtypes.uint: "unsigned int"}

    # buf_to_name must have unchanged order from hereafter. We rely on known ordering to map weights from JS
    buf_to_name = tuple((f"{name}", f"{weightbuf_to_name[data[2]]}") for name, data in bufs_to_save.items())
    if bufs_to_save:
      cprog.append(f"void* bufs[{len(buf_to_name)}];")
      cprog.append(f"""void set_buf(size_t index, void* ptr) {{\n  bufs[index] = ptr;\n}}""")

    # declare zero-filled intermediate buffers
    for name in set(bufs.keys()) - set(bufs_to_save.keys()) - set(special_names.values()):
      n_bytes, dtype, weightbuf_id = bufs[name]
      cprog += [f"{dtype_map[dtype]} {name}[{n_bytes // dtype.itemsize}];"]

    inputs = sorted([(name, bufs[name][1], True) for name in special_names.values() if "input" in name], key=lambda x: x[0].split("input")[1]) # (name, dtype, True)
    symbolic_vars = set()
    for i, (_, args, _, _) in enumerate(statements):
      for j, var in enumerate(args):
        if getattr(var, "op", None) is Ops.DEFINE_VAR and isinstance(getattr(var, "arg", None), tuple) and isinstance(var.arg[0], str):
          symbolic_vars.add(var)
          statements[i][1][j] = var.arg[0] # name assigned in Variable(name, ...), e.g. "start_pos"

    inputs += sorted([(var.arg[0], var.dtype, False) for var in symbolic_vars]) # (name, dtype, False)
    input_c_args = ", ".join(f"{dtype_map[dtype]}* {name}" if isArray else f"{dtype_map[dtype]} {name}" for name,dtype,isArray in inputs)
    outputs = sorted([name for name in special_names.values() if "output" in name], key=lambda x: x.split("output")[1])
    output_c_args = ", ".join([f'{dtype_map[bufs[output][1]]}* {output}' for output in outputs]) # TODO: always arrays only?
    cprog += list(functions.values())
    cprog += [f"void net({output_c_args}, {input_c_args}) {{"]
    # TODO: tell CLANG to assume specified ranges for symbolic vars, and for I/O buffer sizes?
    
    conv_map = {buf_name: i for i, (buf_name, weight_name) in enumerate(buf_to_name)}
    convert = lambda x: f"({dtype_map[bufs_to_save[x][1]]} *)bufs[{conv_map[x]}]" if x in bufs_to_save else x
    cprog += [f"  {name}({', '.join(map(convert, args))});" for (name, args, _global_size, _local_size) in statements] + ["}"]
    cprog = "\n".join(cprog)
    # TODO: make tinygrad output wasm-compatible absolute value of fp16
    # the rendered clang for absolute value of fp16 numbers, compiles/works fine with clang, but causes exceptions while instantiating wasm module in browser
    # for example: __fp16 alu23 = (alu10*(((_Bool)(alu10))?((alu10<((__fp16)(0.0f)))?((__fp16)(-1.0f)):((__fp16)(1.0f))):((__fp16)(0.0f))));
    pattern = re.compile(r'__fp16\s+(\w+)\s*=\s*\((\w+)\s*\*\s*\(\(\(_Bool\)\(\2\)\)\?\(\(\2<\(\(__fp16\)\(0\.0f\)\)\)\?\(\(__fp16\)\(-1\.0f\)\):\(\(__fp16\)\(1\.0f\)\)\):\(\(__fp16\)\(0\.0f\)\)\)\);')
    replacement = r'__fp16 \1 = (__fp16)fabsf((float)\2);'
    cprog = pattern.sub(replacement, cprog)

    with open(os.path.join(os.path.dirname(__file__), f"{step.name}.c"), "w") as text_file:
      text_file.write(cprog)

    input_ptrs = OrderedDict((f"inputPtr{name.split("input")[1]}", (name, bufs[name][0])) for name,_,isArray in inputs if isArray)
    output_ptrs = OrderedDict((f"outputPtr{name.split("output")[1]}", (name, bufs[name][0])) for name in outputs)
    top = f"import {step.name}Module from './{step.name}.js'\n"
    prg = f"""\nvar {step.name} = async function{f"(metadata, progress)" if bufs_to_save else "()"} {{

  const wasm = await {step.name}Module();
  {f"""const weightNames = [{", ".join([f"\"{weight_name}\"" for buf, weight_name in buf_to_name])}];
  for (const [i, name] of weightNames.entries()) {{
    const bufPtr = wasm._malloc(metadata[name].size);
    wasm.HEAPU8.set(metadata[name].bytes, bufPtr);
    metadata[name].bytes = null;
    wasm._set_buf(i, bufPtr);
  }}""" if bufs_to_save else ""}

  return {{
    run: ({",".join(name for name,_,_ in inputs)}) => {{
      {"\n      ".join(f"const {inputPtr} = wasm._malloc({n_bytes});" for inputPtr, (name, n_bytes) in input_ptrs.items())}
      {"\n      ".join(f"const {outputPtr} = wasm._malloc({n_bytes});" for outputPtr, (name, n_bytes) in output_ptrs.items())}
      {"\n      ".join(f"wasm.HEAPU8.set({name}, {inputPtr});" for inputPtr, (name, n_bytes) in input_ptrs.items())}
      wasm._net({", ".join(list(output_ptrs.keys()) + list(input_ptrs.keys()) + sorted([var.arg[0] for var in symbolic_vars]))});
      {"\n      ".join(f"const {name} = wasm.HEAPU8.slice({outputPtr}, {outputPtr} + {n_bytes});" for outputPtr, (name, n_bytes) in output_ptrs.items())}
      {"\n      ".join(f"wasm._free({ptr});" for ptr in list(output_ptrs.keys()) + list(input_ptrs.keys()))}
      return [{", ".join(f"{name}" for name, n_bytes in output_ptrs.values())}];
    }},
    wasm: wasm
  }}
}}\n"""

    return top, prg

  top, prg = "", ""

  for step in sub_steps:
    print(f'Executing step={step.name}')
    # TODO: jit_model crashes with BEAM=2 for decompression stuff, fix CLANG BEAM=2 bug
    context = Context(BEAM=2) if step.name == "transformer" else Context(BEAM=0)
    with context:
      step_top, step_prg = compile_step(step)
    top += step_top
    prg += step_prg

  prg += f"export {{{", ".join(step.name for step in sub_steps)}}};"
  with open(os.path.join(os.path.dirname(__file__), "net_clang.js"), "w") as text_file:
    text_file.write(top + prg)

  done = 1