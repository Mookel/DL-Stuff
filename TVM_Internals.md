# Table of contents

- [PrimFunc Build实现](#primfunc-build)
  - [基本flow](#flow)
  - [_build_for_device](#_build_for_device)
  - [codegen.build_module](#codegenbuild_module)
- [Relay Build Flow](#relay-build-flow)
  - [关键函数&Flow](#flow)
  - [关键抽象](#)
  - [简化用例（dnnl）](#dnnl)
- [Lower实现](#lower)
- [Tensorization](#tensorization)
  - [What's Tensorization](#whats-tensorization)
  - [How to Tensorize?](#how-to-tensorize)
  - [Implement Tensorization for a Specific Target](#implement-tensorization-for-a-specific-target)
- [Custom Accelarator Support](#custom-accelarator-support)
  - [相关DISCUSS](#discuss)
  - [有哪些集成方式？](#)
- [Runtime相关](#runtime)
  - [Cuda ModuleNode创建过程](#cuda-modulenode)
  - [tvm_call_packed是什么？](#tvm_call_packed)
  - [Host Module调用Device Module里面kernel函数的机制是怎样的？](#host-moduledevice-modulekernel)
  - [Host和Device的数据交互?](#hostdevice)

# PrimFunc Build实现

## 基本flow

tvm的low-level build flow，如下所示：

![lower build flow](C:\Users\tianping\OneDrive - Xilinx, Inc\Work-2021\TVM\VTA\lower build flow.png)

总流程比较简单，先lower把schedule lower成PrimFunc；然后再编译（支持异构编译）成不同的运行时Module，最后打包一个Module；

> 引用自：https://tvm.apache.org/docs/dev/codebase_walkthrough.html
>
> `tvm.build()`, defined in `python/tvm/driver/build_module.py`, takes a schedule, input and output `Tensor`, and a target, and returns a [`tvm.runtime.Module`](https://tvm.apache.org/docs/api/python/runtime.html#tvm.runtime.Module) object. A [`tvm.runtime.Module`](https://tvm.apache.org/docs/api/python/runtime.html#tvm.runtime.Module) object contains a compiled function which can be invoked with function call syntax.
>
> The process of `tvm.build()` can be divided into two steps:
>
> - Lowering, where a high level, initial loop nest structures are transformed into a final, low level IR
> - Code generation, where target machine code is generated from the low level IR
>
> Lowering is done by `tvm.lower()` function, defined in `python/tvm/build_module.py`. First, bound inference is performed, and an initial loop nest structure is created.

## _build_for_device

_build_for_device负责：

- 把IR module partition成dev ir mod和host ir mod，并且分别要经过一些特定的pass；（partition之后dev_mod可能不包含任何PrimFunc）;
- 为dev ir mod生成device-specific code和打包成rt dev mod(Module)，即运行时module，并且返回host ir module；

在_build_for_device接口中完成calling_convention属性标注的pass是tvm.tir.transform.SplitHostDevice()；

```python
    opt_mixed += [
        tvm.tir.transform.ThreadSync("shared"),
        tvm.tir.transform.ThreadSync("warp"),
        tvm.tir.transform.InferFragment(),
        tvm.tir.transform.LowerThreadAllreduce(),
        tvm.tir.transform.MakePackedAPI(),
        # 整个pass会去标识calling_conv属性值；
        tvm.tir.transform.SplitHostDevice(),
    ]
```

这个pass的具体实现在（tvm/src/tir/transforms/split_host_device.cc)；其会把PrimFunc中的stmtNode存在属性（attr::thread_extend || attr::pipeline_exec_scope || attr::device_scope）时split成2个func,并且存入opt_mod中：

- 一个external device function symbol，需要在device上实现或者编译；后续会被过滤到dev ir module中；
- 一个负责调用上述external device 函数的PrimFunc；后续会被过滤到host ir module中；

```c++
class HostDeviceSplitter : public StmtMutator {
 public:
  explicit HostDeviceSplitter(IRModule* device_mod, Target device_target, std::string name_prefix)
      : device_mod_(device_mod), device_target_(device_target), name_prefix_(name_prefix) {}
...

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    /// 具有如下属性的会split device function？
    if (op->attr_key == attr::thread_extent || op->attr_key == attr::pipeline_exec_scope ||
        op->attr_key == attr::device_scope) {
      return SplitDeviceFunc(GetRef<Stmt>(op));
    }
    return StmtMutator::VisitStmt_(op);
  }
...

Stmt SplitDeviceFunc(Stmt body) {
    ...
    PrimFunc device_func(params, Substitute(body, remap_vars));
    device_func = WithAttr(std::move(device_func), tir::attr::kDeviceThreadAxis, m.thread_axis_);
    /// 注：会被标注CallingConv::kDeviceKernelLaunch属性
    device_func = WithAttr(std::move(device_func), tvm::attr::kCallingConv,
                           Integer(CallingConv::kDeviceKernelLaunch));
    device_func =
        WithAttr(std::move(device_func), tvm::attr::kGlobalSymbol, runtime::String(kernel_symbol));
    device_func = WithAttr(std::move(device_func), tir::attr::kNoAlias, Integer(1));
    device_func = WithAttr(std::move(device_func), tvm::attr::kTarget, device_target_);
    (*device_mod_)->Add(GlobalVar(kernel_symbol), device_func);

    // generate calls to the device function
    Array<PrimExpr> call_args;
    call_args.push_back(StringImm(kernel_symbol));
    for (PrimExpr arg : arguments) {
      call_args.push_back(arg);
    }
    for (PrimExpr ext : m.thread_extent_) {
      call_args.push_back(ext);
    }
    return Evaluate(Call(DataType::Int(32), builtin::tvm_call_packed(), call_args));
  }
```

完成calling convention的属性标注后，会被后续的pass过滤成dev ir module和host ir module，而dev ir module会被进一步被codegen.build_module编译打包成dev runtime Module。而host ir module不会在_build_for_device中被编译打包，而时返回后在上层统一调用codegen.build_module打包成rt host Module。

## codegen.build_module

codegen.build_module调用的是_ffi_api中的Build接口：

```python
def build_module(mod, target):
    """Build IRModule into Module.

    Parameters
    ----------
    mod : tvm.IRModule
        The ir module.

    target : str
        The target module type.

    Returns
    -------
    module : runtime.Module
        The corressponding module.
    """
    target = Target(target) if isinstance(target, str) else target
    return _ffi_api.Build(mod, target)
```

而_ffi_api.Build(mod, target) 会调用C++里面的实现：

```c++
runtime::Module Build(IRModule mod, Target target) {
  if (transform::PassContext::Current()
          ->GetConfig<Bool>("tir.disable_assert", Bool(false))
          .value()) {
    mod = tir::transform::SkipAssert()(mod);
  }

  // the build function.
  std::string build_f_name = "target.build." + target->kind->name;
  const PackedFunc* bf = runtime::Registry::Get(build_f_name);
  ICHECK(bf != nullptr) << build_f_name << " is not enabled";
  return (*bf)(mod, target);
}
```

在这个接口里或获取对应target的build函数(PackedFunc类型)，然后利用该函数进行实际的build过程去生成具体的运行时Module；

# Relay Build Flow

## 关键函数&Flow

### `python relay.build`

```python
def build(ir_mod, target=None, target_host=None, params=None, mod_name="default"):
    # fmt: off
    # pylint: disable=line-too-long
    """Helper function that builds a Relay function to run on TVM graph executor.

    Parameters
    ----------
    ir_mod : :py:class:`~tvm.IRModule`
        The IR module to build. Using relay.Function is deprecated.

    target : str, :any:`tvm.target.Target`, or dict of str(i.e. device/context name) to str/tvm.target.Target, optional
        For heterogeneous compilation, it is a dictionary indicating context to
        target mapping. For homogeneous compilation, it is a build target.

    target_host : str or :any:`tvm.target.Target`, optional
        Host compilation target, if target is device.
        When TVM compiles device specific program such as CUDA,
        we also need host(CPU) side code to interact with the driver
        setup the dimensions and parameters correctly.
        target_host is used to specify the host side codegen target.
        By default, llvm is used if it is enabled,
        otherwise a stackvm intepreter is used.

    params : dict of str to NDArray
        Input parameters to the graph that do not change
        during inference time. Used for constant folding.

    mod_name: Optional[str]
        The module name we will build

    Returns
    -------
    factory_module : tvm.relay.backend.executor_factory.ExecutorFactoryModule
            The runtime factory for the TVM graph executor.
    """
    # pylint: enable=line-too-long
    # fmt: on
    if not isinstance(ir_mod, (IRModule, _function.Function)):
        raise ValueError("Type of input parameter mod must be tvm.IRModule")

    if isinstance(ir_mod, _function.Function):
        if params:
            ir_mod = bind_params_by_name(ir_mod, params)
        ir_mod = IRModule.from_expr(ir_mod)
        warnings.warn(
            "Please use input parameter mod (tvm.IRModule) "
            "instead of deprecated parameter mod (tvm.relay.function.Function)",
            DeprecationWarning,
        )
    target = _update_target(target)
    if isinstance(target_host, (str, Target)):
        target_host = Target(target_host)
    elif target_host:
        raise ValueError("target host must be the type of str, " + "tvm.target.Target, or None")

    target, target_host = Target.check_and_update_host_consist(
        target, target_host, target_is_dict_key=False
    )

    # Retrieve the executor from the target
    executor = get_executor_from_target(target, target_host)

    # If current dispatch context is fallback context (the default root context),
    # then load pre-tuned parameters from TopHub
    if isinstance(autotvm.DispatchContext.current, autotvm.FallbackContext):
        tophub_context = autotvm.tophub.context(list(target.values()))
    else:
        tophub_context = autotvm.utils.EmptyContext()

    with tophub_context:
        bld_mod = BuildModule()
        # python buildModule() --> c++ RelayBuildModule
        # bld_mod.build = c++ RelayBuildModule.build
        executor_config, runtime_mod, params = bld_mod.build(
            mod=ir_mod, target=target, params=params, executor=executor
        )
        #print("exector_config: ", executor_config)
        #print("runtime_mod:", runtime_mod)
        func_metadata = bld_mod.get_function_metadata()
        #print("func_metadata:", func_metadata)

        if executor == "aot":
            executor_factory = _executor_factory.AOTExecutorFactoryModule(
                ir_mod, target, runtime_mod, mod_name, params, func_metadata
            )
        elif executor == "graph":
            executor_factory = _executor_factory.GraphExecutorFactoryModule(
                ir_mod, target, executor_config, runtime_mod, mod_name, params, func_metadata
            )
        else:
            assert False, "Executor " + executor + " not supported"

        return executor_factory
```

该函数会调用底层的`RelayBuildModule`来完成实际的build工作，RelayBuildModule返回值被`BuildOutput`封装：

```c++
/*!
 * \brief Output of building module
 */
struct BuildOutput {
  std::string graph_json;
  runtime::Module mod;
  std::unordered_map<std::string, tvm::runtime::NDArray> params;
};
```

- executor_config: 如果是用的是graph excutor codegen，则是graph json，否则是None；
- metadata runtime module
- params

python侧会用RelayBuildModule的get_function机制来查询结果中的特定信息：

```python
class BuildModule(object):
    """Build an IR module to run on TVM graph executor. This class is used
    to expose the `RelayBuildModule` APIs implemented in C++.
    """

    def __init__(self):
        self.mod = _build_module._BuildModule()
        self._get_graph_json = self.mod["get_graph_json"]
        self._get_module = self.mod["get_module"]
        ...
        self._set_params_func = self.mod["set_params"]
        self._get_params_func = self.mod["get_params"]
        self._get_function_metadata = self.mod["get_function_metadata"]
```

### `RelayBuildModule: Build`

该函数会利用BuildRelay完成所有的编译工作；

```c++
  void Build(IRModule mod, const TargetsMap& targets, const tvm::Target& target_host,
             const String executor) {
    // Create protected variable targets_ from ground up
    targets_ = targets;
    target_host_ = target_host;
    executor_ = executor;
    CheckAndUpdateHostConsistency(&targets_, &target_host_);
    BuildRelay(mod, params_);
    // Clear compile engine so that tuning schedules can be changed between runs. See issue #6096.
    CompileEngine::Global()->Clear();
  }
```

### `RelayBuildModule: BuildRelay`

完成真正的Relay Function的编译工作：

```c++
void BuildRelay(IRModule relay_module,
                  const std::unordered_map<std::string, tvm::runtime::NDArray>& params) {
    Target target_host = GetTargetHost();
    // If no target_host has been set, we choose a default one, which is
    // llvm if "codegen.LLVMModuleCreate" is accessible.
    const runtime::PackedFunc* pf = runtime::Registry::Get("codegen.LLVMModuleCreate");
    if (!target_host.defined()) target_host = (pf != nullptr) ? Target("llvm") : Target("stackvm");

    // Update all the targets in the targets_ TargetsMap
    CheckAndUpdateHostConsistency(&targets_, &target_host);

    // Relay IRModule -> IRModule optimizations.
    relay_module = Optimize(relay_module, targets_, params);

    // Get the updated function.
    auto func = Downcast<Function>(relay_module->Lookup("main"));

    // Generate code for the updated function.
    executor_codegen_ = MakeExecutorCodegen(executor_);
    executor_codegen_->Init(nullptr, targets_);
    executor_codegen_->Codegen(func);
    executor_codegen_->UpdateOutput(&ret_);
    ret_.params = executor_codegen_->GetParams();

    auto lowered_funcs = executor_codegen_->GetIRModule();

    // Generate a placeholder function that attaches linked params as its arguments.
    if (target_host->GetAttr<Bool>("link-params").value_or(Bool(false))) {
      CHECK(pf != nullptr) << "Unable to link-params with no target_host and no llvm codegen.";
      auto param_ids = executor_codegen_->GetParamIds();
      auto link_params = Map<String, tir::LinkedParam>();
      for (auto param : ret_.params) {
        link_params.Set(param.first, tir::LinkedParam(param_ids[param.first], param.second));
      }

      Map<String, ObjectRef> dict;
      dict.Set(tvm::tir::attr::kLinkedParams, link_params);
      dict.Set(tvm::attr::kGlobalSymbol, String(::tvm::runtime::symbol::tvm_lookup_linked_param));
      DictAttrs attrs{dict};
      auto prim = tir::PrimFunc(Array<tir::Var>(), tir::SeqStmt(Array<tir::Stmt>()), VoidType(),
                                Map<tir::Var, tir::Buffer>(), attrs);
      if (lowered_funcs.find(target_host->str()) == lowered_funcs.end()) {
        lowered_funcs.Set(target_host->str(), IRModule(Map<GlobalVar, BaseFunc>({})));
      }
      lowered_funcs[target_host->str()]->Add(
          GlobalVar(::tvm::runtime::symbol::tvm_lookup_linked_param), prim);
    }

    // When there is no lowered_funcs due to reasons such as optimization.
    if (lowered_funcs.size() == 0) {
      if (target_host.defined() && target_host->kind->name == "llvm") {
        // If we can decide the target is LLVM, we then create an empty LLVM module.
        ret_.mod = (*pf)(target_host->str(), "empty_module");
      } else {
        // If we cannot decide the target is LLVM, we create an empty CSourceModule.
        // The code content is initialized with ";" to prevent complaining
        // from CSourceModuleNode::SaveToFile.
        ret_.mod = tvm::codegen::CSourceModuleCreate(";", "", Array<String>{});
      }
    } else {
      ret_.mod = tvm::build(lowered_funcs, target_host_);
    }

    auto ext_mods = executor_codegen_->GetExternalModules();
    ret_.mod = tvm::codegen::CreateMetadataModule(ret_.params, ret_.mod, ext_mods, GetTargetHost(),
                                                  executor_codegen_->GetMetadata());
    // Remove external params which were stored in metadata module.
    for (tvm::runtime::Module mod : ext_mods) {
      auto pf_var = mod.GetFunction("get_const_vars");
      if (pf_var != nullptr) {
        Array<String> variables = pf_var();
        for (size_t i = 0; i < variables.size(); i++) {
          auto it = ret_.params.find(variables[i].operator std::string());
          if (it != ret_.params.end()) {
            ret_.params.erase(it);
          }
        }
      }
    }
  }
```

总结起来，其步骤主要为：

- 执行IRModule优化，主要侧重于高层Relay level级别的优化（后续我会深入分析）
- 通过MakeExecutorCodegen根据executor的不同创建不同的high-level codegen引擎，executor主要有AOT和Graph方式；
- 基于创建的codegen引擎完成具体的high-level编译，codegen后会提供如下信息供当前build module查询和后续使用：
  - params
  - lower functions(target -> PrimFunc)
  - external runtime moudles
  - meta data
  - graph json string: 如果是graph executor codegen，返回具体的json string，否则为“”
- 设置ret_.params;
- 设置ret_.graph_json;
- （如果lower_func存在则）调用tvm::build完成low-level PrimFunc的编译，返回host runtime module；
- 把相关参数集合在一起，然后创建MetaData Module，涉及到的参数有：
  - params
  - host runtime module
  - external runtime modules
  - target host
  - meta data
- 从meta module中移除在external module中存在的constants；
- 设置ret_.mod, 此时需要return的所有数据已经ready了；

> 对于lower function的编译时利用`tvm::build`函数，这个和python版本差不了太多，唯一差别是不需要lower操作，其他流程基本一样，会把lower function编译生成不同的runtime module并且统一打包一个返回：
>
> ```c++
> // Build for heterogeneous execution.
> runtime::Module build(const Map<Target, IRModule>& inputs_arg, const Target& target_host_arg) {
>   ...
> }
> ```
>
> 

### `GraphExecutorFactoryModule`

```python
class GraphExecutorFactoryModule(ExecutorFactoryModule):
    """Graph executor factory module.
    This is a module of graph executor factory

    Attributes
    ----------
    graph_json_str : the json graph to be deployed in json format output by graph compiler.
        The graph can contain operator(tvm_op) that points to the name of
        PackedFunc in the libmod.
    target : tvm.Target
        The Target used to build this module.
    libmod : tvm.Module
        The module of the corresponding function
    libmod_name: str
        The name of module
    params : dict of str to NDArray
        The parameters of module
    function_metadata : Map of String to FunctionInfo
        This holds a map function names to their information
    """

    def __init__(
        self, ir_mod, target, graph_json_str, libmod, libmod_name, params, function_metadata
    ):
        assert isinstance(graph_json_str, string_types)
        fcreate = get_global_func("tvm.graph_executor_factory.create")
        args = []
        for k, v in params.items():
            args.append(k)
            args.append(ndarray.array(v))

        self.ir_mod = ir_mod
        self.target = target
        self.module = fcreate(graph_json_str, libmod, libmod_name, *args)
        self.graph_json = graph_json_str
        self.lib = libmod
        self.libmod_name = libmod_name
        self.params = params
        self.iter_cnt = 0
        self.function_metadata = function_metadata
```

通过调用如下的接口返回真正的runtime module；

- `tvm.graph_executor_factory.create`

```c++
TVM_REGISTER_GLOBAL("tvm.graph_executor_factory.create")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      ICHECK_GE(args.num_args, 3) << "The expected number of arguments for "
                                     "graph_executor_factory.create needs at least 3, "
                                     "but it has "
                                  << args.num_args;
      // The argument order is graph_json, module, module_name, param0_name, param0_tensor,
      // [param1_name, param1_tensor], ...
      ICHECK_EQ((args.size() - 3) % 2, 0);
      std::unordered_map<std::string, tvm::runtime::NDArray> params;
      for (size_t i = 3; i < static_cast<size_t>(args.size()); i += 2) {
        std::string name = args[i].operator String();
        params[name] = args[i + 1].operator tvm::runtime::NDArray();
      }
      auto exec = make_object<GraphExecutorFactory>(args[0], params, args[2]);
      exec->Import(args[1]);
      *rv = Module(exec);
    });
```

## 关键抽象

### `GraphCodegen`

`GraphCodegen`主要负责Relay Function的编译，输出结果主要有：

- lower functions(target -> PrimFunc)
- constant params
- meta data
- external runtime modules

对GraphCodegen实现有2个间接层，GraphCodeGen实际上真正的实现链条：

```c++
         GraphCodeGen    -->   GraphExecutorCodegenModule   -->   GraphExecutorCodegen
   (对外使用固定wrapper接口)     （Get Function by string key）          （真正实现）
```

GraphExecutorCodegenModule负责提供getFunction接口，主要有：

- init
- codegen
- get_graph_json
- list_params_name
- get_param_by_name
- get_param_id
- get_irmodule
- get_external_modules
- get_metadata
- get_function_metadata

这些function的所有实现都要依赖底层真正的实现者`GraphExecutorCodegen`；

GraphExecutorCodegen的Codegen实现如下：

```c++
class GraphExecutorCodegen : public backend::MemoizedExprTranslator<std::vector<GraphNodeRef>> {
 public:
  GraphExecutorCodegen(runtime::Module* mod, const TargetsMap& targets) : mod_(mod) {
    compile_engine_ = CompileEngine::Global();
    targets_ = targets;
  }
  ...
      
  LoweredOutput Codegen(relay::Function func) {
    auto pf = GetPackedFunc("relay.backend.GraphPlanMemory");
    storage_device_map_ = (*pf)(func);
    UpdateMainWorkspaceSize(func);
    // First we convert all the parameters into input nodes.
    for (auto param : func->params) {
      auto node_ptr = GraphInputNode::make_node_ptr(param->name_hint(), GraphAttrs());
      var_map_[param.get()] = AddNode(node_ptr, param);
    }
    heads_ = VisitExpr(func->body);
    std::ostringstream os;
    dmlc::JSONWriter writer(&os);
    GetJSON(&writer);
    LoweredOutput ret;
    ret.graph_json = os.str();
    ret.params = std::unordered_map<std::string, std::pair<int, const tvm::runtime::NDArray>>();
    for (auto param : params_) {
      ret.params.emplace(std::make_pair(
          param.first,
          std::make_pair(static_cast<int>(param_storage_ids_[param.first]), param.second)));
    }

    for (auto& kv : lowered_funcs_) {
      if (ret.lowered_funcs.count(kv.first) == 0) {
        ret.lowered_funcs.Set(kv.first, IRModule(Map<GlobalVar, BaseFunc>({})));
      }
      auto& mod = ret.lowered_funcs[kv.first];
      mod->Update(kv.second);
      ret.lowered_funcs.Set(kv.first, mod);
    }
    ret.external_mods = compile_engine_->LowerExternalFunctions();
    ret.function_metadata = std::move(function_metadata_);
    return ret;
  }
```

> 注：送到codegen接口的Function应该都是要经过如下pass而partition之后的func，以dnnl为例：
>
> - transform.MergeComposite(dnnl_patterns);
> - transform.AnnotateTarget("dnnl");
> - transform.PartitionGraph();
> - ...

#### codegen流程

TODO



### `CompileEngine`



### `MetadataModule`



### `GraphExecutorFactory`



### `GraphExecutor`



## 简化用例（dnnl）

### 完整用例代码



### 深入剖析



# Lower实现



# Tensorization

## What's Tensorization



## How to Tensorize?

针对tensorization的独立介绍的tutorial:https://tvm.apache.org/docs/tutorials/language/tensorize.html#sphx-glr-tutorials-language-tensorize-py 



## Implement Tensorization for a Specific Target



# Custom Accelarator Support

## 相关DISCUSS

- https://discuss.tvm.apache.org/t/which-is-the-best-way-to-port-tvm-to-a-new-ai-accelerator/6905

> Now, I think there are four ways to port tvm to a new AI accelerator.
>
> 1 ： BYOC, BYOC can offload the ops to your new device which your new device support. BYOC is simple and graceful, But we can’t use AutoTVM in BYOC. I think AutoTVM is the very import feature of TVM.
>
> 2 : Tensorize, By using TVM’s schedule primitive Tensorize, we can replace a unit of computation with the corresponding intrinsic, such as GEMM instruction. We can use AutoTVM in this way, but we may need to use tensorize to modify every ops’s schedule.
>
> 3 : like cuDNN, we can use tvm to call new device like use cuDNN to call GPU. this way is not better than BYOC;
>
> 4 : like GPU/CPU, we can add a new target in tvm like GPU/CPU, we need develop compute and schedule for every op, we also need to develop graph optimize for this new device. we can use AutoTVM in this way. But this way is the most time-consuming and the most difficult;
>
> I think if we only have the op level’s api of new device, BYOC is the best way.
>
> If we have ISA level’s interface of new device, which way is the best?

- https://discuss.tvm.apache.org/t/feedback-on-tvm-port-to-custom-accelerator/9548

## 有哪些集成方式？

我理解目前tvm支持第三方的accelarator有几种方式：

1. BYOC based（待验证）:基于Relay层面的BYOC框架来添加其他framework/target的支持；这种偏向于上层算子，例如框架本身就支持独立的conv，pooling甚至是整个graph/subgraph的支持，类似的框架如TensorRT，dnnl，vitis-ai等等，这种方式最大的坏处是没有办法利用tvm本身te/tir的优化能力，需要由第三方提供Relay operator到其自身硬件指令的完整映射和优化流程；
2. target Based: 在target level增加新的target支持，这个偏向于原子指令的支持，因为在这一层是PrimFunc TIR到硬件指令的映射，更加偏向于底层；现有的如GPU, OPENCL都是用这种方式进行支持：当然，有些accelator由于有粗粒度指令，对其的支持也在采用这种方法基础上引入了tensorization机制，例如：https://discuss.tvm.apache.org/t/feedback-on-tvm-port-to-custom-accelerator/9548 里面提到的。这种方式的最大好处是可以利用TE/TIR的优化；但是坏处是有时候tensorization可能不是那么方便；
3. Pure Intrinsics or tensorization Based：类似于target level，但是我理解需要JIT或者extern runtime library的支持，现有的VTA采用的是这种方式；
4. External Library Based: 类似于Cudnn的支持，直接把operator映射到外部device的library API;

# Runtime相关

## Cuda ModuleNode创建过程

```c++
runtime::Module BuildCUDA(IRModule mod, Target target) {
  using tvm::runtime::Registry;
  bool output_ssa = false;
  /// Cuda的code gen模块；
  CodeGenCUDA cg;
  cg.Init(output_ssa);

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>()) << "CodeGenCUDA: Can only take PrimFunc";
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
        << "CodeGenCUDA: expect calling_conv equals CallingConv::kDeviceKernelLaunch";
    /// 将primFunc放入code gen；
    cg.AddFunction(f);
  }
  
  /// 完成code gen返回string形式的code；
  std::string code = cg.Finish();

  if (const auto* f = Registry::Get("tvm_callback_cuda_postproc")) {
    code = (*f)(code).operator std::string();
  }
  std::string fmt = "ptx";
  std::string ptx;
  if (const auto* f = Registry::Get("tvm_callback_cuda_compile")) {
    ptx = (*f)(code).operator std::string();
    // Dirty matching to check PTX vs cubin.
    // TODO(tqchen) more reliable checks
    if (ptx[0] != '/') fmt = "cubin";
  } else {
    ptx = NVRTCCompile(code, cg.need_include_path());
  }
  /// 创建cuda module node
  return CUDAModuleCreate(ptx, fmt, ExtractFuncInfo(mod), code);
}

TVM_REGISTER_GLOBAL("target.build.cuda").set_body_typed(BuildCUDA);
```

## tvm_call_packed是什么？

是一个built-in op，一般与Call配套使用，负责PackFunc的调用（例如host端调用device端的kernel function）；

## Host Module调用Device Module里面kernel函数的机制是怎样的？

利用tvm_call_packed机制实现，并且要遵循相应的PrimFunc calling convention标准，否则会出错；

## Host和Device的数据交互?

核心在于tvm.nd.array:

```python
def array(arr, device=cpu(0)):
   #...

    if not isinstance(arr, (np.ndarray, NDArray)):
        arr = np.array(arr)
    return empty(arr.shape, arr.dtype, device).copyfrom(arr)
```

这个·empty(arr.shape, arr.dtype, device).copyfrom(arr)·里面会涉及到DeviceAPI的调用，分两步：

- 在目标设备上分配空间：

```c++
NDArray NDArray::Empty(ShapeTuple shape, DLDataType dtype, Device dev, Optional<String> mem_scope) {
  NDArray ret = Internal::Create(shape, dtype, dev);
  ret.get_mutable()->dl_tensor.data =
      DeviceAPI::Get(ret->device)
          ->AllocDataSpace(ret->device, shape.size(), shape.data(), ret->dtype, mem_scope);
  return ret;
}
```

- 完成数据拷贝：

```c++
void ArrayCopyFromBytes(DLTensor* handle, const void* data, size_t nbytes) {
  size_t arr_size = GetDataSize(*handle);
  ICHECK_EQ(arr_size, nbytes) << "ArrayCopyFromBytes: size mismatch";
  ICHECK(IsContiguous(*handle)) << "ArrayCopyFromBytes only support contiguous array for now";

  DLTensor from;
  from.data = const_cast<void*>(data);
  from.device = Device{kDLCPU, 0};
  from.ndim = handle->ndim;
  from.dtype = handle->dtype;
  from.shape = handle->shape;
  from.strides = nullptr;
  from.byte_offset = 0;
  DeviceAPI::Get(handle->device)->CopyDataFromTo(&from, handle, nullptr);
  // Synchronize in case data become unavailable later.
  DeviceAPI::Get(handle->device)->StreamSync(handle->device, nullptr);
}
```
