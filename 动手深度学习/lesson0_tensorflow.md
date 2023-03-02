```python
import numpy as np
import tensorflow as tf
```

# 简单实现


```python
w = tf.Variable(0,dtype=tf.float32)
optimizer = tf.keras.optimizers.Adam(0.1)

def train_step():
    with tf.GradientTape as tape:
        cost = w ** 2 -10 * w + 25
    trainable_variables = [w]
    grads = tape.gradient(cost,trainable_variables)
    optimizer.apply_gradients(zip(grads,trainable_variables))
    
print(w)
```

    2022-10-05 15:08:44.245148: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2022-10-05 15:08:44.245267: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties: 
    pciBusID: 0000:01:00.0 name: NVIDIA GeForce RTX 3060 Laptop GPU computeCapability: 8.6
    coreClock: 1.702GHz coreCount: 30 deviceMemorySize: 5.81GiB deviceMemoryBandwidth: 312.97GiB/s
    2022-10-05 15:08:44.245297: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
    2022-10-05 15:08:44.245323: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
    2022-10-05 15:08:44.245331: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcufft.so.10
    2022-10-05 15:08:44.245338: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcurand.so.10
    2022-10-05 15:08:44.245346: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusolver.so.10
    2022-10-05 15:08:44.248313: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusparse.so.10
    2022-10-05 15:08:44.251059: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
    2022-10-05 15:08:44.251157: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2022-10-05 15:08:44.251265: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2022-10-05 15:08:44.251312: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1858] Adding visible gpu devices: 0



    ---------------------------------------------------------------------------

    InternalError                             Traceback (most recent call last)

    Cell In [5], line 1
    ----> 1 w = tf.Variable(0,dtype=tf.float32)
          2 optimizer = tf.keras.optimizers.Adam(0.1)
          4 def train_step():


    File /opt/miniconda3/envs/tf23/lib/python3.8/site-packages/tensorflow/python/ops/variables.py:262, in VariableMetaclass.__call__(cls, *args, **kwargs)
        260   return cls._variable_v1_call(*args, **kwargs)
        261 elif cls is Variable:
    --> 262   return cls._variable_v2_call(*args, **kwargs)
        263 else:
        264   return super(VariableMetaclass, cls).__call__(*args, **kwargs)


    File /opt/miniconda3/envs/tf23/lib/python3.8/site-packages/tensorflow/python/ops/variables.py:244, in VariableMetaclass._variable_v2_call(cls, initial_value, trainable, validate_shape, caching_device, name, variable_def, dtype, import_scope, constraint, synchronization, aggregation, shape)
        242 if aggregation is None:
        243   aggregation = VariableAggregation.NONE
    --> 244 return previous_getter(
        245     initial_value=initial_value,
        246     trainable=trainable,
        247     validate_shape=validate_shape,
        248     caching_device=caching_device,
        249     name=name,
        250     variable_def=variable_def,
        251     dtype=dtype,
        252     import_scope=import_scope,
        253     constraint=constraint,
        254     synchronization=synchronization,
        255     aggregation=aggregation,
        256     shape=shape)


    File /opt/miniconda3/envs/tf23/lib/python3.8/site-packages/tensorflow/python/ops/variables.py:237, in VariableMetaclass._variable_v2_call.<locals>.<lambda>(**kws)
        223 def _variable_v2_call(cls,
        224                       initial_value=None,
        225                       trainable=None,
       (...)
        234                       aggregation=VariableAggregation.NONE,
        235                       shape=None):
        236   """Call on Variable class. Useful to force the signature."""
    --> 237   previous_getter = lambda **kws: default_variable_creator_v2(None, **kws)
        238   for _, getter in ops.get_default_graph()._variable_creator_stack:  # pylint: disable=protected-access
        239     previous_getter = _make_getter(getter, previous_getter)


    File /opt/miniconda3/envs/tf23/lib/python3.8/site-packages/tensorflow/python/ops/variable_scope.py:2633, in default_variable_creator_v2(next_creator, **kwargs)
       2630 aggregation = kwargs.get("aggregation", None)
       2631 shape = kwargs.get("shape", None)
    -> 2633 return resource_variable_ops.ResourceVariable(
       2634     initial_value=initial_value,
       2635     trainable=trainable,
       2636     validate_shape=validate_shape,
       2637     caching_device=caching_device,
       2638     name=name,
       2639     dtype=dtype,
       2640     constraint=constraint,
       2641     variable_def=variable_def,
       2642     import_scope=import_scope,
       2643     distribute_strategy=distribute_strategy,
       2644     synchronization=synchronization,
       2645     aggregation=aggregation,
       2646     shape=shape)


    File /opt/miniconda3/envs/tf23/lib/python3.8/site-packages/tensorflow/python/ops/variables.py:264, in VariableMetaclass.__call__(cls, *args, **kwargs)
        262   return cls._variable_v2_call(*args, **kwargs)
        263 else:
    --> 264   return super(VariableMetaclass, cls).__call__(*args, **kwargs)


    File /opt/miniconda3/envs/tf23/lib/python3.8/site-packages/tensorflow/python/ops/resource_variable_ops.py:1507, in ResourceVariable.__init__(self, initial_value, trainable, collections, validate_shape, caching_device, name, dtype, variable_def, import_scope, constraint, distribute_strategy, synchronization, aggregation, shape)
       1505   self._init_from_proto(variable_def, import_scope=import_scope)
       1506 else:
    -> 1507   self._init_from_args(
       1508       initial_value=initial_value,
       1509       trainable=trainable,
       1510       collections=collections,
       1511       caching_device=caching_device,
       1512       name=name,
       1513       dtype=dtype,
       1514       constraint=constraint,
       1515       synchronization=synchronization,
       1516       aggregation=aggregation,
       1517       shape=shape,
       1518       distribute_strategy=distribute_strategy)


    File /opt/miniconda3/envs/tf23/lib/python3.8/site-packages/tensorflow/python/ops/resource_variable_ops.py:1650, in ResourceVariable._init_from_args(self, initial_value, trainable, collections, caching_device, name, dtype, constraint, synchronization, aggregation, distribute_strategy, shape)
       1648 with ops.get_default_graph()._attr_scope({"_class": attr}):
       1649   with ops.name_scope("Initializer"), device_context_manager(None):
    -> 1650     initial_value = ops.convert_to_tensor(
       1651         initial_value() if init_from_fn else initial_value,
       1652         name="initial_value", dtype=dtype)
       1653   if shape is not None:
       1654     if not initial_value.shape.is_compatible_with(shape):


    File /opt/miniconda3/envs/tf23/lib/python3.8/site-packages/tensorflow/python/framework/ops.py:1499, in convert_to_tensor(value, dtype, name, as_ref, preferred_dtype, dtype_hint, ctx, accepted_result_types)
       1494       raise TypeError("convert_to_tensor did not convert to "
       1495                       "the preferred dtype: %s vs %s " %
       1496                       (ret.dtype.base_dtype, preferred_dtype.base_dtype))
       1498 if ret is None:
    -> 1499   ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
       1501 if ret is NotImplemented:
       1502   continue


    File /opt/miniconda3/envs/tf23/lib/python3.8/site-packages/tensorflow/python/framework/tensor_conversion_registry.py:52, in _default_conversion_function(***failed resolving arguments***)
         50 def _default_conversion_function(value, dtype, name, as_ref):
         51   del as_ref  # Unused.
    ---> 52   return constant_op.constant(value, dtype, name=name)


    File /opt/miniconda3/envs/tf23/lib/python3.8/site-packages/tensorflow/python/framework/constant_op.py:263, in constant(value, dtype, shape, name)
        166 @tf_export("constant", v1=[])
        167 def constant(value, dtype=None, shape=None, name="Const"):
        168   """Creates a constant tensor from a tensor-like object.
        169 
        170   Note: All eager `tf.Tensor` values are immutable (in contrast to
       (...)
        261     ValueError: if called on a symbolic tensor.
        262   """
    --> 263   return _constant_impl(value, dtype, shape, name, verify_shape=False,
        264                         allow_broadcast=True)


    File /opt/miniconda3/envs/tf23/lib/python3.8/site-packages/tensorflow/python/framework/constant_op.py:275, in _constant_impl(value, dtype, shape, name, verify_shape, allow_broadcast)
        273     with trace.Trace("tf.constant"):
        274       return _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
    --> 275   return _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
        277 g = ops.get_default_graph()
        278 tensor_value = attr_value_pb2.AttrValue()


    File /opt/miniconda3/envs/tf23/lib/python3.8/site-packages/tensorflow/python/framework/constant_op.py:300, in _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
        298 def _constant_eager_impl(ctx, value, dtype, shape, verify_shape):
        299   """Implementation of eager constant."""
    --> 300   t = convert_to_eager_tensor(value, ctx, dtype)
        301   if shape is None:
        302     return t


    File /opt/miniconda3/envs/tf23/lib/python3.8/site-packages/tensorflow/python/framework/constant_op.py:97, in convert_to_eager_tensor(value, ctx, dtype)
         95   except AttributeError:
         96     dtype = dtypes.as_dtype(dtype).as_datatype_enum
    ---> 97 ctx.ensure_initialized()
         98 return ops.EagerTensor(value, ctx.device_name, dtype)


    File /opt/miniconda3/envs/tf23/lib/python3.8/site-packages/tensorflow/python/eager/context.py:539, in Context.ensure_initialized(self)
        537   if self._use_tfrt is not None:
        538     pywrap_tfe.TFE_ContextOptionsSetTfrt(opts, self._use_tfrt)
    --> 539   context_handle = pywrap_tfe.TFE_NewContext(opts)
        540 finally:
        541   pywrap_tfe.TFE_DeleteContextOptions(opts)


    InternalError: CUDA runtime implicit initialization on GPU:0 failed. Status: device kernel image is invalid



```python
train_step()
print(w)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In [6], line 1
    ----> 1 train_step()
          2 print(w)


    NameError: name 'train_step' is not defined



```python
for i in range(1000):
    train_step()
print(w)
```

# 向量化实现


```python
w = tf.Variable(0,dtype=tf.float32)
x = np.array([1.0,-10.0,25.0],dtype=np.float32)
optimizer = tf.keras.optimizers.Adam(0.1)

def training(x,w,optimizer):
    def cost_fn():
        return x[0] * w ** 2 + x[1] * w + x[2]
    for i in range(1000):
        optimizer.minimize(cost_fn,[w])
        
    return w

w = training(x,w,optimizer)
print(w)
```

    2022-10-05 15:33:24.137130: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2022-10-05 15:33:24.137241: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties: 
    pciBusID: 0000:01:00.0 name: NVIDIA GeForce RTX 3060 Laptop GPU computeCapability: 8.6
    coreClock: 1.702GHz coreCount: 30 deviceMemorySize: 5.81GiB deviceMemoryBandwidth: 312.97GiB/s
    2022-10-05 15:33:24.137261: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
    2022-10-05 15:33:24.137274: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
    2022-10-05 15:33:24.137279: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcufft.so.10
    2022-10-05 15:33:24.137284: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcurand.so.10
    2022-10-05 15:33:24.137288: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusolver.so.10
    2022-10-05 15:33:24.137293: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusparse.so.10
    2022-10-05 15:33:24.137297: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
    2022-10-05 15:33:24.137324: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2022-10-05 15:33:24.137384: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2022-10-05 15:33:24.137428: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1858] Adding visible gpu devices: 0



    ---------------------------------------------------------------------------

    InternalError                             Traceback (most recent call last)

    Cell In [7], line 1
    ----> 1 w = tf.Variable(0,dtype=tf.float32)
          2 x = np.array([1.0,-10.0,25.0],dtype=np.float32)
          3 optimizer = tf.keras.optimizers.Adam(0.1)


    File /opt/miniconda3/envs/tf23/lib/python3.8/site-packages/tensorflow/python/ops/variables.py:262, in VariableMetaclass.__call__(cls, *args, **kwargs)
        260   return cls._variable_v1_call(*args, **kwargs)
        261 elif cls is Variable:
    --> 262   return cls._variable_v2_call(*args, **kwargs)
        263 else:
        264   return super(VariableMetaclass, cls).__call__(*args, **kwargs)


    File /opt/miniconda3/envs/tf23/lib/python3.8/site-packages/tensorflow/python/ops/variables.py:244, in VariableMetaclass._variable_v2_call(cls, initial_value, trainable, validate_shape, caching_device, name, variable_def, dtype, import_scope, constraint, synchronization, aggregation, shape)
        242 if aggregation is None:
        243   aggregation = VariableAggregation.NONE
    --> 244 return previous_getter(
        245     initial_value=initial_value,
        246     trainable=trainable,
        247     validate_shape=validate_shape,
        248     caching_device=caching_device,
        249     name=name,
        250     variable_def=variable_def,
        251     dtype=dtype,
        252     import_scope=import_scope,
        253     constraint=constraint,
        254     synchronization=synchronization,
        255     aggregation=aggregation,
        256     shape=shape)


    File /opt/miniconda3/envs/tf23/lib/python3.8/site-packages/tensorflow/python/ops/variables.py:237, in VariableMetaclass._variable_v2_call.<locals>.<lambda>(**kws)
        223 def _variable_v2_call(cls,
        224                       initial_value=None,
        225                       trainable=None,
       (...)
        234                       aggregation=VariableAggregation.NONE,
        235                       shape=None):
        236   """Call on Variable class. Useful to force the signature."""
    --> 237   previous_getter = lambda **kws: default_variable_creator_v2(None, **kws)
        238   for _, getter in ops.get_default_graph()._variable_creator_stack:  # pylint: disable=protected-access
        239     previous_getter = _make_getter(getter, previous_getter)


    File /opt/miniconda3/envs/tf23/lib/python3.8/site-packages/tensorflow/python/ops/variable_scope.py:2633, in default_variable_creator_v2(next_creator, **kwargs)
       2630 aggregation = kwargs.get("aggregation", None)
       2631 shape = kwargs.get("shape", None)
    -> 2633 return resource_variable_ops.ResourceVariable(
       2634     initial_value=initial_value,
       2635     trainable=trainable,
       2636     validate_shape=validate_shape,
       2637     caching_device=caching_device,
       2638     name=name,
       2639     dtype=dtype,
       2640     constraint=constraint,
       2641     variable_def=variable_def,
       2642     import_scope=import_scope,
       2643     distribute_strategy=distribute_strategy,
       2644     synchronization=synchronization,
       2645     aggregation=aggregation,
       2646     shape=shape)


    File /opt/miniconda3/envs/tf23/lib/python3.8/site-packages/tensorflow/python/ops/variables.py:264, in VariableMetaclass.__call__(cls, *args, **kwargs)
        262   return cls._variable_v2_call(*args, **kwargs)
        263 else:
    --> 264   return super(VariableMetaclass, cls).__call__(*args, **kwargs)


    File /opt/miniconda3/envs/tf23/lib/python3.8/site-packages/tensorflow/python/ops/resource_variable_ops.py:1507, in ResourceVariable.__init__(self, initial_value, trainable, collections, validate_shape, caching_device, name, dtype, variable_def, import_scope, constraint, distribute_strategy, synchronization, aggregation, shape)
       1505   self._init_from_proto(variable_def, import_scope=import_scope)
       1506 else:
    -> 1507   self._init_from_args(
       1508       initial_value=initial_value,
       1509       trainable=trainable,
       1510       collections=collections,
       1511       caching_device=caching_device,
       1512       name=name,
       1513       dtype=dtype,
       1514       constraint=constraint,
       1515       synchronization=synchronization,
       1516       aggregation=aggregation,
       1517       shape=shape,
       1518       distribute_strategy=distribute_strategy)


    File /opt/miniconda3/envs/tf23/lib/python3.8/site-packages/tensorflow/python/ops/resource_variable_ops.py:1650, in ResourceVariable._init_from_args(self, initial_value, trainable, collections, caching_device, name, dtype, constraint, synchronization, aggregation, distribute_strategy, shape)
       1648 with ops.get_default_graph()._attr_scope({"_class": attr}):
       1649   with ops.name_scope("Initializer"), device_context_manager(None):
    -> 1650     initial_value = ops.convert_to_tensor(
       1651         initial_value() if init_from_fn else initial_value,
       1652         name="initial_value", dtype=dtype)
       1653   if shape is not None:
       1654     if not initial_value.shape.is_compatible_with(shape):


    File /opt/miniconda3/envs/tf23/lib/python3.8/site-packages/tensorflow/python/framework/ops.py:1499, in convert_to_tensor(value, dtype, name, as_ref, preferred_dtype, dtype_hint, ctx, accepted_result_types)
       1494       raise TypeError("convert_to_tensor did not convert to "
       1495                       "the preferred dtype: %s vs %s " %
       1496                       (ret.dtype.base_dtype, preferred_dtype.base_dtype))
       1498 if ret is None:
    -> 1499   ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
       1501 if ret is NotImplemented:
       1502   continue


    File /opt/miniconda3/envs/tf23/lib/python3.8/site-packages/tensorflow/python/framework/tensor_conversion_registry.py:52, in _default_conversion_function(***failed resolving arguments***)
         50 def _default_conversion_function(value, dtype, name, as_ref):
         51   del as_ref  # Unused.
    ---> 52   return constant_op.constant(value, dtype, name=name)


    File /opt/miniconda3/envs/tf23/lib/python3.8/site-packages/tensorflow/python/framework/constant_op.py:263, in constant(value, dtype, shape, name)
        166 @tf_export("constant", v1=[])
        167 def constant(value, dtype=None, shape=None, name="Const"):
        168   """Creates a constant tensor from a tensor-like object.
        169 
        170   Note: All eager `tf.Tensor` values are immutable (in contrast to
       (...)
        261     ValueError: if called on a symbolic tensor.
        262   """
    --> 263   return _constant_impl(value, dtype, shape, name, verify_shape=False,
        264                         allow_broadcast=True)


    File /opt/miniconda3/envs/tf23/lib/python3.8/site-packages/tensorflow/python/framework/constant_op.py:275, in _constant_impl(value, dtype, shape, name, verify_shape, allow_broadcast)
        273     with trace.Trace("tf.constant"):
        274       return _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
    --> 275   return _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
        277 g = ops.get_default_graph()
        278 tensor_value = attr_value_pb2.AttrValue()


    File /opt/miniconda3/envs/tf23/lib/python3.8/site-packages/tensorflow/python/framework/constant_op.py:300, in _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
        298 def _constant_eager_impl(ctx, value, dtype, shape, verify_shape):
        299   """Implementation of eager constant."""
    --> 300   t = convert_to_eager_tensor(value, ctx, dtype)
        301   if shape is None:
        302     return t


    File /opt/miniconda3/envs/tf23/lib/python3.8/site-packages/tensorflow/python/framework/constant_op.py:97, in convert_to_eager_tensor(value, ctx, dtype)
         95   except AttributeError:
         96     dtype = dtypes.as_dtype(dtype).as_datatype_enum
    ---> 97 ctx.ensure_initialized()
         98 return ops.EagerTensor(value, ctx.device_name, dtype)


    File /opt/miniconda3/envs/tf23/lib/python3.8/site-packages/tensorflow/python/eager/context.py:539, in Context.ensure_initialized(self)
        537   if self._use_tfrt is not None:
        538     pywrap_tfe.TFE_ContextOptionsSetTfrt(opts, self._use_tfrt)
    --> 539   context_handle = pywrap_tfe.TFE_NewContext(opts)
        540 finally:
        541   pywrap_tfe.TFE_DeleteContextOptions(opts)


    InternalError: CUDA runtime implicit initialization on GPU:0 failed. Status: device kernel image is invalid

