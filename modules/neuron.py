# import torch
# from spikingjelly.activation_based import neuron
# from spikingjelly import visualizing
# from matplotlib import pyplot as plt
# from torch import nn
"""\markdown
多模态，以两个模态为例

$X_1[t],X_2[t]$ 表示多种模态的输入

#### 充电
$$
\begin{aligned}
    H_1[t]  &= f (V_1[t-1],X_1[t]) \\
    H_2[t]  &= f (V_2[t-1],X_2[t]) \\
    H[t] &= \alpha_1 H_1[t] \oplus \alpha_2 H_2[t]
\end{aligned}
$$
#### 放电
$$
\begin{aligned}
    S[t] & =\Theta\left(H[t]-V_{\text {threshold }}\right)
\end{aligned}
$$

#### 重置
$$
\begin{aligned}
    V_1[t]=H_1[t]-V_{\text {threshold }} \cdot S[t] \\
    V_2[t]=H_2[t]-V_{\text {threshold }} \cdot S[t]    
\end{aligned}
$$
"""

from typing import Callable
import torch
from torch import nn
from spikingjelly.clock_driven.surrogate import SurrogateFunctionBase, heaviside

from spikingjelly.clock_driven.neuron import LIFNode
from spikingjelly.clock_driven.neuron import ParametricLIFNode 

from modules.surrogate import Rectangle
# from surrogate import Rectangle

## spikingjelly Multi-model single step version
class MultiLIFNeuron(LIFNode):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = None, surrogate_function: Callable = Rectangle(),alpha1: float =0.5,alpha2: float =0.5,
                 detach_reset: bool = False, cupy_fp32_inference=False, **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, cupy_fp32_inference)
        # self.register_memory('m', 0.)  # Complementary memory
        self.register_memory('alpha1',alpha1)
        self.register_memory('alpha2',alpha2)
        ## 注册 v1,v2
        self.register_memory('v1', 0.) 
        self.register_memory('v2', 0.) 
    def forward(self, x: torch.Tensor):
        ## extract x1,x2
        self.neuronal_charge(x)
        spike = self.neuronal_fire()  # LIF fire
        self.neuronal_reset(spike)
        return spike
    
    def neuronal_charge(self, x: torch.Tensor):
        self._charging_v(x)

    def neuronal_reset(self, spike: torch.Tensor):
        self._reset(spike)
    
    def _charging_v(self, x: torch.Tensor):
        if self.decay_input:
            x = x / self.tau
        ## split
        x1,x2 = torch.chunk(x,chunks=2,dim=1)
        if self.v_reset is None or self.v_reset == 0:
            if type(self.v) is float:
                self.v1 = x1
                self.v2 = x2
                
            else:
                self.v1 = self.v1 * (1 - 1. / self.tau) + x1
                self.v2 = self.v2 * (1 - 1. / self.tau) + x2
            # self.v = torch.cat((self.alpha1*self.v1,self.alpha2*self.v2),dim=0)
        else:
            if type(self.v) is float:
                self.v1 = self.v_reset * (1 - 1. / self.tau) + self.v_reset / self.tau + x1
                self.v2 = self.v_reset * (1 - 1. / self.tau) + self.v_reset / self.tau + x2
                # self.v = torch.cat((self.alpha1*self.v1,self.alpha2*self.v2),dim=1)
            else:
                self.v1 = self.v * (1 - 1. / self.tau) + self.v_reset / self.tau + x1
                self.v2 = self.v_reset * (1 - 1. / self.tau) + self.v_reset / self.tau + x2
        self.v = torch.cat((self.alpha1*self.v1,self.alpha2*self.v2),dim=1)
        
    def _reset(self,spike):
        ## reset function
        # print(spike.shape)
        spike1,spike2 = torch.chunk(spike,chunks=2,dim=1)
        # print(spike.shape,spike.shape)
        if self.v_reset is None:
            # # soft reset
            self.v1 = self.v1 - spike1 * self.v_threshold
            self.v2 = self.v2 - spike2 * self.v_threshold
        else:
            # hard reset
            # self.v = (1. - spike) * self.v + spike * self.v_reset
            self.v1 = (1. - spike1) * self.v1 + spike1 * self.v_reset
            self.v2 = (1. - spike2) * self.v2 + spike2 * self.v_reset
        self.v = torch.cat((self.alpha1*self.v1,self.alpha2*self.v2),dim=1)

class MultiStepMultiLIFNeuron(MultiLIFNeuron):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = None, surrogate_function: Callable = Rectangle(),alpha1: float =0.5,alpha2: float =0.5,
                 detach_reset: bool = False, cupy_fp32_inference=False, **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function,alpha1,alpha2, detach_reset, cupy_fp32_inference)
    
    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() > 1
        # x_seq.shape = [T, *]
        spike_seq = []
        self.v_seq = []
        for t in range(x_seq.shape[0]):
            spike_seq.append(super().forward(x_seq[t]).unsqueeze(0))
            self.v_seq.append(self.v.unsqueeze(0))
        spike_seq = torch.cat(spike_seq, 0)
        self.v_seq = torch.cat(self.v_seq, 0)
        return spike_seq

class ReLU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return torch.relu(x)
    
# LIF
class VanillaNeuron(LIFNode):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = None, surrogate_function: Callable = Rectangle(),
                 detach_reset: bool = False, cupy_fp32_inference=False, **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, cupy_fp32_inference)

## PLIF
class PLIFNeuron(ParametricLIFNode):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = None, surrogate_function: Callable = Rectangle(),
                 detach_reset: bool = False, cupy_fp32_inference=False, **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset)


# CLIF
class ComplementaryLIFNeuron(LIFNode):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = None, surrogate_function: Callable = Rectangle(),
                 detach_reset: bool = False, cupy_fp32_inference=False, **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, cupy_fp32_inference)
        self.register_memory('m', 0.)  # Complementary memory

    def forward(self, x: torch.Tensor):
        self.neuronal_charge(x)  # LIF charging
        self.m = self.m * torch.sigmoid(self.v / self.tau)  # Forming
        spike = self.neuronal_fire()  # LIF fire
        self.m += spike  # Strengthen
        self.neuronal_reset(spike)  # LIF reset
        self.v = self.v - spike * torch.sigmoid(self.m)  # Reset
        return spike

    def neuronal_charge(self, x: torch.Tensor):
        self._charging_v(x)

    def neuronal_reset(self, spike: torch.Tensor):
        self._reset(spike)

    def _charging_v(self, x: torch.Tensor):
        if self.decay_input:
            x = x / self.tau

        if self.v_reset is None or self.v_reset == 0:
            if type(self.v) is float:
                self.v = x
            else:
                self.v = self.v * (1 - 1. / self.tau) + x
        else:
            if type(self.v) is float:
                self.v = self.v_reset * (1 - 1. / self.tau) + self.v_reset / self.tau + x
            else:
                self.v = self.v * (1 - 1. / self.tau) + self.v_reset / self.tau + x

    def _reset(self, spike):
        if self.v_reset is None:
            # soft reset
            self.v = self.v - spike * self.v_threshold
        else:
            # hard reset
            self.v = (1. - spike) * self.v + spike * self.v_reset

# spikingjelly multiple step version
class MultiStepCLIFNeuron(ComplementaryLIFNeuron):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = None, surrogate_function: Callable = Rectangle(),
                 detach_reset: bool = False, cupy_fp32_inference=False, **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, cupy_fp32_inference)

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() > 1
        # x_seq.shape = [T, *]
        spike_seq = []
        self.v_seq = []
        for t in range(x_seq.shape[0]):
            spike_seq.append(super().forward(x_seq[t]).unsqueeze(0))
            self.v_seq.append(self.v.unsqueeze(0))
        spike_seq = torch.cat(spike_seq, 0)
        self.v_seq = torch.cat(self.v_seq, 0)
        return spike_seq

## 
class TwoCompartmentLIF(LIFNode):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,k=2,gamma: float = 0.5, 
                 decay_factor: torch.Tensor = torch.full([1, 2], 0, dtype=torch.float),
                 v_reset: float = None, surrogate_function: Callable = Rectangle(),
                 detach_reset: bool = False, cupy_fp32_inference=False, **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, cupy_fp32_inference)
        self.k = k
        for i in range(1, self.k + 1):
            self.register_memory('v' + str(i), 0.)
        self.names = self._memories
        # self.hard_reset = hard_reset
        self.gamma = gamma
        self.decay = decay_factor
        self.decay_factor = torch.nn.Parameter(decay_factor)
    
    def neuronal_charge(self, x: torch.Tensor):
        # v1: membrane potential of dendritic compartment
        # v2: membrane potential of somatic compartment
        self.names['v1'] = self.names['v1'] - torch.sigmoid(self.decay_factor[0][0]) * self.names['v2'] + x
        self.names['v2'] = self.names['v2'] + torch.sigmoid(self.decay_factor[0][1]) * self.names['v1']
        self.v = self.names['v2']
    
    def jit_soft_reset(self,v: torch.Tensor, spike: torch.Tensor, v_threshold: torch.Tensor):
        v = v - spike * v_threshold
        return v
    
    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if not self.v_reset:
            # soft reset
            self.names['v1'] = self.jit_soft_reset(self.names['v1'], spike_d, self.gamma)
            self.names['v2'] = self.jit_soft_reset(self.names['v2'], spike_d, self.v_threshold)
        else:
            # hard reset
            for i in range(2, self.k + 1):
                self.names['v' + str(i)] = self.jit_hard_reset(self.names['v' + str(i)], spike_d,  self.v_reset)
        
    def extra_repr(self):
        return f"v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, " \
               f"hard_reset={self.hard_reset}, " \
               f"gamma={self.gamma}, k={self.k}, step_mode={self.step_mode}, backend={self.backend}"
                      
    def forward(self, x: torch.Tensor):
        self.neuronal_charge(x)
        spike = self.neuronal_fire()  # LIF fire
        self.neuronal_reset(spike)
        return spike

class AdaptiveLIF(LIFNode):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 k=1,
                 alpha=1,
                 vn_reset=True,
                 decay_factor: torch.Tensor = torch.full([1, 2], 0, dtype=torch.float),gamma = 0.,
                 v_reset: float = None, surrogate_function: Callable = Rectangle(),
                 detach_reset: bool = False, cupy_fp32_inference=False, **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, cupy_fp32_inference)
        self.k = k
        self.v = torch.zeros([0])
        for i in range(1, self.k + 1):
            self.register_memory('v' + str(i), 0.)
        self.register_memory('yita', 0.)
        self.register_memory('threshold', 1.)
        self.names = self._memories
        self.decay_factor = torch.nn.Parameter(decay_factor)
        self.vn_reset = vn_reset
    def jit_soft_reset(self,v: torch.Tensor, spike: torch.Tensor, v_threshold: torch.Tensor):
        v = v - spike * v_threshold
        return v
    def neuronal_charge(self, x: torch.Tensor):
        # print(self.decay_factor)
        spike_lastT = self.neuronal_fire()
        alpha = torch.exp(-1 / torch.sigmoid(self.decay_factor[0][0]))
        r0 = torch.exp(-1 / torch.sigmoid(self.decay_factor[0][1]))
        self.names['yita'] = r0 * self.names['yita'] + (1 - r0) * spike_lastT
        self.names['threshold'] = 0.01 + 1.8 * self.names['yita']
        self.names['v1'] = alpha * self.names['v1'] + (1 - alpha) * x
        self.v = self.names['v1']
        
    # def neuronal_fire(self):
    #     return self.surrogate_function(self.v - self.names['threshold'])
    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.v_reset is None:
            # soft reset
            self.names['v1'] = self.jit_soft_reset(self.names['v1'], spike_d, self.names['threshold'])
            self.v = self.jit_soft_reset(self.v, spike_d, self.names['threshold'])

        else:
            # hard reset
            self.v1 = self.jit_hard_reset(self.v1, spike_d, self.v_reset)
            if self.vn_reset:
                for i in range(2, self.k + 1):
                    self.names['v' + str(i)] = self.jit_hard_reset(self.names['v' + str(i)], spike_d,  self.v_reset)
    
    def forward(self, x: torch.Tensor):
        self.neuronal_charge(x)
        spike = self.neuronal_fire()  # LIF fire
        self.neuronal_reset(spike)
        return spike
    

if __name__ == '__main__':
    T = 8
    mlif = MultiLIFNeuron()
    # alif = AdaptiveLIF()
    tc_lif = TwoCompartmentLIF()
    clif = ComplementaryLIFNeuron()
    plif = PLIFNeuron()
    lif = VanillaNeuron()
    x_input = torch.rand((T,2,6, 32, 32)) * 1.2
    for t in range(T):
        mlifout = mlif(x_input[t])
        
        # alifout = alif(x_input[t])
        tc_lifout = tc_lif(x_input[t])
        clifout = clif(x_input[t])
        plifout = plif(x_input[t])
        lifout = lif(x_input[t])
        ## all test are passed
        # 确保所有神经元输出的形状相同
        assert mlifout.shape == tc_lifout.shape == clifout.shape == plifout.shape == lifout.shape, "Output shapes are not equal"
        # print('mlifout',mlifout.shape)
        # print('tc_lifout',tc_lifout.shape)
        # print('clifout',clifout.shape)
        # print('plifout',plifout.shape)
        # print('lifout',lifout.shape)
#     T = 8
#     mlif = MultiLIFNeuron()
#     mlif_m = MultiStepMultiLIFNeuron()
#     x_input = torch.rand((T,2,6, 32, 32)) * 1.2
    
#     s_list = []
#     for t in range(T):
#         s = mlif(x_input[t])
#         s_list.append(s)

#     s_output = mlif_m(x_input)
#     s_list = torch.stack(s_list, dim=0)
#     print(s_list.mean())
#     print(s_output.mean())

#     assert torch.sum(s_output - torch.Tensor(s_list)) == 0