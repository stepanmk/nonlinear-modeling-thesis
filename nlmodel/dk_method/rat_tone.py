from .components import *
from .dk_model import LinearDKModel


class RATTone(LinearDKModel):
    def __init__(self, sr=44100, alpha_t=0.5):
        super().__init__()
        self.vres = True
        self.T = 1 / sr

        self.alpha_t1 = 0.0

        self.recompute_alphas(alpha_t)

        self.RT = 100000

        self.components = [
            Resistor("R1", [2, 3], 1500),
            Capacitor("C1", [3, 0], 3.3e-9),
            # TONE #
            VariableResistor("VR1_1", [1, 2], self.RT * self.alpha_t1),
            # input & output ports
            InputPort("In", [1, 0], 0),
            OutputPort("Out", [3, 0]),
        ]
        self.components_count = {
            "n_res": 1,
            "n_vres": 1,
            "n_caps": 1,
            "n_inputs": 1,
            "n_outputs": 1,
            "n_nodes": 3,
        }
        self.build_model()

    def recompute_alphas(self, t):
        self.alpha_t1 = t

    def get_matrices(self, device):
        return [
            self.Nr.to(device),
            self.Gr.to(device),
            self.Nx.to(device),
            self.Gx.to(device),
            self.Nv.to(device),
            self.Rv.to(device),
            self.Nu.to(device),
            self.No.to(device),
            self.Z.to(device),
            self.Q.to(device),
        ]
