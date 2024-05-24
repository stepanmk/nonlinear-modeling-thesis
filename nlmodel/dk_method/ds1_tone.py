from .components import *
from .dk_model import LinearDKModel


class DS1Tone(LinearDKModel):
    def __init__(self, sr=44100, alpha_t=0.5):
        super().__init__()
        self.vres = True
        self.T = 1 / sr

        self.alpha_t1 = 0.0
        self.alpha_t2 = 0.0

        self.recompute_alphas(alpha_t)

        self.RT = 20000

        self.components = [
            Resistor("R1", [1, 2], 6800),
            Resistor("R2", [3, 5], 2200),
            Capacitor("C1", [1, 3], 0.022e-6),
            Capacitor("C2", [2, 0], 0.1e-6),
            # TONE #
            VariableResistor("VR1_1", [2, 4], self.RT * self.alpha_t1),
            VariableResistor("VR1_2", [4, 5], self.RT * self.alpha_t2),
            Resistor("R3", [5, 0], 6800),
            Resistor("R4", [4, 0], 100000),
            # input & output ports
            InputPort("In", [1, 0], 0),
            OutputPort("Out", [4, 0]),
        ]
        self.components_count = {
            "n_res": 4,
            "n_vres": 2,
            "n_caps": 2,
            "n_inputs": 1,
            "n_outputs": 1,
            "n_nodes": 5,
        }
        self.build_model()

    def recompute_alphas(self, t):
        self.alpha_t1 = t
        self.alpha_t2 = 1 - t

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
