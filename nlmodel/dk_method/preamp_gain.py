from .components import *
from .dk_model import LinearDKModel


class PreampGain(LinearDKModel):
    def __init__(self, sr=44100, alpha_g=1.0):
        super().__init__()
        self.vres = True
        self.T = 1 / sr

        self.RG = 1000000

        self.alpha_g1 = 0.0
        self.alpha_g2 = 0.0

        self.recompute_alphas(alpha_g)

        self.components = [
            Capacitor("C1", [1, 2], 3.3e-9),
            Resistor("R1", [2, 3], 470000),
            VariableResistor("RGa", [3, 4], self.RG * self.alpha_g1),
            VariableResistor("RGb", [4, 0], self.RG * self.alpha_g2),
            Capacitor("C2", [2, 4], 1e-9),
            # input & output ports
            InputPort("In", [1, 0], 0),
            OutputPort("Out", [4, 0]),
        ]
        self.components_count = {
            "n_res": 1,
            "n_vres": 2,
            "n_caps": 2,
            "n_inputs": 1,
            "n_outputs": 1,
            "n_nodes": 4,
        }
        self.build_model()

    def recompute_alphas(self, g):
        self.alpha_g1 = 1 - g
        self.alpha_g2 = g

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
