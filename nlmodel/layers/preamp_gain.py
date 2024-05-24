import torch
import torch.nn as nn

from ..dk_method.preamp_gain import PreampGain
from .pot import CondBlockTone


class PreampGainLayer(nn.Module):

    def __init__(
        self,
        sr: int = 44100,
        state_size: int = 4096,
        n_targets: int = 1,
        batch_size: int = 80,
        cond_input: str = "labels",
        cond_process: str = "pot",
        device: str = "cuda",
        freeze_cond_block: bool = False,
    ):
        super().__init__()
        self.device = device
        # labels or one hot encoded vectors
        self.cond_input = cond_input
        # process cond input either by mlps or by learnable potentiometer tapers
        self.cond_process = cond_process
        # size of the one hot encoded vector (number of target sounds)
        self.n_targets = n_targets
        #
        self.freeze_cond_block = freeze_cond_block
        self.sr = sr

        self.state_size = state_size
        self.state = None
        self.recurse_state = None

        self.ts = PreampGain(sr=self.sr)
        # matrices for state space computation
        self.A, self.B, self.D, self.E = None, None, None, None
        # resistors and caps
        self.Nr, self.Gr, self.Nx, self.Gx = None, None, None, None
        # variable resistors
        self.Nv, self.Rv, self.Q, self.RvQ = None, None, None, None
        # rest of the matrices
        self.Nu, self.No, self.Z = None, None, None
        self.Nvp, self.Nxp, self.Nop, self.Nup = None, None, None, None

        # frequency sampling vector
        self.z = None

        # initial component values
        self.RG = 1000000
        self.R1 = 470000
        self.C1 = 3.3e-9
        self.C2 = 1e-9

        # trainable scalers for the virtual components
        self.alpha_rg = nn.Parameter(
            torch.tensor([0.0], device=self.device), requires_grad=True
        )
        self.alpha_r1 = nn.Parameter(
            torch.tensor([0.0], device=self.device), requires_grad=True
        )

        self.alpha_c1 = nn.Parameter(
            torch.tensor([0.0], device=self.device), requires_grad=True
        )
        self.alpha_c2 = nn.Parameter(
            torch.tensor([0.0], device=self.device), requires_grad=True
        )

        # conditioning block
        self.cond_block = CondBlockTone(
            n_targets=self.n_targets,
            cond_input=self.cond_input,
            cond_process=self.cond_process,
            pot_char="log",
        )
        if self.freeze_cond_block:
            for param in self.cond_block.parameters():
                param.requires_grad = False
        self.batch_size = batch_size
        self.init(self.batch_size)

    def init(self, new_batch_size):
        # put sampling vector to GPU
        self.z = torch.exp(
            torch.complex(
                torch.zeros(1, dtype=torch.double), -torch.ones(1, dtype=torch.double)
            )
            * torch.linspace(0, torch.pi, 1 + self.state_size // 2)
        )
        self.z = (self.z.to(self.device)).unsqueeze(1).repeat(1, new_batch_size)
        # get dk method matrices
        matrices = self.ts.get_matrices(self.device)
        self.Nr = matrices[0]
        self.Gr = matrices[1]
        self.Nx = matrices[2]
        self.Gx = matrices[3]
        self.Nv = matrices[4]
        self.Rv = matrices[5].unsqueeze(0).repeat(new_batch_size, 1, 1)
        self.Nu = matrices[6]
        self.No = matrices[7]
        self.Z = matrices[8]
        self.Q = matrices[9]

        n_vsrcs = self.Nu.shape[0]
        n_nodes = self.Nu.shape[1]

        self.Nvp = torch.cat(
            [self.Nv, torch.zeros([self.Nv.shape[0], n_vsrcs], device=self.device)],
            dim=1,
        )
        self.Nxp = torch.cat(
            [self.Nx, torch.zeros([self.Nx.shape[0], n_vsrcs], device=self.device)],
            dim=1,
        )
        self.Nop = torch.cat(
            [self.No, torch.zeros([self.No.shape[0], n_vsrcs], device=self.device)],
            dim=1,
        )
        self.Nup = torch.cat(
            [
                torch.zeros([n_nodes, n_vsrcs], device=self.device),
                torch.eye(n_vsrcs, device=self.device),
            ],
            dim=0,
        )

    def change_batch_size(self, new_batch_size):
        self.batch_size = new_batch_size
        # put sampling vector to GPU
        self.z = torch.exp(
            torch.complex(
                torch.zeros(1, dtype=torch.double), -torch.ones(1, dtype=torch.double)
            )
            * torch.linspace(0, torch.pi, 1 + self.state_size // 2)
        )
        # adjust sampling vector and Rv tensor according to the batch size
        self.z = (self.z.to(self.device)).unsqueeze(1).repeat(1, new_batch_size)
        matrices = self.ts.get_matrices(self.device)
        self.Rv = matrices[5].unsqueeze(0).repeat(new_batch_size, 1, 1)

    def forward(self, x, cond):
        processed_cond = self.cond_block(cond)
        out = self.freq_filt(x, processed_cond)
        return out

    def freq_filt(self, x, processed_cond):
        self.update_components(processed_cond)
        self.update_state_space()

        self.state = torch.cat((self.state[:, x.shape[1] :, :], x), dim=1)
        state_fft = torch.fft.rfft(self.state.to(torch.double), dim=1)
        h = self.h_from_state_space()
        out_fft = torch.mul(state_fft, h)
        out = (torch.fft.irfft(out_fft, dim=1)).to(torch.float32)
        return out[:, -x.shape[1] :, :]

    def update_components(self, processed_cond):
        RG = (0.9 + torch.sigmoid(self.alpha_rg) * 0.2) * self.RG
        self.update_pots(processed_cond, RG)
        self.update_resistors()
        self.update_caps()

    def update_pots(self, processed_cond, RG):
        # pot values are different for each cond
        # Rv has a shape of (batch_size, num_vres, num_vres)
        Rv = torch.zeros_like(self.Rv, device=self.device)
        # GAIN
        Rv[:, 0, 0] = (1 - processed_cond[0]) * RG
        Rv[:, 1, 1] = processed_cond[0] * RG

        self.RvQ = torch.linalg.inv(Rv + self.Q)

    def update_resistors(self):
        # resistors are the same for all cond values
        Gr = torch.zeros_like(self.Gr, device=self.device)
        R1 = (0.99 + torch.sigmoid(self.alpha_r1) * 0.02) * self.R1
        Gr[0, 0] = 1 / R1
        self.Gr = Gr

    def update_caps(self):
        # capacitors are the same for all cond values
        Gx = torch.zeros_like(self.Gx, device=self.device)
        # 10% tol
        C1 = (0.9 + torch.sigmoid(self.alpha_c1) * 0.2) * self.C1
        C2 = (0.9 + torch.sigmoid(self.alpha_c2) * 0.2) * self.C2
        Gx[0, 0] = 2 * C1 / self.ts.T
        Gx[1, 1] = 2 * C2 / self.ts.T
        self.Gx = Gx

    def update_state_space(self):
        n_vsrc = self.Nu.shape[0]
        # system matrix So is computed only once, since the resistors and capacitors are the same for
        # all conditioning values
        So = torch.cat(
            [self.Nr.T @ self.Gr @ self.Nr + self.Nx.T @ self.Gx @ self.Nx, self.Nu.T],
            dim=1,
        )
        So = torch.cat(
            [
                So,
                torch.cat(
                    [self.Nu, torch.zeros([n_vsrc, n_vsrc], device=self.device)], dim=1
                ),
            ],
            dim=0,
        )
        So_inverse = torch.linalg.inv(So)

        # following matrices are the same as with single state space representation
        self.Q = self.Nvp @ So_inverse @ self.Nvp.T

        Ux = self.Nxp @ So_inverse @ self.Nvp.T
        Uo = self.Nop @ So_inverse @ self.Nvp.T
        Uu = self.Nup.T @ So_inverse @ self.Nvp.T
        ZGx = 2 * self.Z @ self.Gx

        Ao = ZGx @ self.Nxp @ So_inverse @ self.Nxp.T - self.Z
        Bo = ZGx @ self.Nxp @ So_inverse @ self.Nup
        Do = self.Nop @ So_inverse @ self.Nxp.T
        Eo = self.Nop @ So_inverse @ self.Nup

        # computation of state space matrices with added batch dimension
        # the batch dim is simply added by the fact the self.RvQ has it
        self.A = Ao - ZGx @ Ux @ self.RvQ @ Ux.T
        self.B = Bo - ZGx @ Ux @ self.RvQ @ Uu.T
        self.D = Do - Uo @ self.RvQ @ Ux.T
        self.E = Eo - Uo @ self.RvQ @ Uu.T

    def h_from_state_space(self):
        # transfer funcs polynomial coeffs
        den = self.batch_poly(self.A)
        num = (
            self.batch_poly(self.A - (self.B @ self.D))
            + (self.E.squeeze(-1).repeat(1, den.shape[1]) - 1.0) * den
        )
        # evaluate denominator and numerator polynomials
        a = self.batch_polyval(den)
        b = self.batch_polyval(num)
        # compute freq responses and add dims (batch_size, 1 + state_size // 2, 1)
        return (b / a).unsqueeze(-1)

    # implemented according to np.poly() with added batch dim
    def batch_poly(self, seq_of_zeros):
        # eigvals() func returns complex64 numbers, complex128 when the input is float64
        seq = torch.linalg.eigvals(seq_of_zeros)
        # conv1d() can produce rounding errors when using float32 as a dtype (default)
        # this can result in very inaccurate frequency responses due to badly computed coeffs
        # can be fixed by using complex64 or complex128 as a dtype for kernel and coeffs
        coeffs = torch.ones(
            (1, seq.shape[0], 1), device=self.device, dtype=torch.complex128
        )
        for i in range(seq.shape[1]):
            kernel = torch.ones(
                seq.shape[0], 1, 2, device=self.device, dtype=torch.complex128
            )
            kernel[:, 0, 0] = -seq[:, i]
            coeffs = nn.functional.conv1d(
                coeffs, kernel, padding=1, groups=seq.shape[0], bias=None
            )
        # return only the real part since the imag part is zero anyway
        return coeffs.squeeze(0).real

    # implemented according to np.polyval() with added batch dim
    def batch_polyval(self, coeffs):
        c0 = coeffs[:, -1] + self.z * 0
        for i in range(2, coeffs.shape[1] + 1):
            c0 = coeffs[:, -i] + c0 * self.z
        return c0.permute(1, 0)

    def reset_states(self):
        self.state = torch.zeros(
            (self.batch_size, self.state_size, 1), device=self.device
        )
        self.recurse_state = torch.zeros((self.batch_size, 2, 1), device=self.device)

    def detach_states(self):
        self.state = self.state.clone().detach()
        self.recurse_state = self.recurse_state.clone().detach()
        self.Q = self.Q.clone().detach()
