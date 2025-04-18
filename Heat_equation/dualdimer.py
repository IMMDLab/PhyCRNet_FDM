import torch
from functools import reduce
from torch.optim.optimizer import Optimizer
import numpy as np
from math import cos, sin, atan, pi, sqrt

cuda = False if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class DualDimer(Optimizer):
    """Implements Bi-Dimer algorithm.

    References:

    * Henkelman and Jonsson, JCP 111, 7010 (1999)
    * Olsen, Kroes, Henkelman, Arnaldsson, and Jonsson, JCP 121,
      9776 (2004).
    * Heyden, Bell, and Keil, JCP 123, 224101 (2005).
    * Kastner and Sherwood, JCP 128, 014106 (2008).

    Arguments:
        lr (float): learning rate (default: 0.001)

    """

    def __init__(self, params, dim_max, lr=0.001, betas=(0.9, 0.999), eps=1e-8, max_step=0.005, dR=0.0001, f_rot_min=0.1, f_rot_max=1.0,
                 max_num_rot=1, trial_angle=pi/4.0, use_central_forces=True):

        defaults = dict(dim_max=dim_max, lr=lr, betas=betas, eps=eps, max_step=max_step, dR=dR, f_rot_min=f_rot_min, f_rot_max=f_rot_max,
                 max_num_rot=max_num_rot, trial_angle=trial_angle, use_central_forces=use_central_forces)
        super(DualDimer, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("DualDimer doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        group = self.param_groups[0]
        self.lr = group['lr']
        self.beta1, self.beta2 = group['betas']
        self.eps = group['eps']
        self.max_step = group['max_step']
        self.dR = group['dR']
        self.f_rot_min = group['f_rot_min']
        self.f_rot_max = group['f_rot_max']
        self.max_num_rot = group['max_num_rot']
        self.trial_angle = group['trial_angle']
        self.use_central_forces = group['use_central_forces']
        self._numel_cache = None
        self.num_weights = self._numel()
        self.num_con = group['dim_max']

        # Initialize the counters
        self.counters = {'forcecalls': 0, 'rotcount_min': 0, 'rotcount_max': 0, 'optcount': 0}

        self.curvature_min = None
        self.curvature_max = None


    def _normalize(self, vector):
        """Create a unit vector along *vector*"""
        return vector / torch.norm(vector)

    def _parallel_vector(self, vector, base):
        """Extract the components of *vector* that are parallel to *base*"""
        return torch.dot(vector, base) * base

    def _perpendicular_vector(self, vector, base):
        """Remove the components of *vector* that are parallel to *base*"""
        return vector - self._parallel_vector(vector, base)

    def _rotate_vectors(self, v1i, v2i, angle):
        """Rotate vectors *v1i* and *v2i* by *angle*"""
        cAng = cos(angle)
        sAng = sin(angle)
        v1o = v1i * cAng + v2i * sAng
        v2o = v2i * cAng - v1i * sAng
        # Ensure the length of the input and output vectors is equal
        return self._normalize(v1o) * torch.norm(v1i), self._normalize(v2o) * torch.norm(v2i)

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.data.add_(step_size, update[offset:offset + numel].view_as(p.data))
            offset += numel
        assert offset == self._numel()

    def converge_to_eigenmode(self, closure):
        """Perform an eigenmode search.
        This is the core part of dimer method. Rotate a dimer into lowest curvature mode.
        """
        self.counters['rotcount_min'] = 0
        self.counters['rotcount_max'] = 0
        stoprot_min = False
        stoprot_max = False
        self.eigenmode_min = torch.zeros(self.num_weights)
        self.eigenmode_max = torch.zeros(self.num_weights)
        self.eigenmode_min[:-self.num_con] = torch.randn(self.num_weights - self.num_con)
        self.eigenmode_max[-self.num_con:] = torch.randn(self.num_con)
        self.eigenmode_min = Tensor(self._normalize(self.eigenmode_min))
        self.eigenmode_max = Tensor(self._normalize(self.eigenmode_max))

        while not stoprot_min:

            self.update_virtual_forces_min(closure)
            self.update_curvature_min()
            f_rot_A = self._perpendicular_vector((self.forces1[:-self.num_con] - self.forces2[:-self.num_con]),
                                         self.eigenmode_min[:-self.num_con]) / (2.0 * self.dR)

            # Pre rotation stop criteria
            if torch.norm(f_rot_A) <= self.f_rot_min:
                stoprot_min = True
            else:
                # direction along the dimer N
                n_A = self.eigenmode_min[:-self.num_con]
                # direction perpendicular to the dimer theta
                rot_unit_A = self._normalize(f_rot_A)

                # Get the curvature and its derivative
                c0 = self.curvature_min
                c0d = np.vdot((self.forces2[:-self.num_con] - self.forces1[:-self.num_con]), rot_unit_A) / self.dR

                # Trial rotation (no need to store the curvature)
                # Two updated unit direction after rotation
                n_B, rot_unit_B = self._rotate_vectors(n_A, rot_unit_A, self.trial_angle)
                self.eigenmode_min[:-self.num_con] = n_B
                self.update_virtual_forces_min(closure)

                # Get the curvature's derivative
                c1d = np.vdot((self.forces2[:-self.num_con] - self.forces1[:-self.num_con]), rot_unit_B) / self.dR

                # Calculate the Fourier coefficients
                # Improve dimer method [3]
                a1 = (c0d * cos(2 * self.trial_angle) - c1d) / (2 * sin(2 * self.trial_angle))
                b1 = 0.5 * c0d
                a0 = 2 * (c0 - a1)

                # Estimate the rotational angle
                rotangle = atan(b1 / a1) / 2.0

                # Make sure that you find the minimum
                # Rotate into the (hopefully) lowest eigenmode
                # Store the curvature estimate instead of the old curvature
                cmin = a0 / 2.0 + a1 * cos(2 * rotangle) + b1 * sin(2 * rotangle)
                if c0 < cmin:
                    rotangle += pi / 2.0
                    n_min, dummy = self._rotate_vectors(n_A, rot_unit_A, rotangle)
                    self.eigenmode_min[:-self.num_con] = n_min
                    self.update_virtual_forces_min(closure)
                    self.update_curvature_min()
                else:
                    n_min, dummy = self._rotate_vectors(n_A, rot_unit_A, rotangle)
                    self.eigenmode_min[:-self.num_con] = n_min
                    self.update_curvature_min(cmin)

                self.counters['rotcount_min'] += 1

            # Post rotation stop criteria
            if not stoprot_min:
                if (self.counters['rotcount_min'] >= self.max_num_rot) or (torch.norm(f_rot_A) <= self.f_rot_max):
                    stoprot_min = True

        while not stoprot_max:

            self.update_virtual_forces_max(closure)
            self.update_curvature_max()
            f_rot_A = self._perpendicular_vector((self.forces3[-self.num_con:] - self.forces4[-self.num_con:]),
                                         self.eigenmode_max[-self.num_con:]) / (2.0 * self.dR)

            # Pre rotation stop criteria
            if torch.norm(f_rot_A) <= self.f_rot_min:
                stoprot_max = True
            else:
                # direction along the dimer N
                n_A = self.eigenmode_max[-self.num_con:]
                # direction perpendicular to the dimer theta
                rot_unit_A = self._normalize(f_rot_A)

                # Get the curvature and its derivative
                c0 = self.curvature_max
                c0d = np.vdot((self.forces4[-self.num_con:] - self.forces3[-self.num_con:]), rot_unit_A) / self.dR

                # Trial rotation (no need to store the curvature)
                # Two updated unit direction after rotation
                n_B, rot_unit_B = self._rotate_vectors(n_A, rot_unit_A, self.trial_angle)
                self.eigenmode_max[-self.num_con:] = n_B
                self.update_virtual_forces_max(closure)

                # Get the curvature's derivative
                c1d = np.vdot((self.forces4[-self.num_con:] - self.forces3[-self.num_con:]), rot_unit_B) / self.dR

                # Calculate the Fourier coefficients
                # Improve dimer method [3]
                a1 = (c0d * cos(2 * self.trial_angle) - c1d) / (2 * sin(2 * self.trial_angle))
                b1 = 0.5 * c0d
                a0 = 2 * (c0 - a1)

                # Estimate the rotational angle
                rotangle = atan(b1 / a1) / 2.0

                # Make sure that you find the maximum
                # Rotate into the (hopefully) largest eigenmode
                # Store the curvature estimate instead of the old curvature
                cmax = a0 / 2.0 + a1 * cos(2 * rotangle) + b1 * sin(2 * rotangle)
                if c0 > cmax:
                    rotangle += pi / 2.0
                    n_max, dummy = self._rotate_vectors(n_A, rot_unit_A, rotangle)
                    self.eigenmode_max[-self.num_con:] = n_max
                    self.update_virtual_forces_max(closure)
                    self.update_curvature_max()
                else:
                    n_max, dummy = self._rotate_vectors(n_A, rot_unit_A, rotangle)
                    self.eigenmode_max[-self.num_con:] = n_max
                    self.update_curvature_max(cmax)

                self.counters['rotcount_max'] += 1

            # Post rotation stop criteria
            if not stoprot_max:
                if (self.counters['rotcount_max'] >= self.max_num_rot) or (torch.norm(f_rot_A) <= self.f_rot_max):
                    stoprot_max = True

    def update_curvature_min(self, curv = None):
        """Update the curvature."""
        if curv:
            self.curvature_min = curv
        else:
            self.curvature_min = np.vdot((self.forces2[:-self.num_con] - self.forces1[:-self.num_con]),
                                         self.eigenmode_min[:-self.num_con]) / (2.0 * self.dR)

    def update_curvature_max(self, curv=None):
        """Update the curvature."""
        if curv:
            self.curvature_max = curv
        else:
            self.curvature_max = np.vdot((self.forces4[-self.num_con:] - self.forces3[-self.num_con:]),
                                         self.eigenmode_max[-self.num_con:]) / (2.0 * self.dR)

    def update_virtual_forces_min(self, closure):
        """Get the forces at the endpoints of the dimer."""

        # Calculate the forces at pos1
        self._add_grad(self.dR, self.eigenmode_min)
        vitrual_loss = closure()
        self.forces1 = -self._gather_flat_grad()
        self._add_grad(self.dR, -self.eigenmode_min)

        # Estimate / Calculate the forces at pos2
        if self.use_central_forces:
            self.forces2 = 2 * self.forces0 - self.forces1
        else:
            self._add_grad(self.dR, -self.eigenmode_min)
            vitrual_loss = closure()
            self.forces2 = -self._gather_flat_grad()
            self._add_grad(self.dR, self.eigenmode_min)

    def update_virtual_forces_max(self, closure):
        """Get the forces at the endpoints of the dimer."""

        # Calculate the forces at pos3
        self._add_grad(self.dR, self.eigenmode_max)
        vitrual_loss = closure()
        self.forces3 = -self._gather_flat_grad()
        self._add_grad(self.dR, -self.eigenmode_max)

        # Estimate / Calculate the forces at pos4
        if self.use_central_forces:
            self.forces4 = 2 * self.forces0 - self.forces3
        else:
            self._add_grad(self.dR, -self.eigenmode_max)
            vitrual_loss = closure()
            self.forces4 = -self._gather_flat_grad()
            self._add_grad(self.dR, self.eigenmode_max)

    def print(self):
        print('norm(f): {:.5f},min_curv: {:.5f},max_curv: {:.5f}'.format(
            torch.norm(self.forces0), self.curvature_min, self.curvature_max))


    def step(self, closure, epoch):
        """Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """

        t=epoch+1

        self.start = True
        # evaluate initial loss and gradient
        orig_loss = closure()
        self.forces0 = -self._gather_flat_grad()
        self.start = False

        if epoch % 40 == 0:
            self.converge_to_eigenmode(closure)

        v_min = self._parallel_vector(self.forces0[:-self.num_con], self.eigenmode_min[:-self.num_con])
        v_max = self._parallel_vector(self.forces0[-self.num_con:], self.eigenmode_max[-self.num_con:])

        g = -self.forces0
        g[-self.num_con:] = -g[-self.num_con:]
        # g[:-self.num_con] = g[:-self.num_con] + v_min
        # g[-self.num_con:] = -g[-self.num_con:] - v_max

        if t == 1:
            # Exponential moving average of gradient values
            self.m = torch.zeros_like(g)
            # Exponential moving average of squared gradient values
            self.v = torch.zeros_like(g)

        bias_correction1 = 1 - self.beta1 ** t
        bias_correction2 = 1 - self.beta2 ** t

        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * g ** 2

        denom = self.v.sqrt() / sqrt(bias_correction2) + self.eps

        step_size = self.lr / bias_correction1

        step1 = - step_size * torch.div(self.m, denom)

        step2 = Tensor((np.zeros(self.num_weights)))

        if abs(self.curvature_min) > 1e-3:
            step2[:-self.num_con] = v_min / (np.abs(self.curvature_min))

        if abs(self.curvature_max) > 1e-3:
            step2[-self.num_con:] = -v_max / (np.abs(self.curvature_max))

        # n1 = 0.05*torch.norm(step1[:-self.num_con])
        # n2 = 0.05*torch.norm(step1[-self.num_con:])
        #
        n1 = 1e-5
        n2 = 1e-5
        if torch.norm(step2[:-self.num_con]) > n1:
            step2[:-self.num_con] = n1*self._normalize(step2[:-self.num_con])

        if torch.norm(step2[-self.num_con:]) > n2:
            step2[-self.num_con:] = n2*self._normalize(step2[-self.num_con:])

        step = step1 + step2

        # step = step1

        self._add_grad(1.0, step)

        return orig_loss
