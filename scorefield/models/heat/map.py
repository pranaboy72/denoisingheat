from pathlib import Path
import os
import copy
import numpy as np


class Map:
    def __init__(self, amb_temperature=0, material='Cu', dx=0.01, dy=0.01,
                 dt=0.1, size=(10, 10), file_name=None,
                 boundaries=(0, 0, 0, 0), Q=[], Q0=[], initial_state=False,
                 materials_path=False):
        
        # check the validity of inputs
        boundaries = tuple(boundaries)
        Q = list(Q)
        Q0 = list(Q0)
        cond01 = isinstance(amb_temperature, float)
        cond01 = cond01 or isinstance(amb_temperature, int)
        cond05 = isinstance(dx, int) or isinstance(dx, float)
        cond06 = isinstance(dt, int) or isinstance(dt, float)
        cond07 = isinstance(file_name, str)
        cond07 = cond07 or (file_name is None)
        cond08 = isinstance(boundaries, tuple)
        cond09 = isinstance(Q, list)
        cond10 = isinstance(Q0, list)
        cond11 = isinstance(initial_state, bool)
        condition = cond01 and cond05
        condition = condition and cond06 and cond07 and cond08 and cond09
        condition = condition and cond10 and cond11
        if not condition:
            raise ValueError
        
        self.materials = [material]
        self.materials_name = [material]
        self.boundaries = boundaries
        self.amb_temperature = amb_temperature
        
        self.materials_index = [None]
        self.size = size
        self.file_name = file_name
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.temperature = []
        self.latent_heat = []
        self.lheat = []
        self.Cp = []
        self.rho = []
        self.Q = []
        self.Q0 = []
        self.k = []
        self.materials_index = []
        self.state = []
        
        for i in range(self.size[0]):
            self.state.append([])
            self.materials_index.append([])
            self.temperature.append([])
            self.latent_heat.append([])
            self.lheat.append([])
            self.Cp.append([])
            self.rho.append([])
            self.Q.append([0. for i in range(self.size[1])])
            self.Q0.append([0. for i in range(self.size[1])])
            self.k.append([])
            for j in range(self.size[1]):
                self.materials_index[-1].append(0)
                self.temperature[-1].append([amb_temperature, amb_temperature])
                self.state[-1].append(initial_state)
                if initial_state:
                    value = self.materials[self.materials_index[i][j]]
                    self.Cp[-1].append(value.cpa(self.amb_temperature))
                    value = self.materials[self.materials_index[i][j]]
                    self.rho[-1].append(value.rhoa(self.amb_temperature))
                    value = self.materials[self.materials_index[i][j]]
                    self.k[-1].append(value.ka(self.amb_temperature))
                    self.latent_heat[-1].append(
                        self.materials[self.materials_index[i][j]].lheata()
                    )
                    self.lheat[-1].append([])
                    value = self.materials[self.materials_index[i][j]]
                    for lh in value.lheata():
                        if self.temperature[i][j][1] < lh[0] and lh[1] > 0.:
                            self.lheat[-1][-1].append([lh[0], 0.])
                        if self.temperature[i][j][1] > lh[0] and lh[1] > 0.:
                            self.lheat[-1][-1].append([lh[0], lh[1]])
                        if self.temperature[i][j][1] < lh[0] and lh[1] < 0.:
                            self.lheat[-1][-1].append([lh[0], -lh[1]])
                        if self.temperature[i][j][1] > lh[0] and lh[1] < 0.:
                            self.lheat[-1][-1].append([lh[0], 0.])
                else:
                    value = self.materials[self.materials_index[i][j]]
                    self.Cp[-1].append(value.cp0(self.amb_temperature))
                    value = self.materials[self.materials_index[i][j]]
                    self.rho[-1].append(value.rho0(self.amb_temperature))
                    value = self.materials[self.materials_index[i][j]]
                    self.k[-1].append(value.k0(self.amb_temperature))
                    self.latent_heat[-1].append(
                        self.materials[self.materials_index[i][j]].lheat0()
                    )
                    self.lheat[-1].append([])
                    value = self.materials[self.materials_index[i][j]]
                    for lh in value.lheat0():
                        if self.temperature[i][j][1] < lh[0] and lh[1] > 0.:
                            self.lheat[-1][-1].append([lh[0], 0.])
                        if self.temperature[i][j][1] > lh[0] and lh[1] > 0.:
                            self.lheat[-1][-1].append([lh[0], lh[1]])
                        if self.temperature[i][j][1] < lh[0] and lh[1] < 0.:
                            self.lheat[-1][-1].append([lh[0], -lh[1]])
                        if self.temperature[i][j][1] > lh[0] and lh[1] < 0.:
                            self.lheat[-1][-1].append([lh[0], 0.])

        if Q != []:
            self.Q = Q
        if Q0 != []:
            self.Q0 = Q0

        self.time_passed = 0.

        self.Q_ref = copy.copy(self.Q)
        self.Q0_ref = copy.copy(self.Q0)
        
        
    def activate(self, initial_point, final_point):
        """
        Activates the given material.
        Suppose the obstacle is square shaped.
        initial point is the tuple (x,y) of the bottom left point
        and final point is the tuple(x,y) of the top right point.
        """
        
        initial_point_x = int(initial_point[0])
        initial_point_y = int(initial_point[1])
        final_point_x = int(final_point[0])
        final_point_y = int(final_point[1])
        for i in range(initial_point_x, final_point_x):
            for j in range(initial_point_y, final_point_y):
                if self.state[i][j] is False:
                    value = self.temperature[i][j][0]
                    self.temperature[i][j][0] = value + \
                        self.materials[self.materials_index[i][j]].tadi(
                            self.temperature[i][j][0])
                    value = self.materials_index[i][j]
                    self.rho[i][j] = self.materials[value].rhoa(
                        self.temperature[i][j][0])
                    self.Cp[i][j] = self.materials[value].cpa(
                        self.temperature[i][j][0])
                    self.k[i][j] = self.materials[value].ka(
                        self.temperature[i][j][0])
                    self.lheat[i][j] = []
                    valh = self.materials[value].lheata()
                    self.latent_heat[i][j] = valh
                    for lh in self.latent_heat[i][j]:
                        cond = self.temperature[i][j][0] < lh[0]
                        if cond and lh[1] > 0.:
                            self.lheat[i][j].append([lh[0], 0.])
                        cond = self.temperature[i][j][0] > lh[0]
                        if cond and lh[1] > 0.:
                            self.lheat[i][j].append([lh[0], lh[1]])
                        cond = self.temperature[i][j][0] < lh[0]
                        if cond and lh[1] < 0.:
                            self.lheat[i][j].append([lh[0], -lh[1]])
                        cond = self.temperature[i][j][0] > lh[0]
                        if cond and lh[1] < 0.:
                            self.lheat[i][j].append([lh[0], 0.])
                    self.state[i][j] = True
                else:
                    message = 'point ({:d},{:d})'.format(i, j)
                    message = message + ' already activated'
                    print(message)

    def obstacle(self, material='vaccum', initial_point=(3,3), length=(3,3),
               state=False):
        """
        Adds the obstacle as a square shaped 'vaccum'.
        'Initial point' is the bottom left (x,y) tuple,
        'length' is the length along two axis,
        'state' is the initial state of the material.
        """
        # check the validity of inputs
        value = isinstance(initial_point, tuple)
        if value and isinstance(length, tuple):
            cond1 = len(initial_point) == 2
            cond1 = cond1 and len(length) == 2
        else:
            cond1 = False
        cond2 = isinstance(material, str)
        cond3 = isinstance(state, bool)
        if not cond1 and cond2 and cond3:
            raise ValueError
    
        initial_point_x = int(initial_point[0])
        initial_point_y = int(initial_point[1])
        final_point_x = initial_point_x + int(length[0])
        final_point_y = initial_point_y + int(length[1])
        
        value = self.materials_name[index]
        tadi = Path(materials_path + value + '/' + 'tadi.txt')
        tadd = Path(materials_path + value + '/' + 'tadd.txt')
        cpa = Path(materials_path + value + '/' + 'cpa.txt')
        cp0 = Path(materials_path + value + '/' + 'cp0.txt')
        k0 = Path(materials_path + value + '/' + 'k0.txt')
        ka = Path(materials_path + value + '/' + 'ka.txt')
        rho0 = Path(materials_path + value + '/' + 'rho0.txt')
        rhoa = Path(materials_path + value + '/' + 'rhoa.txt')
        lheat0 = Path(materials_path + value + '/' + 'lheat0.txt')
        lheata = Path(materials_path + value + '/' + 'lheata.txt')
        self.materials.append(mats.CalMatPro(
            tadi, tadd, cpa, cp0, k0, ka, rho0, rhoa, lheat0, lheata))
        
        for i in range(initial_point_x, final_point_x):
            for j in range(initial_point_y, final_point_y):
                self.state[i][j] = False
                self.materials_index[i][j] = index
                self.rho[i][j] = self.materials[index].rho0(
                    self.temperature[i][j][0])
                self.Cp[i][j] = self.materials[index].cp0(
                    self.temperature[i][j][0])
                self.k[i][j] = self.materials[index].k0(
                    self.temperature[i][j][0])
                self.lheat[i][j] = []
                valh = self.materials[index].lheat0()
                self.latent_heat[i][j] = valh
                for lh in self.latent_heat[i][j]:
                    if self.temperature[i][j][0] < lh[0] and lh[1] > 0:
                        self.lheat[i][j].append([lh[0], 0.])
                    if self.temperature[i][j][0] > lh[0] and lh[1] > 0:
                        self.lheat[i][j].append([lh[0], lh[1]])
                    if self.temperature[i][j][0] < lh[0] and lh[1] < 0:
                        self.lheat[i][j].append([lh[0], -lh[1]])
                    if self.temperature[i][j][0] > lh[0] and lh[1] < 0:
                        self.lheat[i][j].append([lh[0], 0.])
                        
        