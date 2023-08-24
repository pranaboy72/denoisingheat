import copy
from . import solver
from . import Map
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

class Heat:
    def __init__(self, amb_temperature=0, material='Cu', dx=0.01, dy=0.01,
                 dt=0.1, size=(10,10), boundaries=(0, 0, 0, 0),
                 Q=[], Q0=[], initial_state=False, materials_path='./materials',
                 draw=['temperature', 'materials'], draw_scale=None):
        
        boundaries = tuple(boundaries)
        cond01 = isinstance(amb_temperature, float)
        cond01 = cond01 or isinstance(amb_temperature, int)
        cond02 = isinstance(material, str)
        cond05 = isinstance(dx, int) or isinstance(dx, float)
        cond06 = isinstance(dy, int) or isinstance(dy, float)
        cond07 = isinstance(dt, int) or isinstance(dt, float)
        cond09 = isinstance(boundaries, tuple)
        cond10 = isinstance(initial_state, bool)
        cond11 = isinstance(Q, list)
        cond12 = isinstance(Q0, list)
        cond13 = isinstance(draw, list)
        cond14 = isinstance(draw_scale, list) or isinstance(draw_scale, tuple)
        cond14 = cond14 or draw_scale is None
        condition = cond01 and cond02 and cond05
        condition = condition and cond06 and cond07 and cond09
        condition = condition and cond10 and cond11 and cond12 and cond13
        condition = condition and cond14
        
        if not condition:
            raise ValueError
        
        self.materials_path = materials_path
        self.time_passed = 0
        self.size = size
        self.dt = dt
        self.dx = dx
        self.dy = dy
        self.map = Map(material=material, dx=dx, dy=dy, 
                          dt=dt, size=size, Q=[], Q0=[],
                          initial_state=initial_state, materials_path=materials_path)
        
                    
    def show_figure(self, figure_type):
        """Plotting
            plot for temperature, materials
        """
        # check the validity of inputs
        condition = isinstance(figure_type, str)
        if not condition:
            raise ValueError

        if figure_type not in self.draw:
            self.draw.append(figure_type)
        if figure_type == 'temperature':
            self.figure = plt.figure()
            self.ax = self.figure.add_subplot(111)
            temp = []
            for i in range(self.map.size[0]):
                temp.append([])
                for j in range(self.map.size[1]):
                    temp[-1].append(self.map.temperature[i][j][0])
            if not self.draw_scale:
                vmax = max(max(temp, key=max))
                vmin = min(min(temp, key=min))
                temp = np.array(temp)
                extent = [0, self.size[0]*self.dx, 0, self.size[1]*self.dy]
                self.im = self.ax.imshow(np.transpose(temp), vmax=vmax,
                                         vmin=vmin, cmap='jet', extent=extent,
                                         origin='lower',
                                         interpolation='hamming')
            else:
                temp = np.array(temp)
                extent = [0, self.size[0]*self.dx, 0, self.size[1]*self.dy]
                self.im = self.ax.imshow(np.transpose(temp),
                                         vmax=self.draw_scale[0],
                                         vmin=self.draw_scale[1],
                                         cmap='jet', extent=extent,
                                         origin='lower',
                                         interpolation='hamming')
            cbar_kw = {}
            cbarlabel = "temperature (K)"
            cbar = self.ax.figure.colorbar(self.im, ax=self.ax, **cbar_kw)
            cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
            self.ax.set_title('Temperature')
            self.ax.set_xlabel('x axis (m)')
            self.ax.set_ylabel('y axis (m)')
            plt.show(block=False)
            
        if figure_type == 'materials' and len(self.map.materials_name) > 1:
            self.figure_materials = plt.figure()
            self.ax_materials = self.figure_materials.add_subplot(111)
            vmax = len(self.map.materials)-1
            vmin = 0
            material_id = np.array(self.map.materials_index)
            cmap = plt.get_cmap("PiYG", vmax+1)
            extent = [0, self.size[0]*self.dx, 0, self.size[1]*self.dy]
            value = np.transpose(material_id)
            self.im_materials = self.ax_materials.imshow(value,
                                                         vmax=vmax,
                                                         vmin=vmin,
                                                         cmap=cmap,
                                                         extent=extent,
                                                         origin='lower')
            cbar_kw = {}
            cbarlabel = ""
            materials_name_list = copy.deepcopy(self.map.materials_name)
            materials_name_list.reverse()
            qrates = np.array(materials_name_list)
            value = np.linspace(0, len(self.map.materials) - 1,
                                len(self.map.materials))
            norm = matplotlib.colors.BoundaryNorm(value,
                                                  len(self.map.materials)-1)
            func = lambda x, pos: qrates[::-1][norm(x)]
            fmt = matplotlib.ticker.FuncFormatter(func)
            cbar_kw = dict(ticks=np.arange(0, len(self.map.materials)+1),
                           format=fmt)
            cbar_m = self.ax_materials.figure.colorbar(self.im_materials,
                                                       ax=self.ax_materials,
                                                       **cbar_kw)
            cbar_m.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
            self.ax_materials.set_title('Materials')
            self.ax_materials.set_xlabel('x axis (m)')
            self.ax_materials.set_ylabel('y axis (m)')
            plt.show(block=False)
            print(self.map.materials_name)
        if figure_type == 'materials' and len(self.map.materials_name) == 1:
            print(self.map.materials_name)
            
            
    def activate(self, initial_point, final_point, obs_point, length):
        """Activation of the material

        Args:
            initial_point (_type_): _description_
            final_point (_type_): _description_
        """
        # check the validity of inputs
        value = isinstance(initial_point, tuple)
        if value and isinstance(final_point, tuple):
            condition = len(initial_point) == 2
            condition = condition and len(final_point) == 2
        else:
            condition = False
            
        if not condition:
            raise ValueError
        
        self.map.activate(initial_point=initial_point,
                             final_point=final_point)
        self.map.obstacle(initial_point=obs_point, length=length)

    def compute(self, time_interval, write_interval, verbose=True):
        """
            Copmute the thermal process.
        """
        cond1 = isinstance(time_interval, float)
        cond1 = cond1 or isinstance(time_interval, int)
        cond2 = isinstance(verbose, bool)
        condition = cond1 and cond2
        if not condition:
            raise ValueError
        
        # Total time steps
        nt = int(time_interval / self.map.dt)
        
        # computes
        for k in range(nt):
            # updates the time passed
            self.time_passed = self.time_passed + self.map.dt
            
            # defines the material properties accoring to the state list
            for i in range(self.map.size[0]):
                for j in range(self.map.size[1]):
                    ix = self.map.materials_index[i][j]
                    self.map.rho[i][j] = self.map.materials[ix].rho0(
                        self.map.temperature[i][j][0])
                    self.map.Cp[i][j] = self.map.materials[ix].cp0(
                        self.map.temperature[i][j][0])
                    self.map.k[i][j] = self.map.materials[ix].k0(
                        self.map.temperature[i][j][0])
                    
            temp = []
            for i in range(self.map.size[0]):
                temp.append([])
                for j in range(self.map.size[1]):
                    temp[-1].append(self.map.temperature[i][j][0])
            value = solver.explicit_k(self.map)
            self.map.temperature, self.map.lheat = value
            temp = []
            for i in range(self.map.size[0]):
                temp.append([])
                for j in range(self.map.size[1]):
                    temp[-1].append(self.map.temperature[i][j][0])

            if verbose:
                print('progress:',int(100*k/nt), '%', end='\r')
                
        if verbose:
            print('Finished simulation')