'''2016-04-28
Analyzing spiking circuit
'''

from __future__ import division
import time
import os
import os.path
import sys
import pickle
import copy
import scipy.optimize
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib.mlab import griddata
from matplotlib import ticker
import matplotlib.cm as colormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import brian_no_units
from brian import *

from SpikingCircuitModel import Model

mpl.rc("savefig", dpi=100)

colors = np.array([ [8,48,107],         # dark-blue
                    [228,26,28],        # red
                    [152,78,163],       # purple
                    [77,175,74]])/255.  # green

pops = ['E','P','S','V']

def print_params():
    model = Model(rng_seed=1)
    p = model.params
    import tabulate as T
    import collections

    NAME = 0
    UNIT = 1
    UNITNAME = 2
    MEANING = 3

    pinfo = collections.OrderedDict([
        ('N'         , ['$N$'                   ,1 ,'',  'Number of neurons']),
        ('C_m'       , ['$C_m$'                 ,pF,'pF','Membrane capacitance']),
        ('g_L'       , ['$g_L$'                 ,nS,'nS','Leak conductance']),
        ('tau'       , [r'$\tau$'               ,ms,'ms','Membrane time constant']),
        ('E_L'       , ['$E_L$'                 ,mV,'mV','Resting potential']),
        ('V_T'       , ['$V_T$'                 ,mV,'mV','Threshold voltage']),
        ('Delta_T'   , ['$\Delta_T$'            ,mV,'mV','EIF slope parameter']),
        ('V_re'      , ['$V_{\mathrm{re}}$'     ,mV,'mV','Reset potential']),
        ('tau_refrac', [r'$\tau_{\mathrm{ref}}$',ms,'ms','Refractory period']),
        ('a'         , ['$a$'                   ,nS,'nS','Subthreshold adaptation']),
        ('b'         , ['$C_m$'                 ,pA,'pA','Spike-triggered adaptation']),
        ('tau_w'     , [r'$\tau_w$'             ,ms,'ms','Adaptation time constant']),
        ('sigma'     , ['$\sigma_{\mathrm{ext}}$',mV,'mV','Standard deviation of external input']),
        ('E_syn'     , ['$E_{\mathrm{syn}}$'    ,mV,'mV','Reversal potential']),
        ('tau_syn'   , [r'$\tau_{\mathrm{syn}}$',ms,'ms','Synaptic time constant'])])


    tabledata = collections.OrderedDict()
    tabledata['name'] = [pinfo[key][NAME] for key in pinfo]
    for pop in pops:
        tabledata[pop] = [p[pop][key]/pinfo[key][UNIT] for key in pinfo.keys()[:-2]]
        # The rest need some special treatment
        tabledata[pop] += [p['E']['E_'+pop]/pinfo[key][UNIT]]
        tabledata[pop] += [p['E']['tau_d'+pop]/pinfo[key][UNIT]]
    tabledata['unit'] = [pinfo[key][UNITNAME] for key in pinfo]
    tabledata['meaning'] = [pinfo[key][MEANING] for key in pinfo]

    T.LATEX_ESCAPE_RULES = {}
    headers = ['']+pops+['Unit','Description']
    print T.tabulate(tabledata,headers,tablefmt='latex')


    for mat in [p['g']/nS, p['p0'], p['p2']]:
        print '\\begin{bmatrix}'
        print " \\\\\n".join([" & ".join(map(str,line)) for line in mat])
        print '\\end{bmatrix}\n'

def delta_method(d1,d0,method='sub'):
    minl = np.min((len(d0),len(d1)))
    if method == 'sub':
        return np.mean(d1[:minl]-d0[:minl])
    elif method == 'div':
        return np.mean((d1[:minl]-d0[:minl])/abs(d0[:minl]))
    else:
        ValueError('Unknown Delta Method')

class DataDict(dict):
    '''
    A new class to hold the common data structure
    '''
    def __init__(self):
        dict.__init__(self)
        self.pops = ['E','P','S','V']
        self.pops_from = self.pops
        for pop in self.pops:
            self['rate'+pop] = list()
            for pop_from in self.pops_from:
                self['g'+pop+pop_from] = list()
                self['I'+pop+pop_from] = list()

    def data_merge(self,new_data):
        for pop in self.pops:
            self['rate'+pop].extend(new_data['rate'+pop])
            for pop_from in self.pops_from:
                self['g'+pop+pop_from].extend(new_data['g'+pop+pop_from])
                self['I'+pop+pop_from].extend(new_data['I'+pop+pop_from])
    
    def to_numpy(self):
        for pop in self.pops:
            self['rate'+pop] = np.array(self['rate'+pop])
            for pop_from in self.pops_from:
                self['g'+pop+pop_from] = np.array(self['g'+pop+pop_from])
                self['I'+pop+pop_from] = np.array(self['I'+pop+pop_from])

class Analysis(object):
    def __init__(self, version=0,recover=True):
        self.start = time.time()
        model = Model(rng_seed=1)
        self.pops = model.pops
        self.pops_from = self.pops
        self.version = version
        self.savefile = 'spiking_PVSSTdensity'+str(self.version)+'.pkl'

        if os.path.isfile('data/'+self.savefile) and recover:
            print 'Loading from ' + self.savefile + '...'
            with open('data/'+self.savefile,'rb') as f:
                self.data = pickle.load(f)
        else:
            self.data = dict()
            model = Model(rng_seed=1)
            p_orig = model.params
            self.data['p_orig'] = p_orig

        self.runtime = 1. # second

        self.mu_step = 0.5*mV

    def get_backgroundinput_all(self,recover=True,fullcircuit=True):
        model = Model(rng_seed=1)
        p_orig = model.params
        savefile = 'spiking_backgroundmu'+str(self.version)+'.pkl'
        if os.path.isfile('data/'+savefile) and recover:
            with open('data/'+savefile,'rb') as f:
                self.data_mu = pickle.load(f)
        else:
            self.data_mu = dict()
            self.data_mu['p_orig'] = p_orig

        #density_P_plot = np.repeat([0.5,0.75,1,1.25,1.5],5)
        #density_S_plot = np.tile([0.5,0.75,1,1.25,1.5],5)

        density_P_plot = np.repeat([0.25,0.5,0.75,1,1.25,1.5,1.75],7)
        density_S_plot = np.tile([0.25,0.5,0.75,1,1.25,1.5,1.75],7)

        #density_P_plot = [0.5,0.5,1.5,1.5]
        #density_S_plot = [0.5,1.5,0.5,1.5]

        #density_P_plot = [0.2,0.2,1.8,1.8]
        #density_S_plot = [0.2,1.8,0.2,1.8]

        print 'Calculating Background inputs'
        for density_P, density_S in zip(density_P_plot,density_S_plot):
            if (density_P,density_S) not in self.data_mu.keys():
                mu_spt_list = list()
                for pop_iso in self.pops:
                    mu_spt = self.get_backgroundinput_popiso(pop_iso,density_P,density_S)
                    mu_spt_list.append(mu_spt)

                if fullcircuit:
                    mu0 = copy(mu_spt_list) # use this as initial guess
                    mu_spt_list = self.get_backgroundinput_fullcircuit(density_P,density_S,mu0)

                self.data_mu[(density_P,density_S)] = mu_spt_list
                print 'density P,S ({:0.2f},{:0.2f})'.format(density_P,density_S),
                print mu_spt_list
                sys.stdout.write('Time spent {:0.1f} second.'.format(time.time()-self.start))
                sys.stdout.flush()

        with open('data/'+savefile,'wb') as f:
            pickle.dump(self.data_mu, f)

    def get_backgroundinput_fullcircuit(self,density_P,density_S,mu0):
        p_orig = self.data['p_orig']
        V_spts = p_orig['V_spts']

        bound_dict = {'E' : (25,30),
                      'P' : (8,20),
                      'S' : (13,18),
                      'V' : (10,18)}
        bounds = [bound_dict[pop] for pop in pops]

        rng_seed = 300

        pe = dict()
        for pop in pops:
            pe[pop] = dict()
        pe['P']['N'] = int(p_orig['P']['N']*density_P)
        pe['S']['N'] = int(p_orig['S']['N']*density_S)

        model = Model(rng_seed=rng_seed,extra_para=pe,random_conn=False)
        model.make_model()
        model.build()

        def get_r(mus, model, rng_seed):
            for pop, mu in zip(pops,mus):
                model.params[pop]['mu'] = mu*mV
            model.rng_seed = rng_seed
            model.reinit()
            net = Network(model)
            net.run(self.runtime*second)
            V_list = np.array([model.monitor['V'+pop].mean.mean()/mV for pop in self.pops])
            #print mus,
            #print V_list
            return V_list

        obj_func = lambda x: np.sum((get_r(x,model,rng_seed)-V_spts)**2)

        #res = scipy.optimize.minimize(obj_func,x0=mu0,bounds=bounds,method='SLSQP',options={'maxiter':100,'eps':0.2})
        res = scipy.optimize.minimize(obj_func,x0=mu0,bounds=bounds,method='L-BFGS-B',options={'maxfun':500,'eps':0.2})

        print res
        
        return res.x

    def get_backgroundinput_popiso(self,pop_iso,density_P,density_S):
        p_orig = self.data['p_orig']
        V_spt = p_orig[pop_iso]['V_spt']
        rng_seed = 300

        pe = dict()
        for pop in pops:
            pe[pop] = dict()
        pe['P']['N'] = int(p_orig['P']['N']*density_P)
        pe['S']['N'] = int(p_orig['S']['N']*density_S)

        model = Model(rng_seed=rng_seed,extra_para=pe,random_conn=False)
        model.isolate_population(pop_iso=pop_iso)
        model.build()

        def get_V(mu_iso, model, rng_seed):
            model.params[pop_iso]['mu'] = mu_iso*mV
            model.rng_seed = rng_seed
            model.reinit()
            net = Network(model)
            net.run(self.runtime*second)
            meanV = model.monitor['V'+pop_iso].mean.mean()/mV
            #print mu_iso,
            #print meanV
            return meanV

        obj_func = lambda x: (get_V(x,model,rng_seed)-V_spt)**2

        bound_dict = {'E' : (25,30),
                      'P' : (8,20),
                      'S' : (13,18),
                      'V' : (10,18)}
        bounds = bound_dict[pop_iso]
        res = scipy.optimize.minimize_scalar(obj_func,bounds=bounds,method='Bounded',
                                       options={'maxiter':100,'xatol':0.01})
        #print res
        return res.x

    def get_backgroundinput_fullcircuit_RATE(self,density_P,density_S,mu0):
        p_orig = self.data['p_orig']
        r_spts = np.array([p_orig[pop]['r_spt'] for pop in self.pops])

        bound_dict = {'E' : (20,45),
                    'P' : (10,30),
                    'S' : (10,30),
                    'V' : (10,30)}
        bounds = [bound_dict[pop] for pop in pops]

        rng_seed = 300

        pe = dict()
        for pop in pops:
            pe[pop] = dict()
        pe['P']['N'] = int(p_orig['P']['N']*density_P)
        pe['S']['N'] = int(p_orig['S']['N']*density_S)

        model = Model(rng_seed=rng_seed,extra_para=pe,random_conn=False)
        model.make_model()
        model.build()

        def get_r(mus, model, rng_seed):
            for pop, mu in zip(pops,mus):
                model.params[pop]['mu'] = mu*mV
            model.rng_seed = rng_seed
            model.reinit()
            net = Network(model)
            net.run(self.runtime*second)
            mon = model.monitor
            p = model.params

            rate_list = list()
            for pop in pops:
                spiketime = np.array([spike[1] for spike in mon['Spike'+pop].spikes])
                rate = np.sum(spiketime>200*ms)/p[pop]['N']/(model.clock.t-200*ms)
                rate_list.append(rate)
            print mus,
            print rate_list
            return np.array(rate_list)

        obj_func = lambda x: np.sum((get_r(x,model,rng_seed)-r_spts)**2)

        res = scipy.optimize.minimize(obj_func,x0=mu0,bounds=bounds,method='SLSQP',
                                       options={'maxiter':300,'ftol':0.001,'eps':0.01})

        print res

        return res.x

    def get_backgroundinput_popiso_RATE(self,pop_iso,density_P,density_S):
        p_orig = self.data['p_orig']
        r_spt = p_orig[pop_iso]['r_spt']
        rng_seed = 300

        pe = dict()
        for pop in pops:
            pe[pop] = dict()
        pe['P']['N'] = int(p_orig['P']['N']*density_P)
        pe['S']['N'] = int(p_orig['S']['N']*density_S)

        model = Model(rng_seed=rng_seed,extra_para=pe,random_conn=False)
        model.isolate_population(pop_iso=pop_iso)
        model.build()

        def get_r(mu_iso, model, rng_seed):
            model.params[pop_iso]['mu'] = mu_iso*mV
            model.rng_seed = rng_seed
            model.reinit()
            net = Network(model)
            net.run(self.runtime*second)
            mon = model.monitor
            p = model.params
            pop = pop_iso
            spiketime = np.array([spike[1] for spike in mon['Spike'+pop].spikes])
            rate = np.sum(spiketime>200*ms)/p[pop]['N']/(model.clock.t-200*ms)
            return rate

        obj_func = lambda x: (get_r(x,model,rng_seed)-r_spt)**2

        bound_dict = {'E' : (20,45),
                    'P' : (10,30),
                    'S' : (10,30),
                    'V' : (10,30)}
        bounds = bound_dict[pop_iso]
        res = scipy.optimize.minimize_scalar(obj_func,bounds=bounds,method='Bounded',
                                       options={'maxiter':30,'xatol':0.1})
        return res.x

    def run_density_all(self,n_rnd_target=30):
        with open('data/spiking_backgroundmu'+str(self.version)+'.pkl','rb') as f:
            self.data_mu = pickle.load(f) # load the background input data

        #density_P_plot = np.repeat(np.repeat([0.5,0.75,1,1.25,1.5],5),2)
        #density_S_plot = np.repeat(np.tile([0.5,0.75,1,1.25,1.5],5),2)
        #input_PV_plot = [0,1]*25

        density_P_plot = np.repeat(np.repeat([0.25,0.5,0.75,1,1.25,1.5,1.75],7),2)
        density_S_plot = np.repeat(np.tile([0.25,0.5,0.75,1,1.25,1.5,1.75],7),2)
        input_PV_plot = [0,1]*49

        #density_P_plot = np.repeat([0.2,0.2,1.8,1.8],2)
        #density_S_plot = np.repeat([0.2,1.8,0.2,1.8],2)
        #density_P_plot = np.repeat([0.5,0.5,1.5,1.5],2)
        #density_S_plot = np.repeat([0.5,1.5,0.5,1.5],2)
        #input_PV_plot = [0,1]*4

        for input_PV, density_P, density_S in zip(input_PV_plot, density_P_plot,density_S_plot):
            if (input_PV, density_P, density_S) in self.data.keys():
                data_temp = self.data[(input_PV, density_P, density_S)]
            else:
                data_temp = DataDict()

            n_ran = len(data_temp['rateE']) # number of realizations already ran
            if n_ran < n_rnd_target:
                data0 = self.run_density(input_PV,density_P,density_S,
                                         n_ran,n_rnd_target)
                data_temp.data_merge(data0)

                self.data[(input_PV, density_P, density_S)] = data_temp

                with open('data/'+self.savefile,'wb') as f:
                    pickle.dump(self.data, f)
                with open('data/backup_'+self.savefile,'wb') as f:
                    pickle.dump(self.data, f)

    def run_density(self,input_PV,density_P,density_S,n_ran,n_rnd_target):
        p_orig = self.data['p_orig']
        print '\nPV input {:d}, density P,S ({:0.2f},{:0.2f})'.format(input_PV,density_P,density_S)
        # Extra parameters
        mu_spt_list = self.data_mu[(density_P,density_S)]
        pe = dict()
        for pop in self.pops:
            pe[pop] = dict()
            pe[pop]['mu'] = mu_spt_list[self.pops.index(pop)]*mV

        pe['P']['N'] = int(p_orig['P']['N']*density_P)
        pe['S']['N'] = int(p_orig['S']['N']*density_S)

        pe['P']['mu'] = pe['P']['mu'] + input_PV*self.mu_step # increase if input_PV is 1

        data0 = DataDict()
        model = Model(rng_seed=300,extra_para=pe)
        model.make_model()
        model.add_monitors(record_full=False)
        model.build()
        for i_rnd in range(n_ran,n_rnd_target): # different random seeds
            model.rng_seed = 300+88*i_rnd
            model.reinit()
            net = Network(model)
            net.run(self.runtime*second)
            mon = model.monitor
            p = model.params

            for pop in data0.pops:
                spiketime = np.array([spike[1] for spike in mon['Spike'+pop].spikes]) # get spike timing
                data0['rate'+pop].append(np.sum(spiketime>200*ms)/p[pop]['N']/(model.clock.t-200*ms)) # calculate population rate
                for pop_from in data0.pops_from:
                    data0['g'+pop+pop_from].append(mon['g'+pop+pop_from].mean.mean()/nS)
                    data0['I'+pop+pop_from].append(mon['I'+pop+pop_from].mean.mean()/pA)

            sys.stdout.write('\r')
            sys.stdout.write('Time spent {:0.1f} second. Repeated {:d} times.'.format(time.time()-self.start,i_rnd+1))
            sys.stdout.flush()

        return data0

    def get_Delta(self):
        '''
        Get the value difference between the two input_PV conditions
        '''
        data = self.data
        # Find out all PV/SST density pairs that have both input value
        self.density_P_list = list()# list of plottable density
        self.density_S_list = list()
        data_Delta = DataDict() # dictionary of list
        data0 = DataDict()
        data1 = DataDict()

        key_list = list()
        for key0 in data.keys():
            if isinstance(key0,tuple):
                input_PV, density_P, density_S = key0
                key1 = (1, density_P, density_S)
                if (input_PV == 0) and (key1 in data.keys()): # if the pair exists
                    key_list.append(key0)

        for key0 in key_list:
            input_PV, density_P, density_S = key0
            key1 = (1, density_P, density_S)
            data[key0].to_numpy()
            data[key1].to_numpy()
            self.density_P_list.append(density_P) # update the list
            self.density_S_list.append(density_S)
            for pop in self.pops:
                # Here should compare the ratio, since the starting value are different
                #coef = np.polyfit(data[key0]['rate'+pop],data[key1]['rate'+pop],1)

                #Delta_r = coef[0]
                #Delta_r = np.median(data[key1]['rate'+pop]/data[key0]['rate'+pop])-1

                rname = 'rate'+pop
                Delta_r = delta_method(data[key1][rname],data[key0][rname],'sub')
                data_Delta[rname].append(Delta_r) # update the list
                data0[rname].append(np.mean(data[key0][rname]))
                data1[rname].append(np.mean(data[key1][rname]))

                for pop_from in self.pops_from:
                    gname = 'g'+pop+pop_from
                    Iname = 'I'+pop+pop_from
                    '''
                    if pop_from == 'S':
                        method = 'div'
                    else:
                        method = 'sub'
                    '''
                    method = 'sub'
                    #method = 'div'
                    Delta_g = delta_method(data[key1][gname],data[key0][gname],method)
                    Delta_I = delta_method(data[key1][Iname],data[key0][Iname],method)

                    data_Delta[gname].append(Delta_g)
                    data_Delta[Iname].append(Delta_I)

                    data0[gname].append(np.mean(data[key0][gname]))
                    data1[gname].append(np.mean(data[key1][gname]))

                    data0[Iname].append(np.mean(data[key0][Iname]))
                    data1[Iname].append(np.mean(data[key1][Iname]))

        for key in data_Delta:
            data_Delta[key] = np.array(data_Delta[key])
            data0[key] = np.array(data0[key])
            data1[key] = np.array(data1[key])
        self.data_Delta = data_Delta
        self.data0 = data0
        self.data1 = data1

    def plot_density(self,plot_val='rateE',plot_type='delta'):
        self.get_Delta()
        x = self.density_P_list
        y = self.density_S_list
        if plot_type == 'delta':
            data_plot = self.data_Delta
        elif plot_type == 'before':
            data_plot = self.data0
        elif plot_type == 'after':
            data_plot = self.data1
        else:
            ValueError('Unknown plot type')

        if '+' in plot_val:
            plot_val1, plot_val2 = plot_val.split('+')
            z = data_plot[plot_val1]+data_plot[plot_val2]
        else:
            z = data_plot[plot_val]

        # Divide by input current strength
        z /= (self.mu_step*self.data['p_orig']['P']['g_L']/pA)
        
        extent = (min(x),max(x),min(y),max(y))

        #xi = np.linspace(extent[0],extent[1],100)
        #yi = np.linspace(extent[2],extent[3],100)

        xi = np.unique(self.density_P_list)
        yi = np.unique(self.density_S_list)

        #xi = np.array([0.5,1.0,1.5])
        #yi = np.array([0.5,1.0,1.5])

        zi = griddata(x, y, z, xi, yi, interp='linear')
        print ''
        print zi[::-1,:]

        if plot_type+plot_val in ['deltaIEP','deltaIES']:
            cmap = 'Blues_r'
        elif plot_type+plot_val in ['deltaIEE','deltarateE']:
            cmap = 'Reds'
        else:
            cmap = 'cool'

        fig = plt.figure(figsize=(2.5,2.5))
        ax = fig.add_axes([0.2,0.2,0.65,0.65])
        im = ax.imshow(zi, cmap=cmap,origin='lower',extent=extent,alpha=0.3,interpolation='none')
        CS = ax.contour(zi,5,extent=extent,colors='k',linestyles='solid')
        ax.clabel(CS, fontsize=7, inline=True, fmt = '%0.3f')

        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[2:])
        ax.set_xticks([0.5,1,1.5])


        titles =  {'deltarateE' : 'Effective PV-E Connectivity\n '+r'$M_{\mathrm{PV\rightarrow E}}$',
                   'deltaIEP'  : 'PV-E Current Response\n'+r'$\Delta I_{\mathrm{PV\rightarrow E}}/I_{\mathrm{PV,\mathrm{ext}}}$',
                   'deltaIES'  : 'SST-E Current Response\n'+r'$\Delta I_{\mathrm{SST\rightarrow E}}/I_{\mathrm{PV,\mathrm{ext}}}$',
                   'deltaIEE'  : 'E-E Current Response\n'+r'$\Delta I_{\mathrm{E\rightarrow E}}/I_{\mathrm{PV,\mathrm{ext}}}$'}

        plt.tick_params(axis='both', which='major', labelsize=7)
        ax.set_xlabel(r'PV normalized density $\rho_P$', fontsize=7)

        if plot_type+plot_val in ['deltaIES','deltaIEE']:
            ax.set_yticks([0.5,1,1.5])
            ax.set_yticklabels([])
        else:
            ax.set_yticks([0.5,1,1.5])
            ax.set_ylabel(r'SST normalized density $\rho_S$', fontsize=7)

        if plot_type+plot_val in titles:
            title = titles[plot_type+plot_val]
        else:
            title = plot_type+' '+plot_val
        #ax.set_title(title,fontsize=7)
        #ax.set_title('Effective '+ names[input_type] +' to '+ names[response_type] +' connectivity ',fontsize=7)
        plt.tick_params(axis='both', which='major', labelsize=7)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax,orientation='horizontal')
        ticks = [np.ceil(zi.min()*1e3)*1e-3,np.floor(zi.max()*1e3)*1e-3]
        cb.set_ticks(ticks)
        cb.set_label(title,labelpad=-25,fontsize=7)
        cb.ax.xaxis.set_ticks_position('top')
        #cb.ax.xaxis.set_label_position('top',labelpad=2.5)
        plt.tick_params(axis='both', which='major', labelsize=7)

        figname = 'spiking'+plot_type+plot_val
        plt.savefig('figure/'+figname+'.pdf', transparent=True)


    def plot_comparison(self, plot_val, dpds_list=None):
        plt.figure()
        m = 0
        k = 100
        if dpds_list is None:
            dpds_list = [(0.5,0.5),(0.5,1.5),(1.5,0.5),(1.5,1.5)]
        colors = iter(colormap.rainbow(np.linspace(0, 1, len(dpds_list))))
        fig = plt.figure(figsize=(4.0,4.0))
        ax = fig.add_axes([0.1,0.1,0.5,0.5])
        for (dp,ds) in dpds_list:
            color = next(colors)
            a10 = self.data[(0,dp,ds)][plot_val]
            a20 = self.data[(1,dp,ds)][plot_val]

            #am = a10.mean()*0
            a1 = a10
            a2 = a20
            ax.scatter(a1,a2,label='P {:0.2f}, S {:0.2f}'.format(dp,ds),color=color)

            m0 = np.max((np.max(a1),np.max(a2)))
            k0 = np.max((np.min(a1),np.min(a2)))
            m = np.max((m,m0))
            k = np.min((k,k0))

            k0 = np.min((k0,0))
            m0 = np.max((m0,0))


            coef1 = np.polyfit(a1,a2,1)
            x = np.linspace(k0,m0,10)
            plt.plot(x,coef1[0]*x+coef1[1],color=color)
            #print dp,
            #print ds,
            #print coef1,
            #print np.mean(a20-a10)
        k = np.min((k,0))
        m = np.max((m,0))
        ax.legend(loc=1,bbox_to_anchor=(1.9,1),title='density')
        plt.xlabel('Before PV input')
        plt.ylabel('After PV input')
        plt.plot([k,m],[k,m],'--',color='black')
        plt.title(plot_val)

def run_PSPs(pop_acts=None,g=None):
    pe = dict()
    if g is not None:
        pe['g'] = g
    
    if pop_acts is None:
        pop_acts = ['E','P','S','V']
    for pop_act in pop_acts:
        model = Model(rng_seed=10,extra_para=pe)
        model.make_simple_model(pop_act=pop_act)
        model.add_monitors(record_full=True)
        model.build()
        model.reinit(simple_reinit=True)
        net = Network(model)
        net.run(0.1*second)
        model.PSP_plot()

def vary_gEE_only(mu=17):
    pops = ['E','P','S','V']
    gEE_plot = np.linspace(0,0.7,5)
    rateE = list()
    for gEE in gEE_plot:
        pe = dict()
        pe['conn_pairs'] = [('E','E')]
        pe['g'] = np.zeros((4,4))
        pe['g'][0,0] = gEE
        for pop in pops:
            pe[pop] = dict()
            pe[pop]['V_Tstd'] = 0*mV
        pe['E']['mu'] = mu*mV
        model = Model(rng_seed=10,extra_para=pe,random_conn=False)
        model.make_model()
        model.add_monitors(record_full=True)
        model.build()
        #for pop_act in model.pops:
        #model.make_simple_model(pop_act=pop_act)

        model.reinit()
        net = Network(model)
        net.run(2.0*second)
        #model.PSP_plot()
        #model.raster_plot()

        mon = model.monitor
        p = model.params

        rate = dict()
        for pop in model.pops:
            #print pop,
            spiketime = np.array([spike[1] for spike in mon['Spike'+pop].spikes]) # get spike timing
            rate[pop]=np.sum(spiketime>200*ms)/p[pop]['N']/(model.clock.t-200*ms) # calculate population rate
            #print ''
        print 'gEE = {:0.2f} nS, rateE = {:0.2f}'.format(gEE,rate['E'])
        rateE.append(rate['E'])

    plt.figure()
    plt.plot(gEE_plot,rateE)
    plt.xlabel('gEE (pF)')
    plt.ylabel('rateE (Hz)')

def vary_muext_isopop(pop_iso):
    rng_seed = 300
    pe = dict()
    for pop in pops:
        pe[pop] = dict()
        pe[pop]['r_spt'] = 0
    pe[pop_iso]['N'] = 1000
    pe['conn_pairs'] = []
    model = Model(rng_seed=rng_seed,extra_para=pe,random_conn=False)
    model.isolate_population(pop_iso=pop_iso)
    model.build()
    if pop_iso == 'E':
        mu_isos = np.linspace(10,30,20)
    else:
        mu_isos = np.linspace(5,20,20)
    rate_list = list()
    for mu_iso in mu_isos:
        model.params[pop_iso]['mu'] = mu_iso*mV
        model.rng_seed = rng_seed
        model.reinit()
        net = Network(model)
        net.run(1*second)
        mon = model.monitor
        p = model.params
        pop = pop_iso
        spiketime = np.array([spike[1] for spike in mon['Spike'+pop].spikes])
        rate = np.sum(spiketime>200*ms)/p[pop]['N']/(model.clock.t-200*ms)
        rate_list.append(rate)
    plt.figure()
    plt.plot(mu_isos, rate_list)
    return rate


def sample_run(mus=None,g=None,Np=None,Ns=None,rng_seed=300):
    runtime = 1.0
    pe = dict()
    if g is not None:
        pe['g'] = g
    if mus is None:
        version = 3
        savefile = 'spiking_backgroundmu'+str(version)+'.pkl'
        with open('data/'+savefile,'rb') as f:
            data_mu = pickle.load(f)
        mus = data_mu[(1,1)]
    for pop, mu in zip(['E','P','S','V'],mus):
        pe[pop] = dict()
        pe[pop]['mu'] = mu*mV
    if Np is not None:
        pe['P']['N'] = Np
    if Ns is not None:
        pe['S']['N'] = Ns
    
    model = Model(rng_seed=rng_seed,extra_para=pe,random_conn=False)
    model.make_model()
    model.add_monitors(record_full=True)
    model.build()
    model.rng_seed = rng_seed
    model.reinit()
    net = Network(model)
    
    print '***************Sample Run*********************'
    
    net.run(runtime*second,report='text')
    mon = model.monitor
    p = model.params
    
    fig = plt.figure(figsize=(3,2))
    ax = fig.add_axes([0.2,0.2,0.7,0.7])
    raster_plot(mon['SpikeV'],mon['SpikeS'],mon['SpikeP'],mon['SpikeE'],
                        showgrouplines=True,color='black',markersize=2)
    xlabel('Time (ms)', fontsize=7)
    xlim([0,1000])
    xticks([0,500,1000])
    yticks([3.5,2.5,1.5,0.5],['E','PV','SST','VIP'],rotation=90)
    ylabel('Population', fontsize=7)
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.savefig('figure/spikingraster.pdf')
    
    
    print '\nNumber of neurons',
    print [p[pop]['N'] for pop in model.pops],
    
    print '\nAverage Membrane potential (mV)',
    for pop in model.pops:
        print '{:0.2f}'.format(mon['V'+pop].values.mean()/mV),
    
    print '\nAverage Membrane potential (mV)',
    for pop in model.pops:
        print '{:0.2f}'.format(mon['V'+pop].mean.mean()/mV),
    
    print '\nInput current to E (pA)',
    for pop in model.pops:
        print '{:0.2f}'.format(mon['IE'+pop].values[:,mon['IE'+pop].times>200*ms].mean().mean()/pA),
    
    print '\nInput current to S (pA)',
    for pop in model.pops:
        print '{:0.2f}'.format(mon['IS'+pop].values[:,mon['IS'+pop].times>200*ms].mean().mean()/pA),
    
    print '\nConductance to E (nS)',
    for pop in model.pops:
        print '{:0.2f}'.format(mon['gE'+pop].values[:,mon['gE'+pop].times>200*ms].mean().mean()/nS),
    
    
    print '\nCV',
    for pop in model.pops:
        CV_list = list()
        for key, val in mon['Spike'+pop].spiketimes.iteritems():
            if len(val)>5:
                CV_list.append(CV(val))
        print '{:0.2f}'.format(np.mean(CV_list)),
    
    print '\nMu external',
    for pop in model.pops:
        print '{:0.2f}'.format(p[pop]['mu']/mV),
    
    print '\nsigma external',
    for pop in model.pops:
        print '{:0.2f}'.format(p[pop]['sigma']/mV),
    
    #plt.figure()
    #_ = plt.plot(mon['VE'].values[:10,:].T)
    


A = Analysis(version=4,recover=True)
#A.get_backgroundinput_all(recover=True)
#for n_rnd_target in range(2,53,15):
#    A.run_density_all(n_rnd_target=n_rnd_target)
#A.run_density_all(n_rnd_target=50)
#A.run_density_all(n_rnd_target=50)


#A.run_density_all(n_rnd_target=target)
#A.run_density_all(n_rnd_target=90)
#A.run_density_all(n_rnd_target=100)
#A.run_density_all(n_rnd_target=120)
#A.get_Delta()
#A.plot_density('rateE')
A.plot_density('IEP')
A.plot_density('IES')
A.plot_density('IEE')

# A.plot_density('rateE',plot_type='before')
#A.plot_density('IEPES_contrast')
#dpds_list = [(0.5,0.5),(0.5,1.5)]
dpds_list = None
#A.plot_comparison('rateE',dpds_list = dpds_list)
#A.plot_comparison('IEP',dpds_list = dpds_list)
#A.plot_comparison('IES',dpds_list = dpds_list)

#A.get_backgroundinput_all()



#run_PSPs(pop_acts=['P'])
#vary_gEE_only(mu=26)

# sample_run()

#os.system('say "your program is finished"')

plt.show()