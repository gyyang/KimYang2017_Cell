'''2016-04-15
Spiking network model based on Litwin-Kumar 2016

Note on 2017-11-28:
This code was written with Brian 1 and it seems to be no longer compatible
with the latest scipy (v1.0.0)
If you are interested in running this code, it's best to update the code to
Brian 2.
'''

from __future__ import division
import time
import numpy.random
import random as pyrand
import matplotlib.pyplot as plt
import brian_no_units
from brian import *

colors = np.array([ [8,48,107],         # dark-blue
                    [228,26,28],        # red
                    [152,78,163],       # purple
                    [77,175,74]])/255.  # green

def get_dist(original_dist): # Get the distance in periodic boundary conditions
    return np.minimum(abs(original_dist),360-abs(original_dist))

class Model(NetworkOperation):
    def __init__(self, extra_para=dict(), rng_seed=None, random_conn=True):
        if rng_seed is not None:
            pyrand.seed(324823+rng_seed)
            numpy.random.seed(324823+rng_seed)
        self.rng_seed = rng_seed
        self.rng = np.random.RandomState(324823+self.rng_seed)

        self.model_built = False # not built yet

        self.random_conn = random_conn # Random Connections?

        self.pops   = ['E','P','S','V'] # Exc, PV, SST, VIP
        self.pops_from = self.pops + ['B']

        p            = dict()
        # Parameters for Exc
        p['E']       = {'tau_refrac': 2*ms,
                        'V_re'      : -60*mV,
                        'C_m'       : 180*pF,
                        'g_L'       : 6.25*nS,
                        'E_L'       : -60*mV,
                        'Delta_T'   : 1.0*mV,
                        'V_Tmean'   : -40*mV,
                        'V_Tstd'    : 0*mV,
                        'a'         : 4*nS,
                        'b'         : 8*pA,
                        'tau_w'     : 150*ms,
                        'E_E'       : 0*mV,
                        'E_P'       : -67*mV,
                        'E_S'       : -67*mV,
                        'E_V'       : -67*mV,
                        'tau_dE'    : 2.0*ms,
                        'tau_dP'    : 3.0*ms,
                        'tau_dS'    : 4.0*ms,
                        'tau_dV'    : 4.0*ms,
                        'sigma'     : 3.5*mV,
                        'mu'        : 10*mV
                        }

        for pop in ['P','S','V']:
            p[pop] = p['E'].copy()
            p[pop]['C_m'] = 80*pF
            p[pop]['g_L'] = 5.0*nS

        p['P']['g_L'] = 10.0*nS

        # Number of neurons
        p['E']['N'] = 4000
        p['P']['N'] = 500
        p['S']['N'] = 250
        p['V']['N'] = 250

        # Adaptation current & Exponential-integrate-and-fire params
        p['P']['a'] = 0*nS
        p['P']['b'] = 0*pA
        p['P']['Delta_T'] = 0.25*mV

        # w initilization
        w_inits = [62,0,48,35]
        for pop, w_init in zip(self.pops,w_inits):
            p[pop]['w_init'] = w_init*pA

        p['S']['V_Tmean'] = -45*mV
        p['V']['V_Tmean'] = -45*mV

        # Target spontaneous activitity
        r_spts = [0.5, 1.0, 1.5, 0.5]
        V_spts = np.array([-44.88,-46.58,-49.91,-50.79])
        for pop, r_spt, V_spt in zip(self.pops, r_spts, V_spts):
            p[pop]['r_spt'] = r_spt
            p[pop]['V_spt'] = V_spt
        p['V_spts'] = V_spts

        # Other parameters
        p['dt'] = 0.3*ms
        p['record_dt'] = 1.0*ms

        # Connections
        p['p0'] = np.array([[0.05, 0.3, 0.3, 0],
                            [0.2,  0.4, 0.4, 0],
                            [0.2,  0,   0,   0.4],
                            [0.2,  0.1, 0.2, 0]])

        p['p2'] = np.array([[0.8, 0,  0,  0],
                            [0.1, 0,  0,  0],
                            [0,   0,  0,  0],
                            [0,   0,  0,  0]])

        p['conn_pairs'] = [('E','E'),('E','P'),('E','S'),('E','V'),
                           ('P','E'),('P','P'),('P','V'),
                           ('S','E'),('S','P'),('S','V'),
                           ('V','S')] # (pop_from, pop_to)

        p['g'] = np.array([[0.15,   1,    1,  0],
                          [0.7,    1,    1,  0],
                          [0.35,   0,    0,  0.25],
                          [0.35, 0.5, 0.25,  0]]) * nS

        # External parameters
        for key in extra_para:
            if isinstance(extra_para[key],dict):
                for subkey in extra_para[key]:
                    p[key][subkey]=extra_para[key][subkey]
            else:
                p[key] = extra_para[key] # overwrite the old value

        self.params = p

        # Derived parameters
        for pop in self.pops:
            self.params[pop]['V_T'] = p[pop]['V_Tmean']

        self.W = dict()
        for pop_from, pop_to in p['conn_pairs']:
            self.W[pop_to+pop_from] = self.generate_weight_matrix(pop_from, pop_to)

        for pop in self.pops:
            self.params[pop]['tau'] = p[pop]['C_m']/p[pop]['g_L']

        self.simu_clock = Clock(self.params['dt'])
        self.record_clock = Clock(dt=self.params['record_dt'])

        NetworkOperation.__init__(self,clock=self.simu_clock)

        # Equations
        self.eqs = dict()
        self.eqs['E'] = '''
                        dV/dt = sigma*xi/tau**.5  + (-w-g_L*(V-E_L-mu_ext)+g_L*Delta_T*exp((V-V_T)/Delta_T)+IE+IP+IS+IV)/C_m : volt
                        dw/dt = (a*(V-E_L)-w)/tau_w  : pA
                        IE = -gE*(V-E_E) : pA
                        IP = -gP*(V-E_P) : pA
                        IS = -gS*(V-E_S) : pA
                        IV = -gV*(V-E_V) : pA
                        dgE/dt = -gE/tau_dE : nS
                        dgP/dt = -gP/tau_dP : nS
                        dgS/dt = -gS/tau_dS : nS
                        dgV/dt = -gV/tau_dV : nS
                        mu_ext : volt
                        '''

        for pop in ['P','S','V']:
            self.eqs[pop] = self.eqs['E']

        # Brian network components
        self.network = dict()
        self.connection = dict()
        self.monitor = dict()

    def generate_weight_matrix(self, pop_from, pop_to):
        '''
        Generate the weight matrix for the network
        '''
        p = self.params

        n_from = p[pop_from]['N']
        n_to = p[pop_to]['N']

        i_to = self.pops.index(pop_to)
        i_from = self.pops.index(pop_from)

        Theta_to, Theta_from = np.mgrid[0:n_to,0:n_from]
        Theta_to = 2*np.pi*((Theta_to+0.5)/n_to-0.5)
        Theta_from = 2*np.pi*((Theta_from+0.5)/n_from-0.5)

        P = p['p0'][i_to,i_from]*(1+p['p2'][i_to,i_from]*np.cos(get_dist(Theta_to-Theta_from)))
        if self.random_conn:
            W = (self.rng.rand(n_to,n_from)<P)*p['g'][i_to,i_from]
        else:
            W = P*p['g'][i_to,i_from] # If not random connection
        return W

    def make_model(self):
        n = self.network
        p = self.params
        c = self.connection

        # Defining Neuron Group
        for pop in self.pops:
            p0 = p[pop]
            n[pop] = NeuronGroup(p0['N'], Equations(self.eqs[pop], **p0),
                                 threshold=+20*mV, reset=p0['V_re'],
                                 refractory=p0['tau_refrac'], clock=self.simu_clock)
            if pop != 'P': # for spike adaptations
                c['spike2adapt'+pop] = IdentityConnection(n[pop], n[pop], 'w', weight=p0['b'])

        for pop_from, pop_to in p['conn_pairs']:
            cname = pop_to+pop_from
            c[cname] = Connection(n[pop_from],n[pop_to],'g'+pop_from)
            c[cname].connect(n[pop_from],n[pop_to],self.W[pop_to+pop_from].T)

        mon = self.monitor
        # Monitors
        for pop in self.pops:
            mon['Spike'+pop] = SpikeMonitor(n[pop]) # always have spike monitor
            mon['V'+pop] = StateMonitor(n[pop], 'V', record=False, clock=self.record_clock)

    def add_monitors(self, record_full=False):
        '''
        Add many monitors
        '''
        n = self.network
        mon = self.monitor
        # Monitors
        for pop in self.pops:
            mon['V'+pop] = StateMonitor(n[pop], 'V', record=record_full, clock=self.record_clock)
            mon['w'+pop] = StateMonitor(n[pop], 'w', record=record_full, clock=self.record_clock)
            for pop_from in self.pops:
                mon['g'+pop+pop_from] = StateMonitor(n[pop], 'g'+pop_from, record=record_full, clock=self.record_clock)
                mon['I'+pop+pop_from] = StateMonitor(n[pop], 'I'+pop_from, record=record_full, clock=self.record_clock)

    def make_simple_model(self,pop_act='E'):
        '''
        Construct a simple model that for the PSP experiments
        '''
        n = self.network
        p = self.params
        mon = self.monitor

        for pop in self.pops:
            p[pop]['N'] = 2
            p[pop]['mu'] = 0
            p[pop]['sigma'] = 0
            if pop_act == 'E': # See Jiang et al.
                p[pop]['E_L'] = -70*mV
            else:
                p[pop]['E_L'] = -57*mV

        # Connections
        p['p0'] = np.ones((len(self.pops),len(self.pops)))
        p['p2'] = np.zeros((len(self.pops),len(self.pops)))
        
        self.W = dict()
        for pop_from, pop_to in p['conn_pairs']:
            self.W[pop_to+pop_from] = self.generate_weight_matrix(pop_from, pop_to)

        self.make_model()

        p['forced_spike_times'] = np.array([20*ms])
        self.pop_act = pop_act
        @network_operation(when='end',clock=self.simu_clock)
        def forced_spikes():
            if(any(abs(p['forced_spike_times']-self.simu_clock.t)<self.simu_clock.dt/2)):
                n[self.pop_act].V[0] = +30*mV

        self.contained_objects += [forced_spikes]

    def isolate_population(self,pop_iso='E'):
        '''
        Construct an isolated population model
        '''
        n = self.network
        p = self.params
        c = self.connection
        mon = self.monitor

        # Defining Neuron Group
        for pop in self.pops:
            p0 = p[pop]
            if pop == pop_iso:
                n[pop] = NeuronGroup(p0['N'], Equations(self.eqs[pop], **p0),
                                     threshold=+20*mV, reset=p0['V_re'],
                                     refractory=p0['tau_refrac'], clock=self.simu_clock)
            else:
                n[pop] = PoissonGroup(p0['N'], rates=p0['r_spt'], clock=self.simu_clock)

        for pop_from, pop_to in p['conn_pairs']:
            if pop_to == pop_iso:
                cname = pop_to+pop_from
                c[cname] = Connection(n[pop_from],n[pop_to],'g'+pop_from)
                c[cname].connect(n[pop_from],n[pop_to],self.W[pop_to+pop_from].T)

        # Monitors
        for pop in self.pops:
            mon['Spike'+pop] = SpikeMonitor(n[pop])
        mon['V'+pop_iso] = StateMonitor(n[pop_iso], 'V', record=False, clock=self.record_clock)

    def build(self):
        '''
        Build the network
        :return:
        '''
        self.contained_objects += self.network.values()
        self.contained_objects += self.connection.values()
        self.contained_objects += self.monitor.values()
        self.model_built = True

    def reinit(self,simple_reinit=False):
        '''
        Reinitialize the network
        :param simple_reinit:
        :return:
        '''
        n = self.network
        p = self.params
        c = self.connection
        mon = self.monitor

        if not self.model_built:
            ValueError("Model not built yet!!")

        # Reset random seed
        pyrand.seed(324823+self.rng_seed)
        numpy.random.seed(324823+self.rng_seed)
        self.rng = np.random.RandomState(324823+self.rng_seed)

        # Reset all network components
        for g in n.values():
            g.reinit()
        for g in mon.values():
            g.reinit()

        # Add external input
        for pop in self.pops:
            n[pop].mu_ext = p[pop]['mu']

        # Initialization at random voltage
        if simple_reinit:
            for pop in self.pops:
                n[pop].V = p[pop]['E_L']
        else:
            for pop in self.pops:
                #n[pop].V = 0.5*self.rng.rand(p[pop]['N'])*(p[pop]['V_Tmean']-p[pop]['E_L'])+p[pop]['E_L']
                #n[pop].w = (self.rng.rand(p[pop]['N'])*30+20)*pA
                n[pop].V = p[pop]['V_Tmean']-5*mV
                n[pop].w = p[pop]['w_init']

        # Reset
        self.simu_clock.reinit()
        self.record_clock.reinit()

        #print 'Network Reset'

    def raster_plot(self):
        '''
        Plot rasters
        :return:
        '''
        mon = self.monitor
        p = self.params
        plt.figure()
        print 'Rate',
        for pop in self.pops:
            spiketime = np.array([spike[1] for spike in mon['Spike'+pop].spikes])
            print '{:0.2f}'.format(np.sum(spiketime>200*ms)/p[pop]['N']/(self.clock.t-200*ms)),

        raster_plot(mon['SpikeV'],mon['SpikeS'],mon['SpikeP'],mon['SpikeE'],
                    showgrouplines=True)
        xlim([0,1000])

    def PSP_plot(self,plot_type='PSP'):
        '''
        Plot PSPs
        :return:
        '''
        mon = self.monitor
        p = self.params
        pop_act = self.pop_act

        colors = {'E':'blue','P':'green','S':'orange','V':'purple'}
        fig = plt.figure(figsize=(3.0,1.5))
        ax = fig.add_axes([0.2,0.2,0.5,0.7])
        for pop in self.pops:
            if plot_type == 'PSP':
                y = mon['V'+pop].values[1,:]/mV
            elif plot_type == 'PSC':
                y = mon['I'+pop+pop_act].values[1,:]/pA
            ax.plot(mon['V'+pop].times,y,color=colors[pop],label=pop)
        ax.legend(loc=1,bbox_to_anchor=(1.6,1))
        figname = 'figure/'+plot_type+'_act'+pop_act+'.pdf'
        #plt.savefig(figname)
        print 'figure saved at ' + figname

if __name__ == '__main__':
    pass
