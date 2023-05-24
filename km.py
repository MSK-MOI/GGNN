import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


def plot_KM(pred,T,E,name=" ",split='median'):
    pred=pred.cpu().detach().numpy()
    T=T.cpu().detach().numpy()
    E=E.cpu().detach().numpy()
    if split=='mean':
        cut=np.mean(pred)
    elif split=='median':
        cut=np.median(pred)
    
    set=(pred<cut)
    kmf = KaplanMeierFitter()
    plt.cla()
    plt.rcParams.update({'font.size': 15})
    ax = plt.subplot(111)
    kmf.fit(T[set], event_observed=E[set], label="Low risk")
    kmf.plot_survival_function(ax=ax)
    kmf.fit(T[~set], event_observed=E[~set], label="High risk")
    kmf.plot_survival_function(ax=ax)
    
    font = {'family': 'Arial',
        'color':  'black',
        'weight': 'normal',
        'size': 15,
        }
    ax.set_ylabel('Survival probability',fontdict=font)
    ax.set_xlabel('Time (Months)',fontdict=font)
    mytitle="KM "+name 
    plt.ylim(0, 1)
    plt.title(name)
    plt.savefig("figures/"+mytitle+'.png')
    results = logrank_test(T[set], T[~set], event_observed_A=E[set], event_observed_B=E[~set],t_0=60)
    results.print_summary()