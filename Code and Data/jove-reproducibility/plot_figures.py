"""Five vector figures generated only from the current reproduction run."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter, MaxNLocator

def make_figures(output, annual, capacity, recalibration, paired, policies):
    OUT = Path(output) / 'figures'
    (OUT / 'preview').mkdir(parents=True, exist_ok=True)
    models = ['Logistic', 'SVM', 'Random forest', 'XGBoost']
    A = annual.merge(policies[['horizon','year','model','recall']], on=['horizon','year','model'], validate='one_to_one').rename(columns={'recall':'threshold_recall'})
    C, R = capacity, recalibration
    P = paired.rename(columns={'horizon':'Horizon','comparator':'Comparator','metric':'Metric',
        'difference_favoring_reference':'Difference favoring reference','ci_low':'CI low','ci_high':'CI high'})
    colors = ['#486A9B','#9170A6','#168477','#DC8435']
    markers = ['o','s','^','D']
    ink = '#203449'
    plt.rcParams.update({'font.family':['Arial','DejaVu Sans'],'font.size':18,'axes.titlesize':21,
     'axes.labelsize':18,'xtick.labelsize':17,'ytick.labelsize':17,'legend.fontsize':17,
     'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#8291A0',
     'axes.labelcolor':ink,'text.color':ink,'xtick.color':ink,'ytick.color':ink,
     'axes.linewidth':1,'lines.linewidth':2.6,'lines.markersize':8,
     'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})

    def panel(ax,letter,title):
     ax.set_title(title,loc='left',pad=18)
     ax.text(-.16,1.065,letter,transform=ax.transAxes,fontweight='bold',fontsize=25)
     ax.grid(axis='y',color='#E4E9EF',linewidth=.8); ax.set_axisbelow(True)
    def legend(fig,extra=None):
     handles=[Line2D([],[],color=c,marker=k,label=m) for m,c,k in zip(models,colors,markers)]
     fig.legend(handles=handles if extra is None else extra,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,.012))
    def save(fig,name):
     fig.savefig(OUT/(name+'.svg'))
     fig.savefig(OUT/(name+'.pdf'))
     fig.savefig(OUT/'preview'/(name+'.png'), dpi=150)
     plt.close(fig)

    # Figure 1: method schematic and strictly separated archived-year update schedule.
    fig=plt.figure(figsize=(16,10)); ax=fig.add_axes([.025,.02,.95,.96]); ax.set_axis_off()
    ax.text(.02,.965,'A',fontsize=25,weight='bold'); ax.text(.055,.965,'Reproducible validation workflow',fontsize=23,weight='bold')
    boxes=[(.025,'1  Validate archive','Align firms and outcomes\nCheck probabilities and cutoffs'),(.355,'2  Audit each year','Ranking and probability error\nLocked-rule and capacity checks'),(.685,'3  Update and evaluate','Fit historical intercept\nApply to the next target year')]
    for x,t,b in boxes:
     ax.add_patch(FancyBboxPatch((x,.72),.29,.185,boxstyle='round,pad=0.012,rounding_size=0.015',facecolor='#F0F5F9',edgecolor='#A7B8C9',lw=1.3))
     ax.text(x+.145,.854,t,ha='center',weight='bold',fontsize=20)
     ax.text(x+.145,.772,b,ha='center',va='center',fontsize=17,linespacing=1.5)
    for x in [.325,.655]: ax.annotate('',xy=(x+.025,.81),xytext=(x-.012,.81),arrowprops=dict(arrowstyle='-|>',lw=2,color=ink))
    ax.text(.025,.652,'B',fontsize=25,weight='bold');ax.text(.06,.652,'Expanding historical recalibration sets',fontsize=23,weight='bold')
    xs=np.linspace(.30,.87,5)
    for x,y in zip(xs,range(2020,2025)): ax.text(x,.591,str(y),ha='center',weight='bold',fontsize=19)
    for i,t in enumerate(range(2021,2025)):
     yy=.518-i*.092
     ax.text(.035,yy,f'Evaluate {t}',va='center',fontsize=19)
     for j,x in enumerate(xs):
      yr=2020+j; color='#BFE1D9' if yr<t else '#E6A763' if yr==t else '#F0F2F5'
      ax.add_patch(Rectangle((x-.059,yy-.03),.118,.06,facecolor=color,edgecolor='white',lw=2))
      ax.text(x,yy,'Fit' if yr<t else 'Test' if yr==t else '',ha='center',va='center',fontsize=17)
    ax.text(.03,.136,r'$p^{*}_{i,t}=\mathrm{expit}\,[\mathrm{logit}(p_{i,t})+\alpha_{<t}]$',fontsize=25)
    ax.text(.56,.14,'One intercept per model and horizon',fontsize=18)
    ax.text(.03,.059,'Historical target years only; verify label availability at the intended forecast origin.',fontsize=18,color='#596C7F')
    save(fig,'Figure_1_Protocol')

    # Figure 2: annual AUC and prevalence-referenced Brier skill.
    fig,axs=plt.subplots(2,2,figsize=(15,10));fig.subplots_adjust(left=.10,right=.975,top=.93,bottom=.14,wspace=.27,hspace=.48)
    for col,h in enumerate([1,2]):
     for row,metric in enumerate(['auc','brier_skill']):
      ax=axs[row,col];panel(ax,'ABCD'[row*2+col],f'{h}-year horizon')
      for m,c,mk in zip(models,colors,markers):
       g=A[(A.horizon==h)&(A.model==m)]; ax.plot(g.year,g[metric],color=c,marker=mk)
      ax.set_xticks(range(2020,2025));ax.set_xlim(2019.8,2024.2)
      if row==0: ax.set_ylim(.6,1);ax.set_ylabel('ROC AUC')
      else: ax.axhline(0,color='#7F8C99',ls='--',lw=1.4); ax.set_ylim(-3.3,.6);ax.set_ylabel('Brier skill score');ax.set_xlabel('Target year')
    legend(fig); save(fig,'Figure_2_Temporal_Validation')

    # Figure 3: newly recomputed paired bootstrap intervals.
    fig,axs=plt.subplots(1,2,figsize=(15,8));fig.subplots_adjust(left=.20,right=.965,top=.86,bottom=.18,wspace=.26)
    for j,(metric,title) in enumerate([('auc','AUC difference'),('brier','Brier score improvement')]):
     ax=axs[j];panel(ax,'AB'[j],title)
     labels=[]
     for i,(h,m) in enumerate([(1,'Logistic'),(1,'SVM'),(1,'XGBoost'),(2,'Logistic'),(2,'SVM'),(2,'XGBoost')]):
      r=P[(P.Horizon==h)&(P.Comparator==m)&(P.Metric==metric)].iloc[0]
      mid=r['Difference favoring reference'];lo=r['CI low'];hi=r['CI high']
      row=5-i
      ax.hlines(row,lo,hi,color=colors[2],linewidth=2.6)
      ax.plot([lo,hi],[row,row],linestyle='none',marker='|',markersize=12,color=colors[2])
      ax.plot(mid,row,marker='o' if h==1 else 's',color=colors[2],markersize=10)
      labels.append(f'{h}-year  |  {m}')
     ax.axvline(0,color='#8B98A5',ls='--',lw=1.6);ax.axhline(2.5,color='#CAD3DE',lw=1)
     ax.set_yticks(range(6),labels[::-1] if j==0 else ['']*6);ax.set_ylim(-.65,5.65)
     ax.set_xlabel('RF - comparator' if metric=='auc' else 'Comparator - RF')
     ax.xaxis.set_major_locator(MaxNLocator(5));ax.grid(axis='x',color='#E4E9EF');ax.yaxis.grid(False)
    fig.text(.56,.06,'Positive values favor random forest (RF)',ha='center',fontsize=19)
    save(fig,'Figure_3_Paired_Comparisons')

    # Figure 4: representative XGBoost recalibration.
    fig,axs=plt.subplots(2,2,figsize=(15,10));fig.subplots_adjust(left=.11,right=.975,top=.93,bottom=.14,wspace=.28,hspace=.48)
    for col,h in enumerate([1,2]):
     g=R[(R.horizon==h)&(R.model=='XGBoost')]
     for row in [0,1]:
      ax=axs[row,col];panel(ax,'ABCD'[row*2+col],f'XGBoost | {h}-year horizon')
      if row==0:
       ax.plot(g.year,g.raw_mean,color=colors[3],marker='D');ax.plot(g.year,g.updated_mean,color=colors[2],marker='o')
       ax.plot(g.year,g.prevalence,color=ink,marker='s',ls='--');ax.set_ylabel('Mean probability / event rate');ax.set_ylim(0,.18);ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
      else:
       ax.plot(g.year,g.raw_brier,color=colors[3],marker='D');ax.plot(g.year,g.updated_brier,color=colors[2],marker='o');ax.set_ylabel('Brier score');ax.set_ylim(0,.04);ax.set_xlabel('Target year')
      ax.set_xticks(range(2021,2025));ax.set_xlim(2020.85,2024.15)
    legend(fig,[Line2D([],[],color=colors[3],marker='D',label='Raw'),Line2D([],[],color=colors[2],marker='o',label='Recalibrated'),Line2D([],[],color=ink,ls='--',marker='s',label='Observed event rate')])
    save(fig,'Figure_4_Recalibration')

    # Figure 5: capacity curves and RF policy trajectories.
    fig,axs=plt.subplots(2,2,figsize=(15,10));fig.subplots_adjust(left=.11,right=.975,top=.93,bottom=.15,wspace=.28,hspace=.53)
    for col,h in enumerate([1,2]):
     ax=axs[0,col];panel(ax,'AB'[col],f'{h}-year horizon | capacity')
     for m,c,mk in zip(models,colors,markers):
      g=C[(C.horizon==h)&(C.model==m)].groupby('q').recall.mean();ax.plot(g.index,g.values,color=c,marker=mk)
     ax.set_xticks([.01,.02,.05,.1]);ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.set_xlabel('Review capacity');ax.set_ylabel('Mean annual event capture');ax.set_ylim(0,1);ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
     ax=axs[1,col];panel(ax,'CD'[col],f'{h}-year horizon | RF policies')
     g=C[(C.horizon==h)&(C.model=='Random forest')&(C.q==.05)]
     raw=A[(A.horizon==h)&(A.model=='Random forest')]
     ax.plot(g.year,g.recall,color=colors[2],marker='^',label='Top 5%')
     ax.plot(raw.year,raw.threshold_recall,color=ink,marker='s',ls='--',label='Archived cutoff')
     ax.set_xticks(range(2020,2025));ax.set_xlabel('Target year');ax.set_ylabel('Event capture (recall)');ax.set_ylim(0,1);ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
     ax.legend(loc='upper left',frameon=False,ncol=2,fontsize=15)
    legend(fig);save(fig,'Figure_5_Capacity_Policies')
