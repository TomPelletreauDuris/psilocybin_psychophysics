# psychophysics_Experiment
Repository for the psychophysics experiment code. 

this file gather both the work of NinaVreugdenhil & TomPelletreauDuris, interns under the supervision of PhD candidate Marco Aqil in the lab of Serge Dumoulin at the SPINOZA center for neuroimaging, Amsterdam, NL

Requirements: psychopy, exptools2, panda, matplotlib, numpy, sklearn, scipy, json, pymc3, bambi.

Python : 3.8.11

# Usage

## To launch an experiment : 

Create setting files named expsettings_*Task*.yml within the Experiment folder. Change *Task* to your actual task name. Run the following line from within the Experiment folder. This particular version is also compatibility with the eyetracker. Eyetracker can be set within the settings file.

```python
python main.py sub-*xxx* ses-*x* task-*NameTask* run-*x* path-to-settings-file
```

Subject SHOULD be specified according the the BIDS convention (sub-001, sub-002 and so on), Task MUST match one of the settings files in the Experiment folder.

Available experiments and relative NameTask, settings file:

```python
Ebbinghaus illusion  EH  expsettings_EH.yml
Center-surround contrast illusion  CS  expsettings_CS.yml
Contrast discrimination CD  expsettings_CD.yml
Delboeuf illusion   DB  expsettings_DB.yml
Altered states of consciousness questionnaire  ASC  expsettings_ASC.yml
Sujective experience questionnaire  SE  expsettings_SE.yml
```

## To make plots : 

- PPViz : a python script aiming to automatically load the data from experiments, fit sigmoid curves, plot subject-wise plots, plot group-wise plots.

PPviz is configured to automatically make the plots with the available data. 

/!\ When run for the first time, be aware that the 'fit_all' function has to have the different fit methods uncommented (as 'CDfit(), CSfit(), etc). These methods create the necessary dataframe for the plot.

in plot_group, uncomment the methods you want to generate the plots. For example :
this_task.EHDBgroupfit(self.out_path)
this_task.EHDBgroupplot(self.out_path)

comment this_task.EHDBgroupfit(self.out_path) if you don't want to re-generate a datafram each time. 

- PPplot_ASC.ipynb : a jupyter notebook using PPViz to plot ASC and SE questionnaires.

- PPplot_psychophysics.ipynb : a jupyter notebook using PPviz to plot the CS and EHDB plots. It also performs the ANOVA, RP ANOVA, MLM and Kruskal-Wallis tests on CS, EHDB, ASC and SE data.

- PPbayes_Bayesian_modeling : a jupyter notebook developing the Bayesian cognitive modelling. It contains a detailed guide to understand, reproduce and develop bayesian models with both PyMC and Bambi. 
