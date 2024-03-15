import pandas as pd
import matplotlib.pyplot as pl
import matplotlib.image as mpimg
import matplotlib.widgets as widgets
import matplotlib.lines as mlines
from copy import deepcopy
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import r2_score
from scipy.stats import sem
import os
import yaml
from pathlib import Path
from collections import defaultdict as dd
import json
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.weightstats import DescrStatsW
from datetime import datetime, timedelta

opj = os.path.join

"""
Created on Mon 14 August 2023

@author: TomPelletreauDuris & NinaVreugdenhil
"""

def rp(str):
    sstr = str.replace('_','-')
    return sstr


def getExpAttr(clazz):
    return [name for name, attr in clazz.__dict__.items()
            if not name.startswith("__") 
            and not callable(attr)
            and not type(attr) is staticmethod
            and not type(attr) is str]


class PPviz():

    def __init__(self, log_paths, out_path):

        pl.rcParams.update({'font.size': 22})
        pl.rcParams.update({'pdf.fonttype':42})
        pl.rcParams.update({'figure.max_open_warning': 0})
        pl.rcParams['axes.spines.right'] = False
        pl.rcParams['axes.spines.top'] = False

        self.out_path = out_path
        

        for log_path in [l_p for l_p in log_paths if os.path.isdir(l_p)]:

            id_str = [el.replace('-','_') for el in log_path.split(os.sep)[-1].split('_')]
            
            #---------------------------------------------------------------------------------------------
            #extract the id_str[4] Logs20230502165149 into date and time
            date = id_str[4][4:8] + '-' + id_str[4][8:10] + '-' + id_str[4][10:12]
            time = id_str[4][12:14] + ':' + id_str[4][14:16] + ':' + id_str[4][16:18]
            if id_str[2] == 'task_SE':
                
                if id_str[0] == 'sub_001' and id_str[1] == 'ses_1' and id_str[3] == 'run_1':
                    #set the date and time to the first date and time
                    date = '2023-03-02'
                    time = '09:38:14'
                    print('exception: date and time are not available, using date and time', date, time)
                elif id_str[0] == 'sub_001' and id_str[1] == 'ses_1' and id_str[3] == 'run_2':
                    #set the date and time to the first date and time
                    date = '2023-03-02'
                    time = '12:27:55'
                    print('exception: date and time are not available, using date and time', date, time)
                elif id_str[0] == 'sub_001' and id_str[1] == 'ses_1' and id_str[3] == 'run_3':
                    #set the date and time to the first date and time
                    date = '2023-03-02'
                    time = '13:42:20'
                    print('exception: date and time are not available, using date and time', date, time)
                elif id_str[0] == 'sub_001' and id_str[1] == 'ses_1' and id_str[3] == 'run_4':
                    #set the date and time to the first date and time
                    date = '2023-03-02'
                    time = '14:55:13'
                    print('exception: date and time are not available, using date and time', date, time)

                elif id_str[0] == 'sub_002' and id_str[1] == 'ses_1' and id_str[3] == 'run_1':
                    #set the date and time to the first date and time
                    date = '2023-03-03'
                    time = '13:47:09'
                    print('exception: date and time are not available, using date and time', date, time)
                elif id_str[0] == 'sub_002' and id_str[1] == 'ses_1' and id_str[3] == 'run_2':
                    #set the date and time to the first date and time
                    date = '2023-03-03'
                    time = '16:05:09'
                    print('exception: date and time are not available, using date and time', date, time)
                elif id_str[0] == 'sub_002' and id_str[1] == 'ses_1' and id_str[3] == 'run_3':
                    #set the date and time to the first date and time
                    date = '2023-03-03'
                    time = '19:01:53'
                    print('exception: date and time are not available, using date and time', date, time)

                if time == '::':
                    #check the time from the property of the file itself, modified date
                    time = datetime.fromtimestamp(os.path.getmtime(log_path)).strftime('%H:%M:%S')
                    print('exception: time is not available, using modified time', time)
                # print(date)
                # print(time)
                #-----------------------------------------------------------------------------------------------

            try:
                settings_file = [str(path) for path in Path(log_path).glob('*expsettings.yml')][0]
            
                events_file = [str(path) for path in Path(log_path).glob('*events.tsv')][0]

                order_file = [str(path) for path in Path(log_path).glob('*order.npy')][0]
            except:
                print('no event or settings file found')
                continue

            with open(settings_file) as f:
                settings_dict = yaml.safe_load(f)

            order = np.load(order_file)


            df = pd.read_csv(events_file, sep="\t")
            df_responses = df[df['event_type'] == 'Response'] # The response file with the expected values

            if not hasattr(self, id_str[2]):
                setattr(self, id_str[2], PPviz.Task())

            this_task = getattr(self, id_str[2])

            if not hasattr(this_task, id_str[0]):
                setattr(this_task, id_str[0], PPviz.Subject())

            this_subject = getattr(this_task, id_str[0])          

            if not hasattr(this_subject, id_str[1]):
                setattr(this_subject, id_str[1], PPviz.Session())

            this_session = getattr(this_subject, id_str[1]) 

            if not hasattr(this_session, id_str[3]):
                setattr(this_session, id_str[3], PPviz.Run())
            
            this_run = getattr(this_session, id_str[3])

            this_run.expsettings = settings_dict
            this_run.df_responses = df_responses #The response file with the expected values
            this_run.log_path = log_path
            this_run.order = order
            this_run.time = time

            os.makedirs(opj(self.out_path,rp(id_str[0]),rp(id_str[1])),exist_ok=True)

            this_run.out_path = opj(self.out_path,rp(id_str[0]),rp(id_str[1]))

            if 'sub' not in this_run.expsettings or this_run.expsettings['sub'] != id_str[0]:
                this_run.expsettings['sub'] = id_str[0]
                #print(f'warning: subject in filename and expsettings {settings_file} are different. using filename')
                
            if 'ses' not in this_run.expsettings or this_run.expsettings['ses'] != id_str[1]:
                this_run.expsettings['ses'] = id_str[1]
                #print(f'warning: session in filename and expsettings {settings_file} are different. using filename')

            if 'task' not in this_run.expsettings or this_run.expsettings['task'] != id_str[2]:
                this_run.expsettings['task'] = id_str[2]
                #print(f'warning: task in filename and expsettings {settings_file} are different. using filename')

            if 'run' not in this_run.expsettings or this_run.expsettings['run'] != id_str[3]:
                this_run.expsettings['run'] = id_str[3]
                #print(f'warning: task in filename and expsettings {settings_file} are different. using filename')               

# The warning message you’re seeing is because the exponential function in your lambda function is overflowing. 
# This can happen when the input to the exponential function is too large or too small. 
# You might want to try scaling your input data before fitting the logistic function.
    def fit_sigmoid_curve(xdata,ydata):
        x0s = [[7.5,15],[25,50],[10,50],[40,50],[5,15],[10,15],[4,10],[22,30]]
        curr_min_res = np.inf

        for x0 in x0s:
            try:
                res = minimize(lambda x,a,y:-r2_score(y,1/(np.exp(x[0]-x[1]*a)+1),sample_weight=np.ones_like(y)), x0=x0,
                                args=(xdata,ydata),
                                method='Powell', options={'ftol':1e-3, 'xtol':1e-3})
                if res['fun']<curr_min_res:
                    exp_res = deepcopy(res)
                    curr_min_res = deepcopy(res['fun'])
            except Exception as e:
                print(e)
                x0s.append(10*np.random.rand(2))
                pass

        exp_pred = 1/(np.exp(exp_res['x'][0]-exp_res['x'][1]*np.linspace(xdata.min(),xdata.max(),100))+1)

        # Calculate R²
        r_squared = r2_score(ydata, 1 / (np.exp(exp_res['x'][0] - exp_res['x'][1] * xdata) + 1))

        return exp_res, exp_pred, r_squared
    
    def fit_shifted_sigmoid_curve(xdata,ydata):
        x0s = [[7.5,15],[25,50],[10,50],[40,50],[5,15],[10,15],[4,10], [22,30]]
        curr_min_res = np.inf

        #0.5*len(y)*np.log(1-r2_score(y, 0.75/(np.exp(x[0]-x[1]*a)+1)+0.25,sample_weight=np.ones_like(y))),

        for x0 in x0s:
            try:
                res = minimize(lambda x,a,y:-r2_score(y, 0.75/(np.exp(x[0]-x[1]*a)+1)+0.25,sample_weight=np.ones_like(y)), x0=x0,
                                args=(xdata,ydata),
                                method='Powell', options={'ftol':1e-3, 'xtol':1e-3})
                if res['fun']<curr_min_res:
                    exp_res = deepcopy(res)
                    curr_min_res = deepcopy(res['fun'])
            except Exception as e:
                print(e)
                x0s.append(10*np.random.rand(2))
                pass

        exp_pred = 0.25+0.75/(np.exp(exp_res['x'][0]-exp_res['x'][1]*np.linspace(xdata.min(),xdata.max(),100))+1)

        return exp_res, exp_pred

    def average_time(time_strings):
        # Convert time strings to timedelta objects
        timedelta_list = [datetime.strptime(time_str, '%H:%M:%S').time() for time_str in time_strings]

        # Calculate the average timedelta
        average_timedelta = sum((datetime.combine(datetime.today(), dt) - datetime.combine(datetime.today(), datetime.min.time()) for dt in timedelta_list), timedelta()) / len(timedelta_list)
        
        #select only HH:MM:SS from average_timedelta and convert it to string 
        average_timedelta = str(average_timedelta).split('.')[0]
        return average_timedelta
    
    def subtract_45min(time_string):
        # Convert time string to timedelta object
        time_object = datetime.strptime(time_string, '%H:%M:%S').time()
        
        # Subtract 45 minutes
        adjusted_time = datetime.combine(datetime.today(), time_object) - timedelta(minutes=45)
        
        # Format the adjusted time as a string
        adjusted_time_string = adjusted_time.strftime('%H:%M:%S')
        
        return adjusted_time_string

    def giveDeltaTime(time_string1, time_string2):
        # Convert time strings to datetime objects
        datetime1 = datetime.strptime(time_string1, '%H:%M:%S')
        datetime2 = datetime.strptime(time_string2, '%H:%M:%S')
        
        # Calculate the time difference
        timedelta = datetime2 - datetime1
        
        # Extract the time difference as a string
        delta_time_string = str(timedelta).split('.')[0]
        
        return delta_time_string

    def roundTime(time_string):
        # Convert time string to datetime object
        datetime_object = datetime.strptime(time_string, '%H:%M:%S')
        
        # Round the time to the nearest 30 minutes
        rounded_time = datetime_object + timedelta(minutes=15)
        rounded_time -= timedelta(minutes=rounded_time.minute % 30, seconds=rounded_time.second)

        # Format the rounded time as a string
        rounded_time_string = rounded_time.strftime('%H:%M:%S')

        return rounded_time_string

    def calculateTimeDistance(time_string1, time_string2):
        # Convert time strings to datetime objects
        datetime1 = datetime.strptime(time_string1, '%H:%M:%S')
        datetime2 = datetime.strptime(time_string2, '%H:%M:%S')
        
        # Calculate the time difference
        timedelta = datetime2 - datetime1
        
        # Extract the time difference as a string
        delta_time_string = str(timedelta).split('.')[0]
        
        # Extract the hours, minutes, and seconds from the time difference string
        hours = int(delta_time_string.split(':')[0])
        minutes = int(delta_time_string.split(':')[1])
        seconds = int(delta_time_string.split(':')[2])
        
        # Calculate the time distance in minutes
        time_distance = hours * 60 + minutes + seconds / 60

        # 30 min should be 0.5 in distance
        time_distance = time_distance / 30
        
        return time_distance

    def fit_all(self):
        print("fitting data...")
        for task in getExpAttr(self):
            for subject in getExpAttr(getattr(self,task)):
                for session in getExpAttr(getattr(getattr(self,task),subject)):
                    for run in getExpAttr(getattr(getattr(getattr(self,task),subject),session)):

                        this_run = getattr(getattr(getattr(getattr(self,task),subject),session),run)

                        if 'CD' in task:
                            # this_run.CDfit()
                            print("CD fit finished.")
                        elif 'CS' in task:
                            this_run.CSfit()
                            print("CS fit finished.")
                        elif 'DB' in task or 'EH' in task:
                            # this_run.EHDBfit()
                            print("EHDB fit finished.") 
                        elif 'ASC' in task:
                            # this_run.ASCfit()
                            print("ASC fit finished.")
                        elif 'SE' in task:
                            #this_run.SEfit()
                            print("SE fit finished.")

    def plot_all(self):
        print("plotting data...")
        for task in getExpAttr(self):
            this_task = getattr(self,task)

            for subject in getExpAttr(this_task):
                this_subject = getattr(this_task,subject)
                for session in getExpAttr(this_subject):
                    this_session = getattr(this_subject,session)
                    for run in getExpAttr(this_session):
                        this_run = getattr(this_session, run)

                        if 'CD' in task:
                            this_run.CDplot()
                            print("CD plot finished.")
                        elif 'CS' in task:
                            this_run.CSplot()
                            print("CS plot finished.")
                        elif 'DB' in task or 'EH' in task:
                            this_run.EHDBplot()
                            print("EHDB plot finished.")
                        elif 'ASC' in task:
                            this_run.ASCplot_ses()
                            print("ASC plot finished.")
        print("done")   

    def plot_group(self):
        for task in getExpAttr(self):
            this_task = getattr(self,task)
            
            if 'CD' in task:   
                # this_task.CDgroupfit(self.out_path)            
                # this_task.CDgroupplot(self.out_path)
                print('CD group plot finished')
            elif 'CS' in task:
                this_task.CSgroupfit(self.out_path)
                # this_task.CSgroupplot(self.out_path)
                # this_task.CSgroupplot_placebo(self.out_path)
                # this_task.CSgroupplot_5mg(self.out_path)
                # this_task.CSgroupplot_10mg(self.out_path)
                # this_task.CSgroupplot_average(self.out_path)
                # this_task.CSgroupplot_average2(self.out_path)
                # this_task.CSgroupplot_average3(self.out_path)
                # this_task.CSgroupplot_average_slope(self.out_path)
                # this_task.CSgroupplot_average_r2(self.out_path)
                print("CS group plot finished")
            elif 'DB' in task or 'EH' in task:
                # this_task.EHDBgroupfit(self.out_path)
                # this_task.EHDBgroupplot(self.out_path)
                # this_task.EHDBgroupplot_placebo(self.out_path)
                # this_task.EHDBgroupplot_5mg(self.out_path)
                # this_task.EHDBgroupplot_10mg(self.out_path)
                # this_task.EHDBgroupplot_average(self.out_path)
                # this_task.EHDBgroupplot_average2(self.out_path)
                # this_task.EHDBgroupplot_average3(self.out_path)
                # this_task.EHDBgroupplot_average_slope(self.out_path)
                # this_task.EHDBgroupplot_average_r2(self.out_path)
                print("EHDB group plot finsihed")
                # this_task.EHDB_CSgroupfit(self.out_path)
                # this_task.EHDB_CSgroupplot_average3(self.out_path)
                print("EHDB_CS group plot finsihed")
                # this_task.Average_EHDB_CSgroupfit(self.out_path)
                # this_task.Average_EHDB_CSgroupplot_average3(self.out_path)
                # this_task.Average_EHDB_CSgroupplot_average_slope(self.out_path)
                # this_task.Weigthed_average_EHDB_CSgroupfit(self.out_path)
                # this_task.Weigthed_average_EHDB_CSgroupplot_average_slope(self.out_path)
                # this_task.Weigthed_average_EHDB_CSgroupplot_average_r2(self.out_path)
                print("Average_EHDB_CS group plot finsihed") 
            elif 'ASC' in task:
                #----------------------------------------
                #Activate ASCgroupfit only if you want to
                #fit the data again and fit_all() is used
                #----------------------------------------
                # this_task.ASCgroupfit(self.out_path)
                # this_task.ASCgroupplot(self.out_path)
                # this_task.ASCaverageplot(self.out_path)
                # this_task.ASCaverageplot2(self.out_path)
                # this_task.ASCcategories_5D(self.out_path)
                # this_task.ASCcategories_11D(self.out_path)
                print("ASC group plot finished.")
            elif "SE" in task:
                #---------------------------------------------
                #Activate SEgroupfit for same reasons as above
                #---------------------------------------------
                # this_task.SEgroupfit(self.out_path)
                # this_task.SEgroupplot(self.out_path)
                # this_task.SEaverageplot(self.out_path)
                print("SE group plot finished.")


    class Task():
        def __init__(self):
            pass

        def CDgroupplot(self, out_path):

            preds = dd(list)
            data = dd(list)

            for su, subject in enumerate(getExpAttr(self)):
                this_subject = getattr(self,subject)
                for se,session in enumerate(getExpAttr(this_subject)):
                    this_session = getattr(this_subject,session)
                    for rr,run in enumerate(getExpAttr(this_session)):
                        this_run = getattr(this_session, run)

                        if [su,se,rr] == [0,0,0]:
                            fig, ax = pl.subplots(1,1+len(this_run.sf_values),figsize=(8+8*len(this_run.sf_values),8))
                            fig.suptitle(f"Contrast discrimination task")

                        for ii, sf_value in enumerate(this_run.sf_values):

                            ax[ii].plot(this_run.x_space,this_run.preds[str(sf_value)], c='k', lw=0.8, alpha=0.8)
                            ax[ii].plot(this_run.contrast_values,this_run.probs[str(sf_value)], marker='^', ls='', c='grey', ms=4, alpha=0.8)

                            preds[str(sf_value)].append(this_run.preds[str(sf_value)])
                            data[str(sf_value)].append(this_run.probs[str(sf_value)])

            color = ['red','blue']
            for ii, sf_value in enumerate(this_run.sf_values):
                ax[ii].set_title(f"Sf {sf_value:.1f} c/deg")
                ax[ii].set_xscale('log')
                ax[ii].set_xlabel('Target contrast')
                ax[ii].set_ylabel('Proportion correct responses')

                ax[ii].plot(this_run.x_space,np.ones_like(this_run.x_space)*0.25, ls='--', label = 'Chance', c='k', alpha=0.5)
                ax[ii].plot(this_run.x_space,np.ones_like(this_run.x_space), ls='--', label = 'Ceiling', c='green', alpha=0.5)

                #ax[ii].errorbar(this_run.contrast_values, np.mean(data[str(sf_value)],axis=0), yerr=sem(data[str(sf_value)],axis=0), marker='s', ls='', c='k', ms=6)
                ax[ii].plot(this_run.x_space, np.mean(preds[str(sf_value)],axis=0), c='k', lw=4)

                
                ax[-1].set_xscale('log')
                ax[-1].set_xlabel('Target contrast')
                ax[-1].set_ylabel('Proportion correct responses')

                ax[-1].plot(this_run.x_space,np.ones_like(this_run.x_space)*0.25, ls='--',  c='k', alpha=0.5)
                ax[-1].plot(this_run.x_space,np.ones_like(this_run.x_space), ls='--', c='green', alpha=0.5)

                ax[-1].errorbar(this_run.contrast_values, np.mean(data[str(sf_value)],axis=0), yerr=sem(data[str(sf_value)],axis=0), mec='k', marker='s', ls='', c=color[ii], ms=6)
                ax[-1].plot(this_run.x_space, np.mean(preds[str(sf_value)],axis=0),c=color[ii], lw=2, label=f"Sf {sf_value:.1f} c/deg")
                ax[-1].legend(loc='lower right')


            fig.savefig(opj(out_path,'group_results_CD.pdf'), dpi=600, bbox_inches='tight', transparent=True)

        def CDgroupfit(self, out_path):
            groups_dict = {'placebo':['sub-001_ses-1','sub-002_ses-2','sub-003_ses-3','sub-004_ses-2','sub-005_ses-1','sub-006_ses-3','sub-007_ses-3','sub-008_ses-1',
                         'sub-009_ses-2','sub-010_ses-1','sub-011_ses-3','sub-012_ses-2','sub-013_ses-3','sub-014_ses-2','sub-015_ses-2','sub-018_ses-1','sub-019_ses-2','sub-020_ses-1'],

              '5mg':['sub-001_ses-3','sub-002_ses-1','sub-003_ses-1','sub-004_ses-3','sub-005_ses-2','sub-006_ses-1','sub-007_ses-1','sub-008_ses-3',
                    'sub-009_ses-1','sub-010_ses-2','sub-011_ses-2','sub-012_ses-3','sub-013_ses-2','sub-014_ses-1','sub-015_ses-3','sub-018_ses-3','sub-019_ses-1','sub-020_ses-2'],

              '10mg':['sub-001_ses-2','sub-002_ses-3','sub-003_ses-2','sub-004_ses-1','sub-005_ses-3','sub-006_ses-2','sub-007_ses-2','sub-008_ses-2',
                     'sub-009_ses-3','sub-010_ses-3','sub-011_ses-1','sub-012_ses-1','sub-013_ses-1','sub-014_ses-3','sub-015_ses-1','sub-018_ses-2','sub-019_ses-3','sub-020_ses-3']}


            df = pd.DataFrame()
            CD_data = {}

            for subject in getExpAttr(self):
                this_subject = getattr(self, subject)
                participant_id = rp(subject).split('_')[0]
                CD_data[participant_id] = {}
                for session in getExpAttr(this_subject):
                    this_session = getattr(this_subject, session)
                    session_id = rp(session).split('_')[0]
                    CD_data[participant_id][session_id] = {}
                    for run in getExpAttr(this_session):
                        this_run = getattr(this_session, run)

                        group_subject_session = ''
                        for group in groups_dict:
                            if f"{participant_id}_{session_id}" in groups_dict[group]:
                                group_subject_session = group
                                break

                        # Modify the following lines to extract Ebbinghaus-specific data
                        # Replace 'contrast_values', 'probs', and other attributes with Ebbinghaus data
                        Cdata = {
                            'group_type': group_subject_session,
                            'contrast_values': this_run.contrast_values,
                            'probs': this_run.probs,
                            'preds': this_run.preds,
                            'x_space': this_run.x_space,                            
                        }
                        

                        print('dif')
                        print(participant_id, session_id, group_subject_session)
                        print( this_run.probs.keys())
                        key1 = list(this_run.probs.keys())[0]
                        key2 = list(this_run.probs.keys())[1]                       

                        new = pd.DataFrame({'Subject':subject, 'Session':session,'Group_type':group_subject_session, 'rel_effsize_difference':  [this_run.probs[key1]- this_run.probs[key2]]}, index=[0])
                        df = pd.concat([df, new], ignore_index=True)
                        CD_data[participant_id][session_id] = Cdata

            # Save Ebbinghaus-specific data as a numpy file
            np.save(opj(out_path,'data/CD_data.npy'), CD_data)

            # Save the dataframe
            df.to_csv(opj(out_path,'data/group_results_CD.csv'), index=False)

        def CSgroupplot_placebo(self,out_path):
            CS_data = np.load(opj(out_path,'data/CS_data.npy'), allow_pickle=True).item()

            fig, ax = pl.subplots(1,3,figsize=(24,8))
            fig.suptitle(f"Center-Surround task - placebo")

            preds_placebo = []
            preds_nosurr_placebo = []
            rel_effsize = []
            rel_effsize_nosurr =[]
            subject_name = []

            for participant, sessions in CS_data.items():
                for session, data_ in sessions.items():
                    #is session 0 do nothing
                    if session == 'ses-0':
                        continue
                    else :
                        
                        if data_['group'] == 'placebo':
                            ax[0].set_title('Surround')
                            ax[1].set_title('No surround')
                            ax[0].plot(data_['x_space'],data_['full_pred'], c='grey', lw=0.5, alpha=0.5)
                            ax[0].plot(data_['contrast_values'],data_['probs'], marker='^', ls='', c='grey', ms=4, alpha=0.8)
                            ax[1].plot(data_['x_space_nosurr'],data_['full_pred_nosurr'], c='grey', lw=0.8, alpha=0.8)
                            ax[1].plot(data_['contrast_values_nosurr'],data_['probs_nosurr'], marker='v', ls='', c='grey', ms=4, alpha=0.8)       
                            preds_placebo.append(data_['full_pred'])
                            preds_nosurr_placebo.append(data_['full_pred_nosurr'])
                            rel_effsize.append(data_['rel_effsize'])
                            rel_effsize_nosurr.append(data_['rel_effsize_nosurr'])

                            subject_name.append(participant)

            for a in ax[:-1]:
                a.set_xlabel('Reference contrast')
                a.set_xlabel('Contrast difference (%RMS)')

                a.plot(CS_data['sub-002']['ses-1']['x_space_comb'],np.ones_like(CS_data['sub-002']['ses-1']['x_space_comb'])*0.5, ls='--', c='k', alpha=0.5, label='PSE')
                a.plot(CS_data['sub-002']['ses-1']['x_space_comb'],np.ones_like(CS_data['sub-002']['ses-1']['x_space_comb']), ls='--',  c='green', alpha=0.5)
                a.plot(CS_data['sub-002']['ses-1']['x_space_comb'],np.zeros_like(CS_data['sub-002']['ses-1']['x_space_comb']), ls='--', c='red', alpha=0.5)
                a.plot(CS_data['sub-002']['ses-1']['target_contrast']*np.ones(100),np.linspace(0,1,100), ls='-', c='k', label='Veridical', alpha=0.5)

                a.set_xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7])
                a.set_xticklabels(['-75','-50','-25','0','+25','+50','+75'])

                a.legend(loc='lower right')

                a.set_ylabel('Prob ref perceived as higher contrast')    

            #ax[0].errorbar(this_run.contrast_values, np.mean(data,axis=0), yerr=sem(data,axis=0), marker='s', ls='', c='k', ms=6)
            ax[0].plot(CS_data['sub-002']['ses-1']['x_space'], np.mean(preds_placebo,axis=0), linewidth=4, color='black')

            #ax[1].errorbar(this_run.contrast_values_nosurr, np.mean(data_nosurr,axis=0), yerr=sem(data_nosurr,axis=0), marker='s', ls='', c='k', ms=6)
            ax[1].plot(CS_data['sub-002']['ses-1']['x_space_nosurr'], np.mean(preds_nosurr_placebo,axis=0), linewidth=4, color='black')


            for ii, effsize in enumerate([rel_effsize,rel_effsize_nosurr]):
                ax[-1].bar(ii, np.mean(effsize))

                ax[-1].errorbar(ii*np.ones_like(effsize), effsize, marker='s', mec='k', ls='', ms=8)

            for ss in range(len(rel_effsize)):
                ax[-1].plot(np.arange(2), [rel_effsize[ss], rel_effsize_nosurr[ss]], c='k')           
                
            ax[-1].plot(np.linspace(-1,2,10),np.zeros_like(np.linspace(-1,2,10)), ls='--', c='k', alpha=0.5)
            ax[-1].set_ylim([-90,15])
            ax[-1].set_xlim([-0.5,1.5])
                
            
            ax[-1].set_xticks([0,1])
            ax[-1].set_xticklabels(["Surround", "no surround"])
            ax[-1].set_xlabel('')
            ax[-1].set_ylabel('PSE difference (%RMS)')


            fig.savefig(opj(out_path,'group_results_CS_placebo.pdf'), dpi=600, bbox_inches='tight', transparent=True)
        
        def CSgroupplot_5mg(self,out_path):
            CS_data = np.load(opj(out_path,'data/CS_data.npy'), allow_pickle=True).item()

            fig, ax = pl.subplots(1,3,figsize=(24,8))
            #ax[1].set_yticklabels([])

            fig.suptitle(f"Center-Surround task - 5mg")

           
            preds_5mg = []
            preds_nosurr_5mg = []
            rel_effsize = []
            rel_effsize_nosurr =[]
            subject_name = []

            for participant, sessions in CS_data.items():
                for session, data_ in sessions.items():
                    #is session 0 do nothing
                    if session == 'ses-0':
                        continue
                    else :
                        if data_['group'] == '5mg':
                            ax[0].set_title('Surround')
                            ax[1].set_title('No surround')
                            ax[0].plot(data_['x_space'],data_['full_pred'], c='grey', lw=0.5, alpha=0.5)
                            ax[0].plot(data_['contrast_values'],data_['probs'], marker='^', ls='', c='grey', ms=4, alpha=0.8)
                            ax[1].plot(data_['x_space_nosurr'],data_['full_pred_nosurr'], c='grey', lw=0.8, alpha=0.8)
                            ax[1].plot(data_['contrast_values_nosurr'],data_['probs_nosurr'], marker='v', ls='', c='grey', ms=4, alpha=0.8)       
                            preds_5mg.append(data_['full_pred'])
                            preds_nosurr_5mg.append(data_['full_pred_nosurr'])
                            rel_effsize.append(data_['rel_effsize'])
                            rel_effsize_nosurr.append(data_['rel_effsize_nosurr'])

                            subject_name.append(participant)

            for a in ax[:-1]:
                a.set_xlabel('Reference contrast')
                a.set_xlabel('Contrast difference (%RMS)')

                a.plot(CS_data['sub-002']['ses-1']['x_space_comb'],np.ones_like(CS_data['sub-002']['ses-1']['x_space_comb'])*0.5, ls='--', c='k', alpha=0.5, label='PSE')
                a.plot(CS_data['sub-002']['ses-1']['x_space_comb'],np.ones_like(CS_data['sub-002']['ses-1']['x_space_comb']), ls='--',  c='green', alpha=0.5)
                a.plot(CS_data['sub-002']['ses-1']['x_space_comb'],np.zeros_like(CS_data['sub-002']['ses-1']['x_space_comb']), ls='--', c='red', alpha=0.5)
                a.plot(CS_data['sub-002']['ses-1']['target_contrast']*np.ones(100),np.linspace(0,1,100), ls='-', c='k', label='Veridical', alpha=0.5)

                a.set_xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7])
                a.set_xticklabels(['-75','-50','-25','0','+25','+50','+75'])

                a.legend(loc='lower right')

                a.set_ylabel('Prob ref perceived as higher contrast')

            #ax[0].errorbar(this_run.contrast_values, np.mean(data,axis=0), yerr=sem(data,axis=0), marker='s', ls='', c='k', ms=6)
            ax[0].plot(CS_data['sub-002']['ses-1']['x_space'], np.mean(preds_5mg,axis=0), linewidth=4, color='black')

            #ax[1].errorbar(this_run.contrast_values_nosurr, np.mean(data_nosurr,axis=0), yerr=sem(data_nosurr,axis=0), marker='s', ls='', c='k', ms=6)
            ax[1].plot(CS_data['sub-002']['ses-1']['x_space_nosurr'], np.mean(preds_nosurr_5mg,axis=0), linewidth=4, color='black')
            



            for ii, effsize in enumerate([rel_effsize,rel_effsize_nosurr]):
                ax[-1].bar(ii, np.mean(effsize))

                ax[-1].errorbar(ii*np.ones_like(effsize), effsize, marker='s', mec='k', ls='', ms=8)

            for ss in range(len(rel_effsize)):
                ax[-1].plot(np.arange(2), [rel_effsize[ss], rel_effsize_nosurr[ss]], c='k')           
                
            ax[-1].plot(np.linspace(-1,2,10),np.zeros_like(np.linspace(-1,2,10)), ls='--', c='k', alpha=0.5)
                
            
            ax[-1].set_xticks([0,1])
            ax[-1].set_xticklabels(["Surround", "no surround"])
            ax[-1].set_xlabel('')
            ax[-1].set_ylabel('PSE difference (%RMS)')
            ax[-1].set_ylim([-90,15])
            ax[-1].set_xlim([-0.5,1.5])


            fig.savefig(opj(out_path,'group_results_CS_5mg.pdf'), dpi=600, bbox_inches='tight', transparent=True)

        def CSgroupplot_10mg(self,out_path):
            CS_data = np.load(opj(out_path,'data/CS_data.npy'), allow_pickle=True).item()

            fig, ax = pl.subplots(1,3,figsize=(24,8))
            #ax[1].set_yticklabels([])

            fig.suptitle(f"Center-Surround task - 10mg")

            preds_10mg = []
            preds_nosurr_10mg = []
            rel_effsize = []
            rel_effsize_nosurr =[]
            subject_name = []

            for participant, sessions in CS_data.items():
                for session, data_ in sessions.items():
                    #is session 0 do nothing
                    if session == 'ses-0':
                        continue
                    else :
                        
                        if data_['group'] == '10mg':
                            ax[0].set_title('Surround')
                            ax[1].set_title('No surround')
                            ax[0].plot(data_['x_space'],data_['full_pred'], c='grey', lw=0.5, alpha=0.5)
                            ax[0].plot(data_['contrast_values'],data_['probs'], marker='^', ls='', c='grey', ms=4, alpha=0.8)
                            ax[1].plot(data_['x_space_nosurr'],data_['full_pred_nosurr'], c='grey', lw=0.8, alpha=0.8)
                            ax[1].plot(data_['contrast_values_nosurr'],data_['probs_nosurr'], marker='v', ls='', c='grey', ms=4, alpha=0.8)       
                            preds_10mg.append(data_['full_pred'])
                            preds_nosurr_10mg.append(data_['full_pred_nosurr'])
                            rel_effsize.append(data_['rel_effsize'])
                            rel_effsize_nosurr.append(data_['rel_effsize_nosurr'])

                            subject_name.append(participant)

            for a in ax[:-1]:
                a.set_xlabel('Reference contrast')
                a.set_xlabel('Contrast difference (%RMS)')

                a.plot(CS_data['sub-002']['ses-1']['x_space_comb'],np.ones_like(CS_data['sub-002']['ses-1']['x_space_comb'])*0.5, ls='--', c='k', alpha=0.5, label='PSE')
                a.plot(CS_data['sub-002']['ses-1']['x_space_comb'],np.ones_like(CS_data['sub-002']['ses-1']['x_space_comb']), ls='--',  c='green', alpha=0.5)
                a.plot(CS_data['sub-002']['ses-1']['x_space_comb'],np.zeros_like(CS_data['sub-002']['ses-1']['x_space_comb']), ls='--', c='red', alpha=0.5)
                a.plot(CS_data['sub-002']['ses-1']['target_contrast']*np.ones(100),np.linspace(0,1,100), ls='-', c='k', label='Veridical', alpha=0.5)

                a.set_xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7])
                a.set_xticklabels(['-75','-50','-25','0','+25','+50','+75'])

                a.legend(loc='lower right')

                a.set_ylabel('Prob ref perceived as higher contrast')    

            #ax[0].errorbar(this_run.contrast_values, np.mean(data,axis=0), yerr=sem(data,axis=0), marker='s', ls='', c='k', ms=6)
            ax[0].plot(CS_data['sub-002']['ses-1']['x_space'], np.mean(preds_10mg,axis=0), linewidth=4, color='black')

            #ax[1].errorbar(this_run.contrast_values_nosurr, np.mean(data_nosurr,axis=0), yerr=sem(data_nosurr,axis=0), marker='s', ls='', c='k', ms=6)
            ax[1].plot(CS_data['sub-002']['ses-1']['x_space_nosurr'], np.mean(preds_nosurr_10mg,axis=0), linewidth=4, color='black')   

            
            for ii, effsize in enumerate([rel_effsize,rel_effsize_nosurr]):
                ax[-1].bar(ii, np.mean(effsize))

                ax[-1].errorbar(ii*np.ones_like(effsize), effsize, marker='s', mec='k', ls='', ms=8)

            for ss in range(len(rel_effsize)):
                ax[-1].plot(np.arange(2), [rel_effsize[ss], rel_effsize_nosurr[ss]], c='k')           
                
            ax[-1].plot(np.linspace(-1,2,10),np.zeros_like(np.linspace(-1,2,10)), ls='--', c='k', alpha=0.5)
            ax[-1].set_ylim([-90,15])
            ax[-1].set_xlim([-0.5,1.5])
            ax[-1].set_xticks([0,1])
            ax[-1].set_xticklabels(["Surround", "no surround"])
            ax[-1].set_xlabel('')
            ax[-1].set_ylabel('PSE difference (%RMS)')
            ax[-1].set_xlim([-0.5,1.5])


            fig.savefig(opj(out_path,'group_results_CS_10mg.pdf'), dpi=600, bbox_inches='tight', transparent=True)

        def CSgroupplot(self,out_path):

            CS_data = np.load(opj(out_path,'data/CS_data.npy'), allow_pickle=True).item()
            condition_colors = {'placebo':'blue', '5mg':'orange', '10mg':'green'}

            fig, ax = pl.subplots(1,3,figsize=(24,8))
            fig.suptitle(f"Center-Surround task")

            preds_placebo = []
            preds_5mg = []
            preds_10mg = []
            preds_nosurr_placebo = []
            preds_nosurr_5mg = []
            preds_nosurr_10mg = []
            rel_effsize_placebo = []
            rel_effsize_5mg = []
            rel_effsize_10mg = []
            rel_effsize_nosurr_placebo =[]
            rel_effsize_nosurr_5mg =[]
            rel_effsize_nosurr_10mg =[]
            subject_name_placebo = []
            subject_name_5mg = []
            subject_name_10mg = []


            for participant, sessions in CS_data.items():
                for session, data_ in sessions.items():
                    #is session 0 do nothing
                    if session == 'ses-0':
                        continue
                    else :
                        ax[0].set_title('Surround')
                        ax[1].set_title('No surround')
                        ax[0].plot(data_['x_space'],data_['full_pred'], c='grey', lw=0.5, alpha=0.5)
                        ax[0].plot(data_['contrast_values'],data_['probs'], marker='^', ls='', c='grey', ms=4, alpha=0.8)
                        ax[1].plot(data_['x_space_nosurr'],data_['full_pred_nosurr'], c='grey', lw=0.8, alpha=0.8)
                        ax[1].plot(data_['contrast_values_nosurr'],data_['probs_nosurr'], marker='v', ls='', c='grey', ms=4, alpha=0.8)       

                        if data_['group'] == 'placebo':
                            preds_placebo.append(data_['full_pred'])
                            preds_nosurr_placebo.append(data_['full_pred_nosurr'])
                            rel_effsize_placebo.append(data_['rel_effsize'])
                            rel_effsize_nosurr_placebo.append(data_['rel_effsize_nosurr'])
                            subject_name_placebo.append(participant)
                        elif data_['group'] == '5mg':
                            preds_5mg.append(data_['full_pred'])
                            preds_nosurr_5mg.append(data_['full_pred_nosurr'])
                            rel_effsize_5mg.append(data_['rel_effsize'])
                            rel_effsize_nosurr_5mg.append(data_['rel_effsize_nosurr'])
                            subject_name_5mg.append(participant)
                        elif data_['group'] == '10mg':
                            preds_10mg.append(data_['full_pred'])
                            preds_nosurr_10mg.append(data_['full_pred_nosurr'])
                            rel_effsize_10mg.append(data_['rel_effsize'])
                            rel_effsize_nosurr_10mg.append(data_['rel_effsize_nosurr'])
                            subject_name_10mg.append(participant)

            for a in ax[:-1]:
                a.set_xlabel('Reference contrast')
                a.set_xlabel('Contrast difference (%RMS)')

                a.plot(CS_data['sub-002']['ses-1']['x_space_comb'],np.ones_like(CS_data['sub-002']['ses-1']['x_space_comb'])*0.5, ls='--', c='k', alpha=0.5, label='PSE')
                a.plot(CS_data['sub-002']['ses-1']['x_space_comb'],np.ones_like(CS_data['sub-002']['ses-1']['x_space_comb']), ls='--',  c='green', alpha=0.5)
                a.plot(CS_data['sub-002']['ses-1']['x_space_comb'],np.zeros_like(CS_data['sub-002']['ses-1']['x_space_comb']), ls='--', c='red', alpha=0.5)
                a.plot(CS_data['sub-002']['ses-1']['target_contrast']*np.ones(100),np.linspace(0,1,100), ls='-', c='k', label='Veridical', alpha=0.5)

                a.set_xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7])
                a.set_xticklabels(['-75','-50','-25','0','+25','+50','+75'])

                #a.legend(loc='lower right')

                a.set_ylabel('Prob ref perceived as higher contrast')    

            #ax[0].errorbar(this_run.contrast_values, np.mean(data,axis=0), yerr=sem(data,axis=0), marker='s', ls='', c='k', ms=6)
            ax[0].plot(CS_data['sub-002']['ses-1']['x_space'], np.mean(preds_placebo,axis=0), c=condition_colors['placebo'], lw=2, label='placebo')
            ax[0].plot(CS_data['sub-002']['ses-1']['x_space'], np.mean(preds_5mg,axis=0), c=condition_colors['5mg'], lw=2, label='5mg')
            ax[0].plot(CS_data['sub-002']['ses-1']['x_space'], np.mean(preds_10mg,axis=0), c=condition_colors['10mg'], lw=2, label='10mg')

            #ax[1].errorbar(this_run.contrast_values_nosurr, np.mean(data_nosurr,axis=0), yerr=sem(data_nosurr,axis=0), marker='s', ls='', c='k', ms=6)
            ax[1].plot(CS_data['sub-002']['ses-1']['x_space_nosurr'], np.mean(preds_nosurr_placebo,axis=0), c=condition_colors['placebo'], lw=2, label='placebo')
            ax[1].plot(CS_data['sub-002']['ses-1']['x_space_nosurr'], np.mean(preds_nosurr_5mg,axis=0), c=condition_colors['5mg'], lw=2, label='5mg')
            ax[1].plot(CS_data['sub-002']['ses-1']['x_space_nosurr'], np.mean(preds_nosurr_10mg,axis=0), c=condition_colors['10mg'], lw=2, label='10mg')

            ax[0].legend(loc='lower right')

            
            the_space = np.linspace(-0.10, 0.10, len(subject_name_placebo))

            """ for ii, effsize in enumerate([rel_effsize,rel_effsize_nosurr]):
                ax[-1].bar(ii, np.mean(effsize)) """
            
            for ii, effsize in enumerate([rel_effsize_placebo,rel_effsize_nosurr_placebo]):
                ax[-1].bar(ii-0.3, np.mean(effsize), color=condition_colors['placebo'], alpha=0.5, width=0.3)

                x_axis = ii*np.ones_like(effsize) + the_space
                ax[-1].plot(x_axis-0.3, effsize, marker='s', mec='k', ls='', ms=8, color=condition_colors['placebo'])

            for ii, effsize in enumerate([rel_effsize_5mg,rel_effsize_nosurr_5mg]):
                ax[-1].bar(ii, np.mean(effsize), color=condition_colors['5mg'], alpha=0.5, width=0.3)
                x_axis = ii*np.ones_like(effsize) + the_space
                ax[-1].plot(x_axis, effsize, marker='s', mec='k', ls='', ms=8, color=condition_colors['5mg'])

            for ii, effsize in enumerate([rel_effsize_10mg,rel_effsize_nosurr_10mg]):
                ax[-1].bar(ii+0.3, np.mean(effsize), color=condition_colors['10mg'], alpha=0.5, width=0.3)
                x_axis = ii*np.ones_like(effsize) + the_space
                ax[-1].plot(x_axis+0.3, effsize, marker='s', mec='k', ls='', ms=8, color=condition_colors['10mg'])
            

            for ss in range(len(rel_effsize_placebo)):
                x_axis2 = x_axis[ss]
                ax[-1].plot([x_axis2-1.3, x_axis2-0.3], [rel_effsize_placebo[ss], rel_effsize_nosurr_placebo[ss]], c='k', linewidth=0.5, alpha=0.5)
                ax[-1].text(x_axis2-1.3, rel_effsize_placebo[ss], subject_name_placebo[ss], fontsize=8)

            for ss in range(len(rel_effsize_5mg)):
                x_axis2 = x_axis[ss]
                ax[-1].plot([x_axis2-1, x_axis2], [rel_effsize_5mg[ss], rel_effsize_nosurr_5mg[ss]], c='k', linewidth=0.5, alpha=0.5)
                ax[-1].text(x_axis2-1, rel_effsize_5mg[ss], subject_name_5mg[ss], fontsize=8)

            for ss in range(len(rel_effsize_10mg)):
                x_axis2 = x_axis[ss]
                ax[-1].plot([x_axis2-0.7, x_axis2+0.3], [rel_effsize_10mg[ss], rel_effsize_nosurr_10mg[ss]], c='k', linewidth=0.5, alpha=0.5)
                ax[-1].text(x_axis2-0.7, rel_effsize_10mg[ss], subject_name_10mg[ss], fontsize=8)
                
                
            ax[-1].plot(np.linspace(-1,2,10),np.zeros_like(np.linspace(-1,2,10)), ls='--', c='k', alpha=0.5)
            
            #set ticks with the
            ax[-1].set_xticks([0,1])
            ax[-1].set_xticklabels(["Surround", "no surround"])
            ax[-1].set_xlabel('')
            ax[-1].set_ylabel('PSE difference (%RMS)')
            #ax[-1].set_ylim([-90,10])
            ax[-1].set_xlim([-0.5,1.5])

            fig.savefig(opj(out_path,'group_results_CS.pdf'), dpi=600, bbox_inches='tight', transparent=True)

        def CSgroupplot_average(self,out_path):
            CS_data = np.load(opj(out_path,'data/CS_data.npy'), allow_pickle=True).item()
            condition_colors = {'placebo':'blue', '5mg':'orange', '10mg':'green'}

            fig, ax = pl.subplots(1,3,figsize=(24,8))
            fig.suptitle(f"Center-Surround task - average results with 95% CI")

            preds_placebo = []
            preds_5mg = []
            preds_10mg = []
            preds_nosurr_placebo = []
            preds_nosurr_5mg = []
            preds_nosurr_10mg = []
            rel_effsize_placebo = []
            rel_effsize_5mg = []
            rel_effsize_10mg = []
            rel_effsize_nosurr_placebo =[]
            rel_effsize_nosurr_5mg =[]
            rel_effsize_nosurr_10mg =[]
            subject_name_placebo = []
            subject_name_5mg = []
            subject_name_10mg = []

            for participant, sessions in CS_data.items():
                for session, data_ in sessions.items():
                    #is session 0 do nothing
                    if session == 'ses-0':
                        continue
                    else :
                        ax[0].set_title('Surround')
                        ax[1].set_title('No surround')
                        #ax[0].plot(data_['x_space'],data_['full_pred'], c='grey', lw=0.5, alpha=0.5)
                        #ax[0].plot(data_['contrast_values'],data_['probs'], marker='^', ls='', c='grey', ms=4, alpha=0.8)
                        #ax[1].plot(data_['x_space_nosurr'],data_['full_pred_nosurr'], c='grey', lw=0.8, alpha=0.8)
                        #ax[1].plot(data_['contrast_values_nosurr'],data_['probs_nosurr'], marker='v', ls='', c='grey', ms=4, alpha=0.8)       

                        if data_['group'] == 'placebo':
                            preds_placebo.append(data_['full_pred'])
                            preds_nosurr_placebo.append(data_['full_pred_nosurr'])
                            rel_effsize_placebo.append(data_['rel_effsize'])
                            rel_effsize_nosurr_placebo.append(data_['rel_effsize_nosurr'])
                            subject_name_placebo.append(participant)
                        elif data_['group'] == '5mg':
                            preds_5mg.append(data_['full_pred'])
                            preds_nosurr_5mg.append(data_['full_pred_nosurr'])
                            rel_effsize_5mg.append(data_['rel_effsize'])
                            rel_effsize_nosurr_5mg.append(data_['rel_effsize_nosurr'])
                            subject_name_5mg.append(participant)
                        elif data_['group'] == '10mg':
                            preds_10mg.append(data_['full_pred'])
                            preds_nosurr_10mg.append(data_['full_pred_nosurr'])
                            rel_effsize_10mg.append(data_['rel_effsize'])
                            rel_effsize_nosurr_10mg.append(data_['rel_effsize_nosurr'])
                            subject_name_10mg.append(participant)

            for a in ax[:-1]:
                a.set_xlabel('Reference contrast')
                a.set_xlabel('Contrast difference (%RMS)')

                a.plot(CS_data['sub-002']['ses-1']['x_space_comb'],np.ones_like(CS_data['sub-002']['ses-1']['x_space_comb'])*0.5, ls='--', c='k', alpha=0.5, label='PSE')
                a.plot(CS_data['sub-002']['ses-1']['x_space_comb'],np.ones_like(CS_data['sub-002']['ses-1']['x_space_comb']), ls='--',  c='green', alpha=0.5)
                a.plot(CS_data['sub-002']['ses-1']['x_space_comb'],np.zeros_like(CS_data['sub-002']['ses-1']['x_space_comb']), ls='--', c='red', alpha=0.5)
                a.plot(CS_data['sub-002']['ses-1']['target_contrast']*np.ones(100),np.linspace(0,1,100), ls='-', c='k', label='Veridical', alpha=0.5)

                a.set_xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7])
                a.set_xticklabels(['-75','-50','-25','0','+25','+50','+75'])

                #a.legend(loc='lower right')

                a.set_ylabel('Prob ref perceived as higher contrast')       

            #ax[0].errorbar(this_run.contrast_values, np.mean(data,axis=0), yerr=sem(data,axis=0), marker='s', ls='', c='k', ms=6)
            ax[0].plot(CS_data['sub-002']['ses-1']['x_space'], np.mean(preds_placebo,axis=0), c=condition_colors['placebo'], lw=2, label='placebo')
            ax[0].plot(CS_data['sub-002']['ses-1']['x_space'], np.mean(preds_5mg,axis=0), c=condition_colors['5mg'], lw=2, label='5mg')
            ax[0].plot(CS_data['sub-002']['ses-1']['x_space'], np.mean(preds_10mg,axis=0), c=condition_colors['10mg'], lw=2, label='10mg')

            #add the confidence interval for the average
            ax[0].fill_between(CS_data['sub-002']['ses-1']['x_space'], np.mean(preds_placebo,axis=0)-1.96*sem(preds_placebo,axis=0), np.mean(preds_placebo,axis=0)+1.96*sem(preds_placebo,axis=0), color=condition_colors['placebo'], alpha=0.2)
            ax[0].fill_between(CS_data['sub-002']['ses-1']['x_space'], np.mean(preds_5mg,axis=0)-1.96*sem(preds_5mg,axis=0), np.mean(preds_5mg,axis=0)+1.96*sem(preds_5mg,axis=0), color=condition_colors['5mg'], alpha=0.2)
            ax[0].fill_between(CS_data['sub-002']['ses-1']['x_space'], np.mean(preds_10mg,axis=0)-1.96*sem(preds_10mg,axis=0), np.mean(preds_10mg,axis=0)+1.96*sem(preds_10mg,axis=0), color=condition_colors['10mg'], alpha=0.2)
            
            #ax[1].errorbar(this_run.contrast_values_nosurr, np.mean(data_nosurr,axis=0), yerr=sem(data_nosurr,axis=0), marker='s', ls='', c='k', ms=6)
            ax[1].plot(CS_data['sub-002']['ses-1']['x_space_nosurr'], np.mean(preds_nosurr_placebo,axis=0), c=condition_colors['placebo'], lw=2, label='placebo')
            ax[1].plot(CS_data['sub-002']['ses-1']['x_space_nosurr'], np.mean(preds_nosurr_5mg,axis=0), c=condition_colors['5mg'], lw=2, label='5mg')
            ax[1].plot(CS_data['sub-002']['ses-1']['x_space_nosurr'], np.mean(preds_nosurr_10mg,axis=0), c=condition_colors['10mg'], lw=2, label='10mg')

            ax[1].fill_between(CS_data['sub-002']['ses-1']['x_space_nosurr'], np.mean(preds_nosurr_placebo,axis=0)-1.96*sem(preds_nosurr_placebo,axis=0), np.mean(preds_nosurr_placebo,axis=0)+1.96*sem(preds_nosurr_placebo,axis=0), color=condition_colors['placebo'], alpha=0.2)
            ax[1].fill_between(CS_data['sub-002']['ses-1']['x_space_nosurr'], np.mean(preds_nosurr_5mg,axis=0)-1.96*sem(preds_nosurr_5mg,axis=0), np.mean(preds_nosurr_5mg,axis=0)+1.96*sem(preds_nosurr_5mg,axis=0), color=condition_colors['5mg'], alpha=0.2)
            ax[1].fill_between(CS_data['sub-002']['ses-1']['x_space_nosurr'], np.mean(preds_nosurr_10mg,axis=0)-1.96*sem(preds_nosurr_10mg,axis=0), np.mean(preds_nosurr_10mg,axis=0)+1.96*sem(preds_nosurr_10mg,axis=0), color=condition_colors['10mg'], alpha=0.2)

            ax[0].legend(loc='lower right')

            the_space = np.linspace(-0.10, 0.10, len(subject_name_placebo))

            
            for ii, effsize in enumerate([rel_effsize_placebo,rel_effsize_nosurr_placebo]):
                ax[-1].bar(ii-0.3, np.mean(effsize), color=condition_colors['placebo'], alpha=0.7, width=0.3)

                x_axis = ii*np.ones_like(effsize) + the_space
                ax[-1].plot(x_axis-0.3, effsize, marker='s', mec='k', ls='', ms=6, alpha =0.6, color=condition_colors['placebo'])

            for ii, effsize in enumerate([rel_effsize_5mg,rel_effsize_nosurr_5mg]):
                ax[-1].bar(ii, np.mean(effsize), color=condition_colors['5mg'], alpha=0.7, width=0.3)
                x_axis = ii*np.ones_like(effsize) + the_space
                ax[-1].plot(x_axis, effsize, marker='s', mec='k', ls='', ms=6, alpha =0.6, color=condition_colors['5mg'])

            for ii, effsize in enumerate([rel_effsize_10mg,rel_effsize_nosurr_10mg]):
                ax[-1].bar(ii+0.3, np.mean(effsize), color=condition_colors['10mg'], alpha=0.7, width=0.3)
                x_axis = ii*np.ones_like(effsize) + the_space
                ax[-1].plot(x_axis+0.3, effsize, marker='s', mec='k', ls='', ms=6, alpha =0.6, color=condition_colors['10mg'])
            
            #plot the links and the text
            for ss in range(len(rel_effsize_placebo)):
                x_axis2 = x_axis[ss]
                ax[-1].plot([x_axis2-1.3, x_axis2-1, x_axis2-0.7], [rel_effsize_placebo[ss], rel_effsize_5mg[ss], rel_effsize_10mg[ss]], c='k', linewidth=0.5, alpha=0.5)
                #if sub-013
                if subject_name_placebo[ss] == 'sub-013':
                    ax[-1].text(x_axis2-1.3, rel_effsize_placebo[ss]-0.4, subject_name_placebo[ss], fontsize=8)
                else :
                    ax[-1].text(x_axis2-1.3, rel_effsize_placebo[ss], subject_name_placebo[ss], fontsize=8)

            for ss in range(len(rel_effsize_nosurr_placebo)):
                x_axis2 = x_axis[ss]
                ax[-1].plot([x_axis2-0.3, x_axis2, x_axis2+0.3], [rel_effsize_nosurr_placebo[ss], rel_effsize_nosurr_5mg[ss], rel_effsize_nosurr_10mg[ss]], c='k', linewidth=0.5, alpha=0.5)
                ax[-1].text(x_axis2-0.3, rel_effsize_nosurr_placebo[ss], subject_name_placebo[ss], fontsize=8)

            # for ss in range(len(rel_effsize_5mg)):
            #     x_axis2 = x_axis[ss]
            #     ax[-1].plot([x_axis2-1, x_axis2], [rel_effsize_5mg[ss], rel_effsize_nosurr_5mg[ss]], c='k', linewidth=0.5, alpha=0.5)
            #     ax[-1].text(x_axis2-1, rel_effsize_5mg[ss], subject_name_5mg[ss], fontsize=8)

            # for ss in range(len(rel_effsize_10mg)):
            #     x_axis2 = x_axis[ss]
            #     ax[-1].plot([x_axis2-0.7, x_axis2+0.3], [rel_effsize_10mg[ss], rel_effsize_nosurr_10mg[ss]], c='k', linewidth=0.5, alpha=0.5)
            #     ax[-1].text(x_axis2-0.7, rel_effsize_10mg[ss], subject_name_10mg[ss], fontsize=8)
                
                
            ax[-1].plot(np.linspace(-1,2,10),np.zeros_like(np.linspace(-1,2,10)), ls='--', c='k', alpha=0.5)
            
            #set ticks with the
            ax[-1].set_xticks([0,1])
            ax[-1].set_xticklabels(["Surround", "no surround"])
            ax[-1].set_xlabel('')
            ax[-1].set_ylabel('PSE difference (%RMS)')
            ax[-1].set_ylim([-90,20])
            ax[-1].set_xlim([-0.5,1.5])


            fig.savefig(opj(out_path,'group_results_CS_average.pdf'), dpi=600, bbox_inches='tight', transparent=True)    

        def CSgroupplot_average2(self,out_path):
            CS_data = np.load(opj(out_path,'data/CS_data.npy'), allow_pickle=True).item()
            condition_colors = {'placebo':'blue', '5mg':'orange', '10mg':'green'}

            fig, ax = pl.subplots(1,1,figsize=(8,8))
            fig.suptitle(f"Center-Surround task - relative effect size")

            preds_placebo = []
            preds_5mg = []
            preds_10mg = []
            preds_nosurr_placebo = []
            preds_nosurr_5mg = []
            preds_nosurr_10mg = []
            rel_effsize_placebo = []
            rel_effsize_5mg = []
            rel_effsize_10mg = []
            rel_effsize_nosurr_placebo =[]
            rel_effsize_nosurr_5mg =[]
            rel_effsize_nosurr_10mg =[]
            rel_effsize_difference_placebo = []
            rel_effsize_difference_5mg = []
            rel_effsize_difference_10mg = []
            subject_name_placebo = []
            subject_name_5mg = []
            subject_name_10mg = []

            for participant, sessions in CS_data.items():
                for session, data_ in sessions.items():
                    #is session 0 do nothing
                    if session == 'ses-0':
                        continue
                    else :
                        
                        if data_['group'] == 'placebo':
                            preds_placebo.append(data_['full_pred'])
                            preds_nosurr_placebo.append(data_['full_pred_nosurr'])
                            rel_effsize_placebo.append(data_['rel_effsize'])
                            rel_effsize_nosurr_placebo.append(data_['rel_effsize_nosurr'])
                            rel_effsize_difference_placebo.append(data_['rel_effsize']- data_['rel_effsize_nosurr'])
                            subject_name_placebo.append(participant)
                        elif data_['group'] == '5mg':
                            preds_5mg.append(data_['full_pred'])
                            preds_nosurr_5mg.append(data_['full_pred_nosurr'])
                            rel_effsize_5mg.append(data_['rel_effsize'])
                            rel_effsize_nosurr_5mg.append(data_['rel_effsize_nosurr'])
                            rel_effsize_difference_5mg.append(data_['rel_effsize']- data_['rel_effsize_nosurr'])
                            subject_name_5mg.append(participant)
                        elif data_['group'] == '10mg':
                            preds_10mg.append(data_['full_pred'])
                            preds_nosurr_10mg.append(data_['full_pred_nosurr'])
                            rel_effsize_10mg.append(data_['rel_effsize'])
                            rel_effsize_nosurr_10mg.append(data_['rel_effsize_nosurr'])
                            rel_effsize_difference_10mg.append(data_['rel_effsize']- data_['rel_effsize_nosurr'])
                            subject_name_10mg.append(participant)

            ax.set_ylabel('Differences in relative effect sizes between between surround and nosurroud tasks')       
            the_space = np.linspace(-0.10, 0.10, len(subject_name_placebo))
            
            ax.bar(-0.3, np.mean(rel_effsize_difference_placebo), color=condition_colors['placebo'], alpha=0.5, width=0.3)

            x_axis = 0*np.ones_like(rel_effsize_difference_placebo) + the_space
            ax.plot(x_axis-0.3, rel_effsize_difference_placebo, marker='s', mec='k', ls='', ms=8, color=condition_colors['placebo'])

            ax.bar(0, np.mean(rel_effsize_difference_5mg), color=condition_colors['5mg'], alpha=0.5, width=0.3)
            x_axis = 0*np.ones_like(rel_effsize_difference_5mg) + the_space
            ax.plot(x_axis, rel_effsize_difference_5mg, marker='s', mec='k', ls='', ms=8, color=condition_colors['5mg'])

            ax.bar(0.3, np.mean(rel_effsize_difference_10mg), color=condition_colors['10mg'], alpha=0.5, width=0.3)
            x_axis = 0*np.ones_like(rel_effsize_difference_10mg) + the_space
            ax.plot(x_axis+0.3, rel_effsize_difference_10mg, marker='s', mec='k', ls='', ms=8, color=condition_colors['10mg'])
            

            for ss in range(len(rel_effsize_difference_placebo)):

                x_axis2 = x_axis[ss]
                ax.plot([x_axis2-0.3, x_axis2, x_axis2+0.3], [rel_effsize_difference_placebo[ss], rel_effsize_difference_5mg[ss], rel_effsize_difference_10mg[ss]], c='k', linewidth=0.5, alpha=0.5)

                ax.text(x_axis2-0.3, rel_effsize_difference_placebo[ss], subject_name_placebo[ss], fontsize=8)
                
            ax.plot(np.linspace(-1,2,10),np.zeros_like(np.linspace(-1,2,10)), ls='--', c='k', alpha=0.5)
            
            #set ticks with the
            ax.set_xticks([-0.3, 0, 0.3])
            ax.set_xticklabels(['placebo', '5mg', '10mg'])
            ax.set_xlabel('')
            ax.set_ylabel('PSE difference (%RMS)')
            ax.set_ylim([-90,10])
            ax.set_xlim([-0.5,0.5])


            fig.savefig(opj(out_path,'group_results_CS_average2.pdf'), dpi=600, bbox_inches='tight', transparent=True)

        def CSgroupplot_average3(self,out_path):
            CS_data = np.load(opj(out_path,'data/CS_data.npy'), allow_pickle=True).item()
            df_CS = pd.read_csv(opj(out_path,'data/group_results_CS.csv'))
            condition_colors = {'placebo':'blue', '5mg':'orange', '10mg':'green'}

            fig, ax = pl.subplots(1,4,figsize=(32,8))
            fig.suptitle(f"Center-Surround task - average results with 95% CI")

            preds_placebo = []
            preds_5mg = []
            preds_10mg = []
            preds_nosurr_placebo = []
            preds_nosurr_5mg = []
            preds_nosurr_10mg = []
            rel_effsize_placebo = []
            rel_effsize_5mg = []
            rel_effsize_10mg = []
            rel_effsize_nosurr_placebo =[]
            rel_effsize_nosurr_5mg =[]
            rel_effsize_nosurr_10mg =[]
            rel_effsize_difference_placebo = []
            rel_effsize_difference_5mg = []
            rel_effsize_difference_10mg = []
            subject_name_placebo = []
            subject_name_5mg = []
            subject_name_10mg = []

            for participant, sessions in CS_data.items():
                for session, data_ in sessions.items():
                    #is session 0 do nothing
                    if session == 'ses-0':
                        continue
                    else :
                        
                        if data_['group'] == 'placebo':
                            preds_placebo.append(data_['full_pred'])
                            preds_nosurr_placebo.append(data_['full_pred_nosurr'])
                            rel_effsize_placebo.append(data_['rel_effsize'])
                            rel_effsize_nosurr_placebo.append(data_['rel_effsize_nosurr'])
                            rel_effsize_difference_placebo.append(data_['rel_effsize']- data_['rel_effsize_nosurr'])
                            subject_name_placebo.append(participant)
                        elif data_['group'] == '5mg':
                            preds_5mg.append(data_['full_pred'])
                            preds_nosurr_5mg.append(data_['full_pred_nosurr'])
                            rel_effsize_5mg.append(data_['rel_effsize'])
                            rel_effsize_nosurr_5mg.append(data_['rel_effsize_nosurr'])
                            rel_effsize_difference_5mg.append(data_['rel_effsize']- data_['rel_effsize_nosurr'])
                            subject_name_5mg.append(participant)
                        elif data_['group'] == '10mg':
                            preds_10mg.append(data_['full_pred'])
                            preds_nosurr_10mg.append(data_['full_pred_nosurr'])
                            rel_effsize_10mg.append(data_['rel_effsize'])
                            rel_effsize_nosurr_10mg.append(data_['rel_effsize_nosurr'])
                            rel_effsize_difference_10mg.append(data_['rel_effsize']- data_['rel_effsize_nosurr'])
                            subject_name_10mg.append(participant)

            for a in ax[:2]:
                a.set_xlabel('Reference contrast')
                a.set_xlabel('Contrast difference (%RMS)')

                a.plot(CS_data['sub-002']['ses-1']['x_space_comb'],np.ones_like(CS_data['sub-002']['ses-1']['x_space_comb'])*0.5, ls='--', c='k', alpha=0.5, label='PSE')
                a.plot(CS_data['sub-002']['ses-1']['x_space_comb'],np.ones_like(CS_data['sub-002']['ses-1']['x_space_comb']), ls='--',  c='green', alpha=0.5)
                a.plot(CS_data['sub-002']['ses-1']['x_space_comb'],np.zeros_like(CS_data['sub-002']['ses-1']['x_space_comb']), ls='--', c='red', alpha=0.5)
                a.plot(CS_data['sub-002']['ses-1']['target_contrast']*np.ones(100),np.linspace(0,1,100), ls='-', c='k', label='Veridical', alpha=0.5)

                a.set_xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7])
                a.set_xticklabels(['-75','-50','-25','0','+25','+50','+75'])

                #a.legend(loc='lower right')

                a.set_ylabel('Prob ref perceived as higher contrast')       

            #ax[0].errorbar(this_run.contrast_values, np.mean(data,axis=0), yerr=sem(data,axis=0), marker='s', ls='', c='k', ms=6)
            ax[0].plot(CS_data['sub-002']['ses-1']['x_space'], np.mean(preds_placebo,axis=0), c=condition_colors['placebo'], lw=2, label='placebo')
            ax[0].plot(CS_data['sub-002']['ses-1']['x_space'], np.mean(preds_5mg,axis=0), c=condition_colors['5mg'], lw=2, label='5mg')
            ax[0].plot(CS_data['sub-002']['ses-1']['x_space'], np.mean(preds_10mg,axis=0), c=condition_colors['10mg'], lw=2, label='10mg')

            print('placebo standard error of the mean')
            print(sem(preds_placebo,axis=0))


            #add the confidence interval for the average
            ax[0].fill_between(CS_data['sub-002']['ses-1']['x_space'], np.mean(preds_placebo,axis=0)-1.96*sem(preds_placebo,axis=0), np.mean(preds_placebo,axis=0)+1.96*sem(preds_placebo,axis=0), color=condition_colors['placebo'], alpha=0.2)
            ax[0].fill_between(CS_data['sub-002']['ses-1']['x_space'], np.mean(preds_5mg,axis=0)-1.96*sem(preds_5mg,axis=0), np.mean(preds_5mg,axis=0)+1.96*sem(preds_5mg,axis=0), color=condition_colors['5mg'], alpha=0.2)
            ax[0].fill_between(CS_data['sub-002']['ses-1']['x_space'], np.mean(preds_10mg,axis=0)-1.96*sem(preds_10mg,axis=0), np.mean(preds_10mg,axis=0)+1.96*sem(preds_10mg,axis=0), color=condition_colors['10mg'], alpha=0.2)
            
            #ax[1].errorbar(this_run.contrast_values_nosurr, np.mean(data_nosurr,axis=0), yerr=sem(data_nosurr,axis=0), marker='s', ls='', c='k', ms=6)
            ax[1].plot(CS_data['sub-002']['ses-1']['x_space_nosurr'], np.mean(preds_nosurr_placebo,axis=0), c=condition_colors['placebo'], lw=2, label='placebo')
            ax[1].plot(CS_data['sub-002']['ses-1']['x_space_nosurr'], np.mean(preds_nosurr_5mg,axis=0), c=condition_colors['5mg'], lw=2, label='5mg')
            ax[1].plot(CS_data['sub-002']['ses-1']['x_space_nosurr'], np.mean(preds_nosurr_10mg,axis=0), c=condition_colors['10mg'], lw=2, label='10mg')

            ax[1].fill_between(CS_data['sub-002']['ses-1']['x_space_nosurr'], np.mean(preds_nosurr_placebo,axis=0)-1.96*sem(preds_nosurr_placebo,axis=0), np.mean(preds_nosurr_placebo,axis=0)+1.96*sem(preds_nosurr_placebo,axis=0), color=condition_colors['placebo'], alpha=0.2)
            ax[1].fill_between(CS_data['sub-002']['ses-1']['x_space_nosurr'], np.mean(preds_nosurr_5mg,axis=0)-1.96*sem(preds_nosurr_5mg,axis=0), np.mean(preds_nosurr_5mg,axis=0)+1.96*sem(preds_nosurr_5mg,axis=0), color=condition_colors['5mg'], alpha=0.2)
            ax[1].fill_between(CS_data['sub-002']['ses-1']['x_space_nosurr'], np.mean(preds_nosurr_10mg,axis=0)-1.96*sem(preds_nosurr_10mg,axis=0), np.mean(preds_nosurr_10mg,axis=0)+1.96*sem(preds_nosurr_10mg,axis=0), color=condition_colors['10mg'], alpha=0.2)

            ax[0].legend(loc='lower right')

            the_space = np.linspace(-0.10, 0.10, len(subject_name_placebo))

            
            for ii, effsize in enumerate([rel_effsize_placebo,rel_effsize_nosurr_placebo]):
                ax[2].bar(ii-0.3, np.mean(effsize), color=condition_colors['placebo'], alpha=0.7, width=0.3)

                x_axis = ii*np.ones_like(effsize) + the_space
                ax[2].plot(x_axis-0.3, effsize, marker='s', mec='k', ls='', ms=6, alpha =0.6, color=condition_colors['placebo'])

            for ii, effsize in enumerate([rel_effsize_5mg,rel_effsize_nosurr_5mg]):
                ax[2].bar(ii, np.mean(effsize), color=condition_colors['5mg'], alpha=0.7, width=0.3)
                x_axis = ii*np.ones_like(effsize) + the_space
                ax[2].plot(x_axis, effsize, marker='s', mec='k', ls='', ms=6, alpha =0.6, color=condition_colors['5mg'])

            for ii, effsize in enumerate([rel_effsize_10mg,rel_effsize_nosurr_10mg]):
                ax[2].bar(ii+0.3, np.mean(effsize), color=condition_colors['10mg'], alpha=0.7, width=0.3)
                x_axis = ii*np.ones_like(effsize) + the_space
                ax[2].plot(x_axis+0.3, effsize, marker='s', mec='k', ls='', ms=6, alpha =0.6, color=condition_colors['10mg'])
            
            #plot the links and the text
            for ss in range(len(rel_effsize_placebo)):
                x_axis2 = x_axis[ss]
                ax[2].plot([x_axis2-1.3, x_axis2-1, x_axis2-0.7], [rel_effsize_placebo[ss], rel_effsize_5mg[ss], rel_effsize_10mg[ss]], c='k', linewidth=0.5, alpha=0.5)
                #if sub-013
                if subject_name_placebo[ss] == 'sub-013':
                    ax[2].text(x_axis2-1.3, rel_effsize_placebo[ss]-0.1, subject_name_placebo[ss], fontsize=8)
                else :
                    ax[2].text(x_axis2-1.3, rel_effsize_placebo[ss], subject_name_placebo[ss], fontsize=8)

            for ss in range(len(rel_effsize_nosurr_placebo)):
                x_axis2 = x_axis[ss]
                ax[2].plot([x_axis2-0.3, x_axis2, x_axis2+0.3], [rel_effsize_nosurr_placebo[ss], rel_effsize_nosurr_5mg[ss], rel_effsize_nosurr_10mg[ss]], c='k', linewidth=0.5, alpha=0.5)
                ax[2].text(x_axis2-0.3, rel_effsize_nosurr_placebo[ss], subject_name_placebo[ss], fontsize=8)
 
            ax[2].plot(np.linspace(-1,2,10),np.zeros_like(np.linspace(-1,2,10)), ls='--', c='k', alpha=0.5)
            
            #set ticks with the
            ax[2].set_xticks([0,1])
            ax[2].set_xticklabels(["Surround", "no surround"])
            ax[2].set_xlabel('')
            ax[2].set_ylabel('PSE difference (%RMS)')
            #ax[-1].set_ylim([-90,10])
            ax[2].set_xlim([-0.5,1.5])

            ax[3].set_ylabel('Differences in relative effect sizes between between surround and nosurroud tasks')       
            the_space = np.linspace(-0.10, 0.10, len(subject_name_placebo))
            
            ax[3].bar(-0.3, np.mean(rel_effsize_difference_placebo), color=condition_colors['placebo'], alpha=0.7, width=0.3)

            x_axis = 0*np.ones_like(rel_effsize_difference_placebo) + the_space
            ax[3].plot(x_axis-0.3, rel_effsize_difference_placebo, marker='s', mec='k', ls='', ms=8, color=condition_colors['placebo'])

            ax[3].bar(0, np.mean(rel_effsize_difference_5mg), color=condition_colors['5mg'], alpha=0.7, width=0.3)
            x_axis = 0*np.ones_like(rel_effsize_difference_5mg) + the_space
            ax[3].plot(x_axis, rel_effsize_difference_5mg, marker='s', mec='k', ls='', ms=8, color=condition_colors['5mg'])

            ax[3].bar(0.3, np.mean(rel_effsize_difference_10mg), color=condition_colors['10mg'], alpha=0.7, width=0.3)
            x_axis = 0*np.ones_like(rel_effsize_difference_10mg) + the_space
            ax[3].plot(x_axis+0.3, rel_effsize_difference_10mg, marker='s', mec='k', ls='', ms=8, color=condition_colors['10mg'])
            

            for ss in range(len(rel_effsize_difference_placebo)):

                x_axis2 = x_axis[ss]
                ax[3].plot([x_axis2-0.3, x_axis2, x_axis2+0.3], [rel_effsize_difference_placebo[ss], rel_effsize_difference_5mg[ss], rel_effsize_difference_10mg[ss]], c='k', linewidth=0.5, alpha=0.5)

                ax[3].text(x_axis2-0.3, rel_effsize_difference_placebo[ss], subject_name_placebo[ss], fontsize=8)
                
            ax[3].plot(np.linspace(-1,2,10),np.zeros_like(np.linspace(-1,2,10)), ls='--', c='k', alpha=0.5)
            
            #set ticks with the
            ax[3].set_xticks([-0.3, 0, 0.3])
            ax[3].set_xticklabels(['placebo', '5mg', '10mg'])
            ax[3].set_xlabel('')
            ax[3].set_ylabel('PSE[no surr] - PSE[surr] (%RMS)')
            ax[3].set_ylim([-100,5])
            ax[3].set_xlim([-0.5,0.5])

            #get rid of ses-0
            df_CS = df_CS[df_CS.Session != 'ses_0']
            df_CS_regroup = df_CS.groupby(['Subject','Group_type']).first().reset_index()

            #check the balance
            print(df_CS_regroup['Subject'].value_counts())
            print(df_CS_regroup.groupby(['Group_type'])['rel_effsize_diff'].count())

            #define p value and f value
            aovrm2way = AnovaRM(df_CS_regroup, 'rel_effsize_diff', 'Subject', within=['Group_type'], aggregate_func=np.mean)
            res2way = aovrm2way.fit()
            print("RM ANOVA all subjects Results:")
            #print only the p value
            p = round(res2way.anova_table['Pr > F'][0], 2)
            #print only the f value
            f = round(res2way.anova_table['F Value'][0], 2)

            #Add the ANOVA info to the plot
            ax[3].text(0.5, -40, 'RM Anova', fontsize=12)
            ax[3].text(0.5, -50, f'F value = {f}', fontsize=12)
            ax[3].text(0.5, -60, f'p = {p}', fontsize=12)

            fig.savefig(opj(out_path,'group_results_CS_average.pdf'), dpi=600, bbox_inches='tight', transparent=True) 

        def CSgroupplot_average_slope(self,out_path):
            CS_data = np.load(opj(out_path,'data/CS_data.npy'), allow_pickle=True).item()
            df_CS = pd.read_csv(opj(out_path,'data/group_results_CS.csv'))
            condition_colors = {'placebo':'blue', '5mg':'orange', '10mg':'green'}

            fig, ax = pl.subplots(1,4,figsize=(32,8))
            fig.suptitle(f"Center-Surround task - average results with 95% CI (slope weighted)")

            preds_placebo = []
            preds_5mg = []
            preds_10mg = []
            preds_nosurr_placebo = []
            preds_nosurr_5mg = []
            preds_nosurr_10mg = []
            slope_placebo = []
            slope_5mg = []
            slope_10mg = []
            slope_nosurr_placebo =[]
            slope_nosurr_5mg =[]
            slope_nosurr_10mg =[]
            slope_average_placebo = []
            slope_average_5mg = []
            slope_average_10mg = []
            rel_effsize_placebo = []
            rel_effsize_5mg = []
            rel_effsize_10mg = []
            rel_effsize_nosurr_placebo =[]
            rel_effsize_nosurr_5mg =[]
            rel_effsize_nosurr_10mg =[]
            rel_effsize_difference_placebo = []
            rel_effsize_difference_5mg = []
            rel_effsize_difference_10mg = []
            subject_name_placebo = []
            subject_name_5mg = []
            subject_name_10mg = []

            for participant, sessions in CS_data.items():
                for session, data_ in sessions.items():
                    #is session 0 do nothing
                    if session == 'ses-0':
                        continue
                    else :
                        
                        if data_['group'] == 'placebo':
                            preds_placebo.append(data_['full_pred'])
                            preds_nosurr_placebo.append(data_['full_pred_nosurr'])
                            rel_effsize_placebo.append(data_['rel_effsize'])
                            rel_effsize_nosurr_placebo.append(data_['rel_effsize_nosurr'])
                            rel_effsize_difference_placebo.append(data_['rel_effsize']- data_['rel_effsize_nosurr'])
                            subject_name_placebo.append(participant)
                            if data_['slope'] > 0:
                                slope_placebo.append(round(data_['slope'],2))
                            else:
                                slope_placebo.append(0)
                            if data_['slope_nosurr'] > 0:
                                slope_nosurr_placebo.append(round(data_['slope_nosurr'],2))
                            else:
                                slope_nosurr_placebo.append(0)
                            if data_['slope_average'] > 0:
                                slope_average_placebo.append(round(data_['slope_average'],2))
                            else:
                                slope_average_placebo.append(0)
                        elif data_['group'] == '5mg':
                            preds_5mg.append(data_['full_pred'])
                            preds_nosurr_5mg.append(data_['full_pred_nosurr'])
                            rel_effsize_5mg.append(data_['rel_effsize'])
                            rel_effsize_nosurr_5mg.append(data_['rel_effsize_nosurr'])
                            rel_effsize_difference_5mg.append(data_['rel_effsize']- data_['rel_effsize_nosurr'])
                            subject_name_5mg.append(participant)
                            if data_['slope'] > 0:
                                slope_5mg.append(round(data_['slope'],2))
                            else:
                                slope_5mg.append(0)
                            if data_['slope_nosurr'] > 0:
                                slope_nosurr_5mg.append(round(data_['slope_nosurr'],2))
                            else:
                                slope_nosurr_5mg.append(0)
                            if data_['slope_average'] > 0:
                                slope_average_5mg.append(round(data_['slope_average'],2))
                            else:
                                slope_average_5mg.append(0)
                        elif data_['group'] == '10mg':
                            preds_10mg.append(data_['full_pred'])
                            preds_nosurr_10mg.append(data_['full_pred_nosurr'])
                            rel_effsize_10mg.append(data_['rel_effsize'])
                            rel_effsize_nosurr_10mg.append(data_['rel_effsize_nosurr'])
                            rel_effsize_difference_10mg.append(data_['rel_effsize']- data_['rel_effsize_nosurr'])
                            subject_name_10mg.append(participant)
                            if data_['slope'] > 0:
                                slope_10mg.append(round(data_['slope'],2))
                            else:
                                slope_10mg.append(0)
                            if data_['slope_nosurr'] > 0:
                                slope_nosurr_10mg.append(round(data_['slope_nosurr'],2))
                            else:
                                slope_nosurr_10mg.append(0)
                            if data_['slope_average'] > 0:
                                slope_average_10mg.append(round(data_['slope_average'],2))
                            else:
                                slope_average_10mg.append(0)

            for a in ax[:2]:
                a.set_xlabel('Reference contrast')
                a.set_xlabel('Contrast difference (%RMS)')

                a.plot(CS_data['sub-002']['ses-1']['x_space_comb'],np.ones_like(CS_data['sub-002']['ses-1']['x_space_comb'])*0.5, ls='--', c='k', alpha=0.5, label='PSE')
                a.plot(CS_data['sub-002']['ses-1']['x_space_comb'],np.ones_like(CS_data['sub-002']['ses-1']['x_space_comb']), ls='--',  c='green', alpha=0.5)
                a.plot(CS_data['sub-002']['ses-1']['x_space_comb'],np.zeros_like(CS_data['sub-002']['ses-1']['x_space_comb']), ls='--', c='red', alpha=0.5)
                a.plot(CS_data['sub-002']['ses-1']['target_contrast']*np.ones(100),np.linspace(0,1,100), ls='-', c='k', label='Veridical', alpha=0.5)

                a.set_xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7])
                a.set_xticklabels(['-75','-50','-25','0','+25','+50','+75'])

                #a.legend(loc='lower right')

                a.set_ylabel('Prob ref perceived as higher contrast')       

            #ax[0].errorbar(this_run.contrast_values, np.mean(data,axis=0), yerr=sem(data,axis=0), marker='s', ls='', c='k', ms=6)
            # ax[0].plot(CS_data['sub-002']['ses-1']['x_space'], np.mean(preds_placebo,axis=0), c=condition_colors['placebo'], lw=2, label='placebo')
            # ax[0].plot(CS_data['sub-002']['ses-1']['x_space'], np.mean(preds_5mg,axis=0), c=condition_colors['5mg'], lw=2, label='5mg')
            # ax[0].plot(CS_data['sub-002']['ses-1']['x_space'], np.mean(preds_10mg,axis=0), c=condition_colors['10mg'], lw=2, label='10mg')

            

            # print(preds_placebo[1,:])
            d1 = DescrStatsW(preds_placebo, weights=slope_placebo)
            d2 = DescrStatsW(preds_5mg, weights=slope_5mg)
            d3 = DescrStatsW(preds_10mg, weights=slope_10mg)     
            
            ax[0].plot(CS_data['sub-002']['ses-1']['x_space'], np.average(preds_placebo, axis=0, weights=slope_placebo), c=condition_colors['placebo'], lw=2, label='placebo')
            ax[0].plot(CS_data['sub-002']['ses-1']['x_space'], np.average(preds_5mg, axis=0, weights=slope_5mg), c=condition_colors['5mg'], lw=2, label='5mg')
            ax[0].plot(CS_data['sub-002']['ses-1']['x_space'], np.average(preds_10mg, axis=0, weights=slope_10mg), c=condition_colors['10mg'], lw=2, label='10mg')

            #add the 95% confidence interval for the average
            ax[0].fill_between( CS_data['sub-002']['ses-1']['x_space'], np.average(preds_placebo, axis=0, weights=slope_placebo) - 1.96 * d1.std_mean, np.average(preds_placebo, axis=0, weights=slope_placebo) + 1.96 * d1.std_mean, color=condition_colors['placebo'], alpha=0.2 )
            ax[0].fill_between( CS_data['sub-002']['ses-1']['x_space'], np.average(preds_5mg, axis=0, weights=slope_5mg) - 1.96 * d2.std_mean, np.average(preds_5mg, axis=0, weights=slope_5mg) + 1.96 * d2.std_mean, color=condition_colors['5mg'], alpha=0.2 )
            ax[0].fill_between( CS_data['sub-002']['ses-1']['x_space'], np.average(preds_10mg, axis=0, weights=slope_10mg) - 1.96 * d3.std_mean, np.average(preds_10mg, axis=0, weights=slope_10mg) + 1.96 * d3.std_mean, color=condition_colors['10mg'], alpha=0.2 )
            
            #ax[1].errorbar(this_run.contrast_values_nosurr, np.mean(data_nosurr,axis=0), yerr=sem(data_nosurr,axis=0), marker='s', ls='', c='k', ms=6)
            ax[1].plot(CS_data['sub-002']['ses-1']['x_space_nosurr'], np.mean(preds_nosurr_placebo,axis=0), c=condition_colors['placebo'], lw=2, label='placebo')
            ax[1].plot(CS_data['sub-002']['ses-1']['x_space_nosurr'], np.mean(preds_nosurr_5mg,axis=0), c=condition_colors['5mg'], lw=2, label='5mg')
            ax[1].plot(CS_data['sub-002']['ses-1']['x_space_nosurr'], np.mean(preds_nosurr_10mg,axis=0), c=condition_colors['10mg'], lw=2, label='10mg')

            d1 = DescrStatsW(preds_nosurr_placebo, weights=slope_nosurr_placebo)
            d2 = DescrStatsW(preds_nosurr_5mg, weights=slope_nosurr_5mg)
            d3 = DescrStatsW(preds_nosurr_10mg, weights=slope_nosurr_10mg)

            ax[1].fill_between(CS_data['sub-002']['ses-1']['x_space_nosurr'], np.mean(preds_nosurr_placebo,axis=0)-1.96*d1.std_mean, np.mean(preds_nosurr_placebo,axis=0)+1.96*d1.std_mean, color=condition_colors['placebo'], alpha=0.2)
            ax[1].fill_between(CS_data['sub-002']['ses-1']['x_space_nosurr'], np.mean(preds_nosurr_5mg,axis=0)-1.96*d2.std_mean, np.mean(preds_nosurr_5mg,axis=0)+1.96*d2.std_mean, color=condition_colors['5mg'], alpha=0.2)
            ax[1].fill_between(CS_data['sub-002']['ses-1']['x_space_nosurr'], np.mean(preds_nosurr_10mg,axis=0)-1.96*d3.std_mean, np.mean(preds_nosurr_10mg,axis=0)+1.96*d3.std_mean, color=condition_colors['10mg'], alpha=0.2)
            ax[0].legend(loc='lower right')

            the_space = np.linspace(-0.10, 0.10, len(subject_name_placebo))

            for ii, effsize in enumerate([rel_effsize_placebo,rel_effsize_nosurr_placebo]):
                weighted_average_effsize = np.average(rel_effsize_placebo, weights=slope_placebo)
                weighted_average_effsize_nosurr = np.average(rel_effsize_nosurr_placebo, weights=slope_nosurr_placebo)
                if ii == 0:
                    ax[2].bar(ii-0.3, weighted_average_effsize, color=condition_colors['placebo'], alpha=0.7, width=0.3)
                else :
                    ax[2].bar(ii-0.3, weighted_average_effsize_nosurr, color=condition_colors['placebo'], alpha=0.7, width=0.3)
                
                x_axis = ii*np.ones_like(effsize) + the_space
                ax[2].plot(x_axis-0.3, effsize, marker='s', mec='k', ls='', ms=6, alpha =0.6, color=condition_colors['placebo'])

            for ii, effsize in enumerate([rel_effsize_5mg,rel_effsize_nosurr_5mg]):
                weighted_average_effsize = np.average(rel_effsize_5mg, weights=slope_5mg)
                weighted_average_effsize_nosurr = np.average(rel_effsize_nosurr_5mg, weights=slope_nosurr_5mg)
                if ii == 0:
                    ax[2].bar(ii, weighted_average_effsize, color=condition_colors['5mg'], alpha=0.7, width=0.3)
                else :
                    ax[2].bar(ii, weighted_average_effsize_nosurr, color=condition_colors['5mg'], alpha=0.7, width=0.3)
                x_axis = ii*np.ones_like(effsize) + the_space
                ax[2].plot(x_axis, effsize, marker='s', mec='k', ls='', ms=6, alpha =0.6, color=condition_colors['5mg'])

            for ii, effsize in enumerate([rel_effsize_10mg,rel_effsize_nosurr_10mg]):
                weighted_average_effsize = np.average(rel_effsize_10mg, weights=slope_10mg)
                weighted_average_effsize_nosurr = np.average(rel_effsize_nosurr_10mg, weights=slope_nosurr_10mg)
                if ii == 0:
                    ax[2].bar(ii+0.3, weighted_average_effsize, color=condition_colors['10mg'], alpha=0.7, width=0.3)
                else :
                    ax[2].bar(ii+0.3, weighted_average_effsize_nosurr, color=condition_colors['10mg'], alpha=0.7, width=0.3)
                x_axis = ii*np.ones_like(effsize) + the_space
                ax[2].plot(x_axis+0.3, effsize, marker='s', mec='k', ls='', ms=6, alpha =0.6, color=condition_colors['10mg'])
            
            #plot the links and the text
            for ss in range(len(rel_effsize_placebo)):
                x_axis2 = x_axis[ss]
                ax[2].plot([x_axis2-1.3, x_axis2-1, x_axis2-0.7], [rel_effsize_placebo[ss], rel_effsize_5mg[ss], rel_effsize_10mg[ss]], c='k', linewidth=0.5, alpha=0.5)
                #if sub-013
                if subject_name_placebo[ss] == 'sub-013':
                    ax[2].text(x_axis2-1.3, rel_effsize_placebo[ss]-0.1, subject_name_placebo[ss], fontsize=8)
                else :
                    ax[2].text(x_axis2-1.3, rel_effsize_placebo[ss], subject_name_placebo[ss], fontsize=8)
                ax[2].text(x_axis2-1.3, rel_effsize_placebo[ss]-3, slope_placebo[ss], fontsize=8)
                ax[2].text(x_axis2-1, rel_effsize_5mg[ss]-3, slope_5mg[ss], fontsize=8)
                ax[2].text(x_axis2-0.7, rel_effsize_10mg[ss]-3, slope_10mg[ss], fontsize=8)

            for ss in range(len(rel_effsize_nosurr_placebo)):
                x_axis2 = x_axis[ss]
                ax[2].plot([x_axis2-0.3, x_axis2, x_axis2+0.3], [rel_effsize_nosurr_placebo[ss], rel_effsize_nosurr_5mg[ss], rel_effsize_nosurr_10mg[ss]], c='k', linewidth=0.5, alpha=0.5)
                ax[2].text(x_axis2-0.3, rel_effsize_nosurr_placebo[ss], subject_name_placebo[ss], fontsize=8)
                ax[2].text(x_axis2-0.3, rel_effsize_nosurr_placebo[ss]-3, slope_nosurr_placebo[ss], fontsize=8)
                ax[2].text(x_axis2, rel_effsize_nosurr_5mg[ss]-3, slope_nosurr_5mg[ss], fontsize=8)
                ax[2].text(x_axis2+0.3, rel_effsize_nosurr_10mg[ss]-3, slope_nosurr_10mg[ss], fontsize=8)
 
            ax[2].plot(np.linspace(-1,2,10),np.zeros_like(np.linspace(-1,2,10)), ls='--', c='k', alpha=0.5)
            
            #set ticks with the
            ax[2].set_xticks([0,1])
            ax[2].set_xticklabels(["Surround", "no surround"])
            ax[2].set_xlabel('')
            ax[2].set_ylabel('PSE difference (%RMS)')
            #ax[-1].set_ylim([-90,10])
            ax[2].set_xlim([-0.5,1.5])

            ax[3].set_ylabel('Differences in relative effect sizes between between surround and nosurroud tasks')       
            the_space = np.linspace(-0.10, 0.10, len(subject_name_placebo))
            
            weighted_average_effsize_difference = np.average(rel_effsize_difference_placebo, weights=slope_average_placebo)
            ax[3].bar(-0.3, weighted_average_effsize_difference, color=condition_colors['placebo'], alpha=0.7, width=0.3)
            x_axis = 0*np.ones_like(rel_effsize_difference_placebo) + the_space
            ax[3].plot(x_axis-0.3, rel_effsize_difference_placebo, marker='s', mec='k', ls='', ms=8, color=condition_colors['placebo'])

            weighted_average_effsize_difference = np.average(rel_effsize_difference_5mg, weights=slope_average_5mg)
            ax[3].bar(0, weighted_average_effsize_difference, color=condition_colors['5mg'], alpha=0.7, width=0.3)
            x_axis = 0*np.ones_like(rel_effsize_difference_5mg) + the_space
            ax[3].plot(x_axis, rel_effsize_difference_5mg, marker='s', mec='k', ls='', ms=8, color=condition_colors['5mg'])

            weighted_average_effsize_difference = np.average(rel_effsize_difference_10mg, weights=slope_average_10mg)
            ax[3].bar(0.3, weighted_average_effsize_difference, color=condition_colors['10mg'], alpha=0.7, width=0.3)
            x_axis = 0*np.ones_like(rel_effsize_difference_10mg) + the_space
            ax[3].plot(x_axis+0.3, rel_effsize_difference_10mg, marker='s', mec='k', ls='', ms=8, color=condition_colors['10mg'])
            

            for ss in range(len(rel_effsize_difference_placebo)):

                x_axis2 = x_axis[ss]
                ax[3].plot([x_axis2-0.3, x_axis2, x_axis2+0.3], [rel_effsize_difference_placebo[ss], rel_effsize_difference_5mg[ss], rel_effsize_difference_10mg[ss]], c='k', linewidth=0.5, alpha=0.5)
                ax[3].text(x_axis2-0.3, rel_effsize_difference_placebo[ss], subject_name_placebo[ss], fontsize=8)
                ax[3].text(x_axis2-0.3, rel_effsize_difference_placebo[ss]-3, slope_average_placebo[ss], fontsize=8)
                ax[3].text(x_axis2, rel_effsize_difference_5mg[ss]-3, slope_average_5mg[ss], fontsize=8)
                ax[3].text(x_axis2+0.3, rel_effsize_difference_10mg[ss]-3, slope_average_10mg[ss], fontsize=8)
                
            ax[3].plot(np.linspace(-1,2,10),np.zeros_like(np.linspace(-1,2,10)), ls='--', c='k', alpha=0.5)
            
            #set ticks with the
            ax[3].set_xticks([-0.3, 0, 0.3])
            ax[3].set_xticklabels(['placebo', '5mg', '10mg'])
            ax[3].set_xlabel('')
            ax[3].set_ylabel('PSE[no surr] - PSE[surr] (%RMS)')
            ax[3].set_ylim([-100,5])
            ax[3].set_xlim([-0.5,0.5])

            #get rid of ses-0
            df_CS = df_CS[df_CS.Session != 'ses_0']
            df_CS_regroup = df_CS.groupby(['Subject','Group_type']).first().reset_index()

            """ #check the balance
            print(df_CS_regroup['Subject'].value_counts())
            print(df_CS_regroup.groupby(['Group_type'])['rel_effsize_diff'].count())

            #define p value and f value
            aovrm2way = AnovaRM(df_CS_regroup, 'rel_effsize_diff', 'Subject', within=['Group_type'], aggregate_func=np.mean)
            res2way = aovrm2way.fit()
            print("RM ANOVA all subjects Results:")
            #print only the p value
            p = round(res2way.anova_table['Pr > F'][0], 2)
            #print only the f value
            f = round(res2way.anova_table['F Value'][0], 2)

            #Add the ANOVA info to the plot
            ax[3].text(0.5, -40, 'RM Anova', fontsize=12)
            ax[3].text(0.5, -50, f'F value = {f}', fontsize=12)
            ax[3].text(0.5, -60, f'p = {p}', fontsize=12) """

            fig.savefig(opj(out_path,'group_results_CS_average_slope.pdf'), dpi=600, bbox_inches='tight', transparent=True) 

        def CSgroupplot_average_r2(self,out_path):
            CS_data = np.load(opj(out_path,'data/CS_data.npy'), allow_pickle=True).item()
            df_CS = pd.read_csv(opj(out_path,'data/group_results_CS.csv'))
            condition_colors = {'placebo':'blue', '5mg':'orange', '10mg':'green'}

            fig, ax = pl.subplots(1,4,figsize=(32,8))
            fig.suptitle(f"Center-Surround task - average results with 95% CI (r2 weighted)")

            preds_placebo = []
            preds_5mg = []
            preds_10mg = []
            preds_nosurr_placebo = []
            preds_nosurr_5mg = []
            preds_nosurr_10mg = []
            r2_placebo = []
            r2_5mg = []
            r2_10mg = []
            r2_nosurr_placebo =[]
            r2_nosurr_5mg =[]
            r2_nosurr_10mg =[]
            r2_average_placebo = []
            r2_average_5mg = []
            r2_average_10mg = []
            rel_effsize_placebo = []
            rel_effsize_5mg = []
            rel_effsize_10mg = []
            rel_effsize_nosurr_placebo =[]
            rel_effsize_nosurr_5mg =[]
            rel_effsize_nosurr_10mg =[]
            rel_effsize_difference_placebo = []
            rel_effsize_difference_5mg = []
            rel_effsize_difference_10mg = []
            subject_name_placebo = []
            subject_name_5mg = []
            subject_name_10mg = []

            for participant, sessions in CS_data.items():
                for session, data_ in sessions.items():
                    #is session 0 do nothing
                    if session == 'ses-0':
                        continue
                    else :
                        
                        if data_['group'] == 'placebo':
                            preds_placebo.append(data_['full_pred'])
                            preds_nosurr_placebo.append(data_['full_pred_nosurr'])
                            rel_effsize_placebo.append(data_['rel_effsize'])
                            rel_effsize_nosurr_placebo.append(data_['rel_effsize_nosurr'])
                            rel_effsize_difference_placebo.append(data_['rel_effsize']- data_['rel_effsize_nosurr'])
                            subject_name_placebo.append(participant)
                            if data_['r2'] > 0:
                                r2_placebo.append(round(data_['r2'],2))
                            else:
                                r2_placebo.append(0)
                            if data_['r2_nosurr'] > 0:
                                r2_nosurr_placebo.append(round(data_['r2_nosurr'],2))
                            else:
                                r2_nosurr_placebo.append(0)
                            if data_['r2_average'] > 0:
                                r2_average_placebo.append(round(data_['r2_average'],2))
                            else:
                                r2_average_placebo.append(0)
                        elif data_['group'] == '5mg':
                            preds_5mg.append(data_['full_pred'])
                            preds_nosurr_5mg.append(data_['full_pred_nosurr'])
                            rel_effsize_5mg.append(data_['rel_effsize'])
                            rel_effsize_nosurr_5mg.append(data_['rel_effsize_nosurr'])
                            rel_effsize_difference_5mg.append(data_['rel_effsize']- data_['rel_effsize_nosurr'])
                            subject_name_5mg.append(participant)
                            if data_['r2'] > 0:
                                r2_5mg.append(round(data_['r2'],2))
                            else:
                                r2_5mg.append(0)
                            if data_['r2_nosurr'] > 0:
                                r2_nosurr_5mg.append(round(data_['r2_nosurr'],2))
                            else:
                                r2_nosurr_5mg.append(0)
                            if data_['r2_average'] > 0:
                                r2_average_5mg.append(round(data_['r2_average'],2))
                            else:
                                r2_average_5mg.append(0)
                        elif data_['group'] == '10mg':
                            preds_10mg.append(data_['full_pred'])
                            preds_nosurr_10mg.append(data_['full_pred_nosurr'])
                            rel_effsize_10mg.append(data_['rel_effsize'])
                            rel_effsize_nosurr_10mg.append(data_['rel_effsize_nosurr'])
                            rel_effsize_difference_10mg.append(data_['rel_effsize']- data_['rel_effsize_nosurr'])
                            subject_name_10mg.append(participant)
                            if data_['r2'] > 0:
                                r2_10mg.append(round(data_['r2'],2))
                            else:
                                r2_10mg.append(0)
                            if data_['r2_nosurr'] > 0:
                                r2_nosurr_10mg.append(round(data_['r2_nosurr'],2))
                            else:
                                r2_nosurr_10mg.append(0)
                            if data_['r2_average'] > 0:
                                r2_average_10mg.append(round(data_['r2_average'],2))
                            else:
                                r2_average_10mg.append(0)

            for a in ax[:2]:
                a.set_xlabel('Reference contrast')
                a.set_xlabel('Contrast difference (%RMS)')

                a.plot(CS_data['sub-002']['ses-1']['x_space_comb'],np.ones_like(CS_data['sub-002']['ses-1']['x_space_comb'])*0.5, ls='--', c='k', alpha=0.5, label='PSE')
                a.plot(CS_data['sub-002']['ses-1']['x_space_comb'],np.ones_like(CS_data['sub-002']['ses-1']['x_space_comb']), ls='--',  c='green', alpha=0.5)
                a.plot(CS_data['sub-002']['ses-1']['x_space_comb'],np.zeros_like(CS_data['sub-002']['ses-1']['x_space_comb']), ls='--', c='red', alpha=0.5)
                a.plot(CS_data['sub-002']['ses-1']['target_contrast']*np.ones(100),np.linspace(0,1,100), ls='-', c='k', label='Veridical', alpha=0.5)

                a.set_xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7])
                a.set_xticklabels(['-75','-50','-25','0','+25','+50','+75'])

                #a.legend(loc='lower right')

                a.set_ylabel('Prob ref perceived as higher contrast')       

            #ax[0].errorbar(this_run.contrast_values, np.mean(data,axis=0), yerr=sem(data,axis=0), marker='s', ls='', c='k', ms=6)
            # ax[0].plot(CS_data['sub-002']['ses-1']['x_space'], np.mean(preds_placebo,axis=0), c=condition_colors['placebo'], lw=2, label='placebo')
            # ax[0].plot(CS_data['sub-002']['ses-1']['x_space'], np.mean(preds_5mg,axis=0), c=condition_colors['5mg'], lw=2, label='5mg')
            # ax[0].plot(CS_data['sub-002']['ses-1']['x_space'], np.mean(preds_10mg,axis=0), c=condition_colors['10mg'], lw=2, label='10mg')

            

            # print(preds_placebo[1,:])
            d1 = DescrStatsW(preds_placebo, weights=r2_placebo)
            d2 = DescrStatsW(preds_5mg, weights=r2_5mg)
            d3 = DescrStatsW(preds_10mg, weights=r2_10mg)     
            
            ax[0].plot(CS_data['sub-002']['ses-1']['x_space'], np.average(preds_placebo, axis=0, weights=r2_placebo), c=condition_colors['placebo'], lw=2, label='placebo')
            ax[0].plot(CS_data['sub-002']['ses-1']['x_space'], np.average(preds_5mg, axis=0, weights=r2_5mg), c=condition_colors['5mg'], lw=2, label='5mg')
            ax[0].plot(CS_data['sub-002']['ses-1']['x_space'], np.average(preds_10mg, axis=0, weights=r2_10mg), c=condition_colors['10mg'], lw=2, label='10mg')

            #add the 95% confidence interval for the average
            ax[0].fill_between( CS_data['sub-002']['ses-1']['x_space'], np.average(preds_placebo, axis=0, weights=r2_placebo) - 1.96 * d1.std_mean, np.average(preds_placebo, axis=0, weights=r2_placebo) + 1.96 * d1.std_mean, color=condition_colors['placebo'], alpha=0.2 )
            ax[0].fill_between( CS_data['sub-002']['ses-1']['x_space'], np.average(preds_5mg, axis=0, weights=r2_5mg) - 1.96 * d2.std_mean, np.average(preds_5mg, axis=0, weights=r2_5mg) + 1.96 * d2.std_mean, color=condition_colors['5mg'], alpha=0.2 )
            ax[0].fill_between( CS_data['sub-002']['ses-1']['x_space'], np.average(preds_10mg, axis=0, weights=r2_10mg) - 1.96 * d3.std_mean, np.average(preds_10mg, axis=0, weights=r2_10mg) + 1.96 * d3.std_mean, color=condition_colors['10mg'], alpha=0.2 )
            
            #ax[1].errorbar(this_run.contrast_values_nosurr, np.mean(data_nosurr,axis=0), yerr=sem(data_nosurr,axis=0), marker='s', ls='', c='k', ms=6)
            ax[1].plot(CS_data['sub-002']['ses-1']['x_space_nosurr'], np.mean(preds_nosurr_placebo,axis=0), c=condition_colors['placebo'], lw=2, label='placebo')
            ax[1].plot(CS_data['sub-002']['ses-1']['x_space_nosurr'], np.mean(preds_nosurr_5mg,axis=0), c=condition_colors['5mg'], lw=2, label='5mg')
            ax[1].plot(CS_data['sub-002']['ses-1']['x_space_nosurr'], np.mean(preds_nosurr_10mg,axis=0), c=condition_colors['10mg'], lw=2, label='10mg')

            d1 = DescrStatsW(preds_nosurr_placebo, weights=r2_nosurr_placebo)
            d2 = DescrStatsW(preds_nosurr_5mg, weights=r2_nosurr_5mg)
            d3 = DescrStatsW(preds_nosurr_10mg, weights=r2_nosurr_10mg)

            ax[1].fill_between(CS_data['sub-002']['ses-1']['x_space_nosurr'], np.mean(preds_nosurr_placebo,axis=0)-1.96*d1.std_mean, np.mean(preds_nosurr_placebo,axis=0)+1.96*d1.std_mean, color=condition_colors['placebo'], alpha=0.2)
            ax[1].fill_between(CS_data['sub-002']['ses-1']['x_space_nosurr'], np.mean(preds_nosurr_5mg,axis=0)-1.96*d2.std_mean, np.mean(preds_nosurr_5mg,axis=0)+1.96*d2.std_mean, color=condition_colors['5mg'], alpha=0.2)
            ax[1].fill_between(CS_data['sub-002']['ses-1']['x_space_nosurr'], np.mean(preds_nosurr_10mg,axis=0)-1.96*d3.std_mean, np.mean(preds_nosurr_10mg,axis=0)+1.96*d3.std_mean, color=condition_colors['10mg'], alpha=0.2)
            ax[0].legend(loc='lower right')

            the_space = np.linspace(-0.10, 0.10, len(subject_name_placebo))

            for ii, effsize in enumerate([rel_effsize_placebo,rel_effsize_nosurr_placebo]):
                weighted_average_effsize = np.average(rel_effsize_placebo, weights=r2_placebo)
                weighted_average_effsize_nosurr = np.average(rel_effsize_nosurr_placebo, weights=r2_nosurr_placebo)
                if ii == 0:
                    ax[2].bar(ii-0.3, weighted_average_effsize, color=condition_colors['placebo'], alpha=0.7, width=0.3)
                else :
                    ax[2].bar(ii-0.3, weighted_average_effsize_nosurr, color=condition_colors['placebo'], alpha=0.7, width=0.3)
                
                x_axis = ii*np.ones_like(effsize) + the_space
                ax[2].plot(x_axis-0.3, effsize, marker='s', mec='k', ls='', ms=6, alpha =0.6, color=condition_colors['placebo'])

            for ii, effsize in enumerate([rel_effsize_5mg,rel_effsize_nosurr_5mg]):
                weighted_average_effsize = np.average(rel_effsize_5mg, weights=r2_5mg)
                weighted_average_effsize_nosurr = np.average(rel_effsize_nosurr_5mg, weights=r2_nosurr_5mg)
                if ii == 0:
                    ax[2].bar(ii, weighted_average_effsize, color=condition_colors['5mg'], alpha=0.7, width=0.3)
                else :
                    ax[2].bar(ii, weighted_average_effsize_nosurr, color=condition_colors['5mg'], alpha=0.7, width=0.3)
                x_axis = ii*np.ones_like(effsize) + the_space
                ax[2].plot(x_axis, effsize, marker='s', mec='k', ls='', ms=6, alpha =0.6, color=condition_colors['5mg'])

            for ii, effsize in enumerate([rel_effsize_10mg,rel_effsize_nosurr_10mg]):
                weighted_average_effsize = np.average(rel_effsize_10mg, weights=r2_10mg)
                weighted_average_effsize_nosurr = np.average(rel_effsize_nosurr_10mg, weights=r2_nosurr_10mg)
                if ii == 0:
                    ax[2].bar(ii+0.3, weighted_average_effsize, color=condition_colors['10mg'], alpha=0.7, width=0.3)
                else :
                    ax[2].bar(ii+0.3, weighted_average_effsize_nosurr, color=condition_colors['10mg'], alpha=0.7, width=0.3)
                x_axis = ii*np.ones_like(effsize) + the_space
                ax[2].plot(x_axis+0.3, effsize, marker='s', mec='k', ls='', ms=6, alpha =0.6, color=condition_colors['10mg'])
            
            #plot the links and the text
            for ss in range(len(rel_effsize_placebo)):
                x_axis2 = x_axis[ss]
                ax[2].plot([x_axis2-1.3, x_axis2-1, x_axis2-0.7], [rel_effsize_placebo[ss], rel_effsize_5mg[ss], rel_effsize_10mg[ss]], c='k', linewidth=0.5, alpha=0.5)
                #if sub-013
                if subject_name_placebo[ss] == 'sub-013':
                    ax[2].text(x_axis2-1.3, rel_effsize_placebo[ss]-0.1, subject_name_placebo[ss], fontsize=8)
                else :
                    ax[2].text(x_axis2-1.3, rel_effsize_placebo[ss], subject_name_placebo[ss], fontsize=8)
                ax[2].text(x_axis2-1.3, rel_effsize_placebo[ss]-3, r2_placebo[ss], fontsize=8)
                ax[2].text(x_axis2-1, rel_effsize_5mg[ss]-3, r2_5mg[ss], fontsize=8)
                ax[2].text(x_axis2-0.7, rel_effsize_10mg[ss]-3, r2_10mg[ss], fontsize=8)

            for ss in range(len(rel_effsize_nosurr_placebo)):
                x_axis2 = x_axis[ss]
                ax[2].plot([x_axis2-0.3, x_axis2, x_axis2+0.3], [rel_effsize_nosurr_placebo[ss], rel_effsize_nosurr_5mg[ss], rel_effsize_nosurr_10mg[ss]], c='k', linewidth=0.5, alpha=0.5)
                ax[2].text(x_axis2-0.3, rel_effsize_nosurr_placebo[ss], subject_name_placebo[ss], fontsize=8)
                ax[2].text(x_axis2-0.3, rel_effsize_nosurr_placebo[ss]-3, r2_nosurr_placebo[ss], fontsize=8)
                ax[2].text(x_axis2, rel_effsize_nosurr_5mg[ss]-3, r2_nosurr_5mg[ss], fontsize=8)
                ax[2].text(x_axis2+0.3, rel_effsize_nosurr_10mg[ss]-3, r2_nosurr_10mg[ss], fontsize=8)
 
            ax[2].plot(np.linspace(-1,2,10),np.zeros_like(np.linspace(-1,2,10)), ls='--', c='k', alpha=0.5)
            
            #set ticks with the
            ax[2].set_xticks([0,1])
            ax[2].set_xticklabels(["Surround", "no surround"])
            ax[2].set_xlabel('')
            ax[2].set_ylabel('PSE difference (%RMS)')
            #ax[-1].set_ylim([-90,10])
            ax[2].set_xlim([-0.5,1.5])

            ax[3].set_ylabel('Differences in relative effect sizes between between surround and nosurroud tasks')       
            the_space = np.linspace(-0.10, 0.10, len(subject_name_placebo))
            
            weighted_average_effsize_difference = np.average(rel_effsize_difference_placebo, weights=r2_average_placebo)
            ax[3].bar(-0.3, weighted_average_effsize_difference, color=condition_colors['placebo'], alpha=0.7, width=0.3)
            x_axis = 0*np.ones_like(rel_effsize_difference_placebo) + the_space
            ax[3].plot(x_axis-0.3, rel_effsize_difference_placebo, marker='s', mec='k', ls='', ms=8, color=condition_colors['placebo'])

            weighted_average_effsize_difference = np.average(rel_effsize_difference_5mg, weights=r2_average_5mg)
            ax[3].bar(0, weighted_average_effsize_difference, color=condition_colors['5mg'], alpha=0.7, width=0.3)
            x_axis = 0*np.ones_like(rel_effsize_difference_5mg) + the_space
            ax[3].plot(x_axis, rel_effsize_difference_5mg, marker='s', mec='k', ls='', ms=8, color=condition_colors['5mg'])

            weighted_average_effsize_difference = np.average(rel_effsize_difference_10mg, weights=r2_average_10mg)
            ax[3].bar(0.3, weighted_average_effsize_difference, color=condition_colors['10mg'], alpha=0.7, width=0.3)
            x_axis = 0*np.ones_like(rel_effsize_difference_10mg) + the_space
            ax[3].plot(x_axis+0.3, rel_effsize_difference_10mg, marker='s', mec='k', ls='', ms=8, color=condition_colors['10mg'])
            

            for ss in range(len(rel_effsize_difference_placebo)):

                x_axis2 = x_axis[ss]
                ax[3].plot([x_axis2-0.3, x_axis2, x_axis2+0.3], [rel_effsize_difference_placebo[ss], rel_effsize_difference_5mg[ss], rel_effsize_difference_10mg[ss]], c='k', linewidth=0.5, alpha=0.5)
                ax[3].text(x_axis2-0.3, rel_effsize_difference_placebo[ss], subject_name_placebo[ss], fontsize=8)
                ax[3].text(x_axis2-0.3, rel_effsize_difference_placebo[ss]-3, r2_average_placebo[ss], fontsize=8)
                ax[3].text(x_axis2, rel_effsize_difference_5mg[ss]-3, r2_average_5mg[ss], fontsize=8)
                ax[3].text(x_axis2+0.3, rel_effsize_difference_10mg[ss]-3, r2_average_10mg[ss], fontsize=8)
                
            ax[3].plot(np.linspace(-1,2,10),np.zeros_like(np.linspace(-1,2,10)), ls='--', c='k', alpha=0.5)
            
            #set ticks with the
            ax[3].set_xticks([-0.3, 0, 0.3])
            ax[3].set_xticklabels(['placebo', '5mg', '10mg'])
            ax[3].set_xlabel('')
            ax[3].set_ylabel('PSE[no surr] - PSE[surr] (%RMS)')
            ax[3].set_ylim([-100,5])
            ax[3].set_xlim([-0.5,0.5])

            #get rid of ses-0
            df_CS = df_CS[df_CS.Session != 'ses_0']
            df_CS_regroup = df_CS.groupby(['Subject','Group_type']).first().reset_index()

            """ #check the balance
            print(df_CS_regroup['Subject'].value_counts())
            print(df_CS_regroup.groupby(['Group_type'])['rel_effsize_diff'].count())

            #define p value and f value
            aovrm2way = AnovaRM(df_CS_regroup, 'rel_effsize_diff', 'Subject', within=['Group_type'], aggregate_func=np.mean)
            res2way = aovrm2way.fit()
            print("RM ANOVA all subjects Results:")
            #print only the p value
            p = round(res2way.anova_table['Pr > F'][0], 2)
            #print only the f value
            f = round(res2way.anova_table['F Value'][0], 2)

            #Add the ANOVA info to the plot
            ax[3].text(0.5, -40, 'RM Anova', fontsize=12)
            ax[3].text(0.5, -50, f'F value = {f}', fontsize=12)
            ax[3].text(0.5, -60, f'p = {p}', fontsize=12) """

            fig.savefig(opj(out_path,'group_results_CS_average_r2.pdf'), dpi=600, bbox_inches='tight', transparent=True) 

        def CSgroupfit(self, out_path):
            groups_dict = {'placebo':['sub-001_ses-1','sub-002_ses-2','sub-003_ses-3','sub-004_ses-2','sub-005_ses-1','sub-006_ses-3','sub-007_ses-3','sub-008_ses-1',
                         'sub-009_ses-2','sub-010_ses-1','sub-011_ses-3','sub-012_ses-2','sub-013_ses-3','sub-014_ses-2','sub-015_ses-2','sub-018_ses-1','sub-019_ses-2','sub-020_ses-1'],

              '5mg':['sub-001_ses-3','sub-002_ses-1','sub-003_ses-1','sub-004_ses-3','sub-005_ses-2','sub-006_ses-1','sub-007_ses-1','sub-008_ses-3',
                    'sub-009_ses-1','sub-010_ses-2','sub-011_ses-2','sub-012_ses-3','sub-013_ses-2','sub-014_ses-1','sub-015_ses-3','sub-018_ses-3','sub-019_ses-1','sub-020_ses-2'],

              '10mg':['sub-001_ses-2','sub-002_ses-3','sub-003_ses-2','sub-004_ses-1','sub-005_ses-3','sub-006_ses-2','sub-007_ses-2','sub-008_ses-2',
                     'sub-009_ses-3','sub-010_ses-3','sub-011_ses-1','sub-012_ses-1','sub-013_ses-1','sub-014_ses-3','sub-015_ses-1','sub-018_ses-2','sub-019_ses-3','sub-020_ses-3']}

            df = pd.DataFrame()
            CS_data = {}

            for subject in getExpAttr(self):
                this_subject = getattr(self,subject)
                participant_id = rp(subject).split('_')[0]
                CS_data[participant_id] = {}
                for session in getExpAttr(this_subject):
                    this_session = getattr(this_subject,session)
                    session_id = rp(session).split('_')[0]
                    CS_data[participant_id][session_id] = {}
                    for run in getExpAttr(this_session):
                        this_run = getattr(this_session, run)

                        group_subject_session = ''
                        for group in groups_dict:
                            if f"{participant_id}_{session_id}" in groups_dict[group]:
                                group_subject_session = group
                                break
                        # print(participant_id, session_id, group_subject_session)
                        #print(f'Length of contrast_values: {len(this_run.contrast_values)}')
                        #print(f'Length of probs: {len(this_run.probs)}')
                        # Add similar lines for other arrays
                        
                        new = pd.DataFrame({'Subject':subject, 'Session':session,'Group_type':group_subject_session, 'Contrast_values' : this_run.contrast_values, 'Actual':this_run.probs, 'var':this_run.contrast_values-this_run.probs, 'Contrast_values_nosurr' : this_run.contrast_values_nosurr, 'Actual_nosurr':this_run.probs_nosurr, 'var_nosurr':this_run.contrast_values_nosurr-this_run.probs_nosurr, 'rel_effsize_diff':this_run.rel_effsize-this_run.rel_effsize_nosurr, 'slope':this_run.slope, 'slope_nosurr':this_run.slope_nosurr, 'slope_average':(this_run.slope+this_run.slope_nosurr)/2, 'r2':this_run.r2, 'r2_nosurr':this_run.r2_nosurr, 'r2_average':(this_run.r2+this_run.r2_nosurr)/2})
                        df = pd.concat([df, new], ignore_index=True) 
                        CS_data[participant_id][session_id] = {'group':group_subject_session, 'contrast_values' : this_run.contrast_values, 'probs':this_run.probs, 'nij':this_run.nij, 'rij':this_run.rij, 'diff':this_run.contrast_values-this_run.probs, 'contrast_values_nosurr' : this_run.contrast_values_nosurr, 'probs_nosurr':this_run.probs_nosurr, 'nij_nosurr':this_run.nij_nosurr, 'rij_nosurr':this_run.rij_nosurr , 'diff_nosurr':this_run.contrast_values_nosurr-this_run.probs_nosurr, 'x_space':this_run.x_space, 'full_pred':this_run.full_pred, 'x_space_nosurr':this_run.x_space_nosurr, 'full_pred_nosurr':this_run.full_pred_nosurr, 'rel_effsize':this_run.rel_effsize, 'rel_effsize_nosurr':this_run.rel_effsize_nosurr,'rel_effsize_diff':float(this_run.rel_effsize)-float(this_run.rel_effsize_nosurr), 'x_space_comb':this_run.x_space_comb, 'target_contrast':this_run.expsettings['Stimulus Settings']['Target contrast'], 'slope':this_run.slope, 'slope_nosurr':this_run.slope_nosurr, 'slope_average':(this_run.slope+this_run.slope_nosurr)/2, 'r2':this_run.r2, 'r2_nosurr':this_run.r2_nosurr, 'r2_average':(this_run.r2+this_run.r2_nosurr)/2}
                        
            #change all rel_effsize_diff of subject 9 by minus itself
            if 'sub-009' in CS_data.keys():
                CS_data['sub-009']['ses-1']['rel_effsize_diff'] = CS_data['sub-009']['ses-1']['rel_effsize_diff']*-1
                CS_data['sub-009']['ses-2']['rel_effsize_diff'] = CS_data['sub-009']['ses-2']['rel_effsize_diff']*-1
                CS_data['sub-009']['ses-3']['rel_effsize_diff'] = CS_data['sub-009']['ses-3']['rel_effsize_diff']*-1
                CS_data['sub-009']['ses-1']['rel_effsize'] = CS_data['sub-009']['ses-1']['rel_effsize']*-1
                CS_data['sub-009']['ses-2']['rel_effsize'] = CS_data['sub-009']['ses-2']['rel_effsize']*-1
                CS_data['sub-009']['ses-3']['rel_effsize'] = CS_data['sub-009']['ses-3']['rel_effsize']*-1
                CS_data['sub-009']['ses-1']['rel_effsize_nosurr'] = CS_data['sub-009']['ses-1']['rel_effsize_nosurr']*-1
                CS_data['sub-009']['ses-2']['rel_effsize_nosurr'] = CS_data['sub-009']['ses-2']['rel_effsize_nosurr']*-1
                CS_data['sub-009']['ses-3']['rel_effsize_nosurr'] = CS_data['sub-009']['ses-3']['rel_effsize_nosurr']*-1
            
            if 'sub_009' in df['Subject'].values:
                df.loc[df['Subject'] == 'sub_009', 'rel_effsize_diff'] = df.loc[df['Subject'] == 'sub_009', 'rel_effsize_diff']*-1

            #order the data by subject
            df = df.sort_values(by=['Subject','Session'])
            CS_data = dict(sorted(CS_data.items()))
            
            np.save(opj(out_path,'data/CS_data.npy'), CS_data)

            #save the dataframe
            df.to_csv(opj(out_path,'data/group_results_CS.csv'), index=False)

        def EHDBgroupplot_placebo(self, out_path):
            EHDB_data = np.load(opj(out_path,'data/EHDB_data.npy'), allow_pickle=True).item()

            preds = dd(list)
            data = dd(list)
            rel_effsize = dd(list)

            fig, ax = pl.subplots(1,3,figsize=(24,8))

            for participant, sessions in EHDB_data.items():
                for session, data_ in sessions.items():
                    #is session 0 do nothing
                    if session == 'ses-0':
                        continue
                    else :
                        if data_['group_type'] == 'placebo':
                            for ii, surr_size in enumerate(data_['surr_sizes']):
                                ax[ii].plot(data_['x_spaces'][str(surr_size)],data_['preds'][str(surr_size)], c='k', alpha=0.5)
                                ax[ii].plot(data_['size_values'][str(surr_size)],data_['probs'][str(surr_size)], marker='^', ls='', c='grey', ms=4, alpha=0.5)

                                preds[str(surr_size)].append(data_['preds'][str(surr_size)])
                                data[str(surr_size)].append(data_['probs'][str(surr_size)])

                                rel_effsize[str(surr_size)].append(data_['rel_effsize'][str(surr_size)])

                            #if data_['group'] == 'placebo':
            for ii, surr_size in enumerate(EHDB_data['sub-002']['ses-1']['surr_sizes']):
                ax[ii].set_title(f"Surr size {surr_size:.3f}")
                ax[ii].set_xscale('log')
                ax[ii].set_xlabel('Size difference (% veridical size)')
                ax[ii].set_ylabel('Prob ref perceived as larger')
                ax[ii].minorticks_off()

                true_size = EHDB_data['sub-002']['ses-1']['true_size']

                #ugly
                if surr_size>0:
                    ax[ii].set_xticks(np.linspace(true_size-0.5*true_size,true_size+0.3*true_size,9))
                    ax[ii].set_xticklabels([-50,-40,-30,-20,-10,0,10,20,30])
                else:
                    ax[ii].set_xticks(np.linspace(true_size-0.3*true_size,true_size+0.3*true_size,7))
                    ax[ii].set_xticklabels([-30,-20,-10,0,10,20,30])  
                
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)],np.ones_like(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)])*0.5, ls='--', c='k', alpha=0.5, label='PSE')
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)],np.ones_like(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)]), ls='--', c='green', alpha=0.5)
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)],np.zeros_like(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)]), ls='--', c='red', alpha=0.5)
                #bit hacky way
                ax[ii].plot(true_size*np.ones(100),np.linspace(0,1,100), ls='-', c='k', alpha=0.5, label='Veridical')
                ax[ii].legend(loc='lower right')

                #ax[ii].errorbar(this_run.size_values[str(surr_size)], np.mean(data[str(surr_size)],axis=0), yerr=sem(data[str(surr_size)],axis=0), marker='s', ls='', c='k', ms=6)
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)], np.mean(preds[str(surr_size)],axis=0), c='k', lw=4)


                
                ax[-1].bar(ii, np.mean(rel_effsize[str(surr_size)],axis=0))

                ax[-1].errorbar(ii*np.ones_like(rel_effsize[str(surr_size)]), rel_effsize[str(surr_size)], marker='s', mec='k', ls='', ms=8)
            
            for ss in range(len(rel_effsize[str(surr_size)])):
                ax[-1].plot(np.arange(len(EHDB_data['sub-002']['ses-1']['surr_sizes'])), [rel_effsize[ref][ss] for ref in rel_effsize], c='k')
                
            ax[-1].plot(np.arange(-1,1+len(EHDB_data['sub-002']['ses-1']['surr_sizes'])),np.zeros_like(np.arange(-1,1+len(EHDB_data['sub-002']['ses-1']['surr_sizes']))), ls='--', c='k', alpha=0.5)
                
            
            ax[-1].set_xticks(np.arange(len(EHDB_data['sub-002']['ses-1']['surr_sizes'])))
            ax[-1].set_xticklabels([f"{surr_size:.3f}" for surr_size in EHDB_data['sub-002']['ses-1']['surr_sizes']])
            ax[-1].set_xlabel('Surround size (deg)')
            ax[-1].set_ylabel('PSE difference (% veridical size)')

            ax[-1].set_yticks(np.linspace(0,-30,7))
            ax[-1].set_yticklabels([0,-5,-10,-15,-20,-25,-30])


            fig.suptitle("EHDB task - placebo")


            fig.savefig(opj(out_path,f"group_results_EHDB_placebo.pdf"), dpi=600, bbox_inches='tight', transparent=True)   

        def EHDBgroupplot_5mg(self, out_path):
            EHDB_data = np.load(opj(out_path,'data/EHDB_data.npy'), allow_pickle=True).item()

            preds = dd(list)
            data = dd(list)
            rel_effsize = dd(list)

            fig, ax = pl.subplots(1,3,figsize=(24,8))

            for participant, sessions in EHDB_data.items():
                for session, data_ in sessions.items():
                    #is session 0 do nothing
                    if session == 'ses-0':
                        continue
                    else :
                        if data_['group_type'] == '5mg':
                            for ii, surr_size in enumerate(data_['surr_sizes']):
                                ax[ii].plot(data_['x_spaces'][str(surr_size)],data_['preds'][str(surr_size)], c='k', lw=0.8, alpha=0.8)
                                ax[ii].plot(data_['size_values'][str(surr_size)],data_['probs'][str(surr_size)], marker='^', ls='', c='grey', ms=4, alpha=0.8)

                                preds[str(surr_size)].append(data_['preds'][str(surr_size)])
                                data[str(surr_size)].append(data_['probs'][str(surr_size)])

                                rel_effsize[str(surr_size)].append(data_['rel_effsize'][str(surr_size)])

                            #if data_['group'] == 'placebo':
            for ii, surr_size in enumerate(EHDB_data['sub-002']['ses-1']['surr_sizes']):
                ax[ii].set_title(f"Surr size {surr_size:.3f}")
                ax[ii].set_xscale('log')
                ax[ii].set_xlabel('Size difference (% veridical size)')
                ax[ii].set_ylabel('Prob ref perceived as larger')
                ax[ii].minorticks_off()

                true_size = EHDB_data['sub-002']['ses-1']['true_size']

                #ugly
                if surr_size>0:
                    ax[ii].set_xticks(np.linspace(true_size-0.5*true_size,true_size+0.3*true_size,9))
                    ax[ii].set_xticklabels([-50,-40,-30,-20,-10,0,10,20,30])
                else:
                    ax[ii].set_xticks(np.linspace(true_size-0.3*true_size,true_size+0.3*true_size,7))
                    ax[ii].set_xticklabels([-30,-20,-10,0,10,20,30])  
                
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)],np.ones_like(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)])*0.5, ls='--', c='k', alpha=0.5, label='PSE')
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)],np.ones_like(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)]), ls='--', c='green', alpha=0.5)
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)],np.zeros_like(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)]), ls='--', c='red', alpha=0.5)
                #bit hacky way
                ax[ii].plot(true_size*np.ones(100),np.linspace(0,1,100), ls='-', c='k', alpha=0.5, label='Veridical')
                ax[ii].legend(loc='lower right')

                #ax[ii].errorbar(this_run.size_values[str(surr_size)], np.mean(data[str(surr_size)],axis=0), yerr=sem(data[str(surr_size)],axis=0), marker='s', ls='', c='k', ms=6)
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)], np.mean(preds[str(surr_size)],axis=0), c='k', lw=4)


                
                ax[-1].bar(ii, np.mean(rel_effsize[str(surr_size)],axis=0))

                ax[-1].errorbar(ii*np.ones_like(rel_effsize[str(surr_size)]), rel_effsize[str(surr_size)], marker='s', mec='k', ls='', ms=8)
            
            for ss in range(len(rel_effsize[str(surr_size)])):
                ax[-1].plot(np.arange(len(EHDB_data['sub-002']['ses-1']['surr_sizes'])), [rel_effsize[ref][ss] for ref in rel_effsize], c='k')
                
            ax[-1].plot(np.arange(-1,1+len(EHDB_data['sub-002']['ses-1']['surr_sizes'])),np.zeros_like(np.arange(-1,1+len(EHDB_data['sub-002']['ses-1']['surr_sizes']))), ls='--', c='k', alpha=0.5)
                
            
            ax[-1].set_xticks(np.arange(len(EHDB_data['sub-002']['ses-1']['surr_sizes'])))
            ax[-1].set_xticklabels([f"{surr_size:.3f}" for surr_size in EHDB_data['sub-002']['ses-1']['surr_sizes']])
            ax[-1].set_xlabel('Surround size (deg)')
            ax[-1].set_ylabel('PSE difference (% veridical size)')

            ax[-1].set_yticks(np.linspace(0,-30,7))
            ax[-1].set_yticklabels([0,-5,-10,-15,-20,-25,-30])

            fig.suptitle("EHDB task - 5mg")


            fig.savefig(opj(out_path,f"group_results_EHDB_5mg.pdf"), dpi=600, bbox_inches='tight', transparent=True)   

        def EHDBgroupplot_10mg(self, out_path):
            EHDB_data = np.load(opj(out_path,'data/EHDB_data.npy'), allow_pickle=True).item()

            preds = dd(list)
            data = dd(list)
            rel_effsize = dd(list)

            fig, ax = pl.subplots(1,3,figsize=(24,8))

            for participant, sessions in EHDB_data.items():
                for session, data_ in sessions.items():
                    #is session 0 do nothing
                    if session == 'ses-0':
                        continue
                    else :
                        if data_['group_type'] == '10mg':
                            for ii, surr_size in enumerate(data_['surr_sizes']):
                                ax[ii].plot(data_['x_spaces'][str(surr_size)],data_['preds'][str(surr_size)], c='k', lw=0.8, alpha=0.8)
                                ax[ii].plot(data_['size_values'][str(surr_size)],data_['probs'][str(surr_size)], marker='^', ls='', c='grey', ms=4, alpha=0.8)

                                preds[str(surr_size)].append(data_['preds'][str(surr_size)])
                                data[str(surr_size)].append(data_['probs'][str(surr_size)])

                                rel_effsize[str(surr_size)].append(data_['rel_effsize'][str(surr_size)])

                            #if data_['group'] == 'placebo':
            for ii, surr_size in enumerate(EHDB_data['sub-002']['ses-1']['surr_sizes']):
                ax[ii].set_title(f"Surr size {surr_size:.3f}")
                ax[ii].set_xscale('log')
                ax[ii].set_xlabel('Size difference (% veridical size)')
                ax[ii].set_ylabel('Prob ref perceived as larger')
                ax[ii].minorticks_off()

                true_size = EHDB_data['sub-002']['ses-1']['true_size']

                #ugly
                if surr_size>0:
                    ax[ii].set_xticks(np.linspace(true_size-0.5*true_size,true_size+0.3*true_size,9))
                    ax[ii].set_xticklabels([-50,-40,-30,-20,-10,0,10,20,30])
                else:
                    ax[ii].set_xticks(np.linspace(true_size-0.3*true_size,true_size+0.3*true_size,7))
                    ax[ii].set_xticklabels([-30,-20,-10,0,10,20,30])  
                
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)],np.ones_like(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)])*0.5, ls='--', c='k', alpha=0.5, label='PSE')
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)],np.ones_like(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)]), ls='--', c='green', alpha=0.5)
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)],np.zeros_like(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)]), ls='--', c='red', alpha=0.5)
                #bit hacky way
                ax[ii].plot(true_size*np.ones(100),np.linspace(0,1,100), ls='-', c='k', alpha=0.5, label='Veridical')
                ax[ii].legend(loc='lower right')

                #ax[ii].errorbar(this_run.size_values[str(surr_size)], np.mean(data[str(surr_size)],axis=0), yerr=sem(data[str(surr_size)],axis=0), marker='s', ls='', c='k', ms=6)
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)], np.mean(preds[str(surr_size)],axis=0), c='k', lw=4)


                
                ax[-1].bar(ii, np.mean(rel_effsize[str(surr_size)],axis=0))

                ax[-1].errorbar(ii*np.ones_like(rel_effsize[str(surr_size)]), rel_effsize[str(surr_size)], marker='s', mec='k', ls='', ms=8)
            
            for ss in range(len(rel_effsize[str(surr_size)])):
                ax[-1].plot(np.arange(len(EHDB_data['sub-002']['ses-1']['surr_sizes'])), [rel_effsize[ref][ss] for ref in rel_effsize], c='k')
                
            ax[-1].plot(np.arange(-1,1+len(EHDB_data['sub-002']['ses-1']['surr_sizes'])),np.zeros_like(np.arange(-1,1+len(EHDB_data['sub-002']['ses-1']['surr_sizes']))), ls='--', c='k', alpha=0.5)
                
            
            ax[-1].set_xticks(np.arange(len(EHDB_data['sub-002']['ses-1']['surr_sizes'])))
            ax[-1].set_xticklabels([f"{surr_size:.3f}" for surr_size in EHDB_data['sub-002']['ses-1']['surr_sizes']])
            ax[-1].set_xlabel('Surround size (deg)')
            ax[-1].set_ylabel('PSE difference (% veridical size)')

            ax[-1].set_yticks(np.linspace(0,-30,7))
            ax[-1].set_yticklabels([0,-5,-10,-15,-20,-25,-30])


            fig.suptitle("EHDB task - 10mg")


            fig.savefig(opj(out_path,f"group_results_EHDB_10mg.pdf"), dpi=600, bbox_inches='tight', transparent=True)

        def EHDBgroupplot(self, out_path):
            EHDB_data = np.load(opj(out_path,'data/EHDB_data.npy'), allow_pickle=True).item()
            condition_colors = {'placebo':'blue', '5mg':'orange', '10mg':'green'}

            preds = dd(list)
            preds_placebo = dd(list)
            preds_5mg = dd(list)
            preds_10mg = dd(list)
            data = dd(list)
            rel_effsize = dd(list)
            rel_effsize_placebo = dd(list)
            rel_effsize_5mg = dd(list)
            rel_effsize_10mg = dd(list)
            subject_name_placebo = []
            subject_name_5mg = []
            subject_name_10mg = []

            fig, ax = pl.subplots(1,3,figsize=(8*3,8))

            for participant, sessions in EHDB_data.items():
                for session, data_ in sessions.items():
                    #is session 0 do nothing
                    if session == 'ses-0':
                        continue
                    else :
                        for ii, surr_size in enumerate(data_['surr_sizes']):
                            ax[ii].plot(data_['x_spaces'][str(surr_size)],data_['preds'][str(surr_size)], c='k', lw=0.5, alpha=0.5)
                            ax[ii].plot(data_['size_values'][str(surr_size)],data_['probs'][str(surr_size)], marker='^', ls='', c='grey', ms=4, alpha=0.8)

                            preds[str(surr_size)].append(data_['preds'][str(surr_size)])
                            data[str(surr_size)].append(data_['probs'][str(surr_size)])

                            rel_effsize[str(surr_size)].append(data_['rel_effsize'][str(surr_size)])
                        
                            if data_['group_type'] == 'placebo':
                                preds_placebo[str(surr_size)].append(data_['preds'][str(surr_size)])
                                rel_effsize_placebo[str(surr_size)].append(data_['rel_effsize'][str(surr_size)])
                                subject_name_placebo.append(participant)
                            elif data_['group_type'] == '5mg':
                                preds_5mg[str(surr_size)].append(data_['preds'][str(surr_size)])
                                rel_effsize_5mg[str(surr_size)].append(data_['rel_effsize'][str(surr_size)])
                                subject_name_5mg.append(participant)
                            elif data_['group_type'] == '10mg':
                                preds_10mg[str(surr_size)].append(data_['preds'][str(surr_size)])
                                rel_effsize_10mg[str(surr_size)].append(data_['rel_effsize'][str(surr_size)])
                                subject_name_10mg.append(participant)
            

            for ii, surr_size in enumerate(EHDB_data['sub-002']['ses-1']['surr_sizes']): #example, could be any subject
                ax[ii].set_title(f"Surr size {surr_size:.3f}")
                ax[ii].set_xscale('log')
                ax[ii].set_xlabel('Size difference (% veridical size)')
                ax[ii].set_ylabel('Prob ref perceived as larger')
                ax[ii].minorticks_off()

                true_size = EHDB_data['sub-002']['ses-1']['true_size'] #example, could be any subject

                #ugly
                if surr_size>0:
                    ax[ii].set_xticks(np.linspace(true_size-0.5*true_size,true_size+0.3*true_size,9))
                    ax[ii].set_xticklabels([-50,-40,-30,-20,-10,0,10,20,30])
                else:
                    ax[ii].set_xticks(np.linspace(true_size-0.3*true_size,true_size+0.3*true_size,7))
                    ax[ii].set_xticklabels([-30,-20,-10,0,10,20,30])  
                
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)],np.ones_like(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)])*0.5, ls='--', c='k', alpha=0.5, label='PSE')
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)],np.ones_like(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)]), ls='--', c='green', alpha=0.5)
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)],np.zeros_like(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)]), ls='--', c='red', alpha=0.5)
                #bit hacky way
                ax[ii].plot(true_size*np.ones(100),np.linspace(0,1,100), ls='-', c='k', alpha=0.5, label='Veridical')
                ax[ii].legend(loc='lower right')

                # print(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)])
                # print(preds_placebo[str(surr_size)])
                # print(np.mean(preds_placebo[str(surr_size)],axis=0))
                # print(np.mean(preds[str(surr_size)],axis=0).shape)
                #ax[ii].errorbar(this_run.size_values[str(surr_size)], np.mean(data[str(surr_size)],axis=0), yerr=sem(data[str(surr_size)],axis=0), marker='s', ls='', c='k', ms=6)
                #ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)], np.mean(preds[str(surr_size)],axis=0), c='k', lw=2)
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)], np.mean(preds_placebo[str(surr_size)],axis=0), c=condition_colors['placebo'], lw=2)
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)], np.mean(preds_5mg[str(surr_size)],axis=0), c=condition_colors['5mg'], lw=2)
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)], np.mean(preds_10mg[str(surr_size)],axis=0), c=condition_colors['10mg'], lw=2)
                
                #ax[-1].bar(ii, np.mean(rel_effsize[str(surr_size)],axis=0))
                ax[-1].bar(ii-0.3, np.mean(rel_effsize_placebo[str(surr_size)],axis=0), color=condition_colors['placebo'], alpha=0.5, width=0.3)
                ax[-1].bar(ii, np.mean(rel_effsize_5mg[str(surr_size)],axis=0), color=condition_colors['5mg'], alpha=0.5, width=0.3)
                ax[-1].bar(ii+0.3, np.mean(rel_effsize_10mg[str(surr_size)],axis=0), color=condition_colors['10mg'], alpha=0.5, width=0.3)

                #ax[-1].plot(ii*np.ones_like(rel_effsize[str(surr_size)]), rel_effsize[str(surr_size)], marker='s', mec='k', ls='', ms=8)
                ax[-1].plot(ii-0.3*np.ones_like(rel_effsize_placebo[str(surr_size)]), rel_effsize_placebo[str(surr_size)], marker='s', mec='k',c=condition_colors['placebo'], ls='', ms=8)
                ax[-1].plot(ii*np.ones_like(rel_effsize_5mg[str(surr_size)]), rel_effsize_5mg[str(surr_size)], marker='s', mec='k', c=condition_colors['5mg'], ls='', ms=8)
                ax[-1].plot(ii+0.3*np.ones_like(rel_effsize_10mg[str(surr_size)]), rel_effsize_10mg[str(surr_size)], marker='s', mec='k',c=condition_colors['10mg'], ls='', ms=8)
            
            for ss in range(len(rel_effsize_placebo[str(surr_size)])):
                #ax[-1].plot(np.arange(len(EHDB_data['sub-002']['ses-1']['surr_sizes'])), [rel_effsize[ref][ss] for ref in rel_effsize], c='k')
                ax[-1].plot(np.arange(len(EHDB_data['sub-002']['ses-1']['surr_sizes']))-0.3, [rel_effsize_placebo[ref][ss] for ref in rel_effsize_placebo], c='k', linewidth=0.5, alpha=0.5)
                ax[-1].plot(np.arange(len(EHDB_data['sub-002']['ses-1']['surr_sizes'])), [rel_effsize_5mg[ref][ss] for ref in rel_effsize_5mg], c='k', linewidth=0.5, alpha=0.5)
                
            for ss in range(len(rel_effsize_10mg[str(surr_size)])):
                ax[-1].plot(np.arange(len(EHDB_data['sub-002']['ses-1']['surr_sizes']))+0.3, [rel_effsize_10mg[ref][ss] for ref in rel_effsize_10mg], c='k', linewidth=0.5, alpha=0.5)

                
            ax[-1].plot(np.arange(-1,1+len(EHDB_data['sub-002']['ses-1']['surr_sizes'])),np.zeros_like(np.arange(-1,1+len(EHDB_data['sub-002']['ses-1']['surr_sizes']))), ls='--', c='k', alpha=0.5)
                
            
            ax[-1].set_xticks(np.arange(len(EHDB_data['sub-002']['ses-1']['surr_sizes'])))
            ax[-1].set_xticklabels([f"{surr_size:.3f}" for surr_size in EHDB_data['sub-002']['ses-1']['surr_sizes']])
            ax[-1].set_xlabel('Surround size (deg)')
            ax[-1].set_ylabel('PSE difference (% veridical size)')

            ax[-1].set_yticks(np.linspace(0,-30,7))
            ax[-1].set_yticklabels([0,-5,-10,-15,-20,-25,-30])


            fig.suptitle("EHDB task - group results")


            fig.savefig(opj(out_path,f"group_results_EHDB.pdf"), dpi=600, bbox_inches='tight', transparent=True)   

        def EHDBgroupplot_average(self, out_path):
            condition_colors = {'placebo':'blue', '5mg':'orange', '10mg':'green'}
            EHDB_data = np.load(opj(out_path,'data/EHDB_data.npy'), allow_pickle=True).item()

            preds_placebo = dd(list)
            preds_5mg = dd(list)
            preds_10mg = dd(list)
            data_placebo = dd(list)
            data_5mg = dd(list)
            data_10mg = dd(list)
            rel_effsize_placebo = dd(list)
            rel_effsize_5mg = dd(list)
            rel_effsize_10mg = dd(list)
            subject_name_placebo = []
            subject_name_5mg = []
            subject_name_10mg = []

            fig, ax = pl.subplots(1,3,figsize=(24,8))

            for participant, sessions in EHDB_data.items():
                for session, data_ in sessions.items():
                    #is session 0 do nothing
                    if session == 'ses-0':
                        continue
                    else :
                        

                        if data_['group_type'] == 'placebo':
                            for ii, surr_size in enumerate(data_['surr_sizes']):
                                preds_placebo[str(surr_size)].append(data_['preds'][str(surr_size)])
                                data_placebo[str(surr_size)].append(data_['probs'][str(surr_size)])
                                rel_effsize_placebo[str(surr_size)].append(data_['rel_effsize'][str(surr_size)])
                            subject_name_placebo.append(participant)
                        elif data_['group_type'] == '5mg':
                            for ii, surr_size in enumerate(data_['surr_sizes']):
                                preds_5mg[str(surr_size)].append(data_['preds'][str(surr_size)])
                                data_5mg[str(surr_size)].append(data_['probs'][str(surr_size)])
                                rel_effsize_5mg[str(surr_size)].append(data_['rel_effsize'][str(surr_size)])
                            subject_name_5mg.append(participant)
                        elif data_['group_type'] == '10mg':
                            for ii, surr_size in enumerate(data_['surr_sizes']):
                                preds_10mg[str(surr_size)].append(data_['preds'][str(surr_size)])
                                data_10mg[str(surr_size)].append(data_['probs'][str(surr_size)])
                                rel_effsize_10mg[str(surr_size)].append(data_['rel_effsize'][str(surr_size)])
                            subject_name_10mg.append(participant)

            the_space = np.linspace(-0.10, 0.10, len(subject_name_placebo))
            # print(len(subject_name_placebo))
            # print(the_space)

            for ii, surr_size in enumerate(EHDB_data['sub-002']['ses-1']['surr_sizes']):
                ax[ii].set_title(f"Surr size {surr_size:.3f}")
                ax[ii].set_xscale('log')
                ax[ii].set_xlabel('Size difference (% veridical size)')
                ax[ii].set_ylabel('Prob ref perceived as larger')
                ax[ii].minorticks_off()
                # ax[ii].set_xlim([-30,30])

                true_size = EHDB_data['sub-002']['ses-1']['true_size']

                #ugly
                if surr_size>0:
                    ax[ii].set_xticks(np.linspace(true_size-0.5*true_size,true_size+0.3*true_size,9))
                    ax[ii].set_xticklabels([-50,-40,-30,-20,-10,0,10,20,30])
                else:
                    ax[ii].set_xticks(np.linspace(true_size-0.3*true_size,true_size+0.3*true_size,7))
                    ax[ii].set_xticklabels([-30,-20,-10,0,10,20,30])  
                
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)],np.ones_like(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)])*0.5, ls='--', c='k', alpha=0.5, label='PSE')
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)],np.ones_like(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)]), ls='--', c='green', alpha=0.5)
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)],np.zeros_like(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)]), ls='--', c='red', alpha=0.5)
                #bit hacky way
                ax[ii].plot(true_size*np.ones(100),np.linspace(0,1,100), ls='-', c='k', alpha=0.5, label='Veridical')
                ax[ii].legend(loc='lower right')

                
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)], np.mean(preds_placebo[str(surr_size)],axis=0), c=condition_colors['placebo'], lw=2, label='Placebo')
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)], np.mean(preds_5mg[str(surr_size)],axis=0), c=condition_colors['5mg'], lw=2, label='5mg')
                #np.mean taking into account np.nan
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)], np.nanmean(preds_10mg[str(surr_size)],axis=0), c=condition_colors['10mg'], lw=2, label='10mg')

                ax[ii].fill_between(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)], np.mean(preds_placebo[str(surr_size)],axis=0)-1.98*sem(preds_placebo[str(surr_size)],axis=0), np.mean(preds_placebo[str(surr_size)],axis=0)+1.98*sem(preds_placebo[str(surr_size)],axis=0), color=condition_colors['placebo'], alpha=0.2)
                ax[ii].fill_between(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)], np.mean(preds_5mg[str(surr_size)],axis=0)-1.98*sem(preds_5mg[str(surr_size)],axis=0), np.mean(preds_5mg[str(surr_size)],axis=0)+1.98*sem(preds_5mg[str(surr_size)],axis=0), color=condition_colors['5mg'], alpha=0.2)
                ax[ii].fill_between(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)], np.nanmean(preds_10mg[str(surr_size)],axis=0)-1.98*sem(preds_10mg[str(surr_size)],axis=0), np.mean(preds_10mg[str(surr_size)],axis=0)+1.98*sem(preds_10mg[str(surr_size)],axis=0), color=condition_colors['10mg'], alpha=0.2)          

                #ax[-1].bar(ii, np.mean(rel_effsize[str(surr_size)],axis=0))
                ax[-1].bar(ii-0.3, np.mean(rel_effsize_placebo[str(surr_size)],axis=0), color=condition_colors['placebo'], alpha=0.9, width=0.3)
                ax[-1].bar(ii, np.mean(rel_effsize_5mg[str(surr_size)],axis=0), color=condition_colors['5mg'], alpha=0.9, width=0.3)
                ax[-1].bar(ii+0.3, np.nanmean(rel_effsize_10mg[str(surr_size)],axis=0), color=condition_colors['10mg'], alpha=0.9, width=0.3)

                print(np.ones_like(rel_effsize_placebo[str(surr_size)]))
                x_axis = ii * np.ones_like(rel_effsize_placebo[str(surr_size)]) + the_space

                #ax[-1].plot(ii*np.ones_like(rel_effsize[str(surr_size)]), rel_effsize[str(surr_size)], marker='s', mec='k', ls='', ms=8)
                ax[-1].plot(x_axis-0.3, rel_effsize_placebo[str(surr_size)], marker='s', c=condition_colors['placebo'],mec='k', ls='', ms=6, alpha =0.4)
                ax[-1].plot(x_axis, rel_effsize_5mg[str(surr_size)], marker='s', c=condition_colors['5mg'],mec='k', ls='', ms=6, alpha =0.4)  
                ax[-1].plot(x_axis+0.3, rel_effsize_10mg[str(surr_size)], marker='s', c=condition_colors['10mg'],mec='k', ls='', ms=6, alpha =0.4)
                        
            #plot the links and the text
            for ii, surr_size in enumerate(EHDB_data['sub-002']['ses-1']['surr_sizes']):
                x_axis2 = ii * np.ones_like(rel_effsize_placebo[str(surr_size)]) + the_space
                for ss in range(len(rel_effsize_placebo[str(surr_size)])):
                    #ax[-1].plot(np.arange(len(EHDB_data['sub-002']['ses-1']['surr_sizes'])), [rel_effsize[ref][ss] for ref in rel_effsize], c='k')
                    x_axis3 = x_axis2[ss]
                    ax[-1].plot([x_axis3-0.3, x_axis3,x_axis3+0.3], [rel_effsize_placebo[str(surr_size)][ss],rel_effsize_5mg[str(surr_size)][ss],rel_effsize_10mg[str(surr_size)][ss]] , c='k', linewidth=0.5, alpha=0.5)
                    
                    ax[-1].text(x_axis3-0.3, rel_effsize_placebo[str(surr_size)][ss]-0.1, f"{subject_name_placebo[ss]}", fontsize=6)

            ax[1].legend(loc='lower right')

            ax[-1].plot(np.arange(-1,1+len(EHDB_data['sub-002']['ses-1']['surr_sizes'])),np.zeros_like(np.arange(-1,1+len(EHDB_data['sub-002']['ses-1']['surr_sizes']))), ls='--', c='k', alpha=0.5)

            ax[-1].set_xticks(np.arange(len(EHDB_data['sub-002']['ses-1']['surr_sizes'])))
            ax[-1].set_xticklabels([f"{surr_size:.3f}" for surr_size in EHDB_data['sub-002']['ses-1']['surr_sizes']])
            ax[-1].set_xlabel('Surround size (deg)')
            ax[-1].set_ylabel('PSE difference (% veridical size)')
            ax[-1].legend(loc='lower right')

            ax[-1].set_yticks(np.linspace(0,-30,7))
            ax[-1].set_yticklabels([0,-5,-10,-15,-20,-25,-30])

            ax[-1].set_xlim([-0.5,1.5])

            fig.suptitle("EHDB task - average results with 95% CI")

            fig.savefig(opj(out_path,f"group_results_EHDB_average.pdf"), dpi=600, bbox_inches='tight', transparent=True)

        def EHDBgroupplot_average2(self,out_path):
            condition_colors = {'placebo':'blue', '5mg':'orange', '10mg':'green'}
            EHDB_data = np.load(opj(out_path,'data/EHDB_data.npy'), allow_pickle=True).item()

            fig, ax = pl.subplots(1,1,figsize=(8,8))
            fig.suptitle(f"EHDB task - relative effect size")

            preds_placebo = dd(list)
            preds_5mg = dd(list)
            preds_10mg = dd(list)
            data_placebo = dd(list)
            data_5mg = dd(list)
            data_10mg = dd(list)
            rel_effsize_placebo = dd(list)
            rel_effsize_5mg = dd(list)
            rel_effsize_10mg = dd(list)
            rel_effsize_difference_placebo = []
            rel_effsize_difference_5mg = []
            rel_effsize_difference_10mg = []
            subject_name_placebo = []
            subject_name_5mg = []
            subject_name_10mg = []

            for participant, sessions in EHDB_data.items():
                for session, data_ in sessions.items():
                    #is session 0 do nothing
                    if session == 'ses-0':
                        continue
                    else :
                        

                        if data_['group_type'] == 'placebo':
                            for ii, surr_size in enumerate(data_['surr_sizes']):
                                preds_placebo[str(surr_size)].append(data_['preds'][str(surr_size)])
                                data_placebo[str(surr_size)].append(data_['probs'][str(surr_size)])
                                rel_effsize_placebo[str(surr_size)].append(data_['rel_effsize'][str(surr_size)])
                            rel_effsize_difference_placebo.append(data_['rel_effsize']['0.9375']-data_['rel_effsize']['0.0'])
                            subject_name_placebo.append(participant)
                        elif data_['group_type'] == '5mg':
                            for ii, surr_size in enumerate(data_['surr_sizes']):
                                preds_5mg[str(surr_size)].append(data_['preds'][str(surr_size)])
                                data_5mg[str(surr_size)].append(data_['probs'][str(surr_size)])
                                rel_effsize_5mg[str(surr_size)].append(data_['rel_effsize'][str(surr_size)])
                            print(rel_effsize_5mg['0.9375'])
                            rel_effsize_difference_5mg.append(data_['rel_effsize']['0.9375']-data_['rel_effsize']['0.0'])
                            subject_name_5mg.append(participant)
                        elif data_['group_type'] == '10mg':
                            for ii, surr_size in enumerate(data_['surr_sizes']):
                                preds_10mg[str(surr_size)].append(data_['preds'][str(surr_size)])
                                data_10mg[str(surr_size)].append(data_['probs'][str(surr_size)])
                                rel_effsize_10mg[str(surr_size)].append(data_['rel_effsize'][str(surr_size)])
                            rel_effsize_difference_10mg.append(data_['rel_effsize']['0.9375']-data_['rel_effsize']['0.0'])
                            subject_name_10mg.append(participant)

            ax.set_ylabel('Differences in relative effect sizes between radius 0.938 and radius 0 tasks')       
            the_space = np.linspace(-0.10, 0.10, len(subject_name_placebo))
            
            ax.bar(-0.3, np.mean(rel_effsize_difference_placebo), color=condition_colors['placebo'], alpha=0.5, width=0.3)

            x_axis = 0*np.ones_like(rel_effsize_difference_placebo) + the_space
            ax.plot(x_axis-0.3, rel_effsize_difference_placebo, marker='s', mec='k', ls='', ms=8, color=condition_colors['placebo'])

            ax.bar(0, np.mean(rel_effsize_difference_5mg), color=condition_colors['5mg'], alpha=0.5, width=0.3)
            x_axis = 0*np.ones_like(rel_effsize_difference_5mg) + the_space
            ax.plot(x_axis, rel_effsize_difference_5mg, marker='s', mec='k', ls='', ms=8, color=condition_colors['5mg'])

            ax.bar(0.3, np.nanmean(rel_effsize_difference_10mg), color=condition_colors['10mg'], alpha=0.5, width=0.3)
            x_axis = 0*np.ones_like(rel_effsize_difference_10mg) + the_space
            ax.plot(x_axis+0.3, rel_effsize_difference_10mg, marker='s', mec='k', ls='', ms=8, color=condition_colors['10mg'])
            

            for ss in range(len(rel_effsize_difference_placebo)):

                x_axis2 = x_axis[ss]
                ax.plot([x_axis2-0.3, x_axis2, x_axis2+0.3], [rel_effsize_difference_placebo[ss], rel_effsize_difference_5mg[ss], rel_effsize_difference_10mg[ss]], c='k', linewidth=0.5, alpha=0.5)

                ax.text(x_axis2-0.3, rel_effsize_difference_placebo[ss], subject_name_placebo[ss], fontsize=8)
                
            ax.plot(np.linspace(-1,2,10),np.zeros_like(np.linspace(-1,2,10)), ls='--', c='k', alpha=0.5)
            
            #set ticks with the
            ax.set_xticks([-0.3, 0, 0.3])
            ax.set_xticklabels(['placebo', '5mg', '10mg'])
            ax.set_xlabel('')
            ax.set_ylabel('PSE difference (%RMS)')
            #ax.set_ylim([-90,10])
            ax.set_xlim([-0.5,0.5])


            fig.savefig(opj(out_path,'group_results_EHDB_average2.pdf'), dpi=600, bbox_inches='tight', transparent=True)

        def EHDBgroupplot_average3(self, out_path):
            condition_colors = {'placebo':'blue', '5mg':'orange', '10mg':'green'}
            EHDB_data = np.load(opj(out_path,'data/EHDB_data.npy'), allow_pickle=True).item()
            df_EHDB = pd.read_csv(opj(out_path,'data/group_results_EHDB.csv'))

            fig, ax = pl.subplots(1,4,figsize=(32,8))

            preds_placebo = dd(list)
            preds_5mg = dd(list)
            preds_10mg = dd(list)
            data_placebo = dd(list)
            data_5mg = dd(list)
            data_10mg = dd(list)
            rel_effsize_placebo = dd(list)
            rel_effsize_5mg = dd(list)
            rel_effsize_10mg = dd(list)
            rel_effsize_difference_placebo = []
            rel_effsize_difference_5mg = []
            rel_effsize_difference_10mg = []
            subject_name_placebo = []
            subject_name_5mg = []
            subject_name_10mg = []

            for participant, sessions in EHDB_data.items():
                for session, data_ in sessions.items():
                    #is session 0 do nothing
                    if session == 'ses-0':
                        continue
                    else :
                        if data_['group_type'] == 'placebo':
                            for ii, surr_size in enumerate(data_['surr_sizes']):
                                preds_placebo[str(surr_size)].append(data_['preds'][str(surr_size)])
                                data_placebo[str(surr_size)].append(data_['probs'][str(surr_size)])
                                rel_effsize_placebo[str(surr_size)].append(data_['rel_effsize'][str(surr_size)])
                            rel_effsize_difference_placebo.append(data_['rel_effsize']['0.9375']-data_['rel_effsize']['0.0'])
                            subject_name_placebo.append(participant)
                        elif data_['group_type'] == '5mg':
                            for ii, surr_size in enumerate(data_['surr_sizes']):
                                preds_5mg[str(surr_size)].append(data_['preds'][str(surr_size)])
                                data_5mg[str(surr_size)].append(data_['probs'][str(surr_size)])
                                rel_effsize_5mg[str(surr_size)].append(data_['rel_effsize'][str(surr_size)])
                            print(rel_effsize_5mg['0.9375'])
                            rel_effsize_difference_5mg.append(data_['rel_effsize']['0.9375']-data_['rel_effsize']['0.0'])
                            subject_name_5mg.append(participant)
                        elif data_['group_type'] == '10mg':
                            for ii, surr_size in enumerate(data_['surr_sizes']):
                                preds_10mg[str(surr_size)].append(data_['preds'][str(surr_size)])
                                data_10mg[str(surr_size)].append(data_['probs'][str(surr_size)])
                                rel_effsize_10mg[str(surr_size)].append(data_['rel_effsize'][str(surr_size)])
                            rel_effsize_difference_10mg.append(data_['rel_effsize']['0.9375']-data_['rel_effsize']['0.0'])
                            subject_name_10mg.append(participant)

            the_space = np.linspace(-0.10, 0.10, len(subject_name_placebo))
            # print(len(subject_name_placebo))
            # print(the_space)

            for ii, surr_size in enumerate(EHDB_data['sub-002']['ses-1']['surr_sizes']):
                ax[ii].set_title(f"Surr size {surr_size:.3f}")
                ax[ii].set_xscale('log')
                ax[ii].set_xlabel('Size difference (% veridical size)')
                ax[ii].set_ylabel('Prob ref perceived as larger')
                ax[ii].minorticks_off()
                # ax[ii].set_xlim([-30,30])

                true_size = EHDB_data['sub-002']['ses-1']['true_size']

                #ugly
                if surr_size>0:
                    ax[ii].set_xticks(np.linspace(true_size-0.5*true_size,true_size+0.3*true_size,9))
                    ax[ii].set_xticklabels([-50,-40,-30,-20,-10,0,10,20,30])
                else:
                    ax[ii].set_xticks(np.linspace(true_size-0.3*true_size,true_size+0.3*true_size,7))
                    ax[ii].set_xticklabels([-30,-20,-10,0,10,20,30])  
                
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)],np.ones_like(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)])*0.5, ls='--', c='k', alpha=0.5, label='PSE')
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)],np.ones_like(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)]), ls='--', c='green', alpha=0.5)
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)],np.zeros_like(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)]), ls='--', c='red', alpha=0.5)
                #bit hacky way
                ax[ii].plot(true_size*np.ones(100),np.linspace(0,1,100), ls='-', c='k', alpha=0.5, label='Veridical')
                ax[ii].legend(loc='lower right')

                
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)], np.mean(preds_placebo[str(surr_size)],axis=0), c=condition_colors['placebo'], lw=2, label='Placebo')
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)], np.mean(preds_5mg[str(surr_size)],axis=0), c=condition_colors['5mg'], lw=2, label='5mg')
                #np.mean taking into account np.nan
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)], np.nanmean(preds_10mg[str(surr_size)],axis=0), c=condition_colors['10mg'], lw=2, label='10mg')

                ax[ii].fill_between(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)], np.mean(preds_placebo[str(surr_size)],axis=0)-1.98*sem(preds_placebo[str(surr_size)],axis=0), np.mean(preds_placebo[str(surr_size)],axis=0)+1.98*sem(preds_placebo[str(surr_size)],axis=0), color=condition_colors['placebo'], alpha=0.2)
                ax[ii].fill_between(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)], np.mean(preds_5mg[str(surr_size)],axis=0)-1.98*sem(preds_5mg[str(surr_size)],axis=0), np.mean(preds_5mg[str(surr_size)],axis=0)+1.98*sem(preds_5mg[str(surr_size)],axis=0), color=condition_colors['5mg'], alpha=0.2)
                ax[ii].fill_between(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)], np.nanmean(preds_10mg[str(surr_size)],axis=0)-1.98*sem(preds_10mg[str(surr_size)],axis=0), np.mean(preds_10mg[str(surr_size)],axis=0)+1.98*sem(preds_10mg[str(surr_size)],axis=0), color=condition_colors['10mg'], alpha=0.2)          

                #ax[-1].bar(ii, np.mean(rel_effsize[str(surr_size)],axis=0))
                ax[2].bar(ii-0.3, np.mean(rel_effsize_placebo[str(surr_size)],axis=0), color=condition_colors['placebo'], alpha=0.7, width=0.3)
                ax[2].bar(ii, np.mean(rel_effsize_5mg[str(surr_size)],axis=0), color=condition_colors['5mg'], alpha=0.7, width=0.3)
                ax[2].bar(ii+0.3, np.nanmean(rel_effsize_10mg[str(surr_size)],axis=0), color=condition_colors['10mg'], alpha=0.7, width=0.3)

                print(np.ones_like(rel_effsize_placebo[str(surr_size)]))
                x_axis = ii * np.ones_like(rel_effsize_placebo[str(surr_size)]) + the_space

                #ax[-1].plot(ii*np.ones_like(rel_effsize[str(surr_size)]), rel_effsize[str(surr_size)], marker='s', mec='k', ls='', ms=8)
                ax[2].plot(x_axis-0.3, rel_effsize_placebo[str(surr_size)], marker='s', c=condition_colors['placebo'],mec='k', ls='', ms=6, alpha =0.4)
                ax[2].plot(x_axis, rel_effsize_5mg[str(surr_size)], marker='s', c=condition_colors['5mg'],mec='k', ls='', ms=6, alpha =0.4)  
                ax[2].plot(x_axis+0.3, rel_effsize_10mg[str(surr_size)], marker='s', c=condition_colors['10mg'],mec='k', ls='', ms=6, alpha =0.4)
                        
            #plot the links and the text
            for ii, surr_size in enumerate(EHDB_data['sub-002']['ses-1']['surr_sizes']):
                x_axis2 = ii * np.ones_like(rel_effsize_placebo[str(surr_size)]) + the_space
                for ss in range(len(rel_effsize_placebo[str(surr_size)])):
                    #ax[-1].plot(np.arange(len(EHDB_data['sub-002']['ses-1']['surr_sizes'])), [rel_effsize[ref][ss] for ref in rel_effsize], c='k')
                    x_axis3 = x_axis2[ss]
                    ax[2].plot([x_axis3-0.3, x_axis3,x_axis3+0.3], [rel_effsize_placebo[str(surr_size)][ss],rel_effsize_5mg[str(surr_size)][ss],rel_effsize_10mg[str(surr_size)][ss]] , c='k', linewidth=0.5, alpha=0.5)
                    
                    ax[2].text(x_axis3-0.3, rel_effsize_placebo[str(surr_size)][ss]-0.1, f"{subject_name_placebo[ss]}", fontsize=6)

            ax[1].legend(loc='lower right')

            ax[2].plot(np.arange(-1,1+len(EHDB_data['sub-002']['ses-1']['surr_sizes'])),np.zeros_like(np.arange(-1,1+len(EHDB_data['sub-002']['ses-1']['surr_sizes']))), ls='--', c='k', alpha=0.5)

            ax[2].set_xticks(np.arange(len(EHDB_data['sub-002']['ses-1']['surr_sizes'])))
            ax[2].set_xticklabels([f"{surr_size:.3f}" for surr_size in EHDB_data['sub-002']['ses-1']['surr_sizes']])
            ax[2].set_xlabel('Surround size (deg)')
            ax[2].set_ylabel('PSE difference (% veridical size)')
            ax[2].legend(loc='lower right')

            ax[2].set_yticks(np.linspace(0,-30,7))
            ax[2].set_yticklabels([0,-5,-10,-15,-20,-25,-30])

            ax[2].set_xlim([-0.5,1.5])

            ax[3].set_ylabel('Differences in relative effect sizes between radius 0.938 and radius 0 tasks')       
            the_space = np.linspace(-0.10, 0.10, len(subject_name_placebo))
            
            ax[3].bar(-0.3, np.mean(rel_effsize_difference_placebo), color=condition_colors['placebo'], alpha=0.7, width=0.3)

            x_axis = 0*np.ones_like(rel_effsize_difference_placebo) + the_space
            ax[3].plot(x_axis-0.3, rel_effsize_difference_placebo, marker='s', mec='k', ls='', ms=8, color=condition_colors['placebo'])

            ax[3].bar(0, np.mean(rel_effsize_difference_5mg), color=condition_colors['5mg'], alpha=0.7, width=0.3)
            x_axis = 0*np.ones_like(rel_effsize_difference_5mg) + the_space
            ax[3].plot(x_axis, rel_effsize_difference_5mg, marker='s', mec='k', ls='', ms=8, color=condition_colors['5mg'])

            ax[3].bar(0.3, np.nanmean(rel_effsize_difference_10mg), color=condition_colors['10mg'], alpha=0.7, width=0.3)
            x_axis = 0*np.ones_like(rel_effsize_difference_10mg) + the_space
            ax[3].plot(x_axis+0.3, rel_effsize_difference_10mg, marker='s', mec='k', ls='', ms=8, color=condition_colors['10mg'])
            

            for ss in range(len(rel_effsize_difference_placebo)):

                x_axis2 = x_axis[ss]
                ax[3].plot([x_axis2-0.3, x_axis2, x_axis2+0.3], [rel_effsize_difference_placebo[ss], rel_effsize_difference_5mg[ss], rel_effsize_difference_10mg[ss]], c='k', linewidth=0.5, alpha=0.5)

                ax[3].text(x_axis2-0.3, rel_effsize_difference_placebo[ss], subject_name_placebo[ss], fontsize=8)
                
            ax[3].plot(np.linspace(-1,2,10),np.zeros_like(np.linspace(-1,2,10)), ls='--', c='k', alpha=0.5)
            
            #set ticks with the
            ax[3].set_xticks([-0.3, 0, 0.3])
            ax[3].set_xticklabels(['placebo', '5mg', '10mg'])
            ax[3].set_xlabel('')
            ax[3].set_ylabel('PSE[0.000] - PSE[0.938] (%RMS)')
            #ax.set_ylim([-90,10])
            ax[3].set_xlim([-0.5,0.5])

            #Add the ANOVA info to the plot
            """ ax[3].text(-0.43, 3, 'RM ANOVA :', fontsize=12)
            ax[3].text(-0.15, 3, 'F value = 0.58', fontsize=12)
            ax[3].text(0.25, 3, 'p = 0.56', fontsize=12) """

            #get rid of ses-0
            df_EHDB = df_EHDB[df_EHDB['Session'] != 'ses_0']

            #check the balance of df_EHDB
            print(df_EHDB['Subject'].value_counts())
            print(df_EHDB.groupby(['Group_type'])['rel_effsize_difference'].count())

            #define p value and f value
            aovrm2way = AnovaRM(df_EHDB, 'rel_effsize_difference', 'Subject', within=['Group_type'], aggregate_func=np.mean)
            res2way = aovrm2way.fit()
            print("RM ANOVA all subjects Results:")
            #print only the p value
            p = round(res2way.anova_table['Pr > F'][0], 2)
            #print only the f value
            f = round(res2way.anova_table['F Value'][0], 2)
            print(f"p = {p:.3f}")
            print(f"F = {f:.3f}")

            #Add the ANOVA info to the plot
            ax[3].text(0.5, -10, 'RM Anova', fontsize=12)
            ax[3].text(0.5, -13, f'F value = {f}', fontsize=12)
            ax[3].text(0.5, -16, f'p = {p}', fontsize=12)

            fig.suptitle("EHDB task - average results with 95% CI")

            fig.savefig(opj(out_path,f"group_results_EHDB_average.pdf"), dpi=600, bbox_inches='tight', transparent=True)
        
        def EHDBgroupplot_average_slope(self, out_path):
            condition_colors = {'placebo':'blue', '5mg':'orange', '10mg':'green'}
            EHDB_data = np.load(opj(out_path,'data/EHDB_data.npy'), allow_pickle=True).item()
            df_EHDB = pd.read_csv(opj(out_path,'data/group_results_EHDB.csv'))

            fig, ax = pl.subplots(1,4,figsize=(32,8))

            preds_placebo = dd(list)
            preds_5mg = dd(list)
            preds_10mg = dd(list)
            slope_placebo = dd(list)
            slope_5mg = dd(list)
            slope_10mg = dd(list)
            slope_average_placebo = []
            slope_average_5mg = []
            slope_average_10mg = []
            rel_effsize_placebo = dd(list)
            rel_effsize_5mg = dd(list)
            rel_effsize_10mg = dd(list)
            rel_effsize_difference_placebo = []
            rel_effsize_difference_5mg = []
            rel_effsize_difference_10mg = []
            subject_name_placebo = []
            subject_name_5mg = []
            subject_name_10mg = []

            for participant, sessions in EHDB_data.items():
                for session, data_ in sessions.items():
                    #is session 0 do nothing
                    if session == 'ses-0':
                        continue
                    else :
                        if data_['group_type'] == 'placebo':
                            for ii, surr_size in enumerate(data_['surr_sizes']):
                                preds_placebo[str(surr_size)].append(data_['preds'][str(surr_size)])
                                rel_effsize_placebo[str(surr_size)].append(data_['rel_effsize'][str(surr_size)])
                                print(data_['slope'][str(surr_size)])
                                if data_['slope'][str(surr_size)] > 0:
                                    slope_placebo[str(surr_size)].append(round(data_['slope'][str(surr_size)],2))
                                else:
                                    print('append 0 to slope_placebo')
                                    slope_placebo[str(surr_size)].append(0)
                            rel_effsize_difference_placebo.append(data_['rel_effsize']['0.9375']-data_['rel_effsize']['0.0'])
                            subject_name_placebo.append(participant)
                            if data_['slope_average'] > 0:
                                slope_average_placebo.append(round(data_['slope_average'],2))
                            else:
                                slope_average_placebo.append(0)
                        elif data_['group_type'] == '5mg':
                            for ii, surr_size in enumerate(data_['surr_sizes']):
                                preds_5mg[str(surr_size)].append(data_['preds'][str(surr_size)])
                                rel_effsize_5mg[str(surr_size)].append(data_['rel_effsize'][str(surr_size)])
                                if data_['slope'][str(surr_size)] > 0:
                                    slope_5mg[str(surr_size)].append(round(data_['slope'][str(surr_size)],2))
                                else:
                                    slope_5mg[str(surr_size)].append(0)
                            print(rel_effsize_5mg['0.9375'])
                            rel_effsize_difference_5mg.append(data_['rel_effsize']['0.9375']-data_['rel_effsize']['0.0'])
                            subject_name_5mg.append(participant)
                            if data_['slope_average'] > 0:
                                slope_average_5mg.append(round(data_['slope_average'],2))
                            else:
                                slope_average_5mg.append(0)
                        elif data_['group_type'] == '10mg':
                            for ii, surr_size in enumerate(data_['surr_sizes']):
                                preds_10mg[str(surr_size)].append(data_['preds'][str(surr_size)])
                                rel_effsize_10mg[str(surr_size)].append(data_['rel_effsize'][str(surr_size)])
                                if data_['slope'][str(surr_size)] > 0:
                                    slope_10mg[str(surr_size)].append(round(data_['slope'][str(surr_size)],2))
                                else:
                                    slope_10mg[str(surr_size)].append(0)
                            rel_effsize_difference_10mg.append(data_['rel_effsize']['0.9375']-data_['rel_effsize']['0.0'])
                            subject_name_10mg.append(participant)
                            if data_['slope_average'] > 0:
                                slope_average_10mg.append(round(data_['slope_average'],2))
                            else:
                                slope_average_10mg.append(0)

            the_space = np.linspace(-0.10, 0.10, len(subject_name_placebo))

            for ii, surr_size in enumerate(EHDB_data['sub-002']['ses-1']['surr_sizes']):
                ax[ii].set_title(f"Surr size {surr_size:.3f}")
                ax[ii].set_xscale('log')
                ax[ii].set_xlabel('Size difference (% veridical size)')
                ax[ii].set_ylabel('Prob ref perceived as larger')
                ax[ii].minorticks_off()
                # ax[ii].set_xlim([-30,30])

                true_size = EHDB_data['sub-002']['ses-1']['true_size']

                #ugly
                if surr_size>0:
                    ax[ii].set_xticks(np.linspace(true_size-0.5*true_size,true_size+0.3*true_size,9))
                    ax[ii].set_xticklabels([-50,-40,-30,-20,-10,0,10,20,30])
                else:
                    ax[ii].set_xticks(np.linspace(true_size-0.3*true_size,true_size+0.3*true_size,7))
                    ax[ii].set_xticklabels([-30,-20,-10,0,10,20,30])  
                
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)],np.ones_like(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)])*0.5, ls='--', c='k', alpha=0.5, label='PSE')
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)],np.ones_like(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)]), ls='--', c='green', alpha=0.5)
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)],np.zeros_like(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)]), ls='--', c='red', alpha=0.5)
                #bit hacky way
                ax[ii].plot(true_size*np.ones(100),np.linspace(0,1,100), ls='-', c='k', alpha=0.5, label='Veridical')
                ax[ii].legend(loc='lower right')

                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)], np.average(preds_placebo[str(surr_size)], axis=0, weights=slope_placebo[str(surr_size)]), c=condition_colors['placebo'], lw=2, label='Placebo')
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)], np.average(preds_5mg[str(surr_size)], axis=0, weights=slope_5mg[str(surr_size)]), c=condition_colors['5mg'], lw=2, label='5mg')
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)], np.average(preds_10mg[str(surr_size)], axis=0, weights=slope_10mg[str(surr_size)]), c=condition_colors['10mg'], lw=2, label='10mg')

                d1 = DescrStatsW(preds_placebo[str(surr_size)], weights=slope_placebo[str(surr_size)])
                d2 = DescrStatsW(preds_5mg[str(surr_size)], weights=slope_5mg[str(surr_size)])
                d3 = DescrStatsW(preds_10mg[str(surr_size)], weights=slope_10mg[str(surr_size)])
                ax[ii].fill_between(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)], np.average(preds_placebo[str(surr_size)], axis=0, weights=slope_placebo[str(surr_size)])-1.98*d1.std_mean, np.average(preds_placebo[str(surr_size)], axis=0, weights=slope_placebo[str(surr_size)])+1.98*d1.std_mean, color=condition_colors['placebo'], alpha=0.2)
                ax[ii].fill_between(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)], np.average(preds_5mg[str(surr_size)], axis=0, weights=slope_5mg[str(surr_size)])-1.98*d2.std_mean, np.average(preds_5mg[str(surr_size)], axis=0, weights=slope_5mg[str(surr_size)])+1.98*d2.std_mean, color=condition_colors['5mg'], alpha=0.2)
                ax[ii].fill_between(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)], np.average(preds_10mg[str(surr_size)], axis=0, weights=slope_10mg[str(surr_size)])-1.98*d3.std_mean, np.average(preds_10mg[str(surr_size)], axis=0, weights=slope_10mg[str(surr_size)])+1.98*d3.std_mean, color=condition_colors['10mg'], alpha=0.2)

                #ax[-1].bar(ii, np.mean(rel_effsize[str(surr_size)],axis=0))
                ax[2].bar(ii-0.3, np.average(rel_effsize_placebo[str(surr_size)],axis=0, weights=slope_placebo[str(surr_size)]), color=condition_colors['placebo'], alpha=0.7, width=0.3)
                ax[2].bar(ii, np.average(rel_effsize_5mg[str(surr_size)],axis=0, weights=slope_5mg[str(surr_size)]), color=condition_colors['5mg'], alpha=0.7, width=0.3)
                ax[2].bar(ii+0.3, np.average(rel_effsize_10mg[str(surr_size)],axis=0, weights=slope_10mg[str(surr_size)]), color=condition_colors['10mg'], alpha=0.7, width=0.3)

                # print(np.ones_like(rel_effsize_placebo[str(surr_size)]))
                x_axis = ii * np.ones_like(rel_effsize_placebo[str(surr_size)]) + the_space

                #ax[-1].plot(ii*np.ones_like(rel_effsize[str(surr_size)]), rel_effsize[str(surr_size)], marker='s', mec='k', ls='', ms=8)
                ax[2].plot(x_axis-0.3, rel_effsize_placebo[str(surr_size)], marker='s', c=condition_colors['placebo'],mec='k', ls='', ms=6, alpha =0.4)
                ax[2].plot(x_axis, rel_effsize_5mg[str(surr_size)], marker='s', c=condition_colors['5mg'],mec='k', ls='', ms=6, alpha =0.4)  
                ax[2].plot(x_axis+0.3, rel_effsize_10mg[str(surr_size)], marker='s', c=condition_colors['10mg'],mec='k', ls='', ms=6, alpha =0.4)
                
            #plot the links and the text
            for ii, surr_size in enumerate(EHDB_data['sub-002']['ses-1']['surr_sizes']):
                x_axis2 = ii * np.ones_like(rel_effsize_placebo[str(surr_size)]) + the_space
                for ss in range(len(rel_effsize_placebo[str(surr_size)])):
                    #ax[-1].plot(np.arange(len(EHDB_data['sub-002']['ses-1']['surr_sizes'])), [rel_effsize[ref][ss] for ref in rel_effsize], c='k')
                    x_axis3 = x_axis2[ss]
                    ax[2].plot([x_axis3-0.3, x_axis3,x_axis3+0.3], [rel_effsize_placebo[str(surr_size)][ss],rel_effsize_5mg[str(surr_size)][ss],rel_effsize_10mg[str(surr_size)][ss]] , c='k', linewidth=0.5, alpha=0.5)
                    
                    ax[2].text(x_axis3-0.3, rel_effsize_placebo[str(surr_size)][ss]-0.1, f"{subject_name_placebo[ss]}", fontsize=6)
                    ax[2].text(x_axis3-0.3, rel_effsize_placebo[str(surr_size)][ss]-1, f"{slope_placebo[str(surr_size)][ss]}", fontsize=6)
                    ax[2].text(x_axis3, rel_effsize_5mg[str(surr_size)][ss]-1, f"{slope_5mg[str(surr_size)][ss]}", fontsize=6)
                    ax[2].text(x_axis3+0.3, rel_effsize_10mg[str(surr_size)][ss]-1, f"{slope_10mg[str(surr_size)][ss]}", fontsize=6)


            ax[1].legend(loc='lower right')

            ax[2].plot(np.arange(-1,1+len(EHDB_data['sub-002']['ses-1']['surr_sizes'])),np.zeros_like(np.arange(-1,1+len(EHDB_data['sub-002']['ses-1']['surr_sizes']))), ls='--', c='k', alpha=0.5)

            ax[2].set_xticks(np.arange(len(EHDB_data['sub-002']['ses-1']['surr_sizes'])))
            ax[2].set_xticklabels([f"{surr_size:.3f}" for surr_size in EHDB_data['sub-002']['ses-1']['surr_sizes']])
            ax[2].set_xlabel('Surround size (deg)')
            ax[2].set_ylabel('PSE difference (% veridical size)')
            ax[2].legend(loc='lower right')

            ax[2].set_yticks(np.linspace(0,-30,7))
            ax[2].set_yticklabels([0,-5,-10,-15,-20,-25,-30])

            ax[2].set_xlim([-0.5,1.5])

            ax[3].set_ylabel('Differences in relative effect sizes between radius 0.938 and radius 0 tasks')       
            the_space = np.linspace(-0.10, 0.10, len(subject_name_placebo))
            
            ax[3].bar(-0.3, np.average(rel_effsize_difference_placebo, weights=slope_average_placebo), color=condition_colors['placebo'], alpha=0.7, width=0.3)

            x_axis = 0*np.ones_like(rel_effsize_difference_placebo) + the_space
            ax[3].plot(x_axis-0.3, rel_effsize_difference_placebo, marker='s', mec='k', ls='', ms=8, color=condition_colors['placebo'])

            ax[3].bar(0, np.average(rel_effsize_difference_5mg, weights=slope_average_5mg), color=condition_colors['5mg'], alpha=0.7, width=0.3)
            x_axis = 0*np.ones_like(rel_effsize_difference_5mg) + the_space
            ax[3].plot(x_axis, rel_effsize_difference_5mg, marker='s', mec='k', ls='', ms=8, color=condition_colors['5mg'])

            ax[3].bar(0.3, np.average(rel_effsize_difference_10mg, weights=slope_average_10mg), color=condition_colors['10mg'], alpha=0.7, width=0.3)
            x_axis = 0*np.ones_like(rel_effsize_difference_10mg) + the_space
            ax[3].plot(x_axis+0.3, rel_effsize_difference_10mg, marker='s', mec='k', ls='', ms=8, color=condition_colors['10mg'])

            for ss in range(len(rel_effsize_difference_placebo)):

                x_axis2 = x_axis[ss]
                ax[3].plot([x_axis2-0.3, x_axis2, x_axis2+0.3], [rel_effsize_difference_placebo[ss], rel_effsize_difference_5mg[ss], rel_effsize_difference_10mg[ss]], c='k', linewidth=0.5, alpha=0.5)
                ax[3].text(x_axis2-0.3, rel_effsize_difference_placebo[ss], subject_name_placebo[ss], fontsize=8)
                ax[3].text(x_axis2-0.3, rel_effsize_difference_placebo[ss]-1, f"{slope_average_placebo[ss]}", fontsize=8)
                ax[3].text(x_axis2, rel_effsize_difference_5mg[ss]-1, f"{slope_average_5mg[ss]}", fontsize=8)
                ax[3].text(x_axis2+0.3, rel_effsize_difference_10mg[ss]-1, f"{slope_average_10mg[ss]}", fontsize=8)
                
            ax[3].plot(np.linspace(-1,2,10),np.zeros_like(np.linspace(-1,2,10)), ls='--', c='k', alpha=0.5)
            
            #set ticks with the
            ax[3].set_xticks([-0.3, 0, 0.3])
            ax[3].set_xticklabels(['placebo', '5mg', '10mg'])
            ax[3].set_xlabel('')
            ax[3].set_ylabel('PSE[0.000] - PSE[0.938] (%RMS)')
            #ax.set_ylim([-90,10])
            ax[3].set_xlim([-0.5,0.5])

            #Anova
            """ #get rid of ses-0
            df_EHDB = df_EHDB[df_EHDB['Session'] != 'ses_0']

            #check the balance of df_EHDB
            print(df_EHDB['Subject'].value_counts())
            print(df_EHDB.groupby(['Group_type'])['rel_effsize_difference'].count())

            #define p value and f value
            aovrm2way = AnovaRM(df_EHDB, 'rel_effsize_difference', 'Subject', within=['Group_type'], aggregate_func=np.mean)
            res2way = aovrm2way.fit()
            print("RM ANOVA all subjects Results:")
            #print only the p value
            p = round(res2way.anova_table['Pr > F'][0], 2)
            #print only the f value
            f = round(res2way.anova_table['F Value'][0], 2)
            print(f"p = {p:.3f}")
            print(f"F = {f:.3f}")

            #Add the ANOVA info to the plot
            ax[3].text(0.5, -10, 'RM Anova', fontsize=12)
            ax[3].text(0.5, -13, f'F value = {f}', fontsize=12)
            ax[3].text(0.5, -16, f'p = {p}', fontsize=12) """

            fig.suptitle("EHDB task - average results with 95% CI (slope weighted)")

            fig.savefig(opj(out_path,f"group_results_EHDB_average_slope.pdf"), dpi=600, bbox_inches='tight', transparent=True)

        def EHDBgroupplot_average_r2(self, out_path):
            condition_colors = {'placebo':'blue', '5mg':'orange', '10mg':'green'}
            EHDB_data = np.load(opj(out_path,'data/EHDB_data.npy'), allow_pickle=True).item()
            df_EHDB = pd.read_csv(opj(out_path,'data/group_results_EHDB.csv'))

            fig, ax = pl.subplots(1,4,figsize=(32,8))

            preds_placebo = dd(list)
            preds_5mg = dd(list)
            preds_10mg = dd(list)
            r2_placebo = dd(list)
            r2_5mg = dd(list)
            r2_10mg = dd(list)
            r2_average_placebo = []
            r2_average_5mg = []
            r2_average_10mg = []
            rel_effsize_placebo = dd(list)
            rel_effsize_5mg = dd(list)
            rel_effsize_10mg = dd(list)
            rel_effsize_difference_placebo = []
            rel_effsize_difference_5mg = []
            rel_effsize_difference_10mg = []
            subject_name_placebo = []
            subject_name_5mg = []
            subject_name_10mg = []

            for participant, sessions in EHDB_data.items():
                for session, data_ in sessions.items():
                    #is session 0 do nothing
                    if session == 'ses-0':
                        continue
                    else :
                        if data_['group_type'] == 'placebo':
                            for ii, surr_size in enumerate(data_['surr_sizes']):
                                preds_placebo[str(surr_size)].append(data_['preds'][str(surr_size)])
                                rel_effsize_placebo[str(surr_size)].append(data_['rel_effsize'][str(surr_size)])
                                print(data_['r2'][str(surr_size)])
                                if data_['r2'][str(surr_size)] > 0:
                                    r2_placebo[str(surr_size)].append(round(data_['r2'][str(surr_size)],2))
                                else:
                                    r2_placebo[str(surr_size)].append(0)
                            rel_effsize_difference_placebo.append(data_['rel_effsize']['0.9375']-data_['rel_effsize']['0.0'])
                            subject_name_placebo.append(participant)
                            if data_['r2_average'] > 0:
                                r2_average_placebo.append(round(data_['r2_average'],2))
                            else:
                                r2_average_placebo.append(0)
                        elif data_['group_type'] == '5mg':
                            for ii, surr_size in enumerate(data_['surr_sizes']):
                                preds_5mg[str(surr_size)].append(data_['preds'][str(surr_size)])
                                rel_effsize_5mg[str(surr_size)].append(data_['rel_effsize'][str(surr_size)])
                                if data_['r2'][str(surr_size)] > 0:
                                    r2_5mg[str(surr_size)].append(round(data_['r2'][str(surr_size)],2))
                                else:
                                    r2_5mg[str(surr_size)].append(0)
                            print(rel_effsize_5mg['0.9375'])
                            rel_effsize_difference_5mg.append(data_['rel_effsize']['0.9375']-data_['rel_effsize']['0.0'])
                            subject_name_5mg.append(participant)
                            if data_['r2_average'] > 0:
                                r2_average_5mg.append(round(data_['r2_average'],2))
                            else:
                                r2_average_5mg.append(0)
                        elif data_['group_type'] == '10mg':
                            for ii, surr_size in enumerate(data_['surr_sizes']):
                                preds_10mg[str(surr_size)].append(data_['preds'][str(surr_size)])
                                rel_effsize_10mg[str(surr_size)].append(data_['rel_effsize'][str(surr_size)])
                                if data_['r2'][str(surr_size)] > 0:
                                    r2_10mg[str(surr_size)].append(round(data_['r2'][str(surr_size)],2))
                                else:
                                    r2_10mg[str(surr_size)].append(0)
                            rel_effsize_difference_10mg.append(data_['rel_effsize']['0.9375']-data_['rel_effsize']['0.0'])
                            subject_name_10mg.append(participant)
                            if data_['r2_average'] > 0:
                                r2_average_10mg.append(round(data_['r2_average'],2))
                            else:
                                r2_average_10mg.append(0)

            the_space = np.linspace(-0.10, 0.10, len(subject_name_placebo))

            for ii, surr_size in enumerate(EHDB_data['sub-002']['ses-1']['surr_sizes']):
                ax[ii].set_title(f"Surr size {surr_size:.3f}")
                ax[ii].set_xscale('log')
                ax[ii].set_xlabel('Size difference (% veridical size)')
                ax[ii].set_ylabel('Prob ref perceived as larger')
                ax[ii].minorticks_off()
                # ax[ii].set_xlim([-30,30])

                true_size = EHDB_data['sub-002']['ses-1']['true_size']

                #ugly
                if surr_size>0:
                    ax[ii].set_xticks(np.linspace(true_size-0.5*true_size,true_size+0.3*true_size,9))
                    ax[ii].set_xticklabels([-50,-40,-30,-20,-10,0,10,20,30])
                else:
                    ax[ii].set_xticks(np.linspace(true_size-0.3*true_size,true_size+0.3*true_size,7))
                    ax[ii].set_xticklabels([-30,-20,-10,0,10,20,30])  
                
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)],np.ones_like(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)])*0.5, ls='--', c='k', alpha=0.5, label='PSE')
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)],np.ones_like(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)]), ls='--', c='green', alpha=0.5)
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)],np.zeros_like(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)]), ls='--', c='red', alpha=0.5)
                #bit hacky way
                ax[ii].plot(true_size*np.ones(100),np.linspace(0,1,100), ls='-', c='k', alpha=0.5, label='Veridical')
                ax[ii].legend(loc='lower right')

                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)], np.average(preds_placebo[str(surr_size)], axis=0, weights=r2_placebo[str(surr_size)]), c=condition_colors['placebo'], lw=2, label='Placebo')
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)], np.average(preds_5mg[str(surr_size)], axis=0, weights=r2_5mg[str(surr_size)]), c=condition_colors['5mg'], lw=2, label='5mg')
                ax[ii].plot(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)], np.average(preds_10mg[str(surr_size)], axis=0, weights=r2_10mg[str(surr_size)]), c=condition_colors['10mg'], lw=2, label='10mg')

                d1 = DescrStatsW(preds_placebo[str(surr_size)], weights=r2_placebo[str(surr_size)])
                d2 = DescrStatsW(preds_5mg[str(surr_size)], weights=r2_5mg[str(surr_size)])
                d3 = DescrStatsW(preds_10mg[str(surr_size)], weights=r2_10mg[str(surr_size)])
                ax[ii].fill_between(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)], np.average(preds_placebo[str(surr_size)], axis=0, weights=r2_placebo[str(surr_size)])-1.98*d1.std_mean, np.average(preds_placebo[str(surr_size)], axis=0, weights=r2_placebo[str(surr_size)])+1.98*d1.std_mean, color=condition_colors['placebo'], alpha=0.2)
                ax[ii].fill_between(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)], np.average(preds_5mg[str(surr_size)], axis=0, weights=r2_5mg[str(surr_size)])-1.98*d2.std_mean, np.average(preds_5mg[str(surr_size)], axis=0, weights=r2_5mg[str(surr_size)])+1.98*d2.std_mean, color=condition_colors['5mg'], alpha=0.2)
                ax[ii].fill_between(EHDB_data['sub-002']['ses-1']['x_spaces'][str(surr_size)], np.average(preds_10mg[str(surr_size)], axis=0, weights=r2_10mg[str(surr_size)])-1.98*d3.std_mean, np.average(preds_10mg[str(surr_size)], axis=0, weights=r2_10mg[str(surr_size)])+1.98*d3.std_mean, color=condition_colors['10mg'], alpha=0.2)

                #ax[-1].bar(ii, np.mean(rel_effsize[str(surr_size)],axis=0))
                ax[2].bar(ii-0.3, np.average(rel_effsize_placebo[str(surr_size)],axis=0, weights=r2_placebo[str(surr_size)]), color=condition_colors['placebo'], alpha=0.7, width=0.3)
                ax[2].bar(ii, np.average(rel_effsize_5mg[str(surr_size)],axis=0, weights=r2_5mg[str(surr_size)]), color=condition_colors['5mg'], alpha=0.7, width=0.3)
                ax[2].bar(ii+0.3, np.average(rel_effsize_10mg[str(surr_size)],axis=0, weights=r2_10mg[str(surr_size)]), color=condition_colors['10mg'], alpha=0.7, width=0.3)

                # print(np.ones_like(rel_effsize_placebo[str(surr_size)]))
                x_axis = ii * np.ones_like(rel_effsize_placebo[str(surr_size)]) + the_space

                #ax[-1].plot(ii*np.ones_like(rel_effsize[str(surr_size)]), rel_effsize[str(surr_size)], marker='s', mec='k', ls='', ms=8)
                ax[2].plot(x_axis-0.3, rel_effsize_placebo[str(surr_size)], marker='s', c=condition_colors['placebo'],mec='k', ls='', ms=6, alpha =0.4)
                ax[2].plot(x_axis, rel_effsize_5mg[str(surr_size)], marker='s', c=condition_colors['5mg'],mec='k', ls='', ms=6, alpha =0.4)  
                ax[2].plot(x_axis+0.3, rel_effsize_10mg[str(surr_size)], marker='s', c=condition_colors['10mg'],mec='k', ls='', ms=6, alpha =0.4)
                
            #plot the links and the text
            for ii, surr_size in enumerate(EHDB_data['sub-002']['ses-1']['surr_sizes']):
                x_axis2 = ii * np.ones_like(rel_effsize_placebo[str(surr_size)]) + the_space
                for ss in range(len(rel_effsize_placebo[str(surr_size)])):
                    #ax[-1].plot(np.arange(len(EHDB_data['sub-002']['ses-1']['surr_sizes'])), [rel_effsize[ref][ss] for ref in rel_effsize], c='k')
                    x_axis3 = x_axis2[ss]
                    ax[2].plot([x_axis3-0.3, x_axis3,x_axis3+0.3], [rel_effsize_placebo[str(surr_size)][ss],rel_effsize_5mg[str(surr_size)][ss],rel_effsize_10mg[str(surr_size)][ss]] , c='k', linewidth=0.5, alpha=0.5)
                    
                    ax[2].text(x_axis3-0.3, rel_effsize_placebo[str(surr_size)][ss]-0.1, f"{subject_name_placebo[ss]}", fontsize=6)
                    ax[2].text(x_axis3-0.3, rel_effsize_placebo[str(surr_size)][ss]-1, f"{r2_placebo[str(surr_size)][ss]}", fontsize=6)
                    ax[2].text(x_axis3, rel_effsize_5mg[str(surr_size)][ss]-1, f"{r2_5mg[str(surr_size)][ss]}", fontsize=6)
                    ax[2].text(x_axis3+0.3, rel_effsize_10mg[str(surr_size)][ss]-1, f"{r2_10mg[str(surr_size)][ss]}", fontsize=6)


            ax[1].legend(loc='lower right')

            ax[2].plot(np.arange(-1,1+len(EHDB_data['sub-002']['ses-1']['surr_sizes'])),np.zeros_like(np.arange(-1,1+len(EHDB_data['sub-002']['ses-1']['surr_sizes']))), ls='--', c='k', alpha=0.5)

            ax[2].set_xticks(np.arange(len(EHDB_data['sub-002']['ses-1']['surr_sizes'])))
            ax[2].set_xticklabels([f"{surr_size:.3f}" for surr_size in EHDB_data['sub-002']['ses-1']['surr_sizes']])
            ax[2].set_xlabel('Surround size (deg)')
            ax[2].set_ylabel('PSE difference (% veridical size)')
            ax[2].legend(loc='lower right')

            ax[2].set_yticks(np.linspace(0,-30,7))
            ax[2].set_yticklabels([0,-5,-10,-15,-20,-25,-30])

            ax[2].set_xlim([-0.5,1.5])

            ax[3].set_ylabel('Differences in relative effect sizes between radius 0.938 and radius 0 tasks')       
            the_space = np.linspace(-0.10, 0.10, len(subject_name_placebo))
            
            ax[3].bar(-0.3, np.average(rel_effsize_difference_placebo, weights=r2_average_placebo), color=condition_colors['placebo'], alpha=0.7, width=0.3)

            x_axis = 0*np.ones_like(rel_effsize_difference_placebo) + the_space
            ax[3].plot(x_axis-0.3, rel_effsize_difference_placebo, marker='s', mec='k', ls='', ms=8, color=condition_colors['placebo'])

            ax[3].bar(0, np.average(rel_effsize_difference_5mg, weights=r2_average_5mg), color=condition_colors['5mg'], alpha=0.7, width=0.3)
            x_axis = 0*np.ones_like(rel_effsize_difference_5mg) + the_space
            ax[3].plot(x_axis, rel_effsize_difference_5mg, marker='s', mec='k', ls='', ms=8, color=condition_colors['5mg'])

            ax[3].bar(0.3, np.average(rel_effsize_difference_10mg, weights=r2_average_10mg), color=condition_colors['10mg'], alpha=0.7, width=0.3)
            x_axis = 0*np.ones_like(rel_effsize_difference_10mg) + the_space
            ax[3].plot(x_axis+0.3, rel_effsize_difference_10mg, marker='s', mec='k', ls='', ms=8, color=condition_colors['10mg'])

            for ss in range(len(rel_effsize_difference_placebo)):

                x_axis2 = x_axis[ss]
                ax[3].plot([x_axis2-0.3, x_axis2, x_axis2+0.3], [rel_effsize_difference_placebo[ss], rel_effsize_difference_5mg[ss], rel_effsize_difference_10mg[ss]], c='k', linewidth=0.5, alpha=0.5)
                ax[3].text(x_axis2-0.3, rel_effsize_difference_placebo[ss], subject_name_placebo[ss], fontsize=8)
                ax[3].text(x_axis2-0.3, rel_effsize_difference_placebo[ss]-1, f"{r2_average_placebo[ss]}", fontsize=8)
                ax[3].text(x_axis2, rel_effsize_difference_5mg[ss]-1, f"{r2_average_5mg[ss]}", fontsize=8)
                ax[3].text(x_axis2+0.3, rel_effsize_difference_10mg[ss]-1, f"{r2_average_10mg[ss]}", fontsize=8)
                
            ax[3].plot(np.linspace(-1,2,10),np.zeros_like(np.linspace(-1,2,10)), ls='--', c='k', alpha=0.5)
            
            #set ticks with the
            ax[3].set_xticks([-0.3, 0, 0.3])
            ax[3].set_xticklabels(['placebo', '5mg', '10mg'])
            ax[3].set_xlabel('')
            ax[3].set_ylabel('PSE[0.000] - PSE[0.938] (%RMS)')
            #ax.set_ylim([-90,10])
            ax[3].set_xlim([-0.5,0.5])

            fig.suptitle("EHDB task - average results with 95% CI (r2 weighted)")

            fig.savefig(opj(out_path,f"group_results_EHDB_average_r2.pdf"), dpi=600, bbox_inches='tight', transparent=True)

        def EHDBgroupfit(self, out_path):
            groups_dict = {'placebo':['sub-001_ses-1','sub-002_ses-2','sub-003_ses-3','sub-004_ses-2','sub-005_ses-1','sub-006_ses-3','sub-007_ses-3','sub-008_ses-1',
                         'sub-009_ses-2','sub-010_ses-1','sub-011_ses-3','sub-012_ses-2','sub-013_ses-3','sub-014_ses-2','sub-015_ses-2','sub-018_ses-1','sub-019_ses-2','sub-020_ses-1'],

              '5mg':['sub-001_ses-3','sub-002_ses-1','sub-003_ses-1','sub-004_ses-3','sub-005_ses-2','sub-006_ses-1','sub-007_ses-1','sub-008_ses-3',
                    'sub-009_ses-1','sub-010_ses-2','sub-011_ses-2','sub-012_ses-3','sub-013_ses-2','sub-014_ses-1','sub-015_ses-3','sub-018_ses-3','sub-019_ses-1','sub-020_ses-2'],

              '10mg':['sub-001_ses-2','sub-002_ses-3','sub-003_ses-2','sub-004_ses-1','sub-005_ses-3','sub-006_ses-2','sub-007_ses-2','sub-008_ses-2',
                     'sub-009_ses-3','sub-010_ses-3','sub-011_ses-1','sub-012_ses-1','sub-013_ses-1','sub-014_ses-3','sub-015_ses-1','sub-018_ses-2','sub-019_ses-3','sub-020_ses-3']}


            df = pd.DataFrame()
            EHDB_data = {}

            for subject in getExpAttr(self):
                this_subject = getattr(self, subject)
                participant_id = rp(subject).split('_')[0]
                EHDB_data[participant_id] = {}
                for session in getExpAttr(this_subject):
                    this_session = getattr(this_subject, session)
                    session_id = rp(session).split('_')[0]
                    EHDB_data[participant_id][session_id] = {}
                    for run in getExpAttr(this_session):
                        this_run = getattr(this_session, run)

                        group_subject_session = ''
                        for group in groups_dict:
                            if f"{participant_id}_{session_id}" in groups_dict[group]:
                                group_subject_session = group
                                break

                        # Modify the following lines to extract Ebbinghaus-specific data
                        # Replace 'contrast_values', 'probs', and other attributes with Ebbinghaus data
                        ebbinghaus_data = {
                            'group_type': group_subject_session,
                            'surr_sizes': this_run.surr_sizes,  # List of surround sizes for the run
                            'size_values': this_run.size_values,  # Dictionary with size values for each surround size
                            'x_spaces': this_run.x_spaces,  # Dictionary with X-space values for each surround size
                            'probs': this_run.probs,  # Dictionary with probabilities for each surround size
                            'fits': this_run.fits,  # Dictionary with sigmoid curve fits for each surround size
                            'preds': this_run.preds,  # Dictionary with sigmoid curve predictions for each surround size
                            'rel_effsize': this_run.rel_effsize,  # Dictionary with relative effect sizes for each surround size
                            'true_size' : this_run.df_responses['Center stim radius'].max(),
                            'rel_effsize_difference': this_run.rel_effsize['0.9375']-this_run.rel_effsize['0.0'],
                            'slope': this_run.slope,
                            'slope_average': (this_run.slope['0.9375']+this_run.slope['0.0'])/2,
                            'r2':this_run.r2,
                            'r2_average':(this_run.r2['0.9375']+this_run.r2['0.0'])/2
                            #'x_space': this_run.x_space  # Common X-space for all surround sizes
                        }
                        """ print(len(this_run.surr_sizes))
                        print(len(this_run.size_values))
                        print(len(this_run.x_spaces))
                        print(len(this_run.probs))
                        print(len(this_run.fits))
                        print(len(this_run.preds))
                        print(len(this_run.rel_effsize))
                        print(len(this_run.x_space)) """

                        new = pd.DataFrame({'Subject':subject, 'Session':session,'Group_type':group_subject_session, 'rel_effsize_difference': this_run.rel_effsize['0.9375']-this_run.rel_effsize['0.0'], 'slope': this_run.slope['0.9375'], 'slope_0': this_run.slope['0.0'], 'slope_average':(this_run.slope['0.9375']+this_run.slope['0.0'])/2, 'r2':this_run.r2, 'r2_average':(this_run.r2['0.9375']+this_run.r2['0.0'])/2}, index=[0])
                        df = pd.concat([df, new], ignore_index=True)
                        EHDB_data[participant_id][session_id] = ebbinghaus_data

            #order the data by subject
            df = df.sort_values(by=['Subject','Session'])
            EHDB_data = dict(sorted(EHDB_data.items()))

            # Save Ebbinghaus-specific data as a numpy file
            np.save(opj(out_path, 'data/EHDB_data.npy'), EHDB_data)

            # Save the dataframe
            df.to_csv(opj(out_path, 'data/group_results_EHDB.csv'), index=False)

        def EHDB_CSgroupfit(self, out_path):
            
            CS_data = np.load(opj(out_path,'data/CS_data.npy'), allow_pickle=True).item()
            CS_data2 = {}

            #all sub, all ses but 0
            for k, v in CS_data.items():
                #print(k)
                #print(v)
                CS_data2[k] = {}
                for l, m in v.items():
                    
                    if l != 'ses-0':
                        CS_data2[k].update({l:{}})
                        #print(m)
                        #print(m['rel_effsize'])
                        #print(m['rel_effsize_nosurr'])
                        #print(l)
                        CS_data2[k][l]['rel_effsize'] = m['rel_effsize']
                        CS_data2[k][l]['rel_effsize_nosurr'] = m['rel_effsize_nosurr']
                        CS_data2[k][l]['group'] = m['group']
                        CS_data2[k][l]['slope'] = m['slope']
                        CS_data2[k][l]['rel_effsize_difference'] = m['rel_effsize'] - m['rel_effsize_nosurr']

            CS_data3 = {}

            #all sub, all ses but 0
            for k, v in CS_data2.items():
                #print(k)
                #print(v)
                CS_data3[k] = {}
                for l, m in v.items():
                    #print(m)
                    
                    if m['group'] == 'placebo':
                        CS_data3[k].update({l:{}})
                        #print(m)
                        #print(m['rel_effsize'])
                        #print(m['rel_effsize_nosurr'])
                        #print(l)
                        CS_data3[k][l]['group'] = m['group']
                        CS_data3[k][l]['slope'] = m['slope']
                        CS_data3[k]['rel_effsize_difference'] = m['rel_effsize_difference']
                        CS_data3[k][l]['rel_effsize_difference'] = m['rel_effsize_difference']
                        CS_data3[k][l]['rel_effsize_difference percentage'] = 100

            for k, v in CS_data2.items():
                #print(k)
                placebo_effsize = CS_data3[k]['rel_effsize_difference']
                #CS_data3[k] = {}
                #print(v['rel_effsize_difference'])
                

                for l, m in v.items():
                
                    if m['group'] != 'placebo':
                        CS_data3[k].update({l:{}})
                        if 'rel_effsize_difference' in CS_data3[k]:
                            del CS_data3[k]['rel_effsize_difference']
                        CS_data3[k][l]['group'] = m['group']
                        CS_data3[k][l]['slope'] = m['slope']
                        CS_data3[k][l]['rel_effsize_difference'] = m['rel_effsize_difference']
                        #rel_effsize_difference of the same sub, same ses where group = placebo
                        CS_data3[k][l]['rel_effsize_difference percentage'] = (m['rel_effsize_difference']*100/placebo_effsize)
                        
            #EHDB data
            EHDB_data = np.load(opj(out_path,'data/EHDB_data.npy'), allow_pickle=True).item()
            #get rid of ses 0 
            EHDB_data = {k: v for k, v in EHDB_data.items() if 'ses-0' not in k}
            #create a new variable rel_effesize_difference which is the difference between the rel_effsize['0.9375'] and rel_effsize['0.0]'
            for k, v in EHDB_data.items():
                for b, n in v.items():
                    n['rel_effsize_difference'] = n['rel_effsize']['0.9375'] - n['rel_effsize']['0.0']

            #order EHDB_data by subject (1st items) and sessions (2nd items)
            EHDB_data = dict(sorted(EHDB_data.items()))

            EHDB_data2 = {}

            #all sub, all ses but 0
            for k, v in EHDB_data.items():
                #print(k)
                #print(v)
                EHDB_data2[k] = {}
                for l, m in v.items():
                    
                    #print(l)
                    if l == 'ses-0':
                        print('bla ses-0')
                        #remove ses-0
                        #del CS_data[k][l]
                        
                    elif l != 'ses-0':
                        EHDB_data2[k].update({l:{}})
                        #print(m)
                        #print(m['rel_effsize'])
                        #print(m['rel_effsize_nosurr'])
                        #print(l)
                        
                        EHDB_data2[k][l]['group_type'] = m['group_type']
                        EHDB_data2[k][l]['slope'] = m['slope']
                        EHDB_data2[k][l]['rel_effsize_difference'] = m['rel_effsize_difference']

            EHDB_data3 = {}

            #all sub, all ses but 0
            for k, v in EHDB_data2.items():
                #print(k)
                #print(v)
                EHDB_data3[k] = {}
                for l, m in v.items():
                    print(m)
                    
                    if m['group_type'] == 'placebo':
                        EHDB_data3[k].update({l:{}})
                        #print(m)
                        #print(m['rel_effsize'])
                        #print(m['rel_effsize_nosurr'])
                        #print(l)
                        EHDB_data3[k][l]['group_type'] = m['group_type']
                        EHDB_data3[k][l]['slope'] = m['slope']
                        EHDB_data3[k]['rel_effsize_difference'] = m['rel_effsize_difference']
                        EHDB_data3[k][l]['rel_effsize_difference'] = m['rel_effsize_difference']
                        EHDB_data3[k][l]['rel_effsize_difference percentage'] = 100

            # print(EHDB_data3['sub-010']['rel_effsize_difference'])

            for k, v in EHDB_data2.items():
                #print(k)
                placebo_effsize = EHDB_data3[k]['rel_effsize_difference']
                
                for l, m in v.items():
                
                    if m['group_type'] != 'placebo':
                        EHDB_data3[k].update({l:{}})
                        if 'rel_effsize_difference' in EHDB_data3[k]:
                            del EHDB_data3[k]['rel_effsize_difference']
                        EHDB_data3[k][l]['group_type'] = m['group_type']
                        EHDB_data3[k][l]['slope'] = m['slope']
                        #rel_effsize_difference of the same sub, same ses where group = placebo
                        EHDB_data3[k][l]['rel_effsize_difference percentage'] = (m['rel_effsize_difference']*100/placebo_effsize)

            np.save(opj(out_path, 'data/CS_data3.npy'), CS_data3)
            np.save(opj(out_path, 'data/EHDB_data3.npy'), EHDB_data3)

        def EHDB_CSgroupplot_average3(self, out_path):
            condition_colors = {'placebo':'blue', '5mg':'#c46210', '10mg':'green'}
            condition_colors2 = {'placebo': '#6495ed', '5mg': '#ffbf00', '10mg': '#00ff00'}
            EHDB_data3 = np.load(opj(out_path,'data/EHDB_data3.npy'), allow_pickle=True).item()
            CS_data3 = np.load(opj(out_path,'data/CS_data3.npy'), allow_pickle=True).item()

            fig, ax = pl.subplots(1, 1, figsize=(8,8))

            rel_effsize_difference_placebo = []
            rel_effsize_difference_5mg = []
            rel_effsize_difference_10mg = []
            subject_name_placebo = []
            subject_name_5mg = []
            subject_name_10mg = []

            for participant, sessions in EHDB_data3.items():
                for session, data_ in sessions.items():
                    #is session 0 do nothing
                    if session == 'ses-0':
                        continue
                    else :
                        # print('tada')
                        # print(participant)
                        # print(session)
                        # print(data_)
                        if data_['group_type'] == 'placebo':                             
                            rel_effsize_difference_placebo.append(data_['rel_effsize_difference percentage'])
                            subject_name_placebo.append(participant)
                        elif data_['group_type'] == '5mg':
                            rel_effsize_difference_5mg.append(data_['rel_effsize_difference percentage'])
                            subject_name_5mg.append(participant)
                        elif data_['group_type'] == '10mg':
                            rel_effsize_difference_10mg.append(data_['rel_effsize_difference percentage'])
                            subject_name_10mg.append(participant)

            rel_effsize_difference_placebo_CS = []
            rel_effsize_difference_5mg_CS = []
            rel_effsize_difference_10mg_CS = []
            subject_name_placebo_CS = []
            subject_name_5mg_CS = []
            subject_name_10mg_CS = []

            for participant, sessions in CS_data3.items():
                for session, data_ in sessions.items():
                    #is session 0 do nothing
                    if session == 'ses-0':
                        continue
                    else :
                        if data_['group'] == 'placebo':                             
                            rel_effsize_difference_placebo_CS.append(data_['rel_effsize_difference percentage'])
                            subject_name_placebo_CS.append(participant)
                        elif data_['group'] == '5mg':
                            rel_effsize_difference_5mg_CS.append(data_['rel_effsize_difference percentage'])
                            subject_name_5mg_CS.append(participant)
                        elif data_['group'] == '10mg':
                            rel_effsize_difference_10mg_CS.append(data_['rel_effsize_difference percentage'])
                            subject_name_10mg_CS.append(participant)

            the_space = np.linspace(-0.10, 0.10, len(subject_name_placebo))
            # print(len(subject_name_placebo))
            # print(the_space)

            #EHDB
            ax.set_ylabel('Differences in relative effect sizes percentage for both EHDB and CS tasks')       
            
            ax.bar(-0.3, np.mean(rel_effsize_difference_placebo), color=condition_colors['placebo'], alpha=0.5, width=0.3)

            x_axis = 0*np.ones_like(rel_effsize_difference_placebo) + the_space
            ax.plot(x_axis-0.3, rel_effsize_difference_placebo, marker='s', mec='k', ls='',ms=6.5, color=condition_colors['placebo'])

            ax.bar(0, np.mean(rel_effsize_difference_5mg), color=condition_colors['5mg'], alpha=0.7, width=0.3)
            x_axis = 0*np.ones_like(rel_effsize_difference_5mg) + the_space
            ax.plot(x_axis, rel_effsize_difference_5mg, marker='s', mec='k', ls='',ms=6.5, color=condition_colors['5mg'])

            ax.bar(0.3, np.nanmean(rel_effsize_difference_10mg), color=condition_colors['10mg'], alpha=0.5, width=0.3)
            x_axis = 0*np.ones_like(rel_effsize_difference_10mg) + the_space
            ax.plot(x_axis+0.3, rel_effsize_difference_10mg, marker='s', mec='k', ls='',ms=6.5, color=condition_colors['10mg'])

            #CS
            ax.bar(-0.3, np.mean(rel_effsize_difference_placebo_CS), color=condition_colors2['placebo'], alpha=0.5, width=0.3)

            x_axis = 0*np.ones_like(rel_effsize_difference_placebo_CS) + the_space
            ax.plot(x_axis-0.3, rel_effsize_difference_placebo_CS, marker='o', mec='k', ls='',ms=6.5, color=condition_colors2['placebo'])

            ax.bar(0, np.mean(rel_effsize_difference_5mg_CS), color=condition_colors2['5mg'], alpha=0.5, width=0.3)
            x_axis = 0*np.ones_like(rel_effsize_difference_5mg_CS) + the_space
            ax.plot(x_axis, rel_effsize_difference_5mg_CS, marker='o', mec='k', ls='',ms=6.5, color=condition_colors2['5mg'])

            ax.bar(0.3, np.nanmean(rel_effsize_difference_10mg_CS), color=condition_colors2['10mg'], alpha=0.5, width=0.3)
            x_axis = 0*np.ones_like(rel_effsize_difference_10mg_CS) + the_space
            ax.plot(x_axis+0.3, rel_effsize_difference_10mg_CS, marker='o', mec='k', ls='',ms=6.5, color=condition_colors2['10mg'])

            #define label for the x axis, EHDB is square and CS is circle
            square_patch = mlines.Line2D([], [], color='black', label='EHDB', marker='s')
            circle_patch = mlines.Line2D([], [], color='black', label='CS', marker='o')
            ax.legend(handles=[square_patch, circle_patch])


            for ss in range(len(rel_effsize_difference_placebo)):

                x_axis2 = x_axis[ss]
                ax.plot([x_axis2-0.3, x_axis2, x_axis2+0.3], [rel_effsize_difference_placebo[ss], rel_effsize_difference_5mg[ss], rel_effsize_difference_10mg[ss]], c='k', linewidth=0.5, alpha=0.2)

                ax.text(x_axis2+0.3, rel_effsize_difference_10mg[ss], subject_name_placebo[ss], fontsize=6)

            for ss in range(len(rel_effsize_difference_placebo_CS)):

                x_axis2 = x_axis[ss]
                ax.plot([x_axis2-0.3, x_axis2, x_axis2+0.3], [rel_effsize_difference_placebo_CS[ss], rel_effsize_difference_5mg_CS[ss], rel_effsize_difference_10mg_CS[ss]], c='k', linewidth=0.5, alpha=0.2)

                ax.text(x_axis2+0.3, rel_effsize_difference_10mg_CS[ss], subject_name_placebo_CS[ss], fontsize=6)
                
            ax.plot(np.linspace(-1,2,10),np.zeros_like(np.linspace(-1,2,10)), ls='--', c='k', alpha=0.5)
            
            #set ticks with the
            ax.set_xticks([-0.3, 0, 0.3])
            ax.set_xticklabels(['placebo', '5mg', '10mg'])
            ax.set_xlabel('')
            ax.set_ylabel('relative percentage difference with placebo')
            #ax.set_ylim([-90,10])
            ax.set_xlim([-0.5,0.5])

            fig.suptitle("CS and EHDB task")

            fig.savefig(opj(out_path,f"group_results_EHDB_CS_average.pdf"), dpi=600, bbox_inches='tight', transparent=True)

        def Average_EHDB_CSgroupfit(self, out_path):
            #load the dataframes
            df_CS = pd.read_csv(opj(out_path, 'data/group_results_CS.csv'))
            df_EHDB = pd.read_csv(opj(out_path, 'data/group_results_EHDB.csv'))

            df_average_EHDB_CS = df_EHDB.copy()

            #list subjects from df_CS
            subjects = df_CS['Subject'].unique()
            sessions = df_CS['Session'].unique()

            for s in subjects:
                for ses in sessions:
                    
                    # print(df_CS.loc[(df_CS['Subject'] == s) & (df_CS['Session'] == ses), 'rel_effsize_diff'].values)
                    # print(df_EHDB.loc[(df_EHDB['Subject'] == s) & (df_EHDB['Session'] == ses), 'rel_effsize_difference'].values)
                    # Calculate the average and update the 'rel_effsize_difference' column in df_average_EHDB_CS
                    average_difference = (df_CS.loc[(df_CS['Subject'] == s) & (df_CS['Session'] == ses), 'rel_effsize_diff'].values[0] +
                                        df_EHDB.loc[(df_EHDB['Subject'] == s) & (df_EHDB['Session'] == ses), 'rel_effsize_difference'].values) / 2
                    df_average_EHDB_CS.loc[(df_average_EHDB_CS['Subject'] == s) & (df_average_EHDB_CS['Session'] == ses), 'rel_effsize_difference'] = float(average_difference)

                    #average slope
                    average_slope = (df_CS.loc[(df_CS['Subject'] == s) & (df_CS['Session'] == ses), 'slope_average'].values[0] + df_EHDB.loc[(df_EHDB['Subject'] == s) & (df_EHDB['Session'] == ses), 'slope_average'].values) / 2
                    df_average_EHDB_CS.loc[(df_average_EHDB_CS['Subject'] == s) & (df_average_EHDB_CS['Session'] == ses), 'slope_average'] = float(average_slope)
                    #average r2
                    average_r2 = (df_CS.loc[(df_CS['Subject'] == s) & (df_CS['Session'] == ses), 'r2_average'].values[0] + df_EHDB.loc[(df_EHDB['Subject'] == s) & (df_EHDB['Session'] == ses), 'r2_average'].values) / 2
                    df_average_EHDB_CS.loc[(df_average_EHDB_CS['Subject'] == s) & (df_average_EHDB_CS['Session'] == ses), 'r2_average'] = float(average_r2)

            # print(df_average_EHDB_CS)
            # Save the dataframe
            df_average_EHDB_CS.to_csv(opj(out_path, 'data/group_results_average_EHDB_CS.csv'), index=False)

        def Average_EHDB_CSgroupplot_average3(self, out_path):
            condition_colors = {'placebo':'blue', '5mg':'orange', '10mg':'green'}
            df_average_EHDB_CS = pd.read_csv(opj(out_path, 'data/group_results_average_EHDB_CS.csv'))

            fig, ax = pl.subplots(1,1,figsize=(8,8))

            rel_effsize_difference_placebo = df_average_EHDB_CS.loc[df_average_EHDB_CS['Group_type'] == 'placebo', 'rel_effsize_difference'].values
            rel_effsize_difference_5mg = df_average_EHDB_CS.loc[df_average_EHDB_CS['Group_type'] == '5mg', 'rel_effsize_difference'].values
            rel_effsize_difference_10mg = df_average_EHDB_CS.loc[df_average_EHDB_CS['Group_type'] == '10mg', 'rel_effsize_difference'].values
            subject_name_placebo = df_average_EHDB_CS.loc[df_average_EHDB_CS['Group_type'] == 'placebo', 'Subject'].values
            subject_name_5mg = df_average_EHDB_CS.loc[df_average_EHDB_CS['Group_type'] == '5mg', 'Subject'].values
            subject_name_10mg = df_average_EHDB_CS.loc[df_average_EHDB_CS['Group_type'] == '10mg', 'Subject'].values

            the_space = np.linspace(-0.10, 0.10, len(subject_name_placebo))
            # print(len(subject_name_placebo))
            # print(the_space)

            ax.set_ylabel('Differences in relative effect sizes between radius 0.938 and radius 0 tasks')       
            the_space = np.linspace(-0.10, 0.10, len(subject_name_placebo))
            
            ax.bar(-0.3, np.mean(rel_effsize_difference_placebo), color=condition_colors['placebo'], alpha=0.7, width=0.3)

            x_axis = 0*np.ones_like(rel_effsize_difference_placebo) + the_space
            ax.plot(x_axis-0.3, rel_effsize_difference_placebo, marker='s', mec='k', ls='', ms=8, color=condition_colors['placebo'])

            ax.bar(0, np.mean(rel_effsize_difference_5mg), color=condition_colors['5mg'], alpha=0.7, width=0.3)
            x_axis = 0*np.ones_like(rel_effsize_difference_5mg) + the_space
            ax.plot(x_axis, rel_effsize_difference_5mg, marker='s', mec='k', ls='', ms=8, color=condition_colors['5mg'])

            ax.bar(0.3, np.nanmean(rel_effsize_difference_10mg), color=condition_colors['10mg'], alpha=0.7, width=0.3)
            x_axis = 0*np.ones_like(rel_effsize_difference_10mg) + the_space
            ax.plot(x_axis+0.3, rel_effsize_difference_10mg, marker='s', mec='k', ls='', ms=8, color=condition_colors['10mg'])
            

            for ss in range(len(rel_effsize_difference_placebo)):

                x_axis2 = x_axis[ss]
                ax.plot([x_axis2-0.3, x_axis2, x_axis2+0.3], [rel_effsize_difference_placebo[ss], rel_effsize_difference_5mg[ss], rel_effsize_difference_10mg[ss]], c='k', linewidth=0.5, alpha=0.5)

                ax.text(x_axis2-0.3, rel_effsize_difference_placebo[ss], subject_name_placebo[ss], fontsize=8)
                
            ax.plot(np.linspace(-1,2,10),np.zeros_like(np.linspace(-1,2,10)), ls='--', c='k', alpha=0.5)
            
            #set ticks with the
            ax.set_xticks([-0.3, 0, 0.3])
            ax.set_xticklabels(['placebo', '5mg', '10mg'])
            ax.set_xlabel('')
            ax.set_ylabel('relative effect size difference average (%RMS)')
            #ax.set_ylim([-90,10])
            ax.set_xlim([-0.5,0.5])

            #Add the ANOVA info to the plot
            """ ax.text(-0.43, 3, 'RM ANOVA :', fontsize=12)
            ax.text(-0.15, 3, 'F value = 0.58', fontsize=12)
            ax.text(0.25, 3, 'p = 0.56', fontsize=12) """

            

            #check the balance of df_EHDB
            print(df_average_EHDB_CS['Subject'].value_counts())
            print(df_average_EHDB_CS.groupby(['Group_type'])['rel_effsize_difference'].count())

            #define p value and f value
            aovrm2way = AnovaRM(df_average_EHDB_CS, 'rel_effsize_difference', 'Subject', within=['Group_type'], aggregate_func=np.mean)
            res2way = aovrm2way.fit()
            print("RM ANOVA all subjects Results:")
            #print only the p value
            p = round(res2way.anova_table['Pr > F'][0], 4)
            #print only the f value
            f = round(res2way.anova_table['F Value'][0], 4)
            print(f"p = {p:.3f}")
            print(f"F = {f:.3f}")

            #Add the ANOVA info to the plot
            ax.text(0.5, -10, 'RM Anova', fontsize=12)
            ax.text(0.5, -13, f'F value = {f}', fontsize=12)
            ax.text(0.5, -16, f'p = {p}', fontsize=12)

            fig.suptitle("Average EHDB and CS task")

            fig.savefig(opj(out_path,f"group_results_Average_EHDB_CS_average.pdf"), dpi=600, bbox_inches='tight', transparent=True)

        def Average_EHDB_CSgroupplot_average_slope(self, out_path):
            condition_colors = {'placebo':'blue', '5mg':'orange', '10mg':'green'}
            df_average_EHDB_CS = pd.read_csv(opj(out_path, 'data/group_results_average_EHDB_CS.csv'))

            fig, ax = pl.subplots(1,1,figsize=(8,8))

            rel_effsize_difference_placebo = df_average_EHDB_CS.loc[df_average_EHDB_CS['Group_type'] == 'placebo', 'rel_effsize_difference'].values
            rel_effsize_difference_5mg = df_average_EHDB_CS.loc[df_average_EHDB_CS['Group_type'] == '5mg', 'rel_effsize_difference'].values
            rel_effsize_difference_10mg = df_average_EHDB_CS.loc[df_average_EHDB_CS['Group_type'] == '10mg', 'rel_effsize_difference'].values
            subject_name_placebo = df_average_EHDB_CS.loc[df_average_EHDB_CS['Group_type'] == 'placebo', 'Subject'].values
            subject_name_5mg = df_average_EHDB_CS.loc[df_average_EHDB_CS['Group_type'] == '5mg', 'Subject'].values
            subject_name_10mg = df_average_EHDB_CS.loc[df_average_EHDB_CS['Group_type'] == '10mg', 'Subject'].values
            slope_placebo = df_average_EHDB_CS.loc[df_average_EHDB_CS['Group_type'] == 'placebo', 'slope_average'].values
            slope_5mg = df_average_EHDB_CS.loc[df_average_EHDB_CS['Group_type'] == '5mg', 'slope_average'].values
            slope_10mg = df_average_EHDB_CS.loc[df_average_EHDB_CS['Group_type'] == '10mg', 'slope_average'].values

            the_space = np.linspace(-0.10, 0.10, len(subject_name_placebo))
            # print(len(subject_name_placebo))
            # print(the_space)

            ax.set_ylabel('Differences in relative effect sizes between radius 0.938 and radius 0 tasks')       
            the_space = np.linspace(-0.10, 0.10, len(subject_name_placebo))
            
            ax.bar(-0.3, np.mean(rel_effsize_difference_placebo), color=condition_colors['placebo'], alpha=0.7, width=0.3)

            x_axis = 0*np.ones_like(rel_effsize_difference_placebo) + the_space
            ax.plot(x_axis-0.3, rel_effsize_difference_placebo, marker='s', mec='k', ls='', ms=8, color=condition_colors['placebo'])

            ax.bar(0, np.mean(rel_effsize_difference_5mg), color=condition_colors['5mg'], alpha=0.7, width=0.3)
            x_axis = 0*np.ones_like(rel_effsize_difference_5mg) + the_space
            ax.plot(x_axis, rel_effsize_difference_5mg, marker='s', mec='k', ls='', ms=8, color=condition_colors['5mg'])

            ax.bar(0.3, np.nanmean(rel_effsize_difference_10mg), color=condition_colors['10mg'], alpha=0.7, width=0.3)
            x_axis = 0*np.ones_like(rel_effsize_difference_10mg) + the_space
            ax.plot(x_axis+0.3, rel_effsize_difference_10mg, marker='s', mec='k', ls='', ms=8, color=condition_colors['10mg'])
            

            for ss in range(len(rel_effsize_difference_placebo)):

                x_axis2 = x_axis[ss]
                ax.plot([x_axis2-0.3, x_axis2, x_axis2+0.3], [rel_effsize_difference_placebo[ss], rel_effsize_difference_5mg[ss], rel_effsize_difference_10mg[ss]], c='k', linewidth=0.5, alpha=0.5)
                ax.text(x_axis2-0.3, rel_effsize_difference_placebo[ss], subject_name_placebo[ss], fontsize=8)
                ax.text(x_axis2-0.3, rel_effsize_difference_placebo[ss]-0.5, round(slope_placebo[ss], 2), fontsize=8)
                ax.text(x_axis2, rel_effsize_difference_5mg[ss]-0.5, round(slope_5mg[ss], 2), fontsize=8)
                ax.text(x_axis2+0.3, rel_effsize_difference_10mg[ss]-0.5, round(slope_10mg[ss], 2), fontsize=8)
                
            ax.plot(np.linspace(-1,2,10),np.zeros_like(np.linspace(-1,2,10)), ls='--', c='k', alpha=0.5)
            
            #set ticks with the
            ax.set_xticks([-0.3, 0, 0.3])
            ax.set_xticklabels(['placebo', '5mg', '10mg'])
            ax.set_xlabel('')
            ax.set_ylabel('relative effect size difference average (%RMS)')
            #ax.set_ylim([-90,10])
            ax.set_xlim([-0.5,0.5])

            

            fig.suptitle("Average EHDB and CS task")

            fig.savefig(opj(out_path,f"group_results_Average_EHDB_CS_average_slope.pdf"), dpi=600, bbox_inches='tight', transparent=True)

        def Weigthed_average_EHDB_CSgroupfit(self, out_path):
            #load the dataframes
            df_CS = pd.read_csv(opj(out_path, 'data/group_results_CS.csv'))
            df_EHDB = pd.read_csv(opj(out_path, 'data/group_results_EHDB.csv'))

            df_average_EHDB_CS = df_EHDB.copy()

            #list subjects from df_CS
            subjects = df_CS['Subject'].unique()
            sessions = df_CS['Session'].unique()

            for s in subjects:
                for ses in sessions:
                    
                    # print(df_CS.loc[(df_CS['Subject'] == s) & (df_CS['Session'] == ses), 'rel_effsize_diff'].values)
                    # print(df_EHDB.loc[(df_EHDB['Subject'] == s) & (df_EHDB['Session'] == ses), 'rel_effsize_difference'].values)
                    # Calculate the weighted average and update the 'rel_effsize_difference' column in df_average_EHDB_CS
                    average_difference = (df_CS.loc[(df_CS['Subject'] == s) & (df_CS['Session'] == ses), 'rel_effsize_diff'].values[0] * df_CS.loc[(df_CS['Subject'] == s) & (df_CS['Session'] == ses), 'slope_average'].values[0] +
                                        df_EHDB.loc[(df_EHDB['Subject'] == s) & (df_EHDB['Session'] == ses), 'rel_effsize_difference'].values * df_EHDB.loc[(df_EHDB['Subject'] == s) & (df_EHDB['Session'] == ses), 'slope_average'].values) / (df_CS.loc[(df_CS['Subject'] == s) & (df_CS['Session'] == ses), 'slope_average'].values[0] + df_EHDB.loc[(df_EHDB['Subject'] == s) & (df_EHDB['Session'] == ses), 'slope_average'].values)
                    df_average_EHDB_CS.loc[(df_average_EHDB_CS['Subject'] == s) & (df_average_EHDB_CS['Session'] == ses), 'rel_effsize_difference'] = float(average_difference)
                    df_average_EHDB_CS.loc[(df_average_EHDB_CS['Subject'] == s) & (df_average_EHDB_CS['Session'] == ses), 'slope_average'] = (df_CS.loc[(df_CS['Subject'] == s) & (df_CS['Session'] == ses), 'slope_average'].values[0] + df_EHDB.loc[(df_EHDB['Subject'] == s) & (df_EHDB['Session'] == ses), 'slope_average'].values) / 2
                    df_average_EHDB_CS.loc[(df_average_EHDB_CS['Subject'] == s) & (df_average_EHDB_CS['Session'] == ses), 'r2_average'] = (df_CS.loc[(df_CS['Subject'] == s) & (df_CS['Session'] == ses), 'r2_average'].values[0] + df_EHDB.loc[(df_EHDB['Subject'] == s) & (df_EHDB['Session'] == ses), 'r2_average'].values) / 2

            # print(df_average_EHDB_CS)
            # Save the dataframe
            df_average_EHDB_CS.to_csv(opj(out_path, 'data/group_results_weighted_average_EHDB_CS.csv'), index=False)

        def Weigthed_average_EHDB_CSgroupplot_average_slope(self, out_path):
            condition_colors = {'placebo':'blue', '5mg':'orange', '10mg':'green'}
            df_average_EHDB_CS = pd.read_csv(opj(out_path, 'data/group_results_weighted_average_EHDB_CS.csv'))

            fig, ax = pl.subplots(1,1,figsize=(8,8))

            rel_effsize_difference_placebo = df_average_EHDB_CS.loc[df_average_EHDB_CS['Group_type'] == 'placebo', 'rel_effsize_difference'].values
            rel_effsize_difference_5mg = df_average_EHDB_CS.loc[df_average_EHDB_CS['Group_type'] == '5mg', 'rel_effsize_difference'].values
            rel_effsize_difference_10mg = df_average_EHDB_CS.loc[df_average_EHDB_CS['Group_type'] == '10mg', 'rel_effsize_difference'].values
            subject_name_placebo = df_average_EHDB_CS.loc[df_average_EHDB_CS['Group_type'] == 'placebo', 'Subject'].values
            subject_name_5mg = df_average_EHDB_CS.loc[df_average_EHDB_CS['Group_type'] == '5mg', 'Subject'].values
            subject_name_10mg = df_average_EHDB_CS.loc[df_average_EHDB_CS['Group_type'] == '10mg', 'Subject'].values
            slope_placebo = df_average_EHDB_CS.loc[df_average_EHDB_CS['Group_type'] == 'placebo', 'slope_average'].values
            slope_5mg = df_average_EHDB_CS.loc[df_average_EHDB_CS['Group_type'] == '5mg', 'slope_average'].values
            slope_10mg = df_average_EHDB_CS.loc[df_average_EHDB_CS['Group_type'] == '10mg', 'slope_average'].values

            the_space = np.linspace(-0.10, 0.10, len(subject_name_placebo))
            # print(len(subject_name_placebo))
            # print(the_space)

            ax.set_ylabel('Differences in relative effect sizes between radius 0.938 and radius 0 tasks')       
            the_space = np.linspace(-0.10, 0.10, len(subject_name_placebo))
            
            ax.bar(-0.3, np.mean(rel_effsize_difference_placebo), color=condition_colors['placebo'], alpha=0.7, width=0.3)

            x_axis = 0*np.ones_like(rel_effsize_difference_placebo) + the_space
            ax.plot(x_axis-0.3, rel_effsize_difference_placebo, marker='s', mec='k', ls='', ms=8, color=condition_colors['placebo'])

            ax.bar(0, np.mean(rel_effsize_difference_5mg), color=condition_colors['5mg'], alpha=0.7, width=0.3)
            x_axis = 0*np.ones_like(rel_effsize_difference_5mg) + the_space
            ax.plot(x_axis, rel_effsize_difference_5mg, marker='s', mec='k', ls='', ms=8, color=condition_colors['5mg'])

            ax.bar(0.3, np.nanmean(rel_effsize_difference_10mg), color=condition_colors['10mg'], alpha=0.7, width=0.3)
            x_axis = 0*np.ones_like(rel_effsize_difference_10mg) + the_space
            ax.plot(x_axis+0.3, rel_effsize_difference_10mg, marker='s', mec='k', ls='', ms=8, color=condition_colors['10mg'])
            

            for ss in range(len(rel_effsize_difference_placebo)):

                x_axis2 = x_axis[ss]
                ax.plot([x_axis2-0.3, x_axis2, x_axis2+0.3], [rel_effsize_difference_placebo[ss], rel_effsize_difference_5mg[ss], rel_effsize_difference_10mg[ss]], c='k', linewidth=0.5, alpha=0.5)
                ax.text(x_axis2-0.3, rel_effsize_difference_placebo[ss], subject_name_placebo[ss], fontsize=8)
                ax.text(x_axis2-0.3, rel_effsize_difference_placebo[ss]-0.5, round(slope_placebo[ss], 2), fontsize=8)
                ax.text(x_axis2, rel_effsize_difference_5mg[ss]-0.5, round(slope_5mg[ss], 2), fontsize=8)
                ax.text(x_axis2+0.3, rel_effsize_difference_10mg[ss]-0.5, round(slope_10mg[ss], 2), fontsize=8)
                
            ax.plot(np.linspace(-1,2,10),np.zeros_like(np.linspace(-1,2,10)), ls='--', c='k', alpha=0.5)
            
            #set ticks with the
            ax.set_xticks([-0.3, 0, 0.3])
            ax.set_xticklabels(['placebo', '5mg', '10mg'])
            ax.set_xlabel('')
            ax.set_ylabel('relative effect size difference average (%RMS)')
            #ax.set_ylim([-90,10])
            ax.set_xlim([-0.5,0.5])

            fig.suptitle("Weigthed average EHDB and CS task")

            fig.savefig(opj(out_path,f"group_results_Average_EHDB_CS_average_slope.pdf"), dpi=600, bbox_inches='tight', transparent=True)

        def Weigthed_average_EHDB_CSgroupplot_average_r2(self, out_path):
            condition_colors = {'placebo':'blue', '5mg':'orange', '10mg':'green'}
            df_average_EHDB_CS = pd.read_csv(opj(out_path, 'data/group_results_weighted_average_EHDB_CS.csv'))

            fig, ax = pl.subplots(1,1,figsize=(8,8))

            rel_effsize_difference_placebo = df_average_EHDB_CS.loc[df_average_EHDB_CS['Group_type'] == 'placebo', 'rel_effsize_difference'].values
            rel_effsize_difference_5mg = df_average_EHDB_CS.loc[df_average_EHDB_CS['Group_type'] == '5mg', 'rel_effsize_difference'].values
            rel_effsize_difference_10mg = df_average_EHDB_CS.loc[df_average_EHDB_CS['Group_type'] == '10mg', 'rel_effsize_difference'].values
            subject_name_placebo = df_average_EHDB_CS.loc[df_average_EHDB_CS['Group_type'] == 'placebo', 'Subject'].values
            subject_name_5mg = df_average_EHDB_CS.loc[df_average_EHDB_CS['Group_type'] == '5mg', 'Subject'].values
            subject_name_10mg = df_average_EHDB_CS.loc[df_average_EHDB_CS['Group_type'] == '10mg', 'Subject'].values
            slope_placebo = df_average_EHDB_CS.loc[df_average_EHDB_CS['Group_type'] == 'placebo', 'r2_average'].values
            slope_5mg = df_average_EHDB_CS.loc[df_average_EHDB_CS['Group_type'] == '5mg', 'r2_average'].values
            slope_10mg = df_average_EHDB_CS.loc[df_average_EHDB_CS['Group_type'] == '10mg', 'r2_average'].values

            the_space = np.linspace(-0.10, 0.10, len(subject_name_placebo))
            # print(len(subject_name_placebo))
            # print(the_space)

            ax.set_ylabel('Differences in relative effect sizes between radius 0.938 and radius 0 tasks')       
            the_space = np.linspace(-0.10, 0.10, len(subject_name_placebo))
            
            ax.bar(-0.3, np.mean(rel_effsize_difference_placebo), color=condition_colors['placebo'], alpha=0.7, width=0.3)

            x_axis = 0*np.ones_like(rel_effsize_difference_placebo) + the_space
            ax.plot(x_axis-0.3, rel_effsize_difference_placebo, marker='s', mec='k', ls='', ms=8, color=condition_colors['placebo'])

            ax.bar(0, np.mean(rel_effsize_difference_5mg), color=condition_colors['5mg'], alpha=0.7, width=0.3)
            x_axis = 0*np.ones_like(rel_effsize_difference_5mg) + the_space
            ax.plot(x_axis, rel_effsize_difference_5mg, marker='s', mec='k', ls='', ms=8, color=condition_colors['5mg'])

            ax.bar(0.3, np.nanmean(rel_effsize_difference_10mg), color=condition_colors['10mg'], alpha=0.7, width=0.3)
            x_axis = 0*np.ones_like(rel_effsize_difference_10mg) + the_space
            ax.plot(x_axis+0.3, rel_effsize_difference_10mg, marker='s', mec='k', ls='', ms=8, color=condition_colors['10mg'])
            

            for ss in range(len(rel_effsize_difference_placebo)):

                x_axis2 = x_axis[ss]
                ax.plot([x_axis2-0.3, x_axis2, x_axis2+0.3], [rel_effsize_difference_placebo[ss], rel_effsize_difference_5mg[ss], rel_effsize_difference_10mg[ss]], c='k', linewidth=0.5, alpha=0.5)
                ax.text(x_axis2-0.3, rel_effsize_difference_placebo[ss], subject_name_placebo[ss], fontsize=8)
                ax.text(x_axis2-0.3, rel_effsize_difference_placebo[ss]-0.5, round(slope_placebo[ss], 2), fontsize=8)
                ax.text(x_axis2, rel_effsize_difference_5mg[ss]-0.5, round(slope_5mg[ss], 2), fontsize=8)
                ax.text(x_axis2+0.3, rel_effsize_difference_10mg[ss]-0.5, round(slope_10mg[ss], 2), fontsize=8)
                
            ax.plot(np.linspace(-1,2,10),np.zeros_like(np.linspace(-1,2,10)), ls='--', c='k', alpha=0.5)
            
            #set ticks with the
            ax.set_xticks([-0.3, 0, 0.3])
            ax.set_xticklabels(['placebo', '5mg', '10mg'])
            ax.set_xlabel('')
            ax.set_ylabel('relative effect size difference average (%RMS)')
            #ax.set_ylim([-90,10])
            ax.set_xlim([-0.5,0.5])

            fig.suptitle("Weigthed average EHDB and CS task (r2 weighted)")

            fig.savefig(opj(out_path,f"group_results_Average_EHDB_CS_average_r2.pdf"), dpi=600, bbox_inches='tight', transparent=True)

        def ASCgroupfit(self, out_path):
            Final_data = {}
            groups_dict = {'placebo':['sub-001_ses-1','sub-002_ses-2','sub-003_ses-3','sub-004_ses-2','sub-005_ses-1','sub-006_ses-3','sub-007_ses-3','sub-008_ses-1',
                         'sub-009_ses-2','sub-010_ses-1','sub-011_ses-3','sub-012_ses-2','sub-013_ses-3','sub-014_ses-2','sub-015_ses-2','sub-018_ses-1','sub-019_ses-2','sub-020_ses-1'],

              '5mg':['sub-001_ses-3','sub-002_ses-1','sub-003_ses-1','sub-004_ses-3','sub-005_ses-2','sub-006_ses-1','sub-007_ses-1','sub-008_ses-3',
                    'sub-009_ses-1','sub-010_ses-2','sub-011_ses-2','sub-012_ses-3','sub-013_ses-2','sub-014_ses-1','sub-015_ses-3','sub-018_ses-3','sub-019_ses-1','sub-020_ses-2'],

              '10mg':['sub-001_ses-2','sub-002_ses-3','sub-003_ses-2','sub-004_ses-1','sub-005_ses-3','sub-006_ses-2','sub-007_ses-2','sub-008_ses-2',
                     'sub-009_ses-3','sub-010_ses-3','sub-011_ses-1','sub-012_ses-1','sub-013_ses-1','sub-014_ses-3','sub-015_ses-1','sub-018_ses-2','sub-019_ses-3','sub-020_ses-3']}

            for su, subject in enumerate(getExpAttr(self)):
                this_subject = getattr(self,subject)
                
                subject_name = rp(subject).split('_')[0]
                #print(subject_name)
                Final_data.update({subject_name:{}})

                for se,session in enumerate(getExpAttr(this_subject)):
                    this_session = getattr(this_subject,session)

                    for rr,run in enumerate(getExpAttr(this_session)):
                        this_run = getattr(this_session, run)

                        if this_run.run_result is None:
                            print(f"no run result for {this_run.expsettings['sub']} {this_run.expsettings['ses']} {this_run.expsettings['run']}, you should run fit_all and plot_all first")
                        
                        #print(this_run.run_result)

                        participant_id = rp(this_run.expsettings['sub'])
                        session_id = rp(this_run.expsettings['ses'])
                        
                        #if f"{participant_id}_{session_id}" is in groups_dict['placebo'] then type_id = 'placebo' etc
                        for group in groups_dict:
                            if f"{participant_id}_{session_id}" in groups_dict[group]:
                                type_id = group
                                break

                        this_run.run_result.update({'type':type_id})

                        Final_data[subject_name].update({f"{session_id}":this_run.run_result})
            print('Final_data')            
            print(Final_data)
            
            #save Final_data
            np.save(opj(out_path,'data/group_results_ASC.npy'),Final_data)
            #save pandas
            df = pd.DataFrame.from_dict(Final_data)
            df.to_csv(opj(out_path,'data/group_results_ASC.csv'))
            
            #global average for each group where cat_5D and cat_11D are averaged across participants
            global_average = {}

            # Calculate the mean and standard deviation for each category for each group
            for group in groups_dict:
                global_average[group] = {}
                global_average[group]['cat_5D'] = {}
                global_average[group]['cat_11D'] = {}
                for cat in ['cat_5D', 'cat_11D']:
                    global_average[group][cat]['mean'] = {}
                    global_average[group][cat]['std'] = {}
                    global_average[group][cat]['se'] = {}
                    for sub_cat in Final_data['sub-002']['ses-1'][cat]['mean']:
                        global_average[group][cat]['mean'][sub_cat] = np.round(np.mean([Final_data[participant][session][cat]['mean'][sub_cat] for participant, sessions in Final_data.items() for session, cats in sessions.items() if cats['type'] == group]), 3)
                        global_average[group][cat]['std'][sub_cat] = np.round(np.std([Final_data[participant][session][cat]['mean'][sub_cat] for participant, sessions in Final_data.items() for session, cats in sessions.items() if cats['type'] == group]), 3)
                        global_average[group][cat]['se'][sub_cat] = np.round(np.std([Final_data[participant][session][cat]['mean'][sub_cat] for participant, sessions in Final_data.items() for session, cats in sessions.items() if cats['type'] == group])/np.sqrt(len([Final_data[participant][session][cat]['mean'][sub_cat] for participant, sessions in Final_data.items() for session, cats in sessions.items() if cats['type'] == group])), 3)
            
            print(global_average)
            np.save(opj(out_path,'data/group_average_ASC.npy'),global_average)

            #knowing that the data is stored from 0 to 10, we do another dictionary called Final_data_10 where we change the range from 0 to 9 to 0 to 10
            Final_data_10 = {}
            for participant, sessions in Final_data.items():
                Final_data_10[participant] = {}
                for session, cats in sessions.items():
                    Final_data_10[participant][session] = {}
                    for cat, values in cats.items():
                        if cat != 'type':
                            Final_data_10[participant][session][cat] = {}
                            for calculus, sub_cats in values.items():
                                Final_data_10[participant][session][cat][calculus] = {}
                                if calculus == 'mean':
                                    for sub_cat, value in sub_cats.items():
                                        Final_data_10[participant][session][cat][calculus][sub_cat] = np.round(value*10/9, 3)
                                else:
                                    for sub_cat, value in sub_cats.items():
                                        Final_data_10[participant][session][cat][calculus][sub_cat] = np.round(value * 10/9, 3) #/!\ Apply the same transformation to the std
                        else:
                            Final_data_10[participant][session][cat] = {}
                            Final_data_10[participant][session][cat] = values

            #save Final_data_10
            print('Final_data_10')
            print(Final_data_10)
            np.save(opj(out_path,'data/group_results_ASC_10.npy'),Final_data_10)

            #global average for each group where cat_5D and cat_11D are averaged across participants
            global_average_10 = {}

            # Calculate the mean and standard deviation for each category for each group
            for group in groups_dict:
                global_average_10[group] = {}
                global_average_10[group]['cat_5D'] = {}
                global_average_10[group]['cat_11D'] = {}
                for cat in ['cat_5D', 'cat_11D']:
                    global_average_10[group][cat]['mean'] = {}
                    global_average_10[group][cat]['std'] = {}
                    global_average_10[group][cat]['se'] = {}
                    for sub_cat in Final_data_10['sub-002']['ses-1'][cat]['mean']:
                        global_average_10[group][cat]['mean'][sub_cat] = np.round(np.mean([Final_data_10[participant][session][cat]['mean'][sub_cat] for participant, sessions in Final_data_10.items() for session, cats in sessions.items() if cats['type'] == group]),3)
                        global_average_10[group][cat]['std'][sub_cat] = np.round(np.std([Final_data_10[participant][session][cat]['mean'][sub_cat] for participant, sessions in Final_data_10.items() for session, cats in sessions.items() if cats['type'] == group]), 3)
                        global_average_10[group][cat]['se'][sub_cat] = np.round(np.std([Final_data_10[participant][session][cat]['mean'][sub_cat] for participant, sessions in Final_data_10.items() for session, cats in sessions.items() if cats['type'] == group])/np.sqrt(len([Final_data_10[participant][session][cat]['mean'][sub_cat] for participant, sessions in Final_data_10.items() for session, cats in sessions.items() if cats['type'] == group])), 3)
            print('global_average_10')
            print(global_average_10)

            np.save(opj(out_path,'data/group_average_ASC_10.npy'),global_average_10)

        def ASCgroupplot(self, out_path):
            print('plotting ASC group results')

            Final_data = np.load(opj(out_path,'data/group_results_ASC.npy'), allow_pickle=True).item()
            Final_data_10 = np.load(opj(out_path,'data/group_results_ASC_10.npy'), allow_pickle=True).item()
            global_average = np.load(opj(out_path,'data/group_average_ASC.npy'), allow_pickle=True).item()

            #Second, we're going to do a plot with all the participants in the same plot, with the average of the group in a thin line, using the code from the previous plot
            #order Final_data by participant
            Final_data_10 = dict(sorted(Final_data_10.items()))
            # Initialize the figure
            fig2, ax2 = pl.subplots(3, 2, figsize=(16,24), subplot_kw={'projection': 'polar'})
            fig2.suptitle(f"ASC of all participants")
            # for each condition select the session that have type=condition
            # Iterate over each participant
            for participant, sessions in Final_data_10.items():
                # for each condition select the session that have type=condition
                for i, condition in enumerate(['placebo', '5mg', '10mg']):
                    # for the session with type = condition, plot the data
                    for j, session in enumerate(['ses-1', 'ses-2', 'ses-3']):
                        #check if the session exists
                        if session in sessions:
                            #check if the session is of the current condition
                            if sessions[session]['type'] == condition:
                                #Extract the participant information and scores for cat_5D and cat_11D for the current condition
                                # List of category labels for cat_5D and cat_11D with the order we want to plot them
                                cat_5D_labels = ['OBN', 'DED', 'VRS', 'AUA', 'VIR']
                                cat_11D_labels = ['IS', 'BS', 'SE', 'EU', 'CMP', 'AVS', 'EI', 'CI', 'ANX', 'ICC', 'DE']
                                #print(sessions[session]['cat_5D'])
                                # List of category scores for cat_5D and cat_11D considering the order of the labels
                                cat_5D_scores = [sessions[session]['cat_5D']['mean'][cat] for cat in cat_5D_labels]
                                cat_11D_scores = [sessions[session]['cat_11D']['mean'][cat] for cat in cat_11D_labels]
                                #print(cat_5D_scores)
                                # Convert values to radians for polar plot
                                cat_5D_radians = np.linspace(0, 2*np.pi, len(cat_5D_scores), endpoint=False)
                                cat_11D_radians = np.linspace(0, 2*np.pi, len(cat_11D_scores), endpoint=False)
                                # Close the loop by adding the first data point at the end
                                cat_5D_scores.append(cat_5D_scores[0])
                                cat_11D_scores.append(cat_11D_scores[0])
                                cat_5D_radians = np.append(cat_5D_radians, cat_5D_radians[0])
                                cat_11D_radians = np.append(cat_11D_radians, cat_11D_radians[0])

                                 # Create polar plots for cat_5D and cat_11D on the corresponding row
                                ax2[i, 0].plot(cat_5D_radians, cat_5D_scores, label=participant, alpha=0.4, color='black', linewidth=1)
                                ax2[i, 1].plot(cat_11D_radians, cat_11D_scores, label=participant, alpha=0.4, color='black', linewidth=1)
                                ax2[i, 0].set_thetagrids(np.degrees(cat_5D_radians[:-1]), cat_5D_labels)
                                #remove the circular thin line, leave just one at 5/10 as midpoint, and very faint
                                ax2[i, 0].grid(color='#D5D8DC', linestyle='dashed', linewidth=0.4)
                                ax2[i, 0].spines['polar'].set_color('#D5D8DC')
                                ax2[i, 0].spines['polar'].set_linewidth(0.4)
                                ax2[i, 0].spines['polar'].set_linestyle('dashed')
                                ax2[i, 0].spines['polar'].set_alpha(0.5)
                                #Offset of 90 degrees inverse clockwise in order to start on the top like the image
                                ax2[i, 0].set_theta_offset(np.pi / 2) 
                                ax2[i, 0].set_rlabel_position(22.5)
                                ax2[i, 0].set_ylim(0, 10)  # Adjust the y-axis limits as needed
                                
                                ax2[i, 1].set_thetagrids(np.degrees(cat_11D_radians[:-1]), cat_11D_labels)
                                #The radial line should be thin and grey as well
                                ax2[i, 1].grid(color='#D5D8DC', linestyle='dashed', linewidth=0.4)
                                ax2[i, 1].spines['polar'].set_color('#D5D8DC')
                                ax2[i, 1].spines['polar'].set_linewidth(0.4)
                                ax2[i, 1].spines['polar'].set_linestyle('dashed')
                                ax2[i, 1].spines['polar'].set_alpha(0.5)
                                

                                #ax2[i, 1].tick_params(axis='y', colors='#C059ca')
                                #Offset of 90 degrees inverse clockwise in order to start on the top like the image
                                ax2[i, 1].set_theta_offset(np.pi / 2) 
                                ax2[i, 1].set_rlabel_position(22.5)
                                ax2[i, 1].set_ylim(0, 10)  # Adjust the y-axis limits as needed

            # Add the global averages to the plots with a closed loop
            for i, condition in enumerate(['placebo', '5mg', '10mg']):
                global_average_cat_5D = list(global_average[condition]['cat_5D']['mean'].values())
                global_average_cat_11D = list(global_average[condition]['cat_11D']['mean'].values())
                global_average_cat_5D.append(global_average_cat_5D[0])
                global_average_cat_11D.append(global_average_cat_11D[0])

                global_average_cat_5D_radians = np.linspace(0, 2*np.pi, len(global_average[condition]['cat_5D']['mean']), endpoint=False)
                global_average_cat_11D_radians = np.linspace(0, 2*np.pi, len(global_average[condition]['cat_11D']['mean']), endpoint=False)
                global_average_cat_5D_radians = np.append(global_average_cat_5D_radians, global_average_cat_5D_radians[0])
                global_average_cat_11D_radians = np.append(global_average_cat_11D_radians, global_average_cat_11D_radians[0])

                ax2[i, 0].plot(global_average_cat_5D_radians, global_average_cat_5D, label='group average', linewidth=4, color='black')
                ax2[i, 1].plot(global_average_cat_11D_radians, global_average_cat_11D, label='group average', linewidth=4, color='black')

            
            #set a title for the columns and rows
            ax2[0, 0].set_title('5D categories')
            ax2[0, 1].set_title('11D categories')
            ax2[0, 0].set_ylabel('Placebo')
            ax2[1, 0].set_ylabel('5mg')
            ax2[2, 0].set_ylabel('10mg')
            
            #set a legend for the average line and the participant line
            #ax2[2, 1].legend(loc='lower right', bbox_to_anchor=(1.5, -0.2))
            textstr1 = '\n'.join((f"5D categories:",
                                    f"OBN: oceanic boundlessness",
                                    f"DED: dread of ego dissolution",
                                    f"VRS: visionary restructuralization",
                                    f"AUA: auditory alterations",
                                    f"VIR: vigilance reduction"))
            textstr2 = '\n'.join((f"11D categories:",
                                    f"EU: Experience of unity",
                                    f"SE: Spiritual experience",
                                    f"BS: Blissful state",
                                    f"IS: Insightfulness",
                                    f"DE: Disembodiment",
                                    f"ICC: Impaired control and cognition",
                                    f"ANX: Anxiety",
                                    f"CI: Complex imagery",
                                    f"EI: Elementary imagery",
                                    f"AVS: Audio-visual synesthesia",
                                    f"CMP: Changed meaning of percepts"))
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)        

            #let some space just upper the main title for the text boxes with the accronyms
            fig2.subplots_adjust(top=0.9, right=0.9)

            ax2[0, 0].text(-0.1, 1.5, textstr1, transform=ax2[0, 0].transAxes, fontsize=10,
                            verticalalignment='top', bbox=props)
            ax2[0, 1].text(1, 1.5, textstr2, transform=ax2[0, 1].transAxes, fontsize=10,verticalalignment='top', bbox=props)
            
            # Save the figure
            fig2.savefig(opj(out_path,"All_results_ASC.pdf"), dpi=600, bbox_inches='tight', transparent=True)
            pl.close(fig2)
            print("plotting all participants results done")

        def ASCaverageplot(self, out_path):
            print('plotting average ASC group results')

            Final_data_10 = np.load(opj(out_path,'data/group_results_ASC_10.npy'), allow_pickle=True).item()
            global_average = np.load(opj(out_path,'data/group_average_ASC_10.npy'), allow_pickle=True).item()

            #Second, we're going to do a plot with all the participants in the same plot, with the average of the group in a thin line, using the code from the previous plot
            #order Final_data by participant
            Final_data_10 = dict(sorted(Final_data_10.items()))
            # Initialize the figure
            fig2, ax2 = pl.subplots(1, 2, figsize=(16,8), subplot_kw={'projection': 'polar'})
            condition_colors = {'placebo':'blue', '5mg':'orange', '10mg':'green'}

            # Add the global averages to the plots with a closed loop
            for i, condition in enumerate(['placebo', '5mg', '10mg']):
                global_average_cat_5D = list(global_average[condition]['cat_5D']['mean'].values())
                global_average_cat_11D = list(global_average[condition]['cat_11D']['mean'].values())
                global_average_cat_5D.append(global_average_cat_5D[0])
                global_average_cat_11D.append(global_average_cat_11D[0])

                global_se_cat_5D = list(global_average[condition]['cat_5D']['se'].values())
                global_se_cat_11D = list(global_average[condition]['cat_11D']['se'].values())
                global_se_cat_5D.append(global_se_cat_5D[0])
                global_se_cat_11D.append(global_se_cat_11D[0])

                global_average_cat_5D_radians = np.linspace(0, 2*np.pi, len(global_average[condition]['cat_5D']['mean']), endpoint=False)
                global_average_cat_11D_radians = np.linspace(0, 2*np.pi, len(global_average[condition]['cat_11D']['mean']), endpoint=False)
                global_average_cat_5D_radians = np.append(global_average_cat_5D_radians, global_average_cat_5D_radians[0])
                global_average_cat_11D_radians = np.append(global_average_cat_11D_radians, global_average_cat_11D_radians[0])

                ax2[0].plot(global_average_cat_5D_radians, global_average_cat_5D, label=f'_group average_', linewidth=2, color=condition_colors[condition])
                ax2[1].plot(global_average_cat_11D_radians, global_average_cat_11D, label=f'{condition} mean', linewidth=2, color=condition_colors[condition])

                
                #Add the standard deviation as a shaded area
                ax2[0].fill_between(global_average_cat_5D_radians, np.array(global_average_cat_5D) - np.array(global_se_cat_5D)/2, np.array(global_average_cat_5D) + np.array(global_se_cat_5D)/2, alpha=0.2, color=condition_colors[condition], label=f'{condition} SE')
                ax2[1].fill_between(global_average_cat_11D_radians, np.array(global_average_cat_11D) - np.array(global_se_cat_11D)/2, np.array(global_average_cat_11D) + np.array(global_se_cat_11D)/2, alpha=0.2, color=condition_colors[condition], label=f'{condition} SE')

            ax2[1].legend(loc='lower right', bbox_to_anchor=(1.7, -0.2))
        
            cat_5D_labels = ['OBN', 'DED', 'VRS', 'AUA', 'VIR']
            cat_11D_labels = ['IS', 'BS', 'SE', 'EU', 'CMP', 'AVS', 'EI', 'CI', 'ANX', 'ICC', 'DE']
            ax2[0].set_thetagrids(np.degrees(global_average_cat_5D_radians[:-1]), cat_5D_labels)
            ax2[1].set_thetagrids(np.degrees(global_average_cat_11D_radians[:-1]), cat_11D_labels)

            #remove the circular thin line, leave just one at 5/10 as midpoint, and very faint
            ax2[0].grid(color='#D5D8DC', linestyle='dashed', linewidth=0.4)
            ax2[0].spines['polar'].set_color('#D5D8DC')
            ax2[0].spines['polar'].set_linewidth(0.4)
            ax2[0].spines['polar'].set_linestyle('dashed')
            
            #Offset of 90 degrees inverse clockwise in order to start on the top like the image
            ax2[0].set_theta_offset(np.pi / 2) 
            ax2[0].set_rlabel_position(22.5)
            ax2[0].set_ylim(0, 5.5)  # Adjust the y-axis limits as needed

            ax2[1].grid(color='#D5D8DC', linestyle='dashed', linewidth=0.4)
            ax2[1].spines['polar'].set_color('#D5D8DC')
            ax2[1].spines['polar'].set_linewidth(0.4)
            ax2[1].spines['polar'].set_linestyle('dashed')
            ax2[1].spines['polar'].set_alpha(0.2)
            ax2[1].set_theta_offset(np.pi / 2)
            ax2[1].set_rlabel_position(22.5)
            ax2[1].set_ylim(0, 7)  # Adjust the y-axis limits as needed

            #set a title for the columns and rows
            ax2[0].set_title('5D categories')
            ax2[1].set_title('11D categories')
                        
            textstr1 = '\n'.join((f"5D categories:",
                                    f"OBN: oceanic boundlessness",
                                    f"DED: dread of ego dissolution",
                                    f"VRS: visionary restructuralization",
                                    f"AUA: auditory alterations",
                                    f"VIR: vigilance reduction"))
            textstr2 = '\n'.join((f"11D categories:",
                                    f"EU: Experience of unity",
                                    f"SE: Spiritual experience",
                                    f"BS: Blissful state",
                                    f"IS: Insightfulness",
                                    f"DE: Disembodiment",
                                    f"ICC: Impaired control and cognition",
                                    f"ANX: Anxiety",
                                    f"CI: Complex imagery",
                                    f"EI: Elementary imagery",
                                    f"AVS: Audio-visual synesthesia",
                                    f"CMP: Changed meaning of percepts"))
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)        

            #let some space just upper the main title for the text boxes with the accronyms
            fig2.subplots_adjust(top=0.72)

            ax2[0].text(-0.1, 1.5, textstr1, transform=ax2[0].transAxes, fontsize=10,
                            verticalalignment='top', bbox=props)
            ax2[1].text(1, 1.5, textstr2, transform=ax2[1].transAxes, fontsize=10,verticalalignment='top', bbox=props)

            #set the title of the figure and locate it in the top centered
            fig2.suptitle(f"Average ASC \n of all participants")         

            # Save the figure
            fig2.savefig(opj(out_path,"Average_results_ASC.pdf"), dpi=600, bbox_inches='tight', transparent=True)
            pl.close(fig2)
            print("plotting average participants results done")   
        
        def ASCaverageplot2(self, out_path):
            print('plotting average ASC group results')

            Final_data_10 = np.load(opj(out_path,'data/group_results_ASC_10.npy'), allow_pickle=True).item()
            global_average = np.load(opj(out_path,'data/group_average_ASC_10.npy'), allow_pickle=True).item()

            #Second, we're going to do a plot with all the participants in the same plot, with the average of the group in a thin line, using the code from the previous plot
            #order Final_data by participant
            Final_data_10 = dict(sorted(Final_data_10.items()))
            # Initialize the figure
            fig2, ax2 = pl.subplots(1, 2, figsize=(16, 8), subplot_kw={'projection': 'polar'})
            # condition_colors = {'placebo':(0,0.1,1),'5mg':(0,0.6,0),'10mg':(1,0.7,0)}
            # condition_colors = {'placebo':(0.0,0.15,1) ,'5mg':(0.8,0.05,0.8),'10mg':(1,0.0,0.0)}
            condition_colors =  {'placebo':(0,0.6497,0.7176) ,'5mg':(0.4287*1.8,0.0,0.6130*1.5),'10mg':(0.864*1.1,0.0,0.0)}

            # Add the global averages to the plots with a closed loop
            for i, condition in enumerate(['placebo', '5mg', '10mg']):
                global_average_cat_5D = list(global_average[condition]['cat_5D']['mean'].values())
                global_average_cat_11D = list(global_average[condition]['cat_11D']['mean'].values())
                global_average_cat_5D.append(global_average_cat_5D[0])
                global_average_cat_11D.append(global_average_cat_11D[0])

                global_se_cat_5D = list(global_average[condition]['cat_5D']['se'].values())
                global_se_cat_11D = list(global_average[condition]['cat_11D']['se'].values())
                global_se_cat_5D.append(global_se_cat_5D[0])
                global_se_cat_11D.append(global_se_cat_11D[0])

                global_average_cat_5D_radians = np.linspace(0, 2*np.pi, len(global_average[condition]['cat_5D']['mean']), endpoint=False)
                global_average_cat_11D_radians = np.linspace(0, 2*np.pi, len(global_average[condition]['cat_11D']['mean']), endpoint=False)
                global_average_cat_5D_radians = np.append(global_average_cat_5D_radians, global_average_cat_5D_radians[0])
                global_average_cat_11D_radians = np.append(global_average_cat_11D_radians, global_average_cat_11D_radians[0])

                ax2[0].plot(global_average_cat_5D_radians, global_average_cat_5D, label=f'_group average_', linewidth=2, color=condition_colors[condition])
                ax2[1].plot(global_average_cat_11D_radians, global_average_cat_11D, label=f'{condition} mean', linewidth=2, color=condition_colors[condition])

                
                #Add the standard deviation as a shaded area
                ax2[0].fill_between(global_average_cat_5D_radians, np.array(global_average_cat_5D) - np.array(global_se_cat_5D)/2, np.array(global_average_cat_5D) + np.array(global_se_cat_5D)/2, alpha=0.2, color=condition_colors[condition], label=f'{condition} SE')
                ax2[1].fill_between(global_average_cat_11D_radians, np.array(global_average_cat_11D) - np.array(global_se_cat_11D)/2, np.array(global_average_cat_11D) + np.array(global_se_cat_11D)/2, alpha=0.2, color=condition_colors[condition], label=f'{condition} SE')
        
            cat_5D_labels = ['OBN', 'DED', 'VRS', 'AUA', 'VIR']
            cat_11D_labels = ['IS', 'BS', 'SE', 'EU', 'CMP', 'AVS', 'EI', 'CI', 'ANX', 'ICC', 'DE']
            ax2[0].set_thetagrids(np.degrees(global_average_cat_5D_radians[:-1]), cat_5D_labels)
            ax2[1].set_thetagrids(np.degrees(global_average_cat_11D_radians[:-1]), cat_11D_labels)

            #remove the circular thin line, leave just one at 5/10 as midpoint, and very faint
            ax2[0].grid(color='#D5D8DC', linestyle='dashed', linewidth=0.4)
            ax2[0].spines['polar'].set_color('#D5D8DC')
            ax2[0].spines['polar'].set_linewidth(0.4)
            ax2[0].spines['polar'].set_linestyle('dashed')
            
            #Offset of 90 degrees inverse clockwise in order to start on the top like the image
            ax2[0].set_theta_offset(np.pi / 2) 
            ax2[0].set_rlabel_position(22.5)
            # ax2[0].set_ylim(0, 5.5)  # Adjust the y-axis limits as needed

            ax2[1].grid(color='#D5D8DC', linestyle='dashed', linewidth=0.4)
            ax2[1].spines['polar'].set_color('#D5D8DC')
            ax2[1].spines['polar'].set_linewidth(0.4)
            ax2[1].spines['polar'].set_linestyle('dashed')
            ax2[1].spines['polar'].set_alpha(0.2)
            ax2[1].set_theta_offset(np.pi / 2)
            ax2[1].set_rlabel_position(22.5)
            ax2[1].set_ylim(0, 6)  # Adjust the y-axis limits as needed

            #set a title for the columns and rows
            # ax2[0].set_title('5D categories')
            # ax2[1].set_title('11D categories')  

            # Save the figure
            fig2.savefig(opj(out_path,"Average_results_ASC_color3.pdf"), dpi=600, bbox_inches='tight', transparent=True)
            pl.close(fig2)
            print("plotting average participants results done")   

        def ASCgroupplot_with_colors(self, out_path):
            print('plotting ASC group results')

            Final_data = np.load(opj(out_path,'data/group_results_ASC.npy'), allow_pickle=True).item()
            Final_data_10 = np.load(opj(out_path,'data/group_results_ASC_10.npy'), allow_pickle=True).item()
            global_average = np.load(opj(out_path,'data/group_average_ASC.npy'), allow_pickle=True).item()

            #Second, we're going to do a plot with all the participants in the same plot, with the average of the group in a thin line, using the code from the previous plot
            #order Final_data by participant
            Final_data_10 = dict(sorted(Final_data_10.items()))
            # Initialize the figure
            fig2, ax2 = pl.subplots(3, 2, figsize=(16, 32), subplot_kw={'projection': 'polar'})
            fig2.suptitle(f"ASC of all participants")
            # for each condition select the session that have type=condition
            # Iterate over each participant
            for participant, sessions in Final_data_10.items():
                # for each condition select the session that have type=condition
                for i, condition in enumerate(['placebo', '5mg', '10mg']):
                    # for the session with type = condition, plot the data
                    for j, session in enumerate(['ses-1', 'ses-2', 'ses-3']):
                        #check if the session exists
                        if session in sessions:
                            #check if the session is of the current condition
                            if sessions[session]['type'] == condition:
                                #Extract the participant information and scores for cat_5D and cat_11D for the current condition
                                # List of category labels for cat_5D and cat_11D with the order we want to plot them
                                cat_5D_labels = ['OBN', 'DED', 'VRS', 'AUA', 'VIR']
                                cat_11D_labels = ['IS', 'BS', 'SE', 'EU', 'CMP', 'AVS', 'EI', 'CI', 'ANX', 'ICC', 'DE']
                                #print(sessions[session]['cat_5D'])
                                # List of category scores for cat_5D and cat_11D considering the order of the labels
                                cat_5D_scores = [sessions[session]['cat_5D']['mean'][cat] for cat in cat_5D_labels]
                                cat_11D_scores = [sessions[session]['cat_11D']['mean'][cat] for cat in cat_11D_labels]
                                #print(cat_5D_scores)
                                # Convert values to radians for polar plot
                                cat_5D_radians = np.linspace(0, 2*np.pi, len(cat_5D_scores), endpoint=False)
                                cat_11D_radians = np.linspace(0, 2*np.pi, len(cat_11D_scores), endpoint=False)
                                # Close the loop by adding the first data point at the end
                                cat_5D_scores.append(cat_5D_scores[0])
                                cat_11D_scores.append(cat_11D_scores[0])
                                cat_5D_radians = np.append(cat_5D_radians, cat_5D_radians[0])
                                cat_11D_radians = np.append(cat_11D_radians, cat_11D_radians[0])

                                 # Create polar plots for cat_5D and cat_11D on the corresponding row
                                ax2[i, 0].plot(cat_5D_radians, cat_5D_scores, label=participant, alpha=0.4, color='black', linewidth=1)
                                ax2[i, 1].plot(cat_11D_radians, cat_11D_scores, label=participant, alpha=0.4, color='black', linewidth=1)
                                ax2[i, 0].set_thetagrids(np.degrees(cat_5D_radians[:-1]), cat_5D_labels)
                                #remove the circular thin line, leave just one at 5/10 as midpoint, and very faint
                                ax2[i, 0].yaxis.grid(color='#D5D8DC', linestyle='dashed', linewidth=0.4)
                                ax2[i, 0].spines['polar'].set_color('#D5D8DC')
                                ax2[i, 0].spines['polar'].set_linewidth(0.4)
                                ax2[i, 0].spines['polar'].set_linestyle('dashed')
                                ax2[i, 0].spines['polar'].set_alpha(0.5)
                                #Offset of 90 degrees inverse clockwise in order to start on the top like the image
                                ax2[i, 0].set_theta_offset(np.pi / 2) 
                                ax2[i, 0].set_rlabel_position(22.5)
                                ax2[i, 0].set_ylim(0, 10)  # Adjust the y-axis limits as needed
                                
                                ax2[i, 1].set_thetagrids(np.degrees(cat_11D_radians[:-1]), cat_11D_labels)
                                #The radial line should be thin and grey as well
                                ax2[i, 1].spines['polar'].set_color('#D5D8DC')
                                ax2[i, 1].spines['polar'].set_linewidth(0.4)
                                ax2[i, 1].spines['polar'].set_linestyle('dashed')
                                ax2[i, 1].spines['polar'].set_alpha(0.5)
                                #change the circular thin line to make them grey and thin 0.2
                                ax2[i, 1].yaxis.grid(color='#D5D8DC', linestyle='dashed', linewidth=0.4)
                                
                                # Set the facecolor
                                #purple
                                """ ax2[i, 1].fill_between(np.linspace(0, 3* np.pi * 2 * 2 / 11, 100), 0, 10, facecolor='#C059ca', alpha=0.5)
                                #blue
                                ax2[i, 1].fill_between(np.linspace(3* np.pi * 2 * 2 / 11, 7* np.pi * 2 * 2 / 11, 100), 0, 10, facecolor='#3d95de', alpha=0.5)
                                #green
                                ax2[i, 1].fill_between(np.linspace(7* np.pi * 2 * 2 / 11, 10* np.pi * 2 * 2 / 11, 100), 0, 10, facecolor='#5ad188', alpha=0.5) """

                                # set the labels colors
                                colors = ['#8b0098', '#8b0098', '#8b0098', '#8b0098', '#981100', '#981100', '#981100', '#981100', '#00984e', '#00984e', '#00984e']
                                for label, color in zip(ax2[i, 1].get_xticklabels(), colors):
                                    label.set_color(color)

                                #ax2[i, 1].tick_params(axis='y', colors='#C059ca')
                                #Offset of 90 degrees inverse clockwise in order to start on the top like the image
                                ax2[i, 1].set_theta_offset(np.pi / 2) 
                                ax2[i, 1].set_rlabel_position(22.5)
                                ax2[i, 1].set_ylim(0, 10)  # Adjust the y-axis limits as needed

            # Add the global averages to the plots with a closed loop
            for i, condition in enumerate(['placebo', '5mg', '10mg']):
                global_average_cat_5D = list(global_average[condition]['cat_5D']['mean'].values())
                global_average_cat_11D = list(global_average[condition]['cat_11D']['mean'].values())
                global_average_cat_5D.append(global_average_cat_5D[0])
                global_average_cat_11D.append(global_average_cat_11D[0])

                global_average_cat_5D_radians = np.linspace(0, 2*np.pi, len(global_average[condition]['cat_5D']['mean']), endpoint=False)
                global_average_cat_11D_radians = np.linspace(0, 2*np.pi, len(global_average[condition]['cat_11D']['mean']), endpoint=False)
                global_average_cat_5D_radians = np.append(global_average_cat_5D_radians, global_average_cat_5D_radians[0])
                global_average_cat_11D_radians = np.append(global_average_cat_11D_radians, global_average_cat_11D_radians[0])

                ax2[i, 0].plot(global_average_cat_5D_radians, global_average_cat_5D, label='group average', linewidth=4, color='black')
                ax2[i, 1].plot(global_average_cat_11D_radians, global_average_cat_11D, label='group average', linewidth=4, color='black')

            
            #set a title for the columns and rows
            ax2[0, 0].set_title('5D categories')
            ax2[0, 1].set_title('11D categories')
            ax2[0, 0].set_ylabel('Placebo')
            ax2[1, 0].set_ylabel('5mg')
            ax2[2, 0].set_ylabel('10mg')
            
            #set a legend for the average line and the participant line
            #ax2[2, 1].legend(loc='lower right', bbox_to_anchor=(1.5, -0.2))
            textstr1 = '\n'.join((f"5D categories:",
                                    f"OBN: oceanic boundlessness",
                                    f"DED: dread of ego dissolution",
                                    f"VRS: visionary restructuralization",
                                    f"AUA: auditory alterations",
                                    f"VIR: vigilance reduction"))
            textstr2 = '\n'.join((f"11D categories:",
                                    f"EU: Experience of unity",
                                    f"SE: Spiritual experience",
                                    f"BS: Blissful state",
                                    f"IS: Insightfulness",
                                    f"DE: Disembodiment",
                                    f"ICC: Impaired control and cognition",
                                    f"ANX: Anxiety",
                                    f"CI: Complex imagery",
                                    f"EI: Elementary imagery",
                                    f"AVS: Audio-visual synesthesia",
                                    f"CMP: Changed meaning of percepts"))
            textstr3 = '\n'.join((f"Oceanic boundlessness:",
                                  f"EU: Experience of unity",
                                    f"SE: Spiritual experience",
                                    f"BS: Blissful state",
                                    f"IS: Insightfulness"))
            textstr4 = '\n'.join((f"Anxious ego dissolution:",
                                  f"DE: Disembodiment",
                                    f"ICC: Impaired control and cognition",
                                    f"ANX: Anxiety"))
            textstr5 = '\n'.join((f"Visionary restructuralization:",
                                    f"CI: Complex imagery",
                                    f"EI: Elementary imagery",
                                    f"AVS: Audio-visual synesthesia",
                                    f"CMP: Changed meaning of percepts"))
            textstr6 = "11D categories:"
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)        

            #let some space just upper the main title for the text boxes with the accronyms
            fig2.subplots_adjust(top=0.9, right=0.9)

            ax2[0, 0].text(-0.1, 1.5, textstr1, transform=ax2[0, 0].transAxes, fontsize=10,
                            verticalalignment='top', bbox=props)
            #ax2[0, 1].text(1, 1.5, textstr2, transform=ax2[0, 1].transAxes, fontsize=10,verticalalignment='top', bbox=props)
            ax2[0, 1].text(1, 1.5, textstr6, transform=ax2[0, 1].transAxes, fontsize=10,verticalalignment='top', bbox=props)            
            ax2[0, 1].text(1, 1.45, textstr3, transform=ax2[0, 1].transAxes, fontsize=10,
                            verticalalignment='top', bbox=props).set_color('#8b0098')
            ax2[0, 1].text(1, 1.25, textstr5, transform=ax2[0, 1].transAxes, fontsize=10,
                            verticalalignment='top', bbox=props).set_color('#981100')
            ax2[0, 1].text(1, 1.05, textstr4, transform=ax2[0, 1].transAxes, fontsize=10,
                            verticalalignment='top', bbox=props).set_color('#00984e')
            
            # Save the figure
            fig2.savefig(opj(out_path,"All_results_ASC.pdf"), dpi=600, bbox_inches='tight', transparent=True)
            pl.close(fig2)
            print("plotting all participants results done")

        def ASCgroupplot_individuals(self, out_path):
            print('plotting ASC group results')

            Final_data = np.load(opj(out_path,'data/group_results_ASC.npy'), allow_pickle=True).item()
            Final_data_10 = np.load(opj(out_path,'data/group_results_ASC_10.npy'), allow_pickle=True).item()
            global_average = np.load(opj(out_path,'data/group_average_ASC.npy'), allow_pickle=True).item()

            #Plot the polar plot for each subject, a 3*2 plot where columns are cat_5D, cat_11D and rows are the 3 different conditions (placebo, 5mg, 10mg) and with the average of the group in a thin line
            
            # Iterate over each participant
            for participant, sessions in Final_data_10.items():
                #print(sessions)
                # Initialize the figure
                fig, ax = pl.subplots(3, 2, figsize=(16, 32), subplot_kw={'projection': 'polar'})
                fig.suptitle(f"ASC of {participant}")
                # for each condition select the session that have type=condition
                for i, condition in enumerate(['placebo', '5mg', '10mg']):
                    # for the session with type = condition, plot the data
                    for j, session in enumerate(['ses-1', 'ses-2', 'ses-3']):
                        #check if the session exists
                        if session in sessions:
                            #check if the session is of the current condition
                            if sessions[session]['type'] == condition:
                                # List of category labels for cat_5D and cat_11D with the order we want to plot them
                                cat_5D_labels = ['OBN', 'DED', 'VRS', 'AUA', 'VIR']
                                cat_11D_labels = ['IS', 'BS', 'SE', 'EU', 'CMP', 'AVS', 'EI', 'CI', 'ANX', 'ICC', 'DE']
                                #print('cat_5D')
                                #print(sessions[session]['cat_5D']['mean'])
                                # List of category scores for cat_5D and cat_11D considering the order of the labels
                                cat_5D_scores = [sessions[session]['cat_5D']['mean'][cat] for cat in cat_5D_labels]
                                cat_11D_scores = [sessions[session]['cat_11D']['mean'][cat] for cat in cat_11D_labels]
                                #print(cat_5D_scores)
                                
                                # Convert values to radians for polar plot
                                cat_5D_radians = np.linspace(0, 2*np.pi, len(cat_5D_scores), endpoint=False)
                                cat_11D_radians = np.linspace(0, 2*np.pi, len(cat_11D_scores), endpoint=False)
                                # Close the loop by adding the first data point at the end
                                cat_5D_scores.append(cat_5D_scores[0])
                                cat_11D_scores.append(cat_11D_scores[0])
                                cat_5D_radians = np.append(cat_5D_radians, cat_5D_radians[0])
                                cat_11D_radians = np.append(cat_11D_radians, cat_11D_radians[0])
                                # Create polar plots for cat_5D and cat_11D on the corresponding row
                                #make a dictionary with the colors for each participant as a shade of blue
                                
                                ax[i, 0].plot(cat_5D_radians, cat_5D_scores, label=participant, marker='.', color='black', linewidth=0.5, alpha=0.9)
                                ax[i, 1].plot(cat_11D_radians, cat_11D_scores, label=participant, marker='.',color='black', linewidth=0.5, alpha=0.9)
                                ax[i, 0].set_thetagrids(np.degrees(cat_5D_radians[:-1]), cat_5D_labels)
                                #remove the circular thin line, leave just one at 5/10 as midpoint, and very faint
                                ax[i, 0].yaxis.grid(color='#D5D8DC', linestyle='dashed', linewidth=0.5)
                                ax[i, 0].spines['polar'].set_color('#D5D8DC')
                                ax[i, 0].spines['polar'].set_linewidth(0.5)
                                ax[i, 0].spines['polar'].set_linestyle('dashed')
                                #Offset of -pi/8 inverse clockwise in order to start on the top like the image
                                ax[i, 0].set_theta_offset(np.pi / 2) 
                                ax[i, 0].set_rlabel_position(22.5)
                                ax[i, 0].set_ylim(0, 10)  # Adjust the y-axis limits as needed
                                
                                ax[i, 1].set_thetagrids(np.degrees(cat_11D_radians[:-1]), cat_11D_labels)
                                #change the circular thin line to make them grey and thin 0.2
                                ax[i, 1].yaxis.grid(color='#D5D8DC', linestyle='dashed', linewidth=0.5)
                                #The radial line should be thin and grey as well
                                ax[i, 1].spines['polar'].set_color('#D5D8DC')
                                ax[i, 1].spines['polar'].set_linewidth(0.4)
                                ax[i, 1].spines['polar'].set_linestyle('dashed')
                                #Offset of 90 degrees inverse clockwise in order to start on the top like the image
                                ax[i, 1].set_theta_offset(-np.pi / 15)
                                ax[i, 1].set_rlabel_position(22.5)
                                ax[i, 1].set_ylim(0, 10)  # Adjust the y-axis limits as needed
                                

                                # Add the global averages to the plots with a closed loop
                                global_average_cat_5D = list(global_average[condition]['cat_5D']['mean'].values())
                                global_average_cat_11D = list(global_average[condition]['cat_11D']['mean'].values())
                                global_average_cat_5D.append(global_average_cat_5D[0])
                                global_average_cat_11D.append(global_average_cat_11D[0])

                                global_average_cat_5D_radians = np.linspace(0, 2*np.pi, len(global_average[condition]['cat_5D']['mean']), endpoint=False)
                                global_average_cat_11D_radians = np.linspace(0, 2*np.pi, len(global_average[condition]['cat_11D']['mean']), endpoint=False)
                                global_average_cat_5D_radians = np.append(global_average_cat_5D_radians, global_average_cat_5D_radians[0])
                                global_average_cat_11D_radians = np.append(global_average_cat_11D_radians, global_average_cat_11D_radians[0])
                                
                                ax[i, 0].plot(global_average_cat_5D_radians, global_average_cat_5D, label='group average', linewidth=3, color='black')
                                ax[i, 1].plot(global_average_cat_11D_radians, global_average_cat_11D, label='group average', linewidth=3, color='black')

                
                #set a title for the columns and rows
                ax[0, 0].set_title('5D categories')
                ax[0, 1].set_title('11D categories')
                ax[0, 0].set_ylabel('Placebo')
                ax[1, 0].set_ylabel('5mg')
                ax[2, 0].set_ylabel('10mg')
                
                #set a legend for the average line and the participant line
                ax[2, 1].legend(loc='lower right', bbox_to_anchor=(1.5, -0.2))

                #let some space just upper the main title for the text boxes with the accronyms
                fig.subplots_adjust(top=0.9, right=0.9)
                                
                # Set a text box on the upper center listing the accronyms of the 5D categories
                textstr1 = '\n'.join((f"5D categories:",
                                    f"OBN: oceanic boundlessness",
                                    f"DED: dread of ego dissolution",
                                    f"VRS: visionary restructuralization",
                                    f"AUA: auditory alterations",
                                    f"VIR: vigilance reduction"))
                props = dict(boxstyle='round', facecolor='white', alpha=0.5)
                ax[0, 0].text(-0.1, 1.5, textstr1, transform=ax[0, 0].transAxes, fontsize=10,
                            verticalalignment='top', bbox=props)
                # Set a text box on the upper right listing the accronyms of the 11D categories
                textstr2 = '\n'.join((f"11D categories:",
                                    f"EU: Experience of unity",
                                    f"SE: Spiritual experience",
                                    f"BS: Blissful state",
                                    f"IS: Insightfulness",
                                    f"DE: Disembodiment",
                                    f"ICC: Impaired control and cognition",
                                    f"ANX: Anxiety",
                                    f"CI: Complex imagery",
                                    f"EI: Elementary imagery",
                                    f"AVS: Audio-visual synesthesia",
                                    f"CMP: Changed meaning of percepts"))
                ax[0, 1].text(1, 1.5, textstr2, transform=ax[0, 1].transAxes, fontsize=10,
                            verticalalignment='top', bbox=props)
                
                # Save the figure
                fig.savefig(opj(out_path,participant,f"{participant}_results_ASC.pdf"), dpi=600, bbox_inches='tight', transparent=True)
                pl.close(fig)
                print("plotting group result done for subject", participant)
            
            #Second, we're going to do a plot with all the participants in the same plot, with the average of the group in a thin line, using the code from the previous plot
            #order Final_data by participant
            Final_data_10 = dict(sorted(Final_data_10.items()))
            # Initialize the figure
            fig2, ax2 = pl.subplots(3, 2, figsize=(16, 32), subplot_kw={'projection': 'polar'})
            fig2.suptitle(f"ASC of all participants")
            # for each condition select the session that have type=condition
            # Iterate over each participant
            for participant, sessions in Final_data_10.items():
                # for each condition select the session that have type=condition
                for i, condition in enumerate(['placebo', '5mg', '10mg']):
                    # for the session with type = condition, plot the data
                    for j, session in enumerate(['ses-1', 'ses-2', 'ses-3']):
                        #check if the session exists
                        if session in sessions:
                            #check if the session is of the current condition
                            if sessions[session]['type'] == condition:
                                #Extract the participant information and scores for cat_5D and cat_11D for the current condition
                                # List of category labels for cat_5D and cat_11D with the order we want to plot them
                                cat_5D_labels = ['OBN', 'DED', 'VRS', 'AUA', 'VIR']
                                cat_11D_labels = ['IS', 'BS', 'SE', 'EU', 'CMP', 'AVS', 'EI', 'CI', 'ANX', 'ICC', 'DE']
                                #print(sessions[session]['cat_5D'])
                                # List of category scores for cat_5D and cat_11D considering the order of the labels
                                cat_5D_scores = [sessions[session]['cat_5D']['mean'][cat] for cat in cat_5D_labels]
                                cat_11D_scores = [sessions[session]['cat_11D']['mean'][cat] for cat in cat_11D_labels]
                                #print(cat_5D_scores)
                                # Convert values to radians for polar plot
                                cat_5D_radians = np.linspace(0, 2*np.pi, len(cat_5D_scores), endpoint=False)
                                cat_11D_radians = np.linspace(0, 2*np.pi, len(cat_11D_scores), endpoint=False)
                                # Close the loop by adding the first data point at the end
                                cat_5D_scores.append(cat_5D_scores[0])
                                cat_11D_scores.append(cat_11D_scores[0])
                                cat_5D_radians = np.append(cat_5D_radians, cat_5D_radians[0])
                                cat_11D_radians = np.append(cat_11D_radians, cat_11D_radians[0])
                                 # Create polar plots for cat_5D and cat_11D on the corresponding row
                                ax2[i, 0].plot(cat_5D_radians, cat_5D_scores, label=participant, alpha=0.4, color='black', linewidth=1)
                                ax2[i, 1].plot(cat_11D_radians, cat_11D_scores, label=participant, alpha=0.4, color='black', linewidth=1)
                                ax2[i, 0].set_thetagrids(np.degrees(cat_5D_radians[:-1]), cat_5D_labels)
                                #remove the circular thin line, leave just one at 5/10 as midpoint, and very faint
                                ax2[i, 0].yaxis.grid(color='#D5D8DC', linestyle='dashed', linewidth=0.4)
                                ax2[i, 0].spines['polar'].set_color('#D5D8DC')
                                #Offset of 90 degrees inverse clockwise in order to start on the top like the image
                                ax2[i, 0].set_theta_offset(np.pi / 2) 
                                ax2[i, 0].set_rlabel_position(22.5)
                                ax2[i, 0].set_ylim(0, 10)  # Adjust the y-axis limits as needed
                                """ ax[i, 0].legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
                                ax[i, 0].set_title(f'Session {j+1} ({condition})') """
                                ax2[i, 1].set_thetagrids(np.degrees(cat_11D_radians[:-1]), cat_11D_labels)
                                #The radial line should be thin and grey as well
                                ax2[i, 1].spines['polar'].set_color('#D5D8DC')
                                ax2[i, 1].spines['polar'].set_linewidth(0.4)
                                ax2[i, 1].spines['polar'].set_linestyle('dashed')
                                #change the circular thin line to make them grey and thin 0.2
                                ax2[i, 1].yaxis.grid(color='#D5D8DC', linestyle='dashed', linewidth=0.4)
                                #Offset of 90 degrees inverse clockwise in order to start on the top like the image
                                ax2[i, 1].set_theta_offset(np.pi / 2) 
                                ax2[i, 1].set_rlabel_position(22.5)
                                ax2[i, 1].set_ylim(0, 10)  # Adjust the y-axis limits as needed
                                """ ax[i, 1].legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
                                ax[i, 1].set_title(f'Session {j+1} ({condition})') """

            # Add the global averages to the plots with a closed loop
            for i, condition in enumerate(['placebo', '5mg', '10mg']):
                global_average_cat_5D = list(global_average[condition]['cat_5D']['mean'].values())
                global_average_cat_11D = list(global_average[condition]['cat_11D']['mean'].values())
                global_average_cat_5D.append(global_average_cat_5D[0])
                global_average_cat_11D.append(global_average_cat_11D[0])

                global_average_cat_5D_radians = np.linspace(0, 2*np.pi, len(global_average[condition]['cat_5D']['mean']), endpoint=False)
                global_average_cat_11D_radians = np.linspace(0, 2*np.pi, len(global_average[condition]['cat_11D']['mean']), endpoint=False)
                global_average_cat_5D_radians = np.append(global_average_cat_5D_radians, global_average_cat_5D_radians[0])
                global_average_cat_11D_radians = np.append(global_average_cat_11D_radians, global_average_cat_11D_radians[0])

                ax2[i, 0].plot(global_average_cat_5D_radians, global_average_cat_5D, label='group average', linewidth=4, color='black')
                ax2[i, 1].plot(global_average_cat_11D_radians, global_average_cat_11D, label='group average', linewidth=4, color='black')

            
            #set a title for the columns and rows
            ax2[0, 0].set_title('5D categories')
            ax2[0, 1].set_title('11D categories')
            ax2[0, 0].set_ylabel('Placebo')
            ax2[1, 0].set_ylabel('5mg')
            ax2[2, 0].set_ylabel('10mg')
            
            #set a legend for the average line and the participant line
            #ax2[2, 1].legend(loc='lower right', bbox_to_anchor=(1.5, -0.2))

            #let some space just upper the main title for the text boxes with the accronyms
            #fig2.subplots_adjust(top=0.99, right=0.99)
            fig2.subplots_adjust(top=0.9, right=0.9)
            ax2[0, 0].text(-0.1, -0.3, textstr1, transform=ax[0, 0].transAxes, fontsize=10,
                            verticalalignment='top', bbox=props)
            ax2[0, 1].text(-0.1, 0.5, textstr2, transform=ax[0, 1].transAxes, fontsize=10,
                            verticalalignment='top', bbox=props)
            # Save the figure
            fig2.savefig(opj(out_path,"All_results_ASC.pdf"), dpi=600, bbox_inches='tight', transparent=True)
            pl.close(fig2)
            print("plotting all participants results done")

        def ASCcategories_5D(self, out_path):
            group_results_ASC = np.load(opj(out_path,'data/group_results_ASC_10.npy'), allow_pickle=True).item()
            group_average_ASC = np.load(opj(out_path,'data/group_average_ASC_10.npy'),  allow_pickle=True).item()

            # Initialize the figure
            fig3, ax3 = pl.subplots(1, 6, figsize=(24, 8))
            run_labels = np.array([0.0, 1.0, 2.0])
            categories_labels = ['Oceanic \n Boundlessness', 'Dread of Ego \n Dissolution', 'Visionary \n Restructuralization', 'Auditory \n alterations', 'Vigilance \n reduction']
            
            #for each cat_5D, plot the bar plot of group_average_ASC for eahc of the three conditions (placebo, 5mg, 10mg)
            for i, cat in enumerate(['OBN', 'DED', 'VRS', 'AUA', 'VIR']):
                #print(cat)
                #print(group_average_ASC[cat])
                ax3[i].bar(run_labels, [group_average_ASC['placebo']['cat_5D']['mean'][cat],group_average_ASC['5mg']['cat_5D']['mean'][cat],group_average_ASC['10mg']['cat_5D']['mean'][cat]], color=['blue', 'orange', 'green'])
                ax3[i].set_title(categories_labels[i])
                ax3[i].set_ylim([0, 8.2])
                ax3[i].set_ylabel('Score')

            #6th plot with the average of the 5D categories
            group_average_ASC_placebo_cat_5D_mean = np.mean([group_average_ASC['placebo']['cat_5D']['mean'][cat] for cat in ['OBN', 'DED', 'VRS', 'AUA', 'VIR']])
            group_average_ASC_5mg_cat_5D_mean = np.mean([group_average_ASC['5mg']['cat_5D']['mean'][cat] for cat in ['OBN', 'DED', 'VRS', 'AUA', 'VIR']])
            group_average_ASC_10mg_cat_5D_mean = np.mean([group_average_ASC['10mg']['cat_5D']['mean'][cat] for cat in ['OBN', 'DED', 'VRS', 'AUA', 'VIR']])
            ax3[5].bar(run_labels, [group_average_ASC_placebo_cat_5D_mean, group_average_ASC_5mg_cat_5D_mean, group_average_ASC_10mg_cat_5D_mean], color=['blue', 'orange', 'green'])
            ax3[5].set_title('Total score')
            ax3[5].set_ylim([0, 8.2])
            ax3[5].set_ylabel('Score')
                
            the_space = np.linspace(-0.15, 0.15, (group_results_ASC.items().__len__()))
            participant_list = list(group_results_ASC.keys())

            #then, for each participant, each session, each condition, plot the scatter plot of the cat_5D score for each of the three conditions (placebo, 5mg, 10mg)
            for i, cat in enumerate(['OBN', 'DED', 'VRS', 'AUA', 'VIR']):

                run_labels = np.array([0.0, 1.0, 2.0])

                for participant, sessions in group_results_ASC.items():
                    P510 = {}
                    S510 = {}
                    for session, res in sessions.items() :
                        #we want, in order, placebo, 5mg, 10mg
                        if res['type'] == 'placebo':
                            P510['placebo'] = res['cat_5D']['mean'][cat]
                            S510['placebo'] = res['cat_5D']['std_dev'][cat]
                        elif res['type'] == '5mg':
                            P510['5mg'] = res['cat_5D']['mean'][cat]
                            S510['5mg'] = res['cat_5D']['std_dev'][cat]
                        elif res['type'] == '10mg':
                            P510['10mg'] = res['cat_5D']['mean'][cat]
                            S510['10mg'] = res['cat_5D']['std_dev'][cat]
                        #if a type is missing but not the two others, then add np.nan to P510. FOr example, if type placebo doesn't exist but 5mg and 10mg exists, then add np.nan to P510['placebo']
                        if 'placebo' not in P510:
                            P510['placebo'] = np.nan
                            S510['placebo'] = np.nan
                        elif '5mg' not in P510:
                            P510['5mg'] = np.nan
                            S510['5mg'] = np.nan
                        elif '10mg' not in P510:
                            P510['10mg'] = np.nan
                            S510['10mg'] = np.nan

                    run_labels += the_space[participant_list.index(participant)]  

                    #loop over conditions to plot the points with colors corresponding to the condition
                    for j, condition in enumerate(['placebo', '5mg', '10mg']):
                        ax3[i].plot(run_labels[j], P510[condition], alpha=0.8, marker='s', color=['blue', 'orange', 'green'][j], markeredgecolor='black', markeredgewidth=0.4)

                    #Add the links between the points
                    ax3[i].plot(run_labels, [P510['placebo'], P510['5mg'], P510['10mg']], alpha=0.4, color='black')
                    
                    #standard deviation
                    #ax3[i].errorbar(run_labels, [P510['placebo'], P510['5mg'], P510['10mg']], yerr=[S510['placebo']/2, S510['5mg']/2, S510['10mg']/2], color='black', linewidth=0.4)

                    #Add participant number to the plot
                    ax3[i].text(run_labels[2], P510['10mg']+0.05, participant, fontsize=8)
                    run_labels = np.array([0.0, 1.0, 2.0])
                    
                ax3[i].set_xticks([0, 1, 2])
                ax3[i].set_xticklabels(['placebo', '5mg', '10mg'])

            #6th plot with the average of the 5D categories
            group_results_ASC_placebo_cat_5D_mean = []
            group_results_ASC_5mg_cat_5D_mean = []
            group_results_ASC_10mg_cat_5D_mean = []

            group_results_ASC_placebo_cat_5D_std = []
            group_results_ASC_5mg_cat_5D_std = []
            group_results_ASC_10mg_cat_5D_std = []

            for participant, sessions in group_results_ASC.items():
                P510 = {}
                S510 = {}
                for session, res in sessions.items() :
                    #we want, in order, placebo, 5mg, 10mg
                    if res['type'] == 'placebo':
                        P510['placebo'] = np.mean([res['cat_5D']['mean'][cat] for cat in ['OBN', 'DED', 'VRS', 'AUA', 'VIR']])
                        S510['placebo'] = np.std([res['cat_5D']['std_dev'][cat] for cat in ['OBN', 'DED', 'VRS', 'AUA', 'VIR']])
                    elif res['type'] == '5mg':
                        P510['5mg'] = np.mean([res['cat_5D']['mean'][cat] for cat in ['OBN', 'DED', 'VRS', 'AUA', 'VIR']])
                        S510['5mg'] = np.std([res['cat_5D']['std_dev'][cat] for cat in ['OBN', 'DED', 'VRS', 'AUA', 'VIR']])
                    elif res['type'] == '10mg':
                        P510['10mg'] = np.mean([res['cat_5D']['mean'][cat] for cat in ['OBN', 'DED', 'VRS', 'AUA', 'VIR']])
                        S510['10mg'] = np.std([res['cat_5D']['std_dev'][cat] for cat in ['OBN', 'DED', 'VRS', 'AUA', 'VIR']])
                    #if a type is missing but not the two others, then add np.nan to P510. FOr example, if type placebo doesn't exist but 5mg and 10mg exists, then add np.nan to P510['placebo']
                    if 'placebo' not in P510:
                        P510['placebo'] = np.nan
                        S510['placebo'] = np.nan
                    elif '5mg' not in P510:
                        P510['5mg'] = np.nan
                        S510['5mg'] = np.nan
                    elif '10mg' not in P510:
                        P510['10mg'] = np.nan
                        S510['10mg'] = np.nan

                group_results_ASC_placebo_cat_5D_mean.append(P510['placebo'])
                group_results_ASC_5mg_cat_5D_mean.append(P510['5mg'])
                group_results_ASC_10mg_cat_5D_mean.append(P510['10mg'])

                group_results_ASC_placebo_cat_5D_std.append(S510['placebo'])
                group_results_ASC_5mg_cat_5D_std.append(S510['5mg'])
                group_results_ASC_10mg_cat_5D_std.append(S510['10mg'])

            for i, participant in enumerate(participant_list):
                run_labels += the_space[participant_list.index(participant)]

                #loop over conditions to plot the points with colors corresponding to the condition
                for j, condition in enumerate(['placebo', '5mg', '10mg']):
                    ax3[5].plot(run_labels[j], [group_results_ASC_placebo_cat_5D_mean[i], group_results_ASC_5mg_cat_5D_mean[i], group_results_ASC_10mg_cat_5D_mean[i]][j], alpha=0.8, marker='s', color=['blue', 'orange', 'green'][j], markeredgecolor='black', markeredgewidth=0.4)

                #Add the links between the points
                ax3[5].plot(run_labels, [group_results_ASC_placebo_cat_5D_mean[i], group_results_ASC_5mg_cat_5D_mean[i], group_results_ASC_10mg_cat_5D_mean[i]], alpha=0.4, color='black')

                #standard deviation
                #ax3[5].errorbar(run_labels, [group_results_ASC_placebo_cat_5D_mean[i], group_results_ASC_5mg_cat_5D_mean[i], group_results_ASC_10mg_cat_5D_mean[i]], yerr=[group_results_ASC_placebo_cat_5D_std[i], group_results_ASC_5mg_cat_5D_std[i], group_results_ASC_10mg_cat_5D_std[i]], color='black', linewidth=0.4)

                #Add participant number to the plot
                ax3[5].text(run_labels[2], group_results_ASC_10mg_cat_5D_mean[i]+0.05, participant, fontsize=8)
                run_labels = np.array([0.0, 1.0, 2.0])

            ax3[5].set_xticks([0, 1, 2])
            ax3[5].set_xticklabels(['placebo', '5mg', '10mg'])

            fig3.subplots_adjust(top=0.8, right=1.2)

            #put title for the whole figure
            fig3.suptitle('Cat_5D Scores')

            fig3.savefig(opj(out_path,"All_results_ASC_cat_5D.pdf"), dpi=600, bbox_inches='tight', transparent=True)

        def ASCcategories_11D(self, out_path):
            group_results_ASC = np.load(opj(out_path,'data/group_results_ASC_10.npy'), allow_pickle=True).item()
            group_average_ASC = np.load(opj(out_path,'data/group_average_ASC_10.npy'),  allow_pickle=True).item()

            # Initialize the figure
            fig4, ax4 = pl.subplots(2, 6, figsize=(32, 16))
            run_labels = np.array([0.0, 1.0, 2.0])
            categories_labels = ['Insightfulness', 'Blissful \n state', 'Spiritual \n experience', 'Experience of unity', 'Changed meaning \n of percepts', 'Audio-visual \n synesthesia', 'Elementary \n imagery', 'Complex \n imagery', 'Anxiety', 'Impaired control \n and cognition', 'Disembodiment']
            #for each cat_11D, plot the bar plot of group_average_ASC for eahc of the three conditions (placebo, 5mg, 10mg)
            for i, cat in enumerate(['IS', 'BS', 'SE', 'EU', 'CMP', 'AVS', 'EI', 'CI', 'ANX', 'ICC', 'DE']):
                #x equal i modulo 6 because we have 6 rows of 2 columns
                y = i % 6
                #y equal i divided by 6 because we have 6 rows of 2 columns
                x = i // 6
                #print(cat)
                #print(group_average_ASC[cat])
                ax4[x,y].bar(run_labels, [group_average_ASC['placebo']['cat_11D']['mean'][cat],group_average_ASC['5mg']['cat_11D']['mean'][cat],group_average_ASC['10mg']['cat_11D']['mean'][cat]], color=['blue', 'orange', 'green'])
                ax4[x,y].set_title(categories_labels[i])
                ax4[x,y].set_ylim([0, 10])
                ax4[x,y].set_ylabel('Score')

            the_space = np.linspace(-0.15, 0.15, (group_results_ASC.items().__len__()))
            participant_list = list(group_results_ASC.keys())

            #then, for each participant, each session, each condition, plot the scatter plot of the cat_11D score for each of the three conditions (placebo, 5mg, 10mg)
            for i, cat in enumerate(['IS', 'BS', 'SE', 'EU', 'CMP', 'AVS', 'EI', 'CI', 'ANX', 'ICC', 'DE']):
                run_labels = np.array([0.0, 1.0, 2.0])
                #x equal i modulo 6 because we have 6 rows of 2 columns
                y = i % 6
                #y equal i divided by 6 because we have 6 rows of 2 columns
                x = i // 6

                for participant, sessions in group_results_ASC.items():
                    P510 = {}
                    for session, res in sessions.items() :
                        #we want, in order, placebo, 5mg, 10mg
                        if res['type'] == 'placebo':
                            P510['placebo'] = res['cat_11D']['mean'][cat]
                        elif res['type'] == '5mg':
                            P510['5mg'] = res['cat_11D']['mean'][cat]
                        elif res['type'] == '10mg':
                            P510['10mg'] = res['cat_11D']['mean'][cat]
                        #if a type is missing but not the two others, then add np.nan to P510. FOr example, if type placebo doesn't exist but 5mg and 10mg exists, then add np.nan to P510['placebo']
                        if 'placebo' not in P510:
                            P510['placebo'] = np.nan
                        elif '5mg' not in P510:
                            P510['5mg'] = np.nan
                        elif '10mg' not in P510:
                            P510['10mg'] = np.nan

                    run_labels += the_space[participant_list.index(participant)]
                    #loop over conditions to plot the points with colors corresponding to the condition
                    for j, condition in enumerate(['placebo', '5mg', '10mg']):
                        ax4[x,y].plot(run_labels[j], P510[condition], alpha=0.8, marker='s', color=['blue', 'orange', 'green'][j], markeredgecolor='black', markeredgewidth=0.4)

                    #Add the links between the points
                    ax4[x,y].plot(run_labels, [P510['placebo'], P510['5mg'], P510['10mg']], alpha=0.4, color='black')
                    
                    run_labels = np.array([0.0, 1.0, 2.0])

                    #Add participant number to the plot
                    ax4[x,y].text(run_labels[2], P510['10mg']+0.05, participant, fontsize=8)

                ax4[x,y].set_xticks([0, 1, 2])
                ax4[x,y].set_xticklabels(['placebo', '5mg', '10mg'])
            """ 
            # Remove the last subplot
            fig4.delaxes(ax4[1, 5])
            # Set a text box on the upper right listing the accronyms of the 11D categories
            textstr2 = '\n'.join((f"11D categories:",
                                f"EU: Experience of unity",
                                f"SE: Spiritual experience",
                                f"BS: Blissful state",
                                f"IS: Insightfulness",
                                f"DE: Disembodiment",
                                f"ICC: Impaired control and cognition",
                                f"ANX: Anxiety",
                                f"CI: Complex imagery",
                                f"EI: Elementary imagery",
                                f"AVS: Audio-visual synesthesia",
                                f"CMP: Changed meaning of percepts"))
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            ax4[1, 4].text(2.9, 1, textstr2, fontsize=10, bbox=props) """

            #the last subplot ax4[1, 4] with the total score i.e. average of all dimensions for each Subject and the group
            group_average_ASC_placebo_total = np.mean([group_average_ASC['placebo']['cat_11D']['mean'][cat] for cat in ['IS', 'BS', 'SE', 'EU', 'CMP', 'AVS', 'EI', 'CI', 'ANX', 'ICC', 'DE']])
            group_average_ASC_5mg_total = np.mean([group_average_ASC['5mg']['cat_11D']['mean'][cat] for cat in ['IS', 'BS', 'SE', 'EU', 'CMP', 'AVS', 'EI', 'CI', 'ANX', 'ICC', 'DE']])
            group_average_ASC_10mg_total = np.mean([group_average_ASC['10mg']['cat_11D']['mean'][cat] for cat in ['IS', 'BS', 'SE', 'EU', 'CMP', 'AVS', 'EI', 'CI', 'ANX', 'ICC', 'DE']])
            ax4[1,5].bar(run_labels, [group_average_ASC_placebo_total, group_average_ASC_5mg_total, group_average_ASC_10mg_total], color=['blue', 'orange', 'green'])
            ax4[1,5].set_title('Total score')
            ax4[1,5].set_ylim([0, 10])
            ax4[1,5].set_ylabel('Score')

            group_results_ASC_participants_placebo_total = []
            group_results_ASC_participants_5mg_total = []
            group_results_ASC_participants_10mg_total = []

            group_results_ASC_participants_placebo_std = []
            group_results_ASC_participants_5mg_std = []
            group_results_ASC_participants_10mg_std = []

            for participant, sessions in group_results_ASC.items():
                P510 = {}
                S510 = {}
                for session, res in sessions.items() :
                    #we want, in order, placebo, 5mg, 10mg
                    if res['type'] == 'placebo':
                        P510['placebo'] = np.mean([res['cat_11D']['mean'][cat] for cat in ['IS', 'BS', 'SE', 'EU', 'CMP', 'AVS', 'EI', 'CI', 'ANX', 'ICC', 'DE']])
                        S510['placebo'] = np.std([res['cat_11D']['std_dev'][cat] for cat in ['IS', 'BS', 'SE', 'EU', 'CMP', 'AVS', 'EI', 'CI', 'ANX', 'ICC', 'DE']])            
                    elif res['type'] == '5mg':
                        P510['5mg'] = np.mean([res['cat_11D']['mean'][cat] for cat in ['IS', 'BS', 'SE', 'EU', 'CMP', 'AVS', 'EI', 'CI', 'ANX', 'ICC', 'DE']])
                        S510['5mg'] = np.std([res['cat_11D']['std_dev'][cat] for cat in ['IS', 'BS', 'SE', 'EU', 'CMP', 'AVS', 'EI', 'CI', 'ANX', 'ICC', 'DE']])
                    elif res['type'] == '10mg':
                        P510['10mg'] = np.mean([res['cat_11D']['mean'][cat] for cat in ['IS', 'BS', 'SE', 'EU', 'CMP', 'AVS', 'EI', 'CI', 'ANX', 'ICC', 'DE']])
                        S510['10mg'] = np.std([res['cat_11D']['std_dev'][cat] for cat in ['IS', 'BS', 'SE', 'EU', 'CMP', 'AVS', 'EI', 'CI', 'ANX', 'ICC', 'DE']])
                    #if a type is missing but not the two others, then add np.nan to P510. FOr example, if type placebo doesn't exist but 5mg and 10mg exists, then add np.nan to P510['placebo']
                    if 'placebo' not in P510:
                        P510['placebo'] = np.nan
                        S510['placebo'] = np.nan
                    elif '5mg' not in P510:
                        P510['5mg'] = np.nan
                        S510['5mg'] = np.nan
                    elif '10mg' not in P510:
                        P510['10mg'] = np.nan
                        S510['10mg'] = np.nan

                group_results_ASC_participants_placebo_total.append(P510['placebo'])
                group_results_ASC_participants_5mg_total.append(P510['5mg'])
                group_results_ASC_participants_10mg_total.append(P510['10mg'])

                group_results_ASC_participants_placebo_std.append(S510['placebo'])
                group_results_ASC_participants_5mg_std.append(S510['5mg'])
                group_results_ASC_participants_10mg_std.append(S510['10mg'])

            for i, participant in enumerate(participant_list):
                run_labels += the_space[participant_list.index(participant)]

                #loop over conditions to plot the points with colors corresponding to the condition
                for j, condition in enumerate(['placebo', '5mg', '10mg']):
                    ax4[1,5].plot(run_labels[j], [group_results_ASC_participants_placebo_total[i], group_results_ASC_participants_5mg_total[i], group_results_ASC_participants_10mg_total[i]][j], alpha=0.8, marker='s', color=['blue', 'orange', 'green'][j], markeredgecolor='black', markeredgewidth=0.4)

                #Add the links between the points
                ax4[1,5].plot(run_labels, [group_results_ASC_participants_placebo_total[i], group_results_ASC_participants_5mg_total[i], group_results_ASC_participants_10mg_total[i]], alpha=0.4, color='black')

                #standard deviation
                #ax4[1,5].errorbar(run_labels, [group_results_ASC_participants_placebo_total[i], group_results_ASC_participants_5mg_total[i], group_results_ASC_participants_10mg_total[i]], yerr=[group_results_ASC_participants_placebo_std[i], group_results_ASC_participants_5mg_std[i], group_results_ASC_participants_10mg_std[i]], color='black', linewidth=0.4)

                #Add participant number to the plot
                ax4[1,5].text(run_labels[2], group_results_ASC_participants_10mg_total[i]+0.05, participant, fontsize=8)
                run_labels = np.array([0.0, 1.0, 2.0])

            ax4[1,5].set_xticks([0, 1, 2])
            ax4[1,5].set_xticklabels(['placebo', '5mg', '10mg'])

            fig4.subplots_adjust(top=0.9, right=1.2, hspace=0.3)
            fig4.suptitle('Cat_11D Scores')

            fig4.savefig(opj(out_path,"All_results_ASC_cat_11D.pdf"), dpi=600, bbox_inches='tight', transparent=True)

        def SEgroupfit(self, out_path):
            print('Fitting SE group results...')

            SE_data = {}
            groups_dict = {'placebo':['sub-001_ses-1','sub-002_ses-2','sub-003_ses-3','sub-004_ses-2','sub-005_ses-1','sub-006_ses-3','sub-007_ses-3','sub-008_ses-1',
                         'sub-009_ses-2','sub-010_ses-1','sub-011_ses-3','sub-012_ses-2','sub-013_ses-3','sub-014_ses-2','sub-015_ses-2','sub-018_ses-1','sub-019_ses-2','sub-020_ses-1'],

              '5mg':['sub-001_ses-3','sub-002_ses-1','sub-003_ses-1','sub-004_ses-3','sub-005_ses-2','sub-006_ses-1','sub-007_ses-1','sub-008_ses-3',
                    'sub-009_ses-1','sub-010_ses-2','sub-011_ses-2','sub-012_ses-3','sub-013_ses-2','sub-014_ses-1','sub-015_ses-3','sub-018_ses-3','sub-019_ses-1','sub-020_ses-2'],

              '10mg':['sub-001_ses-2','sub-002_ses-3','sub-003_ses-2','sub-004_ses-1','sub-005_ses-3','sub-006_ses-2','sub-007_ses-2','sub-008_ses-2',
                     'sub-009_ses-3','sub-010_ses-3','sub-011_ses-1','sub-012_ses-1','sub-013_ses-1','sub-014_ses-3','sub-015_ses-1','sub-018_ses-2','sub-019_ses-3','sub-020_ses-3']}

            for su, subject in enumerate(getExpAttr(self)):
                this_subject = getattr(self,subject)
                
                subject_name = rp(subject).split('_')[0]
                #print(subject_name)
                SE_data.update({subject_name:{}})

                for se,session in enumerate(getExpAttr(this_subject)):
                    this_session = getattr(this_subject,session)
                    this_session_dict = {}

                    for rr,run in enumerate(getExpAttr(this_session)):
                        this_run = getattr(this_session, run)

                        #if this_run doesn't exist print error
                        if not this_run:
                            print(f"ERROR: {run} doesn't exist")

                        if this_run.SE_question_dict is None:
                            print(f"no run result for {this_run.expsettings['sub']} {this_run.expsettings['ses']} {this_run.expsettings['run']}, you should run fit_all and plot_all first")
                        else :
                            SE_question_dict = this_run.SE_question_dict  
                            time = this_run.time                 
                            
                        session_id = rp(this_run.expsettings['ses'])
                        run_id = rp(this_run.expsettings['run'])

                        this_session_dict.update({f"{run_id}":SE_question_dict})
                        this_session_dict.update({f"{run_id}_time":time})

                    #if f"{participant_id}_{session_id}" is in groups_dict['placebo'] then type_id = 'placebo' etc
                    for group in groups_dict:
                        if f"{subject_name}_{session_id}" in groups_dict[group]:
                            type_id = group
                            break

                    this_session_dict.update({'type':type_id})

                    SE_data[subject_name].update({f"{session_id}":this_session_dict})

            #print(SE_data.keys())
            np.save(opj(out_path,'data/group_results_SE.npy'),SE_data)

            SE_average = {}
            # calculate the four means of run_1, 2, 3 and 4 for each participant and each session
            for participant, sessions in SE_data.items():
                SE_average.update({participant:{}})

                for session, data in sessions.items():
                    SE_average[participant].update({session:{}})

                    print(data)
                    #print(data['run_1'])
                    #print(data['run_1'].values())
                    #print(np.mean(data['run_1'].values()))
                    #print(np.mean(data['run_2'].values()))
                    #print(np.mean(data['run_3'].values()))
                    #print(np.mean(data['run_4'].values()))
                    #print(data['type'])
                    #if there is run-1, 2, 3 and 4, then calculate the mean of the 4 runs, if not, check which run is there and update only the mean of the runs that are there
                    if 'run-1' in data:
                        SE_average[participant][session].update({'run-1':np.mean(list(data['run-1'].values()))})
                        SE_average[participant][session].update({'run-1_std':np.std(list(data['run-1'].values()))})
                        SE_average[participant][session].update({'run-1_se':np.std(list(data['run-1'].values()))/np.sqrt(len(list(data['run-1'].values())))})
                        SE_average[participant][session].update({'run-1_time':data['run-1_time']})
                    if 'run-2' in data:
                        SE_average[participant][session].update({'run-2':np.mean(list(data['run-2'].values()))})
                        SE_average[participant][session].update({'run-2_std':np.std(list(data['run-2'].values()))})
                        SE_average[participant][session].update({'run-2_se':np.std(list(data['run-2'].values()))/np.sqrt(len(list(data['run-2'].values())))})
                        SE_average[participant][session].update({'run-2_time':data['run-2_time']})
                    if 'run-3' in data:
                        SE_average[participant][session].update({'run-3':np.mean(list(data['run-3'].values()))})
                        SE_average[participant][session].update({'run-3_std':np.std(list(data['run-3'].values()))})
                        SE_average[participant][session].update({'run-3_se':np.std(list(data['run-3'].values()))/np.sqrt(len(list(data['run-3'].values())))})
                        SE_average[participant][session].update({'run-3_time':data['run-3_time']})
                    if 'run-4' in data:
                        SE_average[participant][session].update({'run-4':np.mean(list(data['run-4'].values()))})
                        SE_average[participant][session].update({'run-4_std':np.std(list(data['run-4'].values()))})
                        SE_average[participant][session].update({'run-4_se':np.std(list(data['run-4'].values()))/np.sqrt(len(list(data['run-4'].values())))})
                        SE_average[participant][session].update({'run-4_time':data['run-4_time']})
                    SE_average[participant][session].update({'type':data['type']})

            print('SE_average')
            print(SE_average)
            #save 
            np.save(opj(out_path,'data/group_results_SE_average.npy'),SE_average)
            SE_average = dict(sorted(SE_average.items()))

            #We have SE_average which gives us the four averages of the fours runs of the different sessions of each participant
            #We want the global average over all participants for each group, so we need to average the four averages of each participant for each group
            global_SE_average = {}

            # Iterate over each group
            for group in groups_dict:
                #HERE CODE TO AVERAGE THE FOUR RUNS OF EACH PARTICIPANT FOR EACH GROUP
                # Initialize the group averages for 'run-1', 'run-2', 'run-3' and 'run-4'
                run_1_avg = 0.0
                run_2_avg = 0.0
                run_3_avg = 0.0
                run_4_avg = 0.0
                run_1_std = []
                run_2_std = []
                run_3_std = []
                run_4_std = []


                # Initialize a counter for the number of participants in this group
                num_participants = 0

                # Iterate over participants and sessions
                for participant, sessions in SE_average.items():
                    for session, data in sessions.items():
                        # Check if the participant is in the current group
                        #print(f"{participant}_{session}")

                        if f"{participant}_{session}" in groups_dict[group]:
                            # Update the number of participants
                            #print(data['run-1'])
                            num_participants += 1
                            if 'run-1' in data:
                                # Update run_1 averages
                                run_1_avg +=data['run-1']
                                run_1_std.append(data['run-1'])
                            if 'run-2' in data:
                                # Update run_2 averages
                                run_2_avg +=data['run-2']
                                run_2_std.append(data['run-2'])
                            if 'run-3' in data:
                                # Update run_3 averages
                                run_3_avg +=data['run-3']
                                run_3_std.append(data['run-3'])
                            if 'run-4' in data:
                                # Update run_4 averages
                                run_4_avg +=data['run-4']
                                run_4_std.append(data['run-4'])

                 # Calculate the mean for 'run-1'
                if run_1_avg is not None:
                    run_1_avg /= num_participants
                    run_1_std_ = np.std(run_1_std)
                # Calculate the mean for 'run-2'
                if run_2_avg is not None:
                    run_2_avg /= num_participants
                    run_2_std_ = np.std(run_2_std)
                # Calculate the mean for 'run-3'
                if run_3_avg is not None:
                    run_3_avg /= num_participants
                    run_3_std_ = np.std(run_3_std)
                # Calculate the mean for 'run-4'
                if run_4_avg is not None:
                    run_4_avg /= num_participants
                    run_4_std_ = np.std(run_4_std)

                # Update the global_average dictionary
                global_SE_average[group] = {'run-1': run_1_avg, 'run-2': run_2_avg, 'run-3': run_3_avg, 'run-4': run_4_avg}
                global_SE_average[group+'_std'] = {'run-1': run_1_std_, 'run-2': run_2_std_, 'run-3': run_3_std_, 'run-4': run_4_std_}
                global_SE_average[group+'_se'] = {'run-1': run_1_std_/np.sqrt(num_participants), 'run-2': run_2_std_/np.sqrt(num_participants), 'run-3': run_3_std_/np.sqrt(num_participants), 'run-4': run_4_std_/np.sqrt(num_participants)}
         
            # Print the global averages
            print('global_SE_average')
            print(global_SE_average)
            #save
            np.save(opj(out_path,'data/group_results_SE_global_average.npy'),global_SE_average)

            print('Fitting SE group results done')

        def SEgroupplot(self, out_path):
            #load the data
            SE_average = np.load(opj(out_path,'data/group_results_SE_average.npy'),allow_pickle='TRUE').item()
            global_SE_average = np.load(opj(out_path,'data/group_results_SE_global_average.npy'),allow_pickle='TRUE').item()

            #We have SE_average and global_SE_average, plot the 4 average scores of the runs in a bar plot, with the average of the group in a thin line
            # Initialize the figure
            fig, ax = pl.subplots(3, 1, figsize=(16, 32))
            fig.suptitle(f"SE of all participants")
            # for each condition select the session that have type=condition
            # Iterate over each participant
            #linspace 18 between -0.25 and 0.25
            the_space = np.linspace(-0.25, 0.25, (SE_average.items().__len__()))
            participant_list = list(SE_average.keys())
            #run_1_time list
            run_1_time = []
            run_2_time = []
            run_3_time = []
            run_4_time = []
            for participant, sessions in SE_average.items():
                # for each condition select the session that have type=condition
                for i, condition in enumerate(['placebo', '5mg', '10mg']):
                    # for the session with type = condition, plot the data
                    
                    for j, session in enumerate(['ses-1', 'ses-2', 'ses-3']):
                        #check if the session exists
                        if session in sessions:
                            #check if the session is of the current condition
                            if sessions[session]['type'] == condition:
                                #Extract the participant information and scores for the current condition
                                if 'run-1' in sessions[session]:
                                    run_1_score = sessions[session]['run-1']
                                    run_1_std = sessions[session]['run-1_std']/2
                                    run_1_time.append(sessions[session]['run-1_time'])
                                
                                else : 
                                    run_1_score = np.nan
                                    run_1_std = np.nan
                                                           
                                if 'run-2' in sessions[session]:
                                    run_2_score = sessions[session]['run-2']
                                    run_2_std = sessions[session]['run-2_std']/2
                                    run_2_time.append(sessions[session]['run-2_time'])
                                else : 
                                    run_2_score = np.nan
                                    run_2_std = np.nan
                                    
                                if 'run-3' in sessions[session]:
                                    run_3_score = sessions[session]['run-3']
                                    run_3_std = sessions[session]['run-3_std']/2
                                    run_3_time.append(sessions[session]['run-3_time'])
                                else : 
                                    run_3_score = np.nan
                                    run_3_std = np.nan
                                    
                                if 'run-4' in sessions[session]:
                                    run_4_score = sessions[session]['run-4']
                                    run_4_std = sessions[session]['run-4_std']/2
                                    run_4_time.append(sessions[session]['run-4_time'])
                                else : 
                                    run_4_score = np.nan
                                    run_4_std = np.nan
                                    
                               
                                # Convert scores to lists for plotting the 4 runs
                                run_scores = [run_1_score, run_2_score, run_3_score, run_4_score]
                                 # List of category labels for plotting the 4 runs
                                #run_labels = ['run-1', 'run-2', 'run-3', 'run-4']
                                run_labels = np.array([0.0, 1.0, 2.0, 3.0])
                                
                                run_labels += the_space[participant_list.index(participant)]
                                
                                # Create plots for the 4 runs on the corresponding row for each participant which can then be used for the standard deviation
                                ax[i].plot(run_labels, run_scores, label=participant, marker='s', alpha=0.4, color='black', linewidth=1)
                                #add the standard deviation of the 4 runs and make it the same color as the line
                                ax[i].errorbar(run_labels, run_scores, yerr=[run_1_std, run_2_std, run_3_std, run_4_std], fmt='none', capsize=5, elinewidth=2, markeredgewidth=2, alpha=0.4, color='black')
                                    
                    # Add time zero to x-axis
                    ax[i].axvline(x=-1.0, color='white', linestyle='--', linewidth=0.1)
                    #calculate the average time for each run accross all participants
                    ax[i].set_ylim(0, 10)
                    ax[i].set_ylabel('Score')
                    ax[i].set_title(f'{condition}')


            run_1_time_avg = PPviz.average_time(run_1_time)
            run_0_time_avg = PPviz.subtract_45min(run_1_time_avg)
            run_2_time_avg = PPviz.average_time(run_2_time)
            run_3_time_avg = PPviz.average_time(run_3_time)
            run_4_time_avg = PPviz.average_time(run_4_time)

            run_1_time_avg = PPviz.giveDeltaTime(run_0_time_avg, run_1_time_avg)
            run_2_time_avg = PPviz.giveDeltaTime(run_0_time_avg, run_2_time_avg)
            run_3_time_avg = PPviz.giveDeltaTime(run_0_time_avg, run_3_time_avg)
            run_4_time_avg = PPviz.giveDeltaTime(run_0_time_avg, run_4_time_avg)

            #round the time 30min
            run_2_time_avg = PPviz.roundTime(run_2_time_avg)
            run_3_time_avg = PPviz.roundTime(run_3_time_avg)
            run_4_time_avg = PPviz.roundTime(run_4_time_avg)

            #add these to global_SE_average
            global_SE_average['run'] = {'run-1': run_1_time_avg, 'run-2': run_2_time_avg, 'run-3': run_3_time_avg, 'run-4': run_4_time_avg}

            #save the global_SE_average
            np.save(opj(out_path,'data/group_results_SE_global_average.npy'),global_SE_average)

            ax[0].set_xticks([-1.0, 0.0, 1.0, 2.0, 3.0],['00:00:00', run_1_time_avg, run_2_time_avg, run_3_time_avg, run_4_time_avg])
            ax[1].set_xticks([-1.0, 0.0, 1.0, 2.0, 3.0],['00:00:00', run_1_time_avg, run_2_time_avg, run_3_time_avg, run_4_time_avg])
            ax[2].set_xticks([-1.0, 0.0, 1.0, 2.0, 3.0],['00:00:00', run_1_time_avg, run_2_time_avg, run_3_time_avg, run_4_time_avg]) 

            # Add the global averages to the plots with a closed loop
            ax[0].bar(global_SE_average['placebo'].keys(), global_SE_average['placebo'].values(), label='group average', color='black', alpha=0.7, linewidth=0.5)
            ax[1].bar(global_SE_average['5mg'].keys(), global_SE_average['5mg'].values(), label='group average', color='black',alpha=0.7,linewidth=0.5)
            ax[2].bar(global_SE_average['10mg'].keys(), global_SE_average['10mg'].values(), label='group average', color='black',alpha=0.7,linewidth=3)
            #add legend for the global average bar
            #ax[2].legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

            # Save the figure
            fig.savefig(opj(out_path,"All_results_SE.pdf"), dpi=600, bbox_inches='tight', transparent=True)
            pl.close(fig)
            print("plotting all participants SE results done")
                        
        def SEaverageplot(self, out_path):
            
            #load the SE data
            #color_dict = {'placebo':(0.0,0.15,1) ,'5mg':(0.8,0.05,0.8),'10mg':(1,0.0,0.0)}
            # color_dict = {'placebo':(0,0.1,1),'5mg':(0,0.6,0),'10mg':(1,0.7,0)}
            color_dict =  {'placebo':(0,0.6497,0.7176) ,'5mg':(0.4287*1.8,0.0,0.6130*1.5),'10mg':(0.864*1.1,0.0,0.0)}
            SE_data= np.load(opj(out_path,'data/group_results_SE.npy'), allow_pickle='TRUE').item()
            SE_average = np.load(opj(out_path,'data/group_results_SE_average.npy'),allow_pickle='TRUE').item()
            global_SE_average = np.load(opj(out_path,'data/group_results_SE_global_average.npy'),allow_pickle='TRUE').item()
            
            run_labels = np.array([0.0, 1.0, 2.0, 3.0])
            run_labels2 = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])

            # run_labels3 with the space between each run proportional to the time between each run, 
            # for example if run-1 is at 00:00:00 and run-2 is at 00:30:00, then the space between run-1 and run-2 is 0.5 and if run-3 is at 03:00:00 then the space between run-2 and run-3 is 2.5
            times = global_SE_average['run'].values()
            times = list(times)
            times.insert(0, '00:00:00')

            run_labels3 = np.array([PPviz.calculateTimeDistance(times[0], times[0]), PPviz.calculateTimeDistance(times[0], times[1]), PPviz.calculateTimeDistance(times[0], times[2]), PPviz.calculateTimeDistance(times[0], times[3]), PPviz.calculateTimeDistance(times[0], times[4])])
            # print(run_labels3)

            #We want to plot the 4 average scores of the runs in a plot, with the average of the group conditions on the same graph with 'placebo' in blue, '5mg' in orange and '10mg' in green
            # Initialize the figure
            fig, ax = pl.subplots(1, 1, figsize=(16, 8))
            fig.suptitle(f"SE of all participants")

            # Add time zero to x-axis
            ax.axvline(x=-1.0, color='white', linestyle='--', linewidth=0.1)

            # for each condition select the session that have type=condition
            # Add the global averages to the plots with a closed loop
            v = list(global_SE_average['placebo'].values())
            v_5mg = list(global_SE_average['5mg'].values())
            v_10mg = list(global_SE_average['10mg'].values())

            ax.plot(run_labels3[1:], global_SE_average['placebo'].values(), label='placebo', color=color_dict['placebo'],linewidth=3)
            ax.plot(run_labels3[1:], global_SE_average['5mg'].values(), label='5mg', color=color_dict['5mg'],linewidth=3)
            ax.plot(run_labels3[1:], global_SE_average['10mg'].values(), label='10mg', color=color_dict['10mg'],linewidth=3)

            v_se = list(global_SE_average['placebo_se'].values())
            v_5mg_se = list(global_SE_average['5mg_se'].values())
            v_10mg_se = list(global_SE_average['10mg_se'].values())
            #add the standard deviation with fill between for the global average
            ax.fill_between(run_labels3[1:], np.array(v)-np.array(v_se), np.array(v)+np.array(v_se), alpha=0.2, color=color_dict['placebo'], label='placebo SE')
            ax.fill_between(run_labels3[1:], np.array(v_5mg)-np.array(v_5mg_se), np.array(v_5mg)+np.array(v_5mg_se), alpha=0.2, color=color_dict['5mg'], label='5mg SE')
            ax.fill_between(run_labels3[1:], np.array(v_10mg)-np.array(v_10mg_se), np.array(v_10mg)+np.array(v_10mg_se), alpha=0.2, color=color_dict['10mg'], label='10mg SE')

            
            #For each participant, plot the runs
            # Iterate over each participant
            #linspace 18 between -0.25 and 0.25
            the_space = np.linspace(-0.05, 0.05, (SE_average.items().__len__()))
            participant_list = list(SE_average.keys())

            # for participant, sessions in SE_average.items():
            #     for session, data in sessions.items():
            #         run_labels3 += the_space[participant_list.index(participant)]
                    
            #         if 'run-1' in data:
            #             ax.plot(run_labels3[1], float(data['run-1']), marker='s', alpha=0.4, color=color_dict[data['type']],linestyle="None")
            #         if 'run-2' in data:
            #             ax.plot(run_labels3[2], float(data['run-2']), marker='s', alpha=0.4, color=color_dict[data['type']],linestyle="None")
            #         if 'run-3' in data:
            #             ax.plot(run_labels3[3], float(data['run-3']), marker='s', alpha=0.4, color=color_dict[data['type']],linestyle="None")
            #         if 'run-4' in data:
            #             ax.plot(run_labels3[4], float(data['run-4']), marker='s', alpha=0.4, color=color_dict[data['type']],linestyle="None")


            #         run_labels3 = np.array([PPviz.calculateTimeDistance(times[0], times[0]), PPviz.calculateTimeDistance(times[0], times[1]), PPviz.calculateTimeDistance(times[0], times[2]), PPviz.calculateTimeDistance(times[0], times[3]), PPviz.calculateTimeDistance(times[0], times[4])])
            
            ax.set_xticks(run_labels3,['00:00', f"+{global_SE_average['run']['run-1'][:-3]}", f"+{global_SE_average['run']['run-2'][:-3]}", f"+{global_SE_average['run']['run-3'][:-3]}", f"+{global_SE_average['run']['run-4'][:-3]}"])
            #reduce the size of the x labels
            ax.tick_params(axis='x', labelsize=14)
            ax.set_xlim(-0.4, 11.4)
            #add legend for the global average bar
            ax.set_ylabel('Mean score')
            ax.set_xlabel('Time (HH:MM)')
            # ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

            # Save the figure
            fig.savefig(opj(out_path,"Average_results_SE_color3.pdf"), dpi=600, bbox_inches='tight', transparent=True)
            pl.close(fig)
            print("plotting all participants SE results done")


    class Subject():
        def __init__(self):
            pass
    class Session():
        def __init__(self):
            pass
    class Run():
        def __init__(self):
            pass

        def CDfit(self):

            self.probs =  dict()
            self.fits = dict()
            self.preds = dict()

            if 'Training' in self.expsettings:
                if self.expsettings['Training']:
                    self.df_responses = self.df_responses[self.df_responses['trial_nr']>self.expsettings['nr_training_trials']]

            self.sf_values = np.sort(self.df_responses['spatial_frequency_cycles'].unique())
            self.contrast_values = np.sort(self.df_responses['target_contrast'].unique())
            self.x_space = np.linspace(self.contrast_values.min(),self.contrast_values.max(),100)


            for sf_value in self.sf_values:
                current_probs = []
                for ctr in self.contrast_values:
                    
                    current_responses = self.df_responses[((self.df_responses['target_contrast'] == ctr) & (self.df_responses['spatial_frequency_cycles'] == sf_value))]
                    
                    current_probs.append(np.mean((([int(el) for el in current_responses['key_pressed']])-current_responses['target_position'])==1))


                self.probs[str(sf_value)] = np.array(current_probs)
                self.fits[str(sf_value)], self.preds[str(sf_value)] = PPviz.fit_shifted_sigmoid_curve(self.contrast_values,self.probs[str(sf_value)])

        def CDplot(self):

            self.fig,self.ax = pl.subplots(1,len(self.sf_values),figsize=(8*len(self.sf_values),8))
            self.fig.suptitle(f"{rp(self.expsettings['task'])} {rp(self.expsettings['sub'])} {rp(self.expsettings['ses'])} {rp(self.expsettings['run'])} ({self.df_responses.shape[0]} trials)")

            for ii, sf_value in enumerate(self.sf_values):

                self.ax[ii].set_xscale('log')
                self.ax[ii].set_xlabel('Target contrast')
                self.ax[ii].set_ylabel('Proportion correct responses')

                
                self.ax[ii].plot(self.x_space,np.ones_like(self.x_space)*0.25, ls='--', label = 'Chance', c='k', alpha=0.5)
                self.ax[ii].plot(self.x_space,np.ones_like(self.x_space), ls='--', label = 'Ceiling', c='green', alpha=0.5)

                self.ax[ii].plot(self.x_space,self.preds[str(sf_value)], c='red', lw=2, label = 'Fit')

                self.ax[ii].plot(self.contrast_values,self.probs[str(sf_value)], marker='s', ls='', c='k', ms=6, label = 'Trials')

                self.ax[ii].set_title(f"Sf {sf_value:.1f} c/deg")

            self.fig.savefig(opj(self.out_path,f"{rp(self.expsettings['sub'])}_{rp(self.expsettings['ses'])}_{rp(self.expsettings['run'])}_{rp(self.expsettings['task'])}_results.pdf"), dpi=600, bbox_inches='tight', transparent=True)

        def CSfit(self):

            self.probs = []
            self.probs_nosurr = []
            #total number of trials for each contrast level and the count of correct responses within those trials
            self.nij = []
            self.rij = []
            self.nij_nosurr = []
            self.rij_nosurr = []

            if 'Training' in self.expsettings:
                if self.expsettings['Training']:
                    self.df_responses = self.df_responses[self.df_responses['trial_nr']>self.expsettings['nr_training_trials']]

            #contrast values with surround and without surround
            self.contrast_values = np.sort(self.df_responses['reference_contrast'][self.df_responses['target_surround_presence']==True].unique())
            self.contrast_values_nosurr = np.sort(self.df_responses['reference_contrast'][self.df_responses['target_surround_presence']==False].unique())
            

            #x_space is the x axis for the fitted curve
            self.x_space = np.linspace(self.contrast_values.min(),self.contrast_values.max(),100)
            self.x_space_nosurr = np.linspace(self.contrast_values_nosurr.min(),self.contrast_values_nosurr.max(),100)

            #x_space_comb is the x axis for the plot
            self.x_space_comb = np.linspace(np.min([self.x_space,self.x_space_nosurr]),np.max([self.x_space,self.x_space_nosurr]),100)

            for ctr in self.contrast_values:
                # for each contrast value, select the trials with that contrast value and with surround
                curr_ctr = self.df_responses[(np.isclose(self.df_responses['reference_contrast'],ctr,atol=1e-4)) & (self.df_responses['target_surround_presence'] == True)]
                # calculate the proportion of correct responses for that contrast value by comparing the key pressed to the target position
                #for some arcane reason sometimes key pressed is a number, sometimes a string
                
                #Sub-009 doesn't have inverted responses but just a different threshold than other participants
                #if self.expsettings['sub'] == 'sub_009':
                        #print('warning: correcting inverted response by sub-009 on all sessions of centred surround')
                        #print(curr_ctr['key_pressed'])
                        #self.probs.append(np.mean(((curr_ctr['key_pressed'] == '2')|(curr_ctr['key_pressed'] == 2.0)|(np.array(curr_ctr['key_pressed']).astype(int) == 2)) ^ (curr_ctr['presentation_order'] ==  'ref first')))
                self.probs.append(np.mean(((curr_ctr['key_pressed'] == '2')|(curr_ctr['key_pressed'] == 2.0)|(np.array(curr_ctr['key_pressed']).astype(int) == 2)) ^ (curr_ctr['presentation_order'] ==  'ref first')))
                
                #total number of trials for each contrast level 
                self.nij.append(curr_ctr.shape[0])

                #count of correct responses within those trials
                self.rij.append(np.sum(((curr_ctr['key_pressed'] == '2')|(curr_ctr['key_pressed'] == 2.0)|(np.array(curr_ctr['key_pressed']).astype(int) == 2)) ^ (curr_ctr['presentation_order'] ==  'ref first')))
                print('Here first try to print :', self.nij)
                print('Here first try to print :', self.rij)
                # Count the total number of trials (nij)
                total_trials = len(curr_ctr)
                print('Here second try to print :', total_trials)

                # Count of correct responses (rij)
                correct_responses = np.sum(((curr_ctr['key_pressed'] == '2') | (curr_ctr['key_pressed'] == 2.0) | (np.array(curr_ctr['key_pressed']).astype(int) == 2)) ^ (curr_ctr['presentation_order'] == 'ref first'))
                print('Here second try to print :', correct_responses)


            for ctr in self.contrast_values_nosurr:
                curr_ctr_nosurr = self.df_responses[(np.isclose(self.df_responses['reference_contrast'],ctr,atol=1e-4)) & (self.df_responses['target_surround_presence'] == False)]
                self.probs_nosurr.append(np.mean(((curr_ctr_nosurr['key_pressed'] == '2')|(curr_ctr_nosurr['key_pressed'] == 2.0)|(np.array(curr_ctr_nosurr['key_pressed']).astype(int) == 2)) ^ (curr_ctr_nosurr['presentation_order'] ==  'ref first')))
                self.nij_nosurr.append(curr_ctr_nosurr.shape[0])
                self.rij_nosurr.append(np.sum(((curr_ctr_nosurr['key_pressed'] == '2')|(curr_ctr_nosurr['key_pressed'] == 2.0)|(np.array(curr_ctr_nosurr['key_pressed']).astype(int) == 2)) ^ (curr_ctr_nosurr['presentation_order'] ==  'ref first')))

            self.nij = np.array(self.nij)
            self.rij = np.array(self.rij)
            self.probs = np.array(self.probs)
            self.probs_nosurr = np.array(self.probs_nosurr)

            self.full_fit, self.full_pred, self.r2 = PPviz.fit_sigmoid_curve(self.contrast_values,self.probs)
            self.full_fit_nosurr, self.full_pred_nosurr, self.r2_nosurr = PPviz.fit_sigmoid_curve(self.contrast_values_nosurr,self.probs_nosurr)

            subject_name = rp(self.expsettings['sub']).split('_')[0]
            session_name = rp(self.expsettings['ses']).split('_')[0]

            # print(f"subject: {subject_name}, session: {session_name}")

            # Find the x-coordinate where the sigmoid curve intersects the 0.5 horizontal axis
            x_index = np.argmin(np.abs(self.full_pred - 0.5))
            if x_index > 0:
                h = self.x_space[x_index + 1] - self.x_space[x_index]
                slope = (self.full_pred[x_index + 1] - self.full_pred[x_index - 1]) / (2 * h)
                self.slope = slope
            else:
                self.slope = (self.full_pred[x_index + 1] - self.full_pred[x_index]) / (self.x_space[x_index + 1] - self.x_space[x_index])
            
            # Find the x-coordinate where the sigmoid curve intersects the 0.5 horizontal axis
            x_index = np.argmin(np.abs(self.full_pred_nosurr - 0.5))
            if x_index > 0:
                h = self.x_space_nosurr[x_index + 1] - self.x_space_nosurr[x_index]
                slope_nosurr = (self.full_pred_nosurr[x_index + 1] - self.full_pred_nosurr[x_index - 1]) / (2 * h)
                self.slope_nosurr = slope_nosurr
            else:
                self.slope_nosurr = (self.full_pred_nosurr[x_index + 1] - self.full_pred_nosurr[x_index]) / (self.x_space_nosurr[x_index + 1] - self.x_space_nosurr[x_index])
                    
            #calculate the relative effect size
            self.rel_effsize = 100*(self.x_space[np.argmin(np.abs(self.full_pred-0.5))] - self.expsettings['Stimulus Settings']['Target contrast'])/self.expsettings['Stimulus Settings']['Target contrast']

            self.rel_effsize_nosurr = 100*(self.x_space_nosurr[np.argmin(np.abs(self.full_pred_nosurr-0.5))] - self.expsettings['Stimulus Settings']['Target contrast'])/self.expsettings['Stimulus Settings']['Target contrast']

        def CSplot(self):
            
            self.fig,self.ax = pl.subplots(1,2,figsize=(16, 8))
            self.fig.suptitle(f"{rp(self.expsettings['task'])} {rp(self.expsettings['sub'])} {rp(self.expsettings['ses'])} {rp(self.expsettings['run'])} ({self.df_responses.shape[0]} trials)")

            for a in self.ax:
                a.set_xlabel('Contrast difference (%RMS)')
                  
                a.plot(self.x_space_comb,np.ones_like(self.x_space_comb)*0.5, ls='--', c='k', alpha=0.5)
                a.plot(self.x_space_comb,np.ones_like(self.x_space_comb), ls='--', c='green', alpha=0.5)
                a.plot(self.x_space_comb,np.zeros_like(self.x_space_comb), ls='--', c='red', alpha=0.5)
                a.plot(self.expsettings['Stimulus Settings']['Target contrast']*np.ones(100),np.linspace(0,1,100), ls='-', label = 'Veridical', c='k', alpha=0.5)

                a.set_xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7])
                a.set_xticklabels(['-75','-50','-25','0','+25','+50','+75'])

                a.legend(loc='lower right')
                

            self.ax[0].set_title('Surround is present')
            self.ax[1].set_title('Surround is absent')
            self.ax[0].set_ylabel('Prob ref perceived as higher contrast')

            self.ax[0].plot(self.x_space,self.full_pred, c='red', lw=2, label = 'Fit')
            self.ax[0].plot(self.contrast_values,self.probs, marker='s', ls='', c='k', ms=6)
            self.ax[1].plot(self.x_space_nosurr,self.full_pred_nosurr, c='red', lw=2)
            self.ax[1].plot(self.contrast_values_nosurr,self.probs_nosurr, marker='s', ls='', c='k', ms=6)

            self.fig.savefig(opj(self.out_path,f"{rp(self.expsettings['sub'])}_{rp(self.expsettings['ses'])}_{rp(self.expsettings['run'])}_{rp(self.expsettings['task'])}_results.pdf"), dpi=600, bbox_inches='tight', transparent=True)

        def EHDBfit(self):

            self.probs =  dict()
            self.fits = dict()
            self.preds = dict()
            self.rel_effsize = dict()
            self.size_values = dict()
            self.x_spaces = dict()
            self.slope = dict()
            self.r2 = dict()

            if 'Training' in self.expsettings:
                if self.expsettings['Training']:
                    self.df_responses = self.df_responses[self.df_responses['trial_nr']>self.expsettings['nr_training_trials']]

            self.surr_sizes = np.sort(self.df_responses['Surround stim radius'].unique())[::-1]

            for surr_size in self.surr_sizes:

                current_probs = []

                self.size_values[str(surr_size)] = self.df_responses['Reference stim radius'][self.df_responses['Surround stim radius'] == surr_size].unique()
                self.x_spaces[str(surr_size)] = np.linspace(self.size_values[str(surr_size)].min(),self.size_values[str(surr_size)].max(),100)

                for size in self.size_values[str(surr_size)]:
                    current_responses = self.df_responses[((self.df_responses['Reference stim radius'] == size) & (self.df_responses['Surround stim radius'] == surr_size))]

                    if self.expsettings['sub'] == 'sub_002' and self.expsettings['ses'] == 'ses_0':
                        print('warning: correcting inverted response by sub-002 on ses-0 ebbinghaus')
                        current_probs.append(np.mean((current_responses['key_pressed'] == 2.0) ^ (current_responses['Reference stim position_   0'] > 0)))

                    else:
                        current_probs.append(np.mean((current_responses['key_pressed'] == 1.0) ^ (current_responses['Reference stim position_   0'] > 0)))

                self.probs[str(surr_size)] = np.array(current_probs)

                self.fits[str(surr_size)], self.preds[str(surr_size)], self.r2[str(surr_size)] = PPviz.fit_sigmoid_curve(self.size_values[str(surr_size)],self.probs[str(surr_size)])

                self.rel_effsize[str(surr_size)] = 100*(self.x_spaces[str(surr_size)][np.argmin(np.abs(self.preds[str(surr_size)]-0.5))] - self.df_responses['Center stim radius'].max())/self.df_responses['Center stim radius'].max()
            
            self.x_space = np.linspace(np.min([self.x_spaces[key] for key in self.x_spaces]), np.max([self.x_spaces[key] for key in self.x_spaces]), 100)
            
            # Find the x-coordinate where the sigmoid curve intersects the 0.5 horizontal axis
            for surr_size in self.surr_sizes:
                
                x_index = np.argmin(np.abs(self.preds[str(surr_size)] - 0.5))
                if x_index > 0:
                    h = self.x_spaces[str(surr_size)][x_index + 1] - self.x_spaces[str(surr_size)][x_index]
                    self.slope[str(surr_size)] = (self.preds[str(surr_size)][x_index + 1] - self.preds[str(surr_size)][x_index - 1]) / (2 * h)
                else:
                    self.slope[str(surr_size)] = (self.preds[str(surr_size)][x_index + 1] - self.preds[str(surr_size)][x_index]) / (self.x_spaces[str(surr_size)][x_index + 1] - self.x_spaces[str(surr_size)][x_index])
                      
        def EHDBplot(self):
            
            self.fig,self.ax = pl.subplots(1,len(self.surr_sizes),figsize=(8*len(self.surr_sizes),8))
            self.fig.suptitle(f"{rp(self.expsettings['task'])} {rp(self.expsettings['sub'])} {rp(self.expsettings['ses'])} {rp(self.expsettings['run'])}  ({self.df_responses.shape[0]} trials)")
    
            for ii, surr_size in enumerate(self.surr_sizes):
                self.ax[ii].set_xscale('log')
                self.ax[ii].set_xlabel('Size difference (% veridical size)')
                self.ax[ii].set_ylabel('Prob ref perceived as larger')

                true_size = self.df_responses['Center stim radius'].max()

                self.ax[ii].minorticks_off()

                #ugly
                if surr_size>0:
                    self.ax[ii].set_xticks(np.linspace(true_size-0.5*true_size,true_size+0.3*true_size,9))
                    self.ax[ii].set_xticklabels([-50,-40,-30,-20,-10,0,10,20,30])
                else:
                    self.ax[ii].set_xticks(np.linspace(true_size-0.3*true_size,true_size+0.3*true_size,7))
                    self.ax[ii].set_xticklabels([-30,-20,-10,0,10,20,30])  
                                   
                
                
                self.ax[ii].plot(self.x_spaces[str(surr_size)],np.ones_like(self.x_spaces[str(surr_size)])*0.5, ls='--', c='k', alpha=0.5)
                self.ax[ii].plot(self.x_spaces[str(surr_size)],np.ones_like(self.x_spaces[str(surr_size)]), ls='--', c='green', alpha=0.5)
                self.ax[ii].plot(self.x_spaces[str(surr_size)],np.zeros_like(self.x_spaces[str(surr_size)]), ls='--', c='red', alpha=0.5)
                #bit hacky way
                self.ax[ii].plot(true_size*np.ones(100),np.linspace(0,1,100), ls='-', label = 'Veridical', c='k', alpha=0.5)

                self.ax[ii].set_title(f"Surr size {surr_size:.3f}")
                
                self.ax[ii].legend(loc='lower right')

                self.ax[ii].plot(self.x_spaces[str(surr_size)],self.preds[str(surr_size)], c='red', lw=2, label = 'Fit')
                self.ax[ii].plot(self.size_values[str(surr_size)],self.probs[str(surr_size)], marker='s', ls='', c='k', ms=6, label = 'Trials')


            self.fig.savefig(opj(self.out_path,f"{rp(self.expsettings['sub'])}_{rp(self.expsettings['ses'])}_{rp(self.expsettings['run'])}_{rp(self.expsettings['task'])}_results.pdf"), dpi=600, bbox_inches='tight', transparent=True)

        def ASCfit(self):

            if 'Training' in self.expsettings:
                if self.expsettings['Training']:
                    self.df_responses = self.df_responses[self.df_responses['trial_nr']>self.expsettings['nr_training_trials']]

            # Get the order and the answers of the questions
            events = self.df_responses
            order = self.order
            
            # Filter rows where event_type is 'Response' and select 'key_pressed'
            response_scores = events.loc[events['event_type'] == 'Response', 'key_pressed'].tolist()
            
            # Convert 'response_scores' to an array of numbers and not just strings
            # Define mapping for scores
            score_mapping = {'0': 0, '1':1, '2':2, '3':3, '4':4, '5':5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'num_0': 0, 'num_1': 1, 'num_2': 2, 'num_3': 3, 'num_4': 4, 'num_5': 5, 'num_6': 6, 'num_7': 7, 'num_8': 8, 'num_9': 9, 'num_10': 10}
            scores = [score_mapping[key] for key in response_scores]
            
            # Initialize a result array to store sorted scores
            result = np.zeros(94, dtype=int)

            # Populate the result array using the order
            for i, idx in enumerate(order):
                result[idx] = scores[i]

            #---------------------------------
            #RESULTS IN ORDER OF ASC.JSON FILE
            self.ASC_result = result
            #---------------------------------

            # Transform result in run_results with categories

            # Load the question category data
            with open('ASC.json', 'r') as asc_file:
                asc_data = json.load(asc_file)

            # Transform the tab result in a dictionary with, for each element of the tab, the corresponding id and element
            answers = {i+1: element for i, element in enumerate(result)}

            # Create dictionaries to store total scores and counts for each category
            cat_5D_scores = {}
            cat_5D_counts = {}
            cat_11D_scores = {}
            cat_11D_counts = {}
            cat_5D_mean_scores = {}
            cat_11D_mean_scores = {}
                
            #Because we ordered the questions in the order of the ASC.json file, we can just iterate over the questions in the order of the ASC.json file
            for question_id, score in answers.items():
                question_id = int(question_id)
                
                # Find the corresponding category information for the question
                question_info = next(q for q in asc_data if q['ID'] == question_id)
                cat_5D = question_info['cat_5D']
                cat_11D = question_info['cat_11D']
                
                # Update cat_5D scores and counts
                if cat_5D:
                    if cat_5D not in cat_5D_scores:
                        cat_5D_scores[cat_5D] = 0
                        cat_5D_counts[cat_5D] = 0
                    cat_5D_scores[cat_5D] += score
                    cat_5D_counts[cat_5D] += 1
                
                # Update cat_11D scores and counts
                if cat_11D:
                    if cat_11D not in cat_11D_scores:
                        cat_11D_scores[cat_11D] = 0
                        cat_11D_counts[cat_11D] = 0
                    cat_11D_scores[cat_11D] += score
                    cat_11D_counts[cat_11D] += 1

            print(cat_5D_scores)
            # Calculate mean scores for cat_5D and cat_11D
            cat_5D_mean_scores = {cat: cat_5D_scores[cat] / cat_5D_counts[cat] for cat in cat_5D_scores}
            cat_11D_mean_scores = {cat: cat_11D_scores[cat] / cat_11D_counts[cat] for cat in cat_11D_scores}
            
            # Calculate standard deviations for cat_5D and cat_11D
            cat_5D_std_devs = {cat: np.std([answers[q['ID']] for q in asc_data if q['cat_5D'] == cat]) for cat in cat_5D_scores}
            cat_11D_std_devs = {cat: np.std([answers[q['ID']] for q in asc_data if q['cat_11D'] == cat]) for cat in cat_11D_scores}

            #round the means and std_devs to 3 decimals
            cat_5D_mean_scores = {cat: np.round(mean, 3) for cat, mean in cat_5D_mean_scores.items()}
            cat_11D_mean_scores = {cat: np.round(mean, 3) for cat, mean in cat_11D_mean_scores.items()}
            cat_5D_std_devs = {cat: np.round(std_dev, 3) for cat, std_dev in cat_5D_std_devs.items()}
            cat_11D_std_devs = {cat: np.round(std_dev, 3) for cat, std_dev in cat_11D_std_devs.items()}

            result_entry = {
                                'cat_5D': {
                                    'mean': cat_5D_mean_scores,
                                    'std_dev': cat_5D_std_devs
                                },
                                'cat_11D': {
                                    'mean': cat_11D_mean_scores,
                                    'std_dev': cat_11D_std_devs
                                }
                            }
            
            print(result_entry)
            self.run_result = result_entry

        def ASCplot_ses(self):

            print("plotting sessions result...")

            result_entry = self.run_result

            # List of category labels for cat_5D[mean] and cat_11D[mean]
            cat_5D_labels = list(result_entry['cat_5D']['mean'].keys())
            cat_11D_labels = list(result_entry['cat_11D']['mean'].keys())

            # Convert scores to lists for plotting
            cat_5D_values = np.array([list(result_entry['cat_5D']['mean'].values())])
            cat_11D_values = np.array([list(result_entry['cat_11D']['mean'].values())])

            # Convert values to radians for polar plot
            theta_5D = np.linspace(0, 2 * np.pi, len(cat_5D_labels) + 1)
            theta_11D = np.linspace(0, 2 * np.pi, len(cat_11D_labels) + 1)

            # Create polar plots for cat_5D and cat_11D
            fig, (ax1, ax2) = pl.subplots(1, 2, figsize=(16, 8), subplot_kw={'projection': 'polar'})

            for value in cat_5D_values:
                value = np.append(value, value[0])  # Duplicate the first dimension value to close the loop
                ax1.plot(theta_5D, value, label=self.expsettings['sub'])

            for value in cat_11D_values:
                value = np.append(value, value[0])  # Duplicate the first dimension value to close the loop
                ax2.plot(theta_11D, value, label=self.expsettings['sub'])

            ax1.set_thetagrids(np.degrees(theta_5D[:-1]), cat_5D_labels)
            ax1.set_rlabel_position(22.5)
            ax1.set_ylim(0, 10)
            ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax1.set_title('Cat_5D Scores')

            ax2.set_thetagrids(np.degrees(theta_11D[:-1]), cat_11D_labels)
            ax2.set_rlabel_position(22.5)
            ax2.set_ylim(0, 10)
            ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax2.set_title('Cat_11D Scores')

            fig.savefig(opj(self.out_path,f"{rp(self.expsettings['sub'])}_{rp(self.expsettings['ses'])}_{rp(self.expsettings['run'])}_{rp(self.expsettings['task'])}_results.pdf"), dpi=600, bbox_inches='tight', transparent=True)
            print("plotting result done")

        def SEfit(self):
            print('SEfit')
            # Get the order of the questions
            SE_question = self.df_responses
            SE_order = self.order
            SE_time = self.time

            #make an ordered dictionnary with, for each question found in response_text content as key the key_pressed as value. Don't forget to remove the 'num_' part of the key_pressed and convert it to int. Don't take into account nan response_text 
            # if the elements of the column 'key_pressed' are strings :
            if type(SE_question['key_pressed'].tolist()[0]) == str :
                #if the key_pressed is one size length :
                if len(SE_question['key_pressed'].tolist()[0]) == 1 :
                    SE_question_dict = {row['response_text']: int(row['key_pressed']) for index, row in SE_question.iterrows() if row['response_text'] is not np.nan}
                else :
                    SE_question_dict = {row['response_text']: int(row['key_pressed'][4:]) for index, row in SE_question.iterrows() if row['response_text'] is not np.nan}

            else :
                #if the key_pressed is a number :
                SE_question_dict = {row['response_text']: int(row['key_pressed']) for index, row in SE_question.iterrows() if row['response_text'] is not np.nan}

            #print(SE_question_dict)
            self.SE_question_dict = SE_question_dict