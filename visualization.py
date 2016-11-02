# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 13:15:39 2016

@author: kaku
"""


import numpy as np
import os,datetime
from bokeh.plotting import output_file, show, figure,vplot
from bokeh.models import PrintfTickFormatter,FixedTicker
result_file = os.path.join('..','result')
now_global = datetime.datetime.now()
file_time_global=str(now_global.strftime("%Y-%m-%d-%H-%M"))

def acc_loss_visual(Acc_all):
    
    Ori_M_ori_D = Acc_all[0]
    Tran_M_ori_D = Acc_all[1]    
    Ori_M_New_D = Acc_all[2]
    Tran_M_New_D = Acc_all[3]    
    Ori_M_All_D = Acc_all[4]
    Tran_M_All_D = Acc_all[5]
    
    Ratio = ['1/200','1/100','1/80','1/60','1/40','1/20','1/10','1/6','1/3','1/1']
    output_file(os.path.join(result_file,"Acc_all"+"_"+file_time_global+".html"))
    p1 = figure(x_range = Ratio)
    p1.title = "Original and Transfered model performance"
    p1.xaxis.axis_label = 'Ratio'
    p1.yaxis.axis_label = 'Accuracy %'
    Ratio = ['1/200','1/100','1/80','1/60','1/40','1/20','1/10','1/6','1/3','1/1']
    #Ratio = [1/200,1/100,1/80,1/60,1/40,1/20,1/10,1/6,1/3,1/1]
    #Ratio = str(Ratio)
    #p1.circle(range(0,len(Ori_M_ori_D)), Ori_M_ori_D ,x_range_name='hhhhh',color="green", legend="Ori_M_ori_D " )
    p1.line(range(0,len(Ori_M_ori_D)), Ori_M_ori_D ,line_color="green", legend="Ori_M_ori_D")
    p1.circle(range(0,len(Ori_M_ori_D)), Ori_M_ori_D,color="green", legend="Ori_M_ori_D")
    
    p1.line(range(0,len(Tran_M_ori_D)), Tran_M_ori_D ,line_color="red", legend="Tran_M_ori_D")
    p1.circle(range(0,len(Tran_M_ori_D)), Tran_M_ori_D,color="red", legend="Tran_M_ori_D")
    
    p1.line(range(0,len(Ori_M_New_D)), Ori_M_New_D ,line_color="yellow", legend="Ori_M_New_D")
    p1.circle(range(0,len(Ori_M_New_D)), Ori_M_New_D,color="yellow", legend="Ori_M_New_D")
    
    p1.line(range(0,len(Tran_M_New_D)), Tran_M_New_D ,line_color="gray", legend="Tran_M_New_D")
    p1.circle(range(0,len(Tran_M_New_D)), Tran_M_New_D,color="gray", legend="Tran_M_New_D")
    
    p1.line(range(0,len(Ori_M_All_D)), Ori_M_All_D ,line_color="pink", legend="Ori_M_All_D")
    p1.circle(range(0,len(Ori_M_All_D)), Ori_M_All_D,color="pink", legend="Ori_M_All_D")
    
    p1.line(range(0,len(Tran_M_All_D)), Tran_M_All_D ,line_color="orange", legend="Tran_M_All_D")
    p1.circle(range(0,len(Tran_M_All_D)), Tran_M_All_D,color="orange", legend="Tran_M_All_D")    
    #p1.xaxis[0].ticker=FixedTicker(ticks=[1/200,1/100,1/80])
    #p1.xaxis[0].formatter = PrintfTickFormatter(format = '1/200','1/100','1/80','1/60','1/40','1/20','1/10','1/6','1/3','1/1')
    #p1.xaxis[0].formatter = PrintfTickFormatter(format=Ratio)
#    p1.circle(range(0,len(val_acc_arr)), val_acc_arr,color="orange", legend="Val_Acc")
#    p1.line(range(0,len(val_acc_arr)), val_acc_arr, line_color="orange",legend="Val_Acc")
    show(p1)

Ori_M_ori_D = np.array([[99.73,99.74,99.72,99.64,99.76,99.68,99.71,99.72,99.72,99.72]])
Tran_M_ori_D = np.array([[40.50,49.18,61.49,72.76,88.92,92.39,95.46,95.96,97.11,97.97]])

Ori_M_New_D = np.array([[35.28,34.26,37.66,34.08,32.25,34.41,34.93,29.63,32.31,33.34]])
Tran_M_New_D = np.array([[99.33,98.40,97.78,96.94,94.95,91.45,89.90,84.87,81.53,73.11]])

Ori_M_All_D = np.array([[68.19,67.69,69.34,67.55,66.71,67.73,68.00,65.41,66.73,67.23]])
Tran_M_All_D = np.array([[69.30,73.27,79.25,84.59,91.87,91.93,92.74,90.53,89.48,85.80]])

Acc_all = np.concatenate((Ori_M_ori_D,Tran_M_ori_D,Ori_M_New_D,Tran_M_New_D,Ori_M_All_D,Tran_M_All_D),axis=0)
Acc_all = Acc_all[:,::-1]

acc_loss_visual(Acc_all)

"""
p = figure(title="line", plot_width=300, plot_height=300)
p.line(x=['a', '2', '3', '4', '5'], y=[6, 7, 2, 4, 5])
show(p)
"""