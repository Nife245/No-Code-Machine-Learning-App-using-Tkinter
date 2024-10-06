import tkinter as tk
import ttkbootstrap as ttk
import customtkinter as ctk
import pandas as pd
from tkinter import filedialog , messagebox
from settings import *
from threading import Thread
import numpy as np
from MLprocess import *

file_name = None

class MLApp(ttk.Window):
    def __init__(self):
        super().__init__(self)
        self.minsize(750 , 450)

        #UI
        self.frame2 = ctk.CTkFrame(self ,corner_radius = 0 , fg_color= OFF_WHITE)
        self.frame2.pack(side = 'left' , fill = 'both' , expand = True)

        radio_font = ctk.CTkFont('Calibri' , 18 , weight = 'bold')

        self.upload_btn = ctk.CTkButton(self.frame2 , text = "Upload File (*csv , *xlsx)" ,
                                        height = 50 , font = radio_font , fg_color = TEAL , 
                                        hover_color = LIGHT_TEAL , command = self.loadFile)
        self.upload_btn.place(relx = .5 , rely = .5 , anchor = 'center')
        
        self.mainloop()

    def loadFile(self):
        # Open a file dialog to select a CSV or Excel file
        global file_name
        file_path = filedialog.askopenfilename(
            title="Select a file",
            filetypes=(("CSV files", "*.csv"), ("Excel files", "*.xlsx;*.xls"))
        )
        
        if file_path:
            try:
                # Check the file extension and load the file accordingly
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif file_path.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file_path)
                else:
                    messagebox.showerror("Error", "Unsupported file format.")
                    return
                
                file_name = file_path.split('/')[-1]
                file_name = file_name.split('.')[0]
                MLProcessFrame(self.frame2 , file_name , df)
                messagebox.showinfo("Success", "File loaded successfully!")
            
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")

class MLProcessFrame(ctk.CTkScrollableFrame):
    def __init__(self , parent , file_name , file):
        super().__init__(parent , fg_color= OFF_WHITE)
        self.place(relx = 0 , rely = 0 , relwidth = 1 , relheight = 1)

        self.file = file
        self.parent = parent

        ctk.CTkButton(self , text = "" , image = ARROW , width = 30
                       , fg_color=TEAL , hover_color=LIGHT_TEAL ,
                       command = self.terminatePage).place(x = 10 , y = 10)

        normal_font = ctk.CTkFont('Calibri' , 18 )
        bold_font = ctk.CTkFont('Calibri' , 18 , weight = 'bold')

        self.dimension = self.file.shape
        self.name_label = ctk.CTkLabel(self , text = file_name , font = bold_font ,
                                        text_color = CHARCOAL)
        self.name_label.pack()
        self.dimension_label = ctk.CTkLabel(self
                     , text = f"number of rows : {self.dimension[0]} , number of columns : {self.dimension[1]}" 
                        , font = normal_font , text_color = CHARCOAL)
        self.dimension_label.pack()
        self.targetFrame = ctk.CTkFrame(self , fg_color=OFF_WHITE)
        self.targetFrame.pack(pady = 10)
        ctk.CTkLabel(self.targetFrame , text = "Target :" , font = bold_font ,
                                        text_color = CHARCOAL).pack(side = "left" , padx = (0 , 4))
        self.target = ctk.CTkComboBox(self.targetFrame, values = self.file.columns.tolist() , button_color = TEAL ,
                                      button_hover_color = LIGHT_TEAL ,text_color= CHARCOAL,
                                        state = 'readonly' , width = 200 , height = 35 , font = bold_font)
        self.target.pack(side = "left")
        self.enter_button = ctk.CTkButton(self.targetFrame , width = 50 , height = 35 , text = "Enter" ,
                                          font= bold_font , fg_color=TEAL , hover_color= LIGHT_TEAL , 
                                          command= self.enterTarget)
        self.enter_button.pack(side = "left" , padx = (4 ,0))

    def clean(self):
        file_c = self.file.dropna(subset=[self.target.get()])
        FeedbackLabel(self , "empty Target rows dropped")
        file_c = file_c.drop_duplicates()
        FeedbackLabel(self , "duplicates dropped")

        return file_c

    def enterTarget(self):
        self.target.configure(state = "disabled")
        self.enter_button.configure(state = "disabled")
        file_c = self.clean()
        DataInput(self , "Train data percent" , 'train-test-spilt' , '' , file_c , self.target.get())

    def terminatePage(self):
        self.place_forget()

class ModelInput(ctk.CTkFrame):
    def __init__(self , parent , X_train):
        super().__init__(parent , fg_color= OFF_WHITE)
        self.pack(pady = 4)
        self.parent = parent
        self.X_train = X_train
        bold_font = ctk.CTkFont('Calibri' , 18 , weight = 'bold')
        models = ["Random Forest Classifier" , "Random Forest Regressor" ,
                  "Decision Tree Classifier" , "Decision Tree Regressor" ,
                  "Logistic Regression" , "Linear Regression" ,
                  "Support Vector Classifier" , "Support Vector Regressor"]

        ctk.CTkLabel(self , text = "Pick Model :" , font = bold_font ,
                                        text_color = CHARCOAL).pack(side = "left" , padx = (0 , 4))
        self.model_entry = ctk.CTkComboBox(self, values = models , button_color = TEAL ,
                                      button_hover_color = LIGHT_TEAL ,text_color= CHARCOAL,
                                        state = 'readonly' , width = 240 , height = 35 , font = bold_font)
        self.model_entry.pack(side = "left")
        self.enter_button = ctk.CTkButton(self , width = 50 , height = 35 , text = "Enter" ,
                                          font= bold_font , fg_color=TEAL , hover_color= LIGHT_TEAL , 
                                          command= self.enterModel)
        self.enter_button.pack(side = "left" , padx = (4 ,0))

        self.frame = ctk.CTkFrame(self.parent , fg_color=OFF_WHITE)
        self.frame.pack()

    def enterModel(self):
        self.frame.pack_forget()
        model = self.model_entry.get()
        self.frame = ctk.CTkFrame(self.parent , fg_color=OFF_WHITE)
        self.frame.pack()
        FeedbackLabel(self.frame , f"Model Received : {model}")
        font = ctk.CTkFont('Calibri' , 18 , weight = 'bold')
        loading_label = ctk.CTkLabel(master=self.frame , text= "Loading..." , font = font , text_color=CHARCOAL)
        loading_label.pack()
        save_model_btn = ctk.CTkButton(self.frame , text = "Save Model" ,
                                        height = 50 , font = font , fg_color = TEAL , 
                                        hover_color = LIGHT_TEAL , command = lambda : saveModel(model_name=model , file_name = file_name))
        Thread(target=createModel , args = ( model ,self.X_train ,self.frame , loading_label , save_model_btn)).start()

class DataInput(ctk.CTkFrame):
    def __init__(self , parent , label , operation , percent , file , target):
        super().__init__(parent , fg_color= OFF_WHITE)
        self.pack()
        self.operation = operation
        self.percent = percent
        self.file = file
        self.parent = parent
        self.target = target

        self.normal_font = ctk.CTkFont('Calibri' , 18 )
        bold_font = ctk.CTkFont('Calibri' , 18 , weight = 'bold')

        ctk.CTkLabel(self , text = label , font = bold_font ,
                                        text_color = CHARCOAL).pack(side = "left" , padx = (0 , 4))
        self.entry = ctk.CTkEntry(self ,text_color= CHARCOAL,
                                   width = 200 , height = 35 , font = bold_font , placeholder_text="1-100")
        self.entry.pack(side = "left")
        self.enter_button = ctk.CTkButton(self , width = 50 , height = 35 , text = "Enter" ,
                                          font= bold_font , fg_color=TEAL , hover_color= LIGHT_TEAL , 
                                          command= self.operationControl)
        self.enter_button.pack(side = "left" , padx = (4 ,0))
    
    def operationControl(self):
        #avoid multiple operatons 
        self.enter_button.configure(state = 'disabled')
        if self.operation == 'drop columns':
            file = dropNullColumns(file = self.file , percent = self.percent
                            , threshold = self.entry.get() , target_column = self.target
                            , parent = self.parent)
            print(columns)
            ColumnInspection(self.parent , file)
        elif self.operation == 'cardinality_limit' :
            #categorical encoding
            file = categoricalEncoding(cardinality_limit = self.entry.get()
                                 , file = self.file , parent= self.parent)
            #Pick Model
            ModelInput(self.parent , file)
        elif self.operation == 'train-test-spilt' :
            X_train, X_test, y_train, y_test = trainTestSplit(file = self.file 
                            , train_percent = self.entry.get()
                            , target = self.target)
            #update values in MLprocess
            data.append(X_train)
            data.append(X_test)
            data.append(y_train)
            data.append(y_test)
            FeedbackLabel(self.parent , 'Train Test Split Done')
            print(X_train)

            #percentage null of each column
            percent = np.array([])
            rows = X_train.shape[0]
            for column in X_train.columns.tolist():
                percent = np.append(percent , np.around((X_train[column].isnull().sum() / rows) , 2))
            print(percent)
            ctk.CTkLabel(master=self.parent , text = "The Limit for column emptiness allowed .e.g. 40% empty" , 
                         font = self.normal_font).pack()
            DataInput(self.parent , "Percent threshold" , 'drop columns' , percent , X_train , target = self.target)

class FeedbackLabel(ctk.CTkFrame):
    def __init__(self , parent , content):
        super().__init__(parent , fg_color=OFF_WHITE)
        self.pack()

        normal_font = ctk.CTkFont('Calibri' , 18 )
        ctk.CTkLabel(self , image=CHECK , text = '').pack(side = 'left')
        ctk.CTkLabel(self , text = content , text_color= CHARCOAL , font=normal_font).pack(side = 'left' , padx = 4)

class ColumnInspection(ctk.CTkFrame):
    def __init__(self , parent , file):
        super().__init__(parent)
        self.pack(pady = 4)
        
        self.file = file
        self.normal_font = ctk.CTkFont('Calibri' , 18 )
        bold_font = ctk.CTkFont('Calibri' , 18 , weight='bold')
        self.parent = parent
        #main UI
        ctk.CTkLabel(self , text = "Do you want to drop other columns ?" 
                     , font = self.normal_font , 
                     text_color= CHARCOAL).pack(pady = 4)
        frame = ctk.CTkFrame(self)
        frame.pack()
        ctk.CTkButton(frame , width = 50 , height = 35 , text = "Yes" ,
                                          font= bold_font , fg_color=TEAL , hover_color= LIGHT_TEAL , 
                    command= self.showDropPage).pack(side = 'left' , padx = (0 , 8))
        
        ctk.CTkButton(frame , width = 50 , height = 35 , text = "No" ,
                                          font= bold_font , fg_color=TEAL , hover_color= LIGHT_TEAL , 
                    command= self.RemovePage).pack(side = 'left')
        
    def showDropPage(self):
        self.pack_forget()
        self.showColumns()
    
    def RemovePage(self):
        self.pack_forget()
        dataImputation(self.file , self.parent)
        DataInput(self.parent , "Cardinality Limit" , 'cardinality_limit', '' ,self.file , '')

    def showColumns(self):
        self.columns_frame = ctk.CTkFrame(self.parent)
        self.columns_frame.pack(pady = 4 , ipadx = 3 , ipady = 3)

        columns = self.file.columns.tolist()
        num_columns = len(columns)
        grid_columns = 3
        grid_rows = int(np.ceil(num_columns / grid_columns)) + 1

        self.columns_frame.columnconfigure(list(np.arange(grid_columns)) , uniform='a')
        self.columns_frame.rowconfigure(list(np.arange(grid_rows)) , uniform='b')

        column_index = 0
        self.check_status = []
        for row in range(grid_rows):
            for column in range(grid_columns):
                if column_index <= (len(columns)-1):
                    column_name = columns[column_index]
                    check = ctk.CTkCheckBox(self.columns_frame , text= column_name,
                                            font = self.normal_font , text_color=CHARCOAL , fg_color= TEAL , 
                                            hover_color=LIGHT_TEAL)
                    check.grid(row = row , column = column , padx = 3 , sticky = 'nsw')
                    self.check_status.append((column_name , check))
                    column_index+=1
            self.drop_btn = ctk.CTkButton(self.columns_frame , text = "Drop" ,
                                        height = 50 , font = self.normal_font , fg_color = TEAL , 
                                        hover_color = LIGHT_TEAL , command = self.dropColumns)
            self.drop_btn.grid(row = (grid_rows-1) , column= 0 , sticky ='nsw' , pady = 1)
            self.done = ctk.CTkButton(self.columns_frame , text = "Done" ,
                                        height = 50 , font = self.normal_font , fg_color = TEAL , 
                                        hover_color = LIGHT_TEAL , command = self.finishDrop)
            self.done.grid(row = (grid_rows-1) , column= 1 , sticky ='nsw' , pady = 1)
 
    def finishDrop(self):
        for column_status in self.check_status :
            column_status[1].configure(state = 'disabled')
        self.drop_btn.configure(state = 'disabled')
        self.done.configure(state = 'disabled')
        FeedbackLabel(self.parent , "Column Drop Done")
        dataImputation(file = self.file , parent = self.parent)
        DataInput(self.parent , "Cardinality Limit" , 'cardinality_limit', '' ,self.file , '')

    def dropColumns(self):
        num = 0
        column_names = []
        for column_status in self.check_status :
            if column_status[1].get() == 1 :
                self.file = self.file.drop(column_status[0] , axis = 1)
                num += 1
                column_names.append(column_status[0])
                #add the dropped columns to the total columns dropped in train data
                columns.append(column_status[0])
        print(columns)
        print(f"{num} columns dropped and names are {column_names} \n total columns left : {self.file.shape[1]}")
        self.columns_frame.pack_forget()
        self.showDropPage()

if __name__ == '__main__':
    MLApp()