import Tkinter as tk     # python 2
import tkFont as tkfont  # python 2
from Tkinter import *
import tkFileDialog as filedialog
import cv2
import ttk
import os
from PIL import Image, ImageTk
import subprocess
import os
import shutil
import re

import pandas as pd
import dill
import re

import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

print("importing stRTED")
classifier = dill.load(open('model2kk_500.sav', 'rb'))
tfvectorizer = dill.load(open('tfidf2kk_500.sav', 'rb'))
vectorizer = dill.load(open('bow2kk_500.sav', 'rb'))

df = pd.read_csv('tagcount.csv')
tag_dict = {}
for i in df['tags']:
    tag_dict[i] = int(1)

print("importing done")

stop=set(stopwords.words('english'))
sno=SnowballStemmer('english')

print("Models Imported...")

def clean(s):
    s=str(s)
    s=s.lower()
    html=re.compile('<.*?>')
    cleaned = re.sub(html,' ',s)
    fil=[]
    for i in cleaned.split():
        if i!='c++':
            cleaned=re.sub('[^A-Za-z]+', '', i)
            fil.append(cleaned)
        else:
            fil.append(i)
    return fil

stop=set(stopwords.words('english'))
sno=SnowballStemmer('english')

def stem(s):
    fil=[]
    for _ in s:
        if _ not in stop:
            s=(sno.stem(_).encode('utf8'))
            fil.append(s)
    s=b' '.join(fil)
    return s

def predict():
    title = str(gui.entry1.get())
    body  = str(gui.entry2.get())
    final_answer = set()
    opt = int(1)
    if(opt == int(1)):
        t1 = title
        t2 = body
        t = str(t1) + ' ' + str(t2)
        print(t)
        l=[]
        l.append(stem(clean(t)))

        ans_list = []
        for keys in l:
            if keys in dict.keys(tag_dict):
                final_answer.add(keys)
                ans_list.append(ans)

        x=tfvectorizer.transform(l)

        t=classifier.predict(x)
        print(t)

        k=vectorizer.inverse_transform(t)

        for ans in k[0]:
            final_answer.add(ans)
            ans_list.append(ans)

        ans_list.sort()

        asso_rules = {}
        df_aso = pd.read_csv('asso_rules.csv')

        for (kk,vv) in zip(df_aso['first'], df_aso['second']):
          asso_rules[kk] = vv

        def recur(idd, str_find, L):
            if(idd == L):
                if(str_find == ""):
                    return
                str_find = str_find[:len(str_find)-1]
                if str_find in asso_rules.keys():
                    xx = asso_rules[str_find]
                    xx_list = xx.split()
                    for ii in xx_list:
                        final_answer.add(ii)
                return
            recur(idd + 1, str_find, L)
            str_find = str_find + ans_list[idd] + ' '
            recur(idd + 1, str_find, L)
            return

        zero = int(0)

        recur(zero, "", len(ans_list))
		
	all_tags = ""
	for at in final_answer:
	    all_tags += str(str(at) + ' ')
	gui.entry3.insert(0, all_tags)

		
def exit(): 
    gui.destroy()

def clear():
    gui.entry1.delete(0,END)
    gui.entry2.delete(0,END)
    gui.entry3.delete(0,END)
    gui.entry1.focus()


# create a GUI window 
gui = Tk() 
  
# set the background colour of GUI window 
gui.configure(background="lightblue") 

# Set the background image
img = ImageTk.PhotoImage(Image.open("bg_cyan.png"))

panel = tk.Label(gui, image = img)
panel.pack(side = "bottom", fill = "both", expand = "yes")

# set the title of GUI window 
gui.title("Stack Overflow Tag Predictor") 
  
# set the configuration of GUI window 
gui.geometry("800x600") 

gui.label1 = Label(text="Input Title")
gui.label1.place(relx=0.180, rely=0.200, height=34, width=200)
gui.label1.configure(bg="lightblue")
gui.label1.configure(font=("Raleway", 15))
gui.label1.configure(foreground="black")

gui.label2 = Label(text="Input Body")
gui.label2.place(relx=0.180, rely=0.300, height=34, width=200)
gui.label2.configure(bg="lightblue")
gui.label2.configure(font=("Raleway", 15))
gui.label2.configure(fg="black")

gui.label3 = Label(text="Predicted Tags")
gui.label3.place(relx=0.180, rely=0.500, height=34, width=200)
gui.label3.configure(bg="lightblue")
gui.label3.configure(font=("Raleway", 15, 'italic'))
gui.label3.configure(fg="black")

gui.entry1 = Entry()
gui.entry1.place(relx=0.45, rely=0.20, height=34, relwidth=0.34)
gui.entry1.configure(background="white")
gui.entry1.configure(foreground="black")

gui.entry2 = Entry()
gui.entry2.place(relx=0.45, rely=0.300, height=74, relwidth=0.34)
gui.entry2.configure(background="white")
gui.entry2.configure(foreground="black")

gui.entry3 = Entry()
gui.entry3.place(relx=0.45, rely=0.50, height=34, relwidth=0.34)
gui.entry3.configure(background="white")
gui.entry3.configure(foreground="black")

gui.button2 = Button(text="Clear", command = clear)
gui.button2.place(relx=0.300, rely=0.689, height=37, width=85)
gui.button2.configure(background="white")
gui.button2.configure(font=("Raleway", 12, 'bold'))
gui.button2.configure(foreground="black")

gui.button3 = Button(text="Predict", command = predict)
gui.button3.place(relx=0.450, rely=0.689, height=37, width=85)
gui.button3.configure(background="white")
gui.button3.configure(font=("Raleway", 12, 'bold'))
gui.button3.configure(foreground="black")

gui.button4 = Button(text="Exit", command = exit)
gui.button4.place(relx=0.600, rely=0.689, height=37, width=85)
gui.button4.configure(background="white")
gui.button4.configure(font=("Raleway", 12, 'bold'))
gui.button4.configure(foreground="black")

        
gui.mainloop()
  
