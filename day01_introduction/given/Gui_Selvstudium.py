# Gui tutorial

# Gui - En f?rste runde

# F?lg skridtene i programmet nedenfor meget n?je, s? du 
# forst?r hvad der sker.Lav sm? ting om. S? du kan se pr?cis
# hvor du er. Osv.
# Og derved f?r et f?rste indtryk af hvordan der tilf?jes
# Gui til din kode. Vi vender tilbage til Gui emnet i Uge 7

from tkinter import *

#First we make a small window
window = Tk()

window.title("Welcome to My super app")

window.mainloop()

#The last line which calls mainloop function, this function calls the endless loop of the window,
# so the window will wait for any user interaction till we close it.
# If you forget to call the mainloop function, nothing will appear to the user.

#Create a label widget
#To add a label to our previous example, we will create a label using the label class like this:

#    lbl = Label(window, text="Hello")
#    Then we will set its position on the form using the grid function and give it the location like this:
#    lbl.grid(column=0, row=0)

# I.e.
# Close the window then this code runs:

window = Tk()
window.title("Welcome to Super app nb 2")
lbl = Label(window, text="Hello")
lbl.grid(column=0, row=0)

window.mainloop()

# Change fonts
window = Tk()
window.title("Welcome to Super app nb 3")
lbl = Label(window, text="Hello", font=("Arial Bold", 50))
lbl.grid(column=0, row=0)

window.mainloop()

#Setting window size
#We can set the default window size using geometry function like this:
window = Tk()
window.title("app nb 4")
window.geometry('350x200')
lbl = Label(window, text="Hello", font=("Arial Bold", 50))
lbl.grid(column=0, row=0)

window.mainloop()


#Adding a button widget
#Let?s start by adding the button to the window, the button is created and added to the window the same as the label:

window = Tk()

window.title("nb 5 app")
window.geometry('350x200')
lbl = Label(window, text="Hello 5")
lbl.grid(column=0, row=0)
btn = Button(window, text="Click Me")
btn.grid(column=1, row=0)

window.mainloop()

#Handle button click event
#First, we will write the function that we need to execute when the button is clicked:

def clicked():
    lbl.configure(text="Button was clicked !!")


window = Tk()

window.title("#6 app")
window.geometry('350x200')
lbl = Label(window, text="Hello")
lbl.grid(column=0, row=0)
btn = Button(window, text="Click Me", command=clicked)
btn.grid(column=1, row=0)

window.mainloop()

#Get input using Entry class (Tkinter textbox)
# In the previous Python GUI examples, we saw how to add simple widgets,
# now let?s try getting the user input using Tkinter Entry class (Tkinter textbox).

window = Tk()

window.title("#7 app")
window.geometry('350x200')
lbl = Label(window, text="Hello")
lbl.grid(column=0, row=0)
txt = Entry(window, width=10)
txt.grid(column=1, row=0)
btn = Button(window, text="Click Me", command=clicked)
btn.grid(column=2, row=0)

window.mainloop()

#what about showing the entered text on the Entry widget?

window = Tk()

def newclicked():
    res = "Welcome to " + txt.get()
    lbl.configure(text=res)

window.title("#8 app")
window.geometry('350x200')
lbl = Label(window, text="Hello")
lbl.grid(column=0, row=0)
txt = Entry(window, width=10)
txt.grid(column=1, row=0)
btn = Button(window, text="Click Me", command=newclicked)
btn.grid(column=2, row=0)

window.mainloop()


#Set focus to entry widget
#That?s super easy, all we need to do is to call focus function like this:
# txt.focus()

window = Tk()

window.title("#9 app")
window.geometry('350x200')
lbl = Label(window, text="Hello")
lbl.grid(column=0, row=0)
txt = Entry(window, width=10)
txt.grid(column=1, row=0)
btn = Button(window, text="Click Me", command=newclicked)
btn.grid(column=2, row=0)
txt.focus()

window.mainloop()

# Lots of stuff can be added, lets Add some radio buttons widgets
#To add radio buttons, simply you can use RadioButton class like this:

from tkinter.ttk import *

window = Tk()

window.title("#10 app")
window.geometry('350x200')

rad1 = Radiobutton(window, text='First', value=1)
rad2 = Radiobutton(window, text='Second', value=2)
rad3 = Radiobutton(window, text='Third', value=3)
rad1.grid(column=1, row=0)
rad2.grid(column=2, row=0)
rad3.grid(column=3, row=1)

window.mainloop()

def radioclicked():
    res = "Welcome to TYT"
    lbl.configure(text=res)

window = Tk()

window.title("#11 app")
window.geometry('350x200')

lbl = Label(window, text="Hello")
lbl.grid(column=0, row=0)

rad1 = Radiobutton(window, text='First', value=1)
rad1 = Radiobutton(window,text='First', value=1, command=radioclicked)
rad2 = Radiobutton(window, text='Second', value=2)
rad3 = Radiobutton(window, text='Third', value=3)
rad1.grid(column=1, row=0)
rad2.grid(column=2, row=0)
rad3.grid(column=3, row=1)

window.mainloop()

# Add a ScrolledText widget (Tkinter textarea)

from tkinter import scrolledtext

window = Tk()

window.title("#12 app")
window.geometry('350x200')
txt = scrolledtext.ScrolledText(window, width=40, height=10)
txt.grid(column=0, row=0)

window.mainloop()

# To set scrolledtext content, you can use the insert method like this:
# txt.insert(INSERT,'You text goes here')

window = Tk()

window.title("#13 app")
window.geometry('350x200')
txt = scrolledtext.ScrolledText(window, width=40, height=10)
txt.insert(INSERT, 'This is y text')
txt.grid(column=0, row=0)

window.mainloop()

#Create a MessageBox
#To show a message box using Tkinter, you can use messagebox library like this:

#from tkinter import messagebox
#messagebox.showinfo('Message title', 'Message content')

# Putting it together
window = Tk()

from tkinter import messagebox

def messageonclicked():
    messagebox.showinfo('Message title', 'Message content from #14')

window.title("#14 app")
window.geometry('350x200')

btn = Button(window, text='Click here', command=messageonclicked)
btn.grid(column=0, row=0)

window.mainloop()


#Add a Progressbar widget
# To create a progress bar, you can use the progressbar class like this:

from tkinter.ttk import Progressbar
from tkinter import ttk

window = Tk()
window.title("#15 app")

window.geometry('350x200')
style = ttk.Style()
style.theme_use('default')
style.configure("black.Horizontal.TProgressbar", background='black')

bar = Progressbar(window, length=200, style='black.Horizontal.TProgressbar')
bar['value'] = 70
bar.grid(column=0, row=0)

window.mainloop()


