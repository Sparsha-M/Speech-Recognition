
from pathlib import Path

from tkinter import *
from winsound import *

# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage

import pickle

myvar = ""

with open('data.pkl', 'rb') as file:
    # Call load method to deserialze
    myvar = pickle.load(file)
    print(myvar)


####################################################
from data_generator import vis_train_features

# extract label and audio features for a single training example
vis_text, vis_raw_audio, vis_mfcc_feature, vis_spectrogram_feature, vis_audio_path = vis_train_features()
from IPython.display import Markdown, display
from data_generator import vis_train_features, plot_raw_audio
from IPython.display import Audio

from data_generator import plot_spectrogram_feature

from data_generator import plot_mfcc_feature


def graph():
    plot_raw_audio(vis_raw_audio)
    plot_spectrogram_feature(vis_spectrogram_feature)
    plot_mfcc_feature(vis_mfcc_feature)
#####################################################


#######################################
# from pygame import mixer

# mixer.init()


# def play_music():
#     mixer.music.load(vis_audio_path)
#     mixer.music.play()
#######################################

def play(): return PlaySound(vis_audio_path, SND_FILENAME)
# button = Button(root, text = 'Play', command = play)


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("./assets")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


window = Tk()

window.geometry("1000x600")
window.configure(bg="#F5F7FA")


canvas = Canvas(
    window,
    bg="#F5F7FA",
    height=600,
    width=1000,
    bd=0,
    highlightthickness=0,
    relief="ridge"
)

canvas.place(x=0, y=0)
image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    1048.0,
    52.000000000000014,
    image=image_image_1
)

image_image_2 = PhotoImage(
    file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(
    87.0,
    186.0,
    image=image_image_2
)

image_image_3 = PhotoImage(
    file=relative_to_assets("image_3.png"))
image_3 = canvas.create_image(
    488.0,
    51.000000000000014,
    image=image_image_3
)

image_image_4 = PhotoImage(
    file=relative_to_assets("image_4.png"))
image_4 = canvas.create_image(
    335.0,
    298.99998474121094,
    image=image_image_4
)

image_image_5 = PhotoImage(
    file=relative_to_assets("image_5.png"))
image_5 = canvas.create_image(
    740.0,
    371.0,
    image=image_image_5
)

entry_image_1 = PhotoImage(
    file=relative_to_assets("entry_1.png"))
entry_bg_1 = canvas.create_image(
    746.0,
    320.0,
    image=entry_image_1
)
entry_1 = Text(
    bd=0,
    bg="#FFFFFF",
    highlightthickness=0
)
entry_1.place(
    x=535.0,
    y=184.0,
    width=422.0,
    height=270.0
)
entry_1.insert(INSERT, myvar)
entry_1.insert(END, "\n")
# entry_1.pack()

#########################################################


# def SpeechToText():
#     speechtotextwindow = Toplevel(window)
#     speechtotextwindow.title('Speech-to-Text Converter')
#     speechtotextwindow.geometry("600x600")
#     speechtotextwindow.configure(bg="#B3DEF8")

#     Label(speechtotextwindow, text='Speech-to-Text Converter',
#           font=("Comic Sans MS", 18), bg='IndianRed').place(x=150, y=20)

#     recordbutton = Button(speechtotextwindow, text='Generate text',
#                           bg='Sienna', command=print_output)
#     recordbutton.place(x=260, y=100)

#     text = Text(speechtotextwindow, font=12, height=3, width=30)
#     text.place(x=135, y=200)
#     text.insert(INSERT, myvar)
#     text.insert(END, "\n")
#     text.pack()

##########################################################


button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    # image=button_image_1,
    text='Play Audio',
    borderwidth=0,
    highlightthickness=0,
    command=play,
    relief="flat"
)
button_1.place(
    x=665.0,
    y=500.0,
    width=166.0,
    height=50.0
)

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=graph,
    relief="flat"
)
button_2.place(
    x=189.0,
    y=360.0,
    width=159.0,
    height=44.0
)
window.resizable(False, False)
window.mainloop()
