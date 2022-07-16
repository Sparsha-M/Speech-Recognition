# Speech Recognition using LSTM-RNN
 Many speech recognition applications and devices are available, but the more advanced solutions use AI and machine learning. They integrate grammar, syntax, structure, and composition of audio and voice signals to understand and process human speech. Ideally, they learn as they go — evolving responses with each interaction.

[//]: # (Image References)

[image1]: ./images/pipeline.png "ASR Pipeline"
[image2]: ./images/select_kernel.png "select speech-recognition kernel"

## Project Overview

In this notebook, We will build a deep neural network that functions as part of an end-to-end automatic speech recognition (ASR) pipeline!  

![ASR Pipeline][image1]

We begin by investigating the [LibriSpeech dataset](http://www.openslr.org/12/) that will be used to train and evaluate our models. The algorithm will first convert any raw audio to feature representations that are commonly used for ASR, and will then move on to building neural networks that can map these audio features to transcribed text. After learning about the basic types of layers that are often used for deep learning-based approaches to ASR, We will engage in our own investigations by creating and testing our own state-of-the-art models. Throughout the notebook, we provide recommended research papers for additional reading and links to GitHub repositories with interesting implementations.

## Project Instructions

### Getting Started

1. Clone the repository, and navigate to the downloaded folder.
```
git clone https://github.com/Shreevathsa1/Speech-Recognition-using-LSTM-RNN.git
cd Speech-Recognition-using-LSTM-RNN
```

2. Create (and activate) a new environment with Python 3.10 and the `numpy` package.

	- __Linux__ or __Mac__: 
	```
	conda create --name speech-recognition python numpy
	source activate speech-recognition
	```
	- __Windows__: 
	```
	conda create --name speech-recognition python numpy scipy
	activate speech-recognition
	```

3. Install TensorFlow.
	- Option 1: __To install TensorFlow with GPU support__, follow [the guide](https://www.tensorflow.org/install/) to install the necessary NVIDIA software on your system:
	```
	pip install tensorflow-gpu==2.9.1
	```
	- Option 2: __To install TensorFlow with CPU support only__,
	```
	pip install tensorflow==2.9.1
	```

4. Install a few pip packages.
```
pip install -r requirements.txt
```

5. Switch [Keras backend](https://keras.io/backend/) to TensorFlow.
	- __Linux__ or __Mac__: 
	```
	KERAS_BACKEND=tensorflow python -c "from keras import backend"
	```
	- __Windows__: 
	```
	set KERAS_BACKEND=tensorflow
	python -c "from keras import backend"
	```

6. Obtain the 'libav' package.
	- __Linux__: 'sudo apt-get install libav-tools'
	- __Mac__: 'brew install libav'
	- __Windows__: Browse to the [Libav website](https://libav.org/download/)
		- Scroll down to "Windows Nightly and Release Builds" and click on the appropriate link for our system (32-bit or 64-bit).
		- Click 'nightly-gpl'.
		- Download most recent archive file.
		- Extract the file.  Move the 'usr' directory to our C: drive.
		- Go back to our terminal window from above.
	```
	rename C:\usr avconv
    set PATH=C:\avconv\bin;%PATH%
	```

7. Obtain the appropriate subsets of the LibriSpeech dataset, and convert all flac files to wav format.
	- __Linux__ or __Mac__: 
	```
	wget http://www.openslr.org/resources/12/dev-clean.tar.gz
	tar -xzvf dev-clean.tar.gz
	wget http://www.openslr.org/resources/12/test-clean.tar.gz
	tar -xzvf test-clean.tar.gz
	mv flac_to_wav.sh LibriSpeech
	cd LibriSpeech
	./flac_to_wav.sh
	```
	- __Windows__: Download two files ([file 1](http://www.openslr.org/resources/12/dev-clean.tar.gz) and [file 2](http://www.openslr.org/resources/12/test-clean.tar.gz)) via browser and save in this directory.  Extract them with an application that is compatible with `tar` and `gz` such as [7-zip](http://www.7-zip.org/) or [WinZip](http://www.winzip.com/). Convert the files from the terminal window:
	```
	move flac_to_wav.sh LibriSpeech
	cd LibriSpeech
	powershell ./flac_to_wav.sh
	```

8. Create JSON files corresponding to the train and validation datasets.
```
cd ..
python create_desc_json.py LibriSpeech/dev-clean/ train_corpus.json
python create_desc_json.py LibriSpeech/test-clean/ valid_corpus.json
```

9. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the 'speech_recog' environment.  Open the notebook.
```
python -m ipykernel install --user --name speech_recog --display-name "speech_recog"
jupyter notebook speech_recognition.ipynb
```

10. Before running code, change the kernel to match the 'speech_recog' environment by using the drop-down menu.  Then, follow the instructions in the notebook.

![select speech_recog kernel][image2]


### Amazon Web Services

If you do not have access to a local GPU, you could use Amazon Web Services to launch an EC2 GPU instance.


#### Work in progress and feel free to add any of the below

#### (1) Add a Language Model to the Decoder

The performance of the decoding step can be greatly enhanced by incorporating a language model.  Build our own language model from scratch, or leverage a repository or toolkit that you find online to improve our predictions.

#### (2) Train on Bigger Data

In the project, we used some of the smaller downloads from the LibriSpeech corpus.  Try training our model on some larger datasets - instead of using `dev-clean.tar.gz`, download one of the larger training sets on the [website](http://www.openslr.org/12/).

#### (3) Try out Different Audio Features

In this project, we had the choice to use _either_ spectrogram or MFCC features.  Take the time to test the performance of _both_ of these features.  For a special challenge, train a network that uses raw audio waveforms!

## Special Thanks

We have borrowed the `create_desc_json.py` and `flac_to_wav.sh` files from the [ba-dls-deepspeech](https://github.com/baidu-research/ba-dls-deepspeech) repository, along with some functions used to generate spectrograms.