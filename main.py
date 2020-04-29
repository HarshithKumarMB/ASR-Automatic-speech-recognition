import os
import pickle
import subprocess as sp
import pandas as pd
import neural_net
import sound_recorder
import speech_recognition as sr 
from langdetect import detect
from langdetect import DetectorFactory

#initialize the recognizer
r=sr.Recognizer()

if __name__ == '__main__':
	while True:
		print('\nMenu')
		print('1. Train Neural Net')
		print('2. Turn my voice into text')
		print('3. Analyse Voice')
		print('4. Recognise language')
		print('5. Exit')
		option = input('Enter Option Number: ')

		if option == '1':
			neural_net.run()
		elif option == '2':
			with sr.Microphone() as source:
				print("you have 20 seconds to speak")
				print("Reding your speech....")
				aud=r.record(source, duration=20)     # recording the audio through source
				text=r.recognize_google(aud)    # conversion of speech to text
				print(" ")
				
				print("This is what was heard") 
				print(" ")
				print(" ")
				print(text)
				break
				
		elif option == '3':
			if not os.path.isfile('trained_neural_net'): 
				                                                                 # checking if neural_net file exists
				print('\nNeural net not trained. First train the neural net.')
			else:
				sound_recorder.run()

				print('\nExtracting data from recorded voice...\n')
				sp.call(['C:/Program Files/R/R-3.4.4/bin/Rscript.exe','getAttributes.r', os.getcwd()], shell=True) 
				print('\n\nPreprocessing extracted data...')
				data = pd.read_csv('output/voiceDetails.csv')
				del data['peak_f'], data['sound.files'], data['selec'], data['duration']
				dataset = pd.read_csv('voice.csv')
				dataset = dataset.iloc[:, :-1]
				
				data = (data - dataset.mean())/ (dataset.max() - dataset.min())                 

				trained_neural_net = pickle.load(open('trained_neural_net', 'rb'))          # load trained neural net from file
				print('\nPrediction: \r')
				print('Female' if trained_neural_net.predict(data)[0] == 1 else 'Male')         #prediction
				break
		elif option =='4':
			with sr.Microphone() as source:
				print("you have 20 seconds to speak")
				print("Recognizing....")
				aud=r.record(source, duration=20)
				text=r.recognize_google(aud)
				lang_=detect(text)
				DetectorFactory.seed = 0
				print(" ")
				print(" ")
				print("Is this what you spoke:", text)
				print(" ")
				print("Is this the language what you are speaking:", lang_)
				break
			
		elif option == '5':
			print('\nExiting...')
			break
		else:
			print('\nInvalid option. Please try again...')
