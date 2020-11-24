# To Train

In our testing, we used a specific commit of the Tensorflow github for our training. 

` ` `
Commit: 1b215ce 
` ` `

[TF Github](https://github.com/tensorflow/tensorflow/blob/1b215ce9f3236d2de1c679530332ffe773ac4168/tensorflow/lite/micro/examples/micro_speech/-train/train_micro_speech_model.ipynb)

Train the model in the cloud using Google Colaboratory or locally using a
Jupyter Notebook.

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/tensorflow/blob/1b215ce9f3236d2de1c679530332ffe773ac4168/tensorflow/lite/micro/examples/micro_speech/train/train_micro_speech_model.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Google Colaboratory</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/tensorflow/blob/1b215ce9f3236d2de1c679530332ffe773ac4168/tensorflow/lite/micro/examples/micro_speech/train/train_micro_speech_model.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />Jupyter Notebook</a>
  </td>
</table>


*Estimated Training Time: ~2 Hours.*




## Implementing the model

Since we are deploying our model to a microcontroller, we will be using the model.cc file. 

To implement the model, we have to change a few files inside of the Final_Project directory. 
```
micro_features_model_speech.cpp
```
We need this file to implement our .cc model by replacing just the values in the new file with the model.cc file. We also need to change the value for the length according to the number at the bottom of our model.cc file. 
```
micro_features_micro_model_settings_speech.cpp
```
We need to add the words we trained for. For example, we trained for yes, no, on, off, so we added those to the list here. Add them in the order you trained them. 
```
micro_features_micro_model_settings_speech.h
```
We need to change the kCategoryCount variable to the correct number of words wanted as shown in the previous file, in this example, 6 with the silence and unknown. 
```
arduino_command_responder_speech.cpp
```
We need to add the responses for our words. Add your words like how our words are set up here, with found_command[] being an array of characters. The variable certainty is also here which we can use to lower or highten the threshold for words to be heard.  
