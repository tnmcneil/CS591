Classification instructions:
To run code from either pytorch tutorial or for the modified cities version, go to the folder containing the appropriate code 
(/assignment3/pytorch_tutorial/classify/ or /assignment3/classify respectively) and run python train.py to train the model yourself.  
Pretrained .pt model is also provided, to use this run predict.py [your_input] is either a city or baby name you wish to classify

Generation instructions:
Again go to appropriate folder (/assignment3/pytorch_tutorial/generate/ or /assignment3/generate/) 
and run python train.py [corpus] where corpus is the txt file containing the text you wish to
train on. (I used Jane Austen and Trump speeches)
Pretrained .pt model is provided for shakespeare, trump and jane austen, to experiment with these
run python generate.py [model] [options] where model is one of jane_austen.pt, trump.pt 
or shakespeare.pt. possible options are: -p: string to prime generation with, -l: desired prediction
length and -t: temperature. but defaults are set for all of these.