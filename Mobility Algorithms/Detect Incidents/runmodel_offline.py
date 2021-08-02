import pandas as pd
import numpy as np
import tensorflow as tf

"""Run incident detection neural network offline using test features file for model evaluation."""

def main():
	"""Parse command line arguments, load test features and create ground truth labels, load neural network model, predict incidents and output them to file."""
	parser = argparse.ArgumentParser(description='Measures Estimation program for offline evaluation of neural network for incident detection.')
    parser.add_argument('test_data')#csv of test features
    parser.add_argument('model_filename') #Where the best performing model is saved
    parser.add_argument('--out', help = 'Output csv file (include .csv)')  
    args = parser.parse_args()

	df_test = pd.read_csv(args.test_data).fillna(0)
	#Ground Truth labeling, must be edited to work for new data.
	actual = (df_test['Link'] >= 249) & (df_test['Link'] <= 250) & (df_test['CurrentTime'] >= 14640) & (df_test['CurrentTime'] <= 14940)

	model = tf.keras.models.load_model(model_filename)
	results = model.evaluate(test,actual,verbose=0)

	print(results)

	prediction = model.predict(test)

	 if args.out:
        out_file = args.out

    else:
        out_file = '/prediction.csv'

	df_test[prediction >= 0.5].to_csv(out_file)

if __name__ == "__main__":
    main()