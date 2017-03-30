package com.ngray.jnet.recurrent.test;

import static org.junit.Assert.*;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.Test;

import com.ngray.jnet.optimizers.RMSPropException;
import com.ngray.jnet.recurrent.Callback;
import com.ngray.jnet.recurrent.DataSet;
import com.ngray.jnet.recurrent.DictionaryException;
import com.ngray.jnet.recurrent.RecurrentNeuralNetwork;
import com.ngray.jnet.recurrent.RecurrentNeuralNetworkException;
import com.ngray.jnet.recurrent.Sequence;
import com.ngray.jnet.recurrent.SequenceException;
import com.ngray.jnet.recurrent.SequenceGenerator;


public final class TestRNNTrainingWithLargeDataSet {

	@Test
	public void test() throws SequenceException, DictionaryException, FileNotFoundException, IOException, RMSPropException, RecurrentNeuralNetworkException {
		DataSet dataSet = createTrainingData();
		RecurrentNeuralNetwork.Builder builder = new RecurrentNeuralNetwork.Builder();
		builder.setInputDimension(Alphabet.getSize());
		builder.setOutputDimension(Alphabet.getSize());
		builder.setNetworkDimension(256);
		RecurrentNeuralNetwork rnn = builder.build();
		
		//RecurrentNeuralNetwork rnn = new RecurrentNeuralNetwork(256, Alphabet.getSize(), Alphabet.getSize(), 1.0, 0.9);
		int epochs = 30;
		int batchSize = 25;
		double learningRate = 0.001;
		int maxBackPropSteps = 2;
		double lossThreshold = 1.0;
		SequenceGenerator<Character> seqGenerator = Alphabet.createSequenceGenerator();
		Callback callback = new CallbackGenerate(seqGenerator.getInputSequence(convert("Ask not")), 200);
		System.out.println("Begin training...");
		rnn.train(dataSet, epochs, batchSize, learningRate, maxBackPropSteps, lossThreshold, callback);
		System.out.println("End training...");
		callback.call(rnn);
	}
	
	private DataSet createTrainingData() throws FileNotFoundException, IOException, SequenceException, DictionaryException {
		String testDataFileName = "./src/test/resources/JFK.txt";
		System.out.println("Reading training data...");
		String text = "";
		try (BufferedReader reader = new BufferedReader(new FileReader(testDataFileName))) {
			String nextLine = null;
			while ((nextLine = reader.readLine()) != null) {
				text += nextLine;
			}
		}
		
		System.out.println("Creating data set...");
		List<Sequence> inputs = new ArrayList<Sequence>();
		Map<Sequence, Sequence> outputs = new HashMap<>();
		SequenceGenerator<Character> seqGen = Alphabet.createSequenceGenerator();
		int subStringLength = 25;
		for (int i = 0; i < text.length(); ++i) {
			String subString = text.substring(i,  Math.min(i+subStringLength,text.length()));
			Sequence input = seqGen.getInputSequence(convert(subString));
			
		
			// the output is the sequence shifted right by one character, except for the last
			// output which we terminate with a random character (hence why we use getInputSequence here)
			String outSubString = text.substring(i+1,  Math.min(i+subStringLength+1,text.length()));
			if (outSubString.length() < subString.length()) {
				outSubString += " ";
			}
			Sequence output = seqGen.getInputSequence(convert(outSubString));
			if (input.getLength() == subStringLength) {
				System.out.println(subString);
				inputs.add(input);
				outputs.put(input, output);
			}
		}
		System.out.println("Data set size = " + inputs.size());
		return new DataSet(inputs, outputs);

	}
	
	public static List<Character> convert(String s) {
		List<Character> list = new ArrayList<>();
		
	    for (Character item: s.toCharArray()) {
	        list.add(item);
	    }
	    return list;
	}
	
	

}
