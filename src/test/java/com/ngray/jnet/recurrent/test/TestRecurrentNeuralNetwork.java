package com.ngray.jnet.recurrent.test;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.Test;

import com.ngray.jnet.algebra.Vector;
import com.ngray.jnet.recurrent.DataSet;
import com.ngray.jnet.recurrent.DictionaryException;
import com.ngray.jnet.recurrent.RecurrentNeuralNetwork;
import com.ngray.jnet.recurrent.Sequence;
import com.ngray.jnet.recurrent.SequenceException;
import com.ngray.jnet.recurrent.SequenceGenerator;

public class TestRecurrentNeuralNetwork {

	private static String[] testSentences = {
			// TODO
			"This is a dull test example."
	};
	
	
	private static String[] testData = {			
			"The notes of the Tristan chord are not unusual; they could be respelled enharmonically to form a common half-diminished seventh chord.",// What distinguishes the chord is its unusual relationship to the implied key of its surroundings.",
			"Much has been written about the Tristan chord's possible harmonic functions or voice leading (melodic function)."//, and the motif has been interpreted in various ways. For instance, Arnold Schering traces the development of the Tristan chord through ten intermediate steps, beginning with the Phrygian cadence (iv6-V).",			
	};
	
	private List<Character> convert(String s) {
		List<Character> list = new ArrayList<>();
		
	    for (Character item: s.toCharArray()) {
	        list.add(item);
	    }
	    return list;
	}
	
	@Test
	public void testEvaluate() throws SequenceException, DictionaryException {
		SequenceGenerator<Character> seqGenerator = Alphabet.createSequenceGenerator();
		RecurrentNeuralNetwork rnn = new RecurrentNeuralNetwork(100, Alphabet.getSize(), Alphabet.getSize());
		Sequence input = seqGenerator.getInputSequence(convert(testSentences[0]));
		Sequence output = rnn.evaluate(input);
		String s = Alphabet.convert(seqGenerator.convertSequence(rnn.predict(output)));
		System.out.println(s);
		assert (s.length() == testSentences[0].length());
	}

	@Test
	public void testCalculateLoss() throws SequenceException, DictionaryException {
		SequenceGenerator<Character> seqGenerator = Alphabet.createSequenceGenerator();
		RecurrentNeuralNetwork rnn = new RecurrentNeuralNetwork(100, Alphabet.getSize(), Alphabet.getSize());
		Sequence input = seqGenerator.getInputSequence(convert(testSentences[0]));
		Sequence expectedOutput = seqGenerator.getOutputSequence(convert(testSentences[0]));
		Sequence output = rnn.evaluate(input);
		double loss = rnn.calculateLoss(output, expectedOutput);
		double expectedLoss = Math.log(Alphabet.SYMBOLS.length);
		
		String s = Alphabet.convert(seqGenerator.convertSequence(rnn.predict(output)));
		System.out.println(s);
		System.out.println("Loss is: " + loss + ", Expected loss: " + Math.log(Alphabet.SYMBOLS.length));
		assert (Math.abs(loss - expectedLoss) < 1.0);
	}
	
	@Test(expected=SequenceException.class)
	public void testEvaluationNullInputSequence() throws SequenceException, DictionaryException {
		RecurrentNeuralNetwork rnn = new RecurrentNeuralNetwork(100, Alphabet.getSize(), Alphabet.getSize());
		Sequence input = null;
		rnn.evaluate(input);
	}
	
	@Test
	public void testEvaluateEmptyInputSequence() throws SequenceException, DictionaryException {
		RecurrentNeuralNetwork rnn = new RecurrentNeuralNetwork(100, Alphabet.getSize(), Alphabet.getSize());
		Sequence input = Sequence.newSequence(new ArrayList<Vector>());
		Sequence output = rnn.evaluate(input);
		assertTrue(output.getLength() == 0);
	}
	
	@Test
	public void testCalculateLossEmptyInputSequence() throws SequenceException, DictionaryException {
		SequenceGenerator<Character> seqGenerator = Alphabet.createSequenceGenerator();
		RecurrentNeuralNetwork rnn = new RecurrentNeuralNetwork(100, Alphabet.getSize(), Alphabet.getSize());
		Sequence input = seqGenerator.getInputSequence(convert(""));
		Sequence expectedOutput = seqGenerator.getOutputSequence(convert(""));
		Sequence output = rnn.evaluate(input);
		double loss = rnn.calculateLoss(output, expectedOutput);
		assertTrue(loss == 0.0);
	}
	
	@Test
	public void testTrain() throws DictionaryException, SequenceException {
		SequenceGenerator<Character> seqGenerator = Alphabet.createSequenceGenerator();
		RecurrentNeuralNetwork rnn = new RecurrentNeuralNetwork(200, Alphabet.getSize(), Alphabet.getSize(), 0.01);
		List<Sequence> inputs = new ArrayList<>();
		Map<Sequence, Sequence> outputs = new HashMap<>();
		for (String s : testData) {
			Sequence input = seqGenerator.getInputSequence(convert(s));
			Sequence output = seqGenerator.getOutputSequence(convert(s));
			inputs.add(input);
			outputs.put(input,output);
		}
		DataSet dataSet = new DataSet(inputs, outputs);
		
		int epochs = 40;
		int batchSize = 2;
		double learningRate = 0.25;
		int maxBackPropSteps = 2;
		rnn.train(dataSet, epochs, batchSize, learningRate, maxBackPropSteps);
		
		System.out.println("Testing predictive power on training data");
		Sequence output = rnn.evaluate(seqGenerator.getInputSequence(convert(testData[0])));
		String s = Alphabet.convert(seqGenerator.convertSequence(rnn.predict(output)));
		System.out.println(s);
		
		output = rnn.evaluate(seqGenerator.getInputSequence(convert(testData[1])));
		s = Alphabet.convert(seqGenerator.convertSequence(rnn.predict(output)));
		System.out.println(s);
		
		System.out.println("Testing generate() - don't expect this to work much in this test");
		output = rnn.generate(seqGenerator.getInputSequence(convert("Much")), 50);
		s = Alphabet.convert(seqGenerator.convertSequence(rnn.predict(output)));
		System.out.println(s);
		
	}

}
